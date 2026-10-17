//! Qwen3-VL style vision tower (24 ViT blocks, patch 16, 2x2 merger to the text hidden size) in fp16 like mlx-vlm runs it.
//! Preprocessing follows preprocessor_config.json: smart resize to multiples of 32 within the pixel bounds,
//! bicubic, rescale 1/255, mean/std 0.5, temporal patch 2 (the frame repeated).

use super::gpu::{contents, Buffer, Gpu, B};
use super::weights::Weights;
use half::f16;
use std::path::Path;

pub struct ImageOut {
    /// [merged tokens, out] row-major (out = text hidden size), values rounded through fp16.
    pub embeds: Vec<f32>,
    /// (t, patch rows, patch cols) like image_grid_thw.
    pub grid: (usize, usize, usize),
}

struct Block {
    n1: (usize, usize),
    qkv: (usize, usize),
    proj: (usize, usize),
    n2: (usize, usize),
    fc1: (usize, usize),
    fc2: (usize, usize),
}

pub struct Vision {
    blocks: Vec<Block>,
    patch: (usize, usize),
    pos: Vec<f16>,
    mnorm: (usize, usize),
    mfc1: (usize, usize),
    mfc2: (usize, usize),
    out: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
}

const H: usize = 1024;

fn wb(w: &Weights, p: &str) -> (usize, usize) {
    (w.off(&format!("{p}.weight")), w.off(&format!("{p}.bias")))
}

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// Qwen2-VL smart_resize with factor 32.
pub fn smart_resize(h: usize, w: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let f = 32.0;
    let (hf, wf) = (h as f64, w as f64);
    let mut hb = round_half_even(hf / f) * f;
    let mut wbar = round_half_even(wf / f) * f;
    if hb * wbar > max_pixels as f64 {
        let beta = (hf * wf / max_pixels as f64).sqrt();
        hb = f.max((hf / beta / f).floor() * f);
        wbar = f.max((wf / beta / f).floor() * f);
    } else if hb * wbar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wbar = (wf * beta / f).ceil() * f;
    }
    (hb as usize, wbar as usize)
}

impl Vision {
    pub fn new(_gpu: &Gpu, w: &Weights) -> Vision {
        let blocks = (0..24)
            .map(|i| {
                let p = format!("vision_tower.blocks.{i}.");
                Block {
                    n1: wb(w, &format!("{p}norm1")),
                    qkv: wb(w, &format!("{p}attn.qkv")),
                    proj: wb(w, &format!("{p}attn.proj")),
                    n2: wb(w, &format!("{p}norm2")),
                    fc1: wb(w, &format!("{p}mlp.linear_fc1")),
                    fc2: wb(w, &format!("{p}mlp.linear_fc2")),
                }
            })
            .collect();
        let pos = w
            .bytes("vision_tower.pos_embed.weight")
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        Vision {
            blocks,
            patch: wb(w, "vision_tower.patch_embed.proj"),
            pos,
            mnorm: wb(w, "vision_tower.merger.norm"),
            mfc1: wb(w, "vision_tower.merger.linear_fc1"),
            mfc2: wb(w, "vision_tower.merger.linear_fc2"),
            out: w.get("vision_tower.merger.linear_fc2.weight").shape[0],
            min_pixels: 65536,
            max_pixels: 16777216,
        }
    }

    /// Decodes and preprocesses an image file into patch rows [N, 1536] (fp16) in merge order.
    pub fn preprocess(&self, path: &Path) -> Result<(Vec<f16>, usize, usize), String> {
        let img = image::open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .to_rgb8();
        let (w0, h0) = img.dimensions();
        let (hb, wbar) = smart_resize(h0 as usize, w0 as usize, self.min_pixels, self.max_pixels);
        let img = if (hb, wbar) != (h0 as usize, w0 as usize) {
            image::imageops::resize(
                &img,
                wbar as u32,
                hb as u32,
                image::imageops::FilterType::CatmullRom,
            )
        } else {
            img
        };
        let (gh, gw) = (hb / 16, wbar / 16);
        let n = gh * gw;
        let mut x = vec![f16::ZERO; n * 1536];
        let raw = img.as_raw();
        let mut p = 0;
        for bh in 0..gh / 2 {
            for bw in 0..gw / 2 {
                for r2 in 0..2 {
                    for c2 in 0..2 {
                        let (row, col) = (bh * 2 + r2, bw * 2 + c2);
                        let dst = &mut x[p * 1536..(p + 1) * 1536];
                        for y in 0..16 {
                            for xx in 0..16 {
                                let src = ((row * 16 + y) * wbar + col * 16 + xx) * 3;
                                for c in 0..3 {
                                    let v = (raw[src + c] as f32 * (1.0 / 255.0) - 0.5) / 0.5;
                                    let hv = f16::from_f32(v);
                                    let i = (y * 16 + xx) * 3 + c;
                                    dst[i] = hv;
                                    dst[768 + i] = hv;
                                }
                            }
                        }
                        p += 1;
                    }
                }
            }
        }
        Ok((x, gh, gw))
    }

    /// Bilinear position embedding interpolation (fp16 arithmetic like mlx-vlm), merge order, [N, 1024].
    fn pos_embed(&self, gh: usize, gw: usize) -> Vec<f16> {
        let side = 48usize;
        let lin = |n: usize| -> Vec<f32> {
            let step = if n > 1 {
                (side - 1) as f32 / (n - 1) as f32
            } else {
                0.0
            };
            (0..n).map(|i| i as f32 * step).collect()
        };
        let (hi, wi) = (lin(gh), lin(gw));
        let mut grid = vec![f16::ZERO; gh * gw * H];
        for r in 0..gh {
            let hf = hi[r] as usize;
            let hc = (hf + 1).min(side - 1);
            let dh = hi[r] - hf as f32;
            for c in 0..gw {
                let wf = wi[c] as usize;
                let wc = (wf + 1).min(side - 1);
                let dw = wi[c] - wf as f32;
                let idx = [
                    hf * side + wf,
                    hf * side + wc,
                    hc * side + wf,
                    hc * side + wc,
                ];
                let wt = [
                    (1.0 - dh) * (1.0 - dw),
                    (1.0 - dh) * dw,
                    dh * (1.0 - dw),
                    dh * dw,
                ]
                .map(f16::from_f32);
                let out = &mut grid[(r * gw + c) * H..][..H];
                for d in 0..H {
                    let mut acc = f16::ZERO;
                    for k in 0..4 {
                        let term =
                            f16::from_f32(self.pos[idx[k] * H + d].to_f32() * wt[k].to_f32());
                        acc = if k == 0 {
                            term
                        } else {
                            f16::from_f32(acc.to_f32() + term.to_f32())
                        };
                    }
                    out[d] = acc;
                }
            }
        }
        let mut out = vec![f16::ZERO; gh * gw * H];
        let mut p = 0;
        for bh in 0..gh / 2 {
            for bw in 0..gw / 2 {
                for r2 in 0..2 {
                    for c2 in 0..2 {
                        let src = ((bh * 2 + r2) * gw + bw * 2 + c2) * H;
                        out[p * H..(p + 1) * H].copy_from_slice(&grid[src..src + H]);
                        p += 1;
                    }
                }
            }
        }
        out
    }

    /// Rotary angles per patch: 16 row frequencies then 16 column frequencies.
    fn angles(gh: usize, gw: usize) -> Vec<f32> {
        let inv: Vec<f32> = (0..16)
            .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / 32.0))
            .collect();
        let mut a = Vec::with_capacity(gh * gw * 32);
        for bh in 0..gh / 2 {
            for bw in 0..gw / 2 {
                for r2 in 0..2 {
                    for c2 in 0..2 {
                        let (row, col) = ((bh * 2 + r2) as f32, (bw * 2 + c2) as f32);
                        a.extend(inv.iter().map(|f| row * f));
                        a.extend(inv.iter().map(|f| col * f));
                    }
                }
            }
        }
        a
    }

    pub fn run(&self, gpu: &Gpu, wbuf: &Buffer, path: &Path) -> Result<ImageOut, String> {
        let (x, gh, gw) = self.preprocess(path)?;
        self.run_patches(gpu, wbuf, &x, gh, gw)
    }

    pub fn run_patches(
        &self,
        gpu: &Gpu,
        wbuf: &Buffer,
        x: &[f16],
        gh: usize,
        gw: usize,
    ) -> Result<ImageOut, String> {
        let n = gh * gw;
        let m = n / 4;
        let upload = |v: &[u8]| {
            let b = gpu.buffer(v.len());
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), contents::<u8>(&b), v.len()) };
            b
        };
        let as_bytes = |v: &[f16]| {
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }.to_vec()
        };
        let xb = upload(&as_bytes(x));
        let h = upload(&as_bytes(&self.pos_embed(gh, gw)));
        let ang = Self::angles(gh, gw);
        let angb =
            upload(unsafe { std::slice::from_raw_parts(ang.as_ptr() as *const u8, ang.len() * 4) });
        let t1 = gpu.buffer(n * H * 2);
        let big = gpu.buffer(n * 4096 * 2);
        let (q, k, v) = (
            gpu.buffer(n * H * 2),
            gpu.buffer(n * H * 2),
            gpu.buffer(n * H * 2),
        );
        let dout = self.out;
        let out = gpu.buffer(m * dout * 2);
        let gemm = |cmd: &mut super::gpu::Cmd,
                    xin: &Buffer,
                    w: (usize, usize),
                    y: &Buffer,
                    rows: usize,
                    nout: usize,
                    kin: usize,
                    act: u32,
                    add: bool| {
            cmd.run(
                "v_gemm",
                &[
                    B(xin, 0).into(),
                    B(wbuf, w.0).into(),
                    B(wbuf, w.1).into(),
                    B(y, 0).into(),
                    (rows as u32).into(),
                    (nout as u32).into(),
                    (kin as u32).into(),
                    act.into(),
                    (add as u32).into(),
                ],
                (nout / 64, rows.div_ceil(32), 1),
                (128, 1, 1),
            );
        };
        let ln = |cmd: &mut super::gpu::Cmd, xin: &Buffer, w: (usize, usize), y: &Buffer| {
            cmd.run(
                "v_layernorm",
                &[
                    B(xin, 0).into(),
                    B(wbuf, w.0).into(),
                    B(wbuf, w.1).into(),
                    B(y, 0).into(),
                    (H as u32).into(),
                ],
                (n, 1, 1),
                (256, 1, 1),
            );
        };
        let mut cmd = gpu.begin();
        gemm(&mut cmd, &xb, self.patch, &h, n, H, 1536, 0, true);
        for b in &self.blocks {
            ln(&mut cmd, &h, b.n1, &t1);
            gemm(&mut cmd, &t1, b.qkv, &big, n, 3 * H, H, 0, false);
            cmd.run(
                "v_rope",
                &[
                    B(&big, 0).into(),
                    B(&angb, 0).into(),
                    B(&q, 0).into(),
                    B(&k, 0).into(),
                    B(&v, 0).into(),
                    (n as u32).into(),
                ],
                (n, 48, 1),
                (64, 1, 1),
            );
            cmd.run(
                "v_flash",
                &[
                    B(&q, 0).into(),
                    B(&k, 0).into(),
                    B(&v, 0).into(),
                    B(&t1, 0).into(),
                    (n as u32).into(),
                ],
                (n.div_ceil(32), 16, 1),
                (128, 1, 1),
            );
            gemm(&mut cmd, &t1, b.proj, &h, n, H, H, 0, true);
            ln(&mut cmd, &h, b.n2, &t1);
            gemm(&mut cmd, &t1, b.fc1, &big, n, 4096, H, 1, false);
            gemm(&mut cmd, &big, b.fc2, &h, n, H, 4096, 0, true);
        }
        ln(&mut cmd, &h, self.mnorm, &t1);
        gemm(&mut cmd, &t1, self.mfc1, &big, m, 4096, 4096, 2, false);
        gemm(&mut cmd, &big, self.mfc2, &out, m, dout, 4096, 0, false);
        cmd.finish();
        let o = contents::<u16>(&out);
        let embeds = (0..m * dout)
            .map(|i| f16::from_bits(unsafe { *o.add(i) }).to_f32())
            .collect();
        Ok(ImageOut {
            embeds,
            grid: (1, gh, gw),
        })
    }
}
