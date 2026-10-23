//! LoRA adapter safetensors (the same files the Metal runtime reads, see runtime/lora.rs) to a GGUF LoRA for
//! libllama. A and B become fp16 with alpha / r and the scale folded into B, as on Metal, and the GGUF's
//! alpha is 0 so libllama applies B as is. Names follow llama.cpp's qwen35 tensors.

use std::collections::HashMap;
use std::path::Path;

use half::f16;

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn read_safetensors(path: &Path) -> Result<HashMap<String, Tensor>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 8 {
        return Err("lora: not a safetensors file".into());
    }
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let header = b.get(8..8 + n).ok_or("lora: bad header")?;
    let h: serde_json::Value = serde_json::from_slice(header).map_err(|e| format!("lora: {e}"))?;
    let body = &b[8 + n..];
    let mut out = HashMap::new();
    for (k, v) in h.as_object().ok_or("lora: bad header")? {
        if k == "__metadata__" {
            continue;
        }
        let o = &v["data_offsets"];
        let (s, e) = (
            o[0].as_u64().unwrap_or(0) as usize,
            o[1].as_u64().unwrap_or(0) as usize,
        );
        let raw = body.get(s..e).ok_or(format!("lora: {k} out of range"))?;
        let data: Vec<f32> = match v["dtype"].as_str().unwrap_or("") {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            d => return Err(format!("lora: {k}: unsupported dtype {d}")),
        };
        let shape = v["shape"]
            .as_array()
            .ok_or("lora: shape")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        out.insert(k.clone(), Tensor { shape, data });
    }
    Ok(out)
}

/// alpha / r from a PEFT adapter_config.json.
fn peft_scaling(dir: &Path) -> Result<f32, String> {
    let p = dir.join("adapter_config.json");
    let bytes =
        std::fs::read(&p).map_err(|e| format!("lora: PEFT keys need {}: {e}", p.display()))?;
    let j: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("lora: {e}"))?;
    for k in ["rank_pattern", "alpha_pattern"] {
        if j[k].as_object().is_some_and(|o| !o.is_empty()) {
            return Err(format!("lora: {k} is not supported"));
        }
    }
    let r = j["r"].as_f64().ok_or("lora: adapter_config r")?;
    let alpha = j["lora_alpha"]
        .as_f64()
        .ok_or("lora: adapter_config lora_alpha")?;
    Ok(if j["use_rslora"].as_bool() == Some(true) {
        alpha / r.sqrt()
    } else {
        alpha / r
    } as f32)
}

/// GDN head layout from config.json: (key heads, value heads, value head dim, key head dim).
#[derive(Clone, Copy)]
pub struct Heads {
    pub k_heads: usize,
    pub v_heads: usize,
    pub v_dim: usize,
    pub k_dim: usize,
}

fn gguf_name(module: &str) -> Option<&'static str> {
    Some(match module {
        "linear_attn.in_proj_qkv" => "attn_qkv",
        "linear_attn.in_proj_z" => "attn_gate",
        "linear_attn.out_proj" => "ssm_out",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        _ => return None,
    })
}

/// llama.cpp stores GDN value heads tiled ([K0, K1, ..., K0, K1, ...]) where the checkpoint groups them by key
/// head. Index map for `n` features of `head_dim` each: new[i] = old[map[i]].
fn tiled(n_k: usize, rep: usize, head_dim: usize) -> Vec<usize> {
    let mut map = Vec::with_capacity(n_k * rep * head_dim);
    for r in 0..rep {
        for k in 0..n_k {
            for d in 0..head_dim {
                map.push((k * rep + r) * head_dim + d);
            }
        }
    }
    map
}

/// Writes the GGUF LoRA for the adapter at `path` to `out`, B multiplied by `scale`.
pub fn convert(path: &Path, scale: f32, heads: Heads, out: &Path) -> Result<usize, String> {
    let tensors = read_safetensors(path)?;
    let peft = tensors.keys().any(|k| k.ends_with(".weight"));
    let fold = if peft {
        peft_scaling(path.parent().unwrap_or(Path::new(".")))?
    } else {
        1.0
    } * scale;
    let rep = heads.v_heads / heads.k_heads.max(1);
    let reorder = rep > 1;

    // (gguf base name, A [r, in], B [out, r]) in a stable order
    let mut pairs: Vec<(String, Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>)> = Vec::new();
    let mut keys: Vec<&String> = tensors.keys().collect();
    keys.sort();
    let mut rank = 0;
    for k in keys {
        let k2 = k.strip_suffix(".weight").unwrap_or(k);
        let Some(stem) = k2.strip_suffix(".lora_A") else {
            if k2.ends_with(".lora_B") {
                continue;
            }
            return Err(format!("lora: unexpected tensor {k}"));
        };
        let i = stem
            .find("layers.")
            .ok_or(format!("lora: {k} is not a layer linear"))?;
        let rest = &stem[i + "layers.".len()..];
        let (layer, module) = rest.split_once('.').ok_or(format!("lora: bad name {k}"))?;
        let layer: usize = layer
            .parse()
            .map_err(|_| format!("lora: bad layer in {k}"))?;
        let name = gguf_name(module).ok_or(format!("lora: unsupported module {module}"))?;
        let b_key = if peft {
            format!("{stem}.lora_B.weight")
        } else {
            format!("{stem}.lora_B")
        };
        let a = &tensors[k];
        let b = tensors
            .get(&b_key)
            .ok_or(format!("lora: {stem} needs lora_A and lora_B"))?;
        let r = a.shape[0];
        if a.shape.len() != 2 || b.shape.len() != 2 || b.shape[1] != r {
            return Err(format!("lora: {stem}: A {:?} B {:?}", a.shape, b.shape));
        }
        if rank != 0 && r != rank {
            return Err("lora: mixed ranks are not supported".into());
        }
        rank = r;
        let (n_in, n_out) = (a.shape[1], b.shape[0]);
        let mut a_data = a.data.clone();
        let mut b_data: Vec<f32> = b.data.iter().map(|x| x * fold).collect();
        if reorder {
            let vd = heads.v_heads * heads.v_dim;
            match name {
                // V rows of qkv (after q and k), all rows of z: B rows
                "attn_qkv" | "attn_gate" => {
                    let start = if name == "attn_qkv" {
                        2 * heads.k_heads * heads.k_dim
                    } else {
                        0
                    };
                    if n_out != start + vd {
                        return Err(format!(
                            "lora: {stem}: {n_out} rows, expected {}",
                            start + vd
                        ));
                    }
                    let map = tiled(heads.k_heads, rep, heads.v_dim);
                    let old = b_data.clone();
                    for (i, &src) in map.iter().enumerate() {
                        b_data[(start + i) * r..(start + i + 1) * r]
                            .copy_from_slice(&old[(start + src) * r..(start + src + 1) * r]);
                    }
                }
                // out_proj reads the tiled activation: A columns
                "ssm_out" => {
                    if n_in != vd {
                        return Err(format!("lora: {stem}: {n_in} inputs, expected {vd}"));
                    }
                    let map = tiled(heads.k_heads, rep, heads.v_dim);
                    let old = a_data.clone();
                    for row in 0..r {
                        for (i, &src) in map.iter().enumerate() {
                            a_data[row * n_in + i] = old[row * n_in + src];
                        }
                    }
                }
                _ => {}
            }
        }
        pairs.push((
            format!("blk.{layer}.{name}.weight"),
            a_data,
            vec![n_in, r],
            b_data,
            vec![r, n_out],
        ));
    }
    if pairs.is_empty() {
        return Err("lora: no adapted linears".into());
    }
    write_gguf(out, &pairs)?;
    Ok(pairs.len())
}

const ALIGN: usize = 32;

fn gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend((s.len() as u64).to_le_bytes());
    buf.extend(s.as_bytes());
}

#[allow(clippy::type_complexity)]
fn write_gguf(
    out: &Path,
    pairs: &[(String, Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>)],
) -> Result<(), String> {
    const STRING: u32 = 8;
    const F32: u32 = 6;
    const TYPE_F16: u32 = 1;
    let kv: [(&str, Result<&str, f32>); 4] = [
        ("general.architecture", Ok("qwen35")),
        ("general.type", Ok("adapter")),
        ("adapter.type", Ok("lora")),
        ("adapter.lora.alpha", Err(0.0)),
    ];
    let mut head = Vec::new();
    head.extend(b"GGUF");
    head.extend(3u32.to_le_bytes());
    head.extend(((pairs.len() * 2) as u64).to_le_bytes());
    head.extend((kv.len() as u64).to_le_bytes());
    for (key, value) in kv {
        gguf_string(&mut head, key);
        match value {
            Ok(s) => {
                head.extend(STRING.to_le_bytes());
                gguf_string(&mut head, s);
            }
            Err(f) => {
                head.extend(F32.to_le_bytes());
                head.extend(f.to_le_bytes());
            }
        }
    }
    let mut data: Vec<u8> = Vec::new();
    for (name, a, a_shape, b, b_shape) in pairs {
        for (suffix, values, shape) in [("lora_a", a, a_shape), ("lora_b", b, b_shape)] {
            while data.len() % ALIGN != 0 {
                data.push(0);
            }
            gguf_string(&mut head, &format!("{name}.{suffix}"));
            head.extend((shape.len() as u32).to_le_bytes());
            for &d in shape {
                head.extend((d as u64).to_le_bytes());
            }
            head.extend(TYPE_F16.to_le_bytes());
            head.extend((data.len() as u64).to_le_bytes());
            data.extend(
                values
                    .iter()
                    .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes()),
            );
        }
    }
    while head.len() % ALIGN != 0 {
        head.push(0);
    }
    head.extend(data);
    std::fs::write(out, head).map_err(|e| format!("{}: {e}", out.display()))
}
