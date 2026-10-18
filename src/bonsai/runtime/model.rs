//! Qwen3.5 hybrid text model (Gated DeltaNet + gated attention layers) encoded as Metal dispatches.
//! All sizes come from config.json (Ternary Bonsai 2 2B and 4B).

use super::gpu::{contents, Arg, Buffer, Cmd, Gpu, B};
use super::lora::{Group, Lora};
use super::weights::Weights;

/// Largest chunk of tokens one forward call takes; longer prompts are split.
pub const TMAX: usize = 512;
/// Largest speculative verify chunk (rows whose logits and GDN inputs are kept per layer).
pub const VMAX: usize = 16;
const ARGMAX_GROUPS: usize = 128;
/// Rotation kernels handle this many values per threadgroup (kernels.metal CHUNK).
const CHUNK: usize = 2048;
/// Prefill attention splits keys into 512-wide chunks; ctx <= 4096 gives at most 8.
const MAX_SPLITS: usize = 8;

/// Model sizes from config.json.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub layer_is_attn: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    pub rope_theta: f64,
    pub rotary_dims: usize,
    pub mrope: Vec<usize>,
    pub eps: f64,
    pub vision_out: usize,
}

impl Config {
    pub fn load(path: &std::path::Path) -> Result<Config, String> {
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
            .map_err(|e| e.to_string())?;
        let t = &j["text_config"];
        let u = |v: &serde_json::Value, k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("config: missing {k}"));
        let r = &t["rope_parameters"];
        let head_dim = u(t, "head_dim")?;
        Ok(Config {
            hidden: u(t, "hidden_size")?,
            intermediate: u(t, "intermediate_size")?,
            vocab: u(t, "vocab_size")?,
            layer_is_attn: t["layer_types"].as_array().ok_or("config: layer_types")?.iter().map(|v| v == "full_attention").collect(),
            heads: u(t, "num_attention_heads")?,
            kv_heads: u(t, "num_key_value_heads")?,
            head_dim,
            lin_k_heads: u(t, "linear_num_key_heads")?,
            lin_v_heads: u(t, "linear_num_value_heads")?,
            lin_k_dim: u(t, "linear_key_head_dim")?,
            lin_v_dim: u(t, "linear_value_head_dim")?,
            conv_kernel: u(t, "linear_conv_kernel_dim")?,
            rope_theta: r["rope_theta"].as_f64().ok_or("config: rope_theta")?,
            rotary_dims: (head_dim as f64 * r["partial_rotary_factor"].as_f64().ok_or("config: partial_rotary_factor")?) as usize,
            mrope: r["mrope_section"].as_array().ok_or("config: mrope_section")?.iter().map(|v| v.as_u64().unwrap() as usize).collect(),
            eps: t["rms_norm_eps"].as_f64().ok_or("config: rms_norm_eps")?,
            vision_out: u(&j["vision_config"], "out_hidden_size")?,
        })
    }

    /// GDN conv channels: q and k (key heads) plus v (value heads).
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_heads * self.lin_k_dim + self.gdn_width()
    }
    pub fn gdn_width(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }
    pub fn attn_width(&self) -> usize {
        self.heads * self.head_dim
    }
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    fn gdn_state(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim * self.lin_k_dim
    }
    fn conv_state(&self) -> usize {
        (self.conv_kernel - 1) * self.conv_dim()
    }

    /// Shapes the kernels fix (shared by 2B and 4B) plus the divisibility the dispatch grids need.
    pub fn check(&self) -> Result<(), String> {
        let fixed = [
            ("head_dim", self.head_dim, 256),
            ("linear_key_head_dim", self.lin_k_dim, 128),
            ("linear_value_head_dim", self.lin_v_dim, 128),
            ("linear_conv_kernel_dim", self.conv_kernel, 4),
            ("q heads per kv head", self.heads / self.kv_heads.max(1), 4),
            ("vision out_hidden_size", self.vision_out, self.hidden),
        ];
        for (k, got, w) in fixed {
            if got != w {
                return Err(format!("unsupported config: {k} = {got}, kernels need {w}"));
            }
        }
        let div = [
            ("hidden_size", self.hidden, 512),
            ("intermediate_size", self.intermediate, 512),
            ("attention width", self.attn_width(), 512),
            ("GDN value width", self.gdn_width(), CHUNK),
            ("GDN conv width", self.conv_dim(), 128),
            ("kv width", self.kv_width(), 64),
            ("linear_num_value_heads", self.lin_v_heads, 2),
        ];
        for (k, got, m) in div {
            if got % m != 0 {
                return Err(format!("unsupported config: {k} = {got} is not a multiple of {m}"));
            }
        }
        if self.heads % self.kv_heads != 0 || self.lin_v_heads % self.lin_k_heads != 0 {
            return Err("unsupported config: heads not divisible by kv heads".into());
        }
        let half = self.rotary_dims / 2;
        if self.rotary_dims % 2 != 0 || self.rotary_dims > self.head_dim || self.mrope.len() != 3 || self.mrope.iter().sum::<usize>() != half {
            return Err("unsupported config: rotary dims / mrope sections".into());
        }
        if self.eps != 1e-6 {
            return Err("unsupported config: rms_norm_eps differs from the kernels (1e-6)".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Lin {
    w: usize,
    s: usize,
    signs: usize,
    pub o: usize,
    pub k: usize,
}

enum Mixer {
    Gdn {
        qkv: Lin,
        z: Lin,
        out: Lin,
        a: usize,
        b: usize,
        alog: usize,
        dtb: usize,
        conv: usize,
        norm: usize,
        idx: usize,
    },
    Attn {
        q: Lin,
        k: Lin,
        v: Lin,
        o: Lin,
        qn: usize,
        kn: usize,
        idx: usize,
    },
}

struct Layer {
    in_norm: usize,
    post_norm: usize,
    mixer: Mixer,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// Recurrent and cache state. Snapshots are full copies made on the GPU.
pub struct State {
    gdn: Buffer,
    conv: Buffer,
    kc: Buffer,
    vc: Buffer,
    /// Tokens in the KV cache.
    pub len: usize,
    /// Next mrope position (differs from len after an image).
    pub pos: i32,
}

struct Scratch {
    h: Buffer,
    xn: Buffer,
    xr: Buffer,
    a1: Buffer,
    a2: Buffer,
    qkvc: Buffer,
    y: Buffer,
    ab: Buffer,
    kp: Buffer,
    vp: Buffer,
    q: Buffer,
    ids: Buffer,
    pos: Buffer,
    logits: Buffer,
    pv: Buffer,
    pi: Buffer,
    apart: Buffer,
    out: Buffer,
    // Pipelined decode: token ring (2 slots) and position ring (2 x 3).
    dtok: Buffer,
    dpos: Buffer,
    // Speculative verify: per-layer GDN inputs and candidate states.
    v_qkv: Buffer,
    v_qkvc: Buffer,
    v_ab: Buffer,
    v_gdn: Buffer,
    v_conv: Buffer,
}

/// LoRA adapter plus its scratch (stacked A outputs u [T, rows]).
struct LoraState {
    lora: Lora,
    u: Buffer,
}

pub struct Text {
    pub cfg: Config,
    layers: Vec<Layer>,
    embed: Lin,
    final_norm: usize,
    inv_freq: Buffer,
    s: Scratch,
    pub state: State,
    pub ctx: usize,
    logit_rows: usize,
    pub n_gdn: usize,
    pub n_attn: usize,
    lora: Option<LoraState>,
}

fn lin(w: &Weights, name: &str) -> Lin {
    let t = w.get(&format!("{name}.weight"));
    Lin {
        w: t.off,
        s: w.off(&format!("{name}.scales")),
        signs: w.off(&format!("{name}.signs")),
        o: t.shape[0],
        k: t.shape[1] * 16,
    }
}

fn expect(name: &str, l: Lin, o: usize, k: usize) -> Lin {
    assert!(l.o == o && l.k == k, "{name}: shape {}x{}, config says {o}x{k}", l.o, l.k);
    l
}

impl State {
    pub fn new(gpu: &Gpu, cfg: &Config, ctx: usize, n_gdn: usize, n_attn: usize) -> State {
        State {
            gdn: gpu.buffer(n_gdn * cfg.gdn_state() * 4),
            conv: gpu.buffer(n_gdn * cfg.conv_state() * 4),
            kc: gpu.buffer(n_attn * ctx * cfg.kv_width() * 4),
            vc: gpu.buffer(n_attn * ctx * cfg.kv_width() * 4),
            len: 0,
            pos: 0,
        }
    }
}

impl Text {
    pub fn new(gpu: &Gpu, w: &Weights, cfg: &Config, ctx: usize) -> Text {
        assert!(ctx <= 4096, "KV capacity is limited to 4096 positions");
        let p = "language_model.model.";
        let (d, ff, conv, gw, aw, kvw) = (cfg.hidden, cfg.intermediate, cfg.conv_dim(), cfg.gdn_width(), cfg.attn_width(), cfg.kv_width());
        let (mut na, mut ng) = (0, 0);
        let layers = (0..cfg.layer_is_attn.len())
            .map(|i| {
                let l = format!("{p}layers.{i}.");
                let li = |n: &str, o, k| expect(n, lin(w, &format!("{l}{n}")), o, k);
                let mixer = if cfg.layer_is_attn[i] {
                    na += 1;
                    Mixer::Attn {
                        q: li("self_attn.q_proj", 2 * aw, d),
                        k: li("self_attn.k_proj", kvw, d),
                        v: li("self_attn.v_proj", kvw, d),
                        o: li("self_attn.o_proj", d, aw),
                        qn: w.off(&format!("{l}self_attn.q_norm.weight")),
                        kn: w.off(&format!("{l}self_attn.k_norm.weight")),
                        idx: na - 1,
                    }
                } else {
                    ng += 1;
                    let a = format!("{l}linear_attn.");
                    Mixer::Gdn {
                        qkv: li("linear_attn.in_proj_qkv", conv, d),
                        z: li("linear_attn.in_proj_z", gw, d),
                        out: li("linear_attn.out_proj", d, gw),
                        a: w.off(&format!("{a}in_proj_a.weight")),
                        b: w.off(&format!("{a}in_proj_b.weight")),
                        alog: w.off(&format!("{a}A_log")),
                        dtb: w.off(&format!("{a}dt_bias")),
                        conv: w.off(&format!("{a}conv1d.weight")),
                        norm: w.off(&format!("{a}norm.weight")),
                        idx: ng - 1,
                    }
                };
                Layer {
                    in_norm: w.off(&format!("{l}input_layernorm.weight")),
                    post_norm: w.off(&format!("{l}post_attention_layernorm.weight")),
                    mixer,
                    gate: li("mlp.gate_proj", ff, d),
                    up: li("mlp.up_proj", ff, d),
                    down: li("mlp.down_proj", d, ff),
                }
            })
            .collect();
        let half = cfg.rotary_dims / 2;
        let inv_freq = gpu.buffer(half * 4);
        let f = contents::<f32>(&inv_freq);
        for i in 0..half {
            unsafe { *f.add(i) = 1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / cfg.rotary_dims as f32) };
        }
        let logit_rows = VMAX;
        let f4 = |n: usize| gpu.buffer(n * 4);
        let wide = ff.max(d).max(gw).max(aw);
        let s = Scratch {
            h: f4(TMAX * d),
            xn: f4(TMAX * d),
            xr: f4(TMAX * wide),
            a1: f4(TMAX * ff.max(conv).max(2 * aw)),
            a2: f4(TMAX * ff.max(gw)),
            qkvc: f4(TMAX * conv),
            y: f4(TMAX * gw.max(aw)),
            ab: f4(TMAX * 2 * cfg.lin_v_heads),
            kp: f4(TMAX * kvw),
            vp: f4(TMAX * kvw),
            q: f4(TMAX * aw),
            ids: f4(TMAX),
            pos: f4(3 * TMAX),
            logits: f4(logit_rows * cfg.vocab),
            pv: f4(logit_rows * ARGMAX_GROUPS),
            apart: f4(TMAX * cfg.heads * MAX_SPLITS * 258),
            pi: f4(logit_rows * ARGMAX_GROUPS),
            out: f4(logit_rows),
            dtok: f4(2),
            dpos: f4(6),
            v_qkv: f4(ng * VMAX * conv),
            v_qkvc: f4(ng * VMAX * conv),
            v_ab: f4(ng * VMAX * 2 * cfg.lin_v_heads),
            v_gdn: f4(ng * cfg.gdn_state()),
            v_conv: f4(ng * cfg.conv_state()),
        };
        Text {
            layers,
            embed: expect("embed_tokens", lin(w, &format!("{p}embed_tokens")), cfg.vocab, d),
            final_norm: w.off(&format!("{p}norm.weight")),
            inv_freq,
            s,
            state: State::new(gpu, cfg, ctx, ng, na),
            n_gdn: ng,
            n_attn: na,
            ctx,
            logit_rows,
            cfg: cfg.clone(),
            lora: None,
        }
    }

    /// Packed linears per layer in LoRA group order: mixer input, mixer output, MLP input, MLP down.
    /// Names are the module paths below `model.layers.N.`.
    pub fn linears(&self, layer: usize) -> [Vec<(&'static str, Lin)>; 4] {
        let l = &self.layers[layer];
        let (inp, out) = match &l.mixer {
            Mixer::Gdn { qkv, z, out, .. } => (
                vec![("linear_attn.in_proj_qkv", *qkv), ("linear_attn.in_proj_z", *z)],
                vec![("linear_attn.out_proj", *out)],
            ),
            Mixer::Attn { q, k, v, o, .. } => (
                vec![("self_attn.q_proj", *q), ("self_attn.k_proj", *k), ("self_attn.v_proj", *v)],
                vec![("self_attn.o_proj", *o)],
            ),
        };
        [inp, out, vec![("mlp.gate_proj", l.gate), ("mlp.up_proj", l.up)], vec![("mlp.down_proj", l.down)]]
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Installs (or with None removes) a LoRA adapter. The caller drops any cached state.
    pub fn set_lora(&mut self, gpu: &Gpu, lora: Option<Lora>) {
        self.lora = lora.map(|lora| LoraState {
            u: gpu.buffer(TMAX * lora.max_rows * 4),
            lora,
        });
    }

    /// (adapted linears, rank) of the installed adapter.
    pub fn lora_info(&self) -> Option<(usize, usize)> {
        self.lora.as_ref().map(|l| (l.lora.linears, l.lora.rank))
    }

    pub fn reset(&mut self) {
        self.state.len = 0;
        self.state.pos = 0;
        unsafe {
            std::ptr::write_bytes(contents::<u8>(&self.state.gdn), 0, self.n_gdn * self.cfg.gdn_state() * 4);
            std::ptr::write_bytes(contents::<u8>(&self.state.conv), 0, self.n_gdn * self.cfg.conv_state() * 4);
        }
    }

    /// Host access to the hidden input rows (for splicing image embeddings).
    pub fn hidden_ptr(&self) -> *mut f32 {
        contents::<f32>(&self.s.h)
    }
    pub fn hidden(&self) -> &Buffer {
        &self.s.h
    }

    pub fn write_ids(&self, ids: &[u32]) {
        let p = contents::<u32>(&self.s.ids);
        for (i, &t) in ids.iter().enumerate() {
            unsafe { *p.add(i) = t };
        }
    }

    /// Positions [3][T] for the chunk; text tokens use the same position in all three rows.
    pub fn write_pos(&self, pos: &[[i32; 3]]) {
        let t = pos.len();
        let p = contents::<i32>(&self.s.pos);
        for (i, ps) in pos.iter().enumerate() {
            for r in 0..3 {
                unsafe { *p.add(r * t + i) = ps[r] };
            }
        }
    }

    /// LoRA group `g` of `layer`, if the adapter has it.
    fn group(&self, layer: usize, g: usize) -> Option<(&Group, &LoraState)> {
        let ls = self.lora.as_ref()?;
        Some((ls.lora.groups.get(layer)?[g].as_ref()?, ls))
    }

    /// Extra arguments and threadgroups for a rotation kernel that computes LoRA A of `group` on the side,
    /// `rpt` rows per threadgroup (1 to LORA_RPT_MAX in kernels.metal).
    fn rot_lora<'a>(&'a self, group: Option<(&'a Group, &'a LoraState)>, rpt: usize) -> ([Arg<'a>; 3], usize) {
        match group {
            Some((g, ls)) => (
                [B(&ls.lora.buf, g.a).into(), B(&ls.u, 0).into(), (g.rows() as u32).into()],
                g.rows().div_ceil(rpt),
            ),
            None => ([B(&self.s.xr, 0).into(), B(&self.s.xr, 0).into(), 0u32.into()], 0),
        }
    }

    /// Matmuls sharing input `x`, plus the LoRA up projection of `group` (its u is already computed).
    #[allow(clippy::too_many_arguments)]
    fn qmm_lora(&self, cmd: &mut Cmd, wb: &Buffer, ls: &[(Lin, B)], x: B, t: usize, add: bool, group: Option<(&Group, &LoraState)>, ab: Option<[Arg; 5]>) {
        if t == 1 && ls.len() > 1 {
            assert!(!add);
            return self.qmm_multi(cmd, wb, ls, x, group, ab);
        }
        if t == 1 {
            let (l, y) = ls[0];
            let (b, u, r) = match group {
                Some((g, st)) => (B(&st.lora.buf, g.b[0]), B(&st.u, 0), g.r),
                None => (B(&self.s.xr, 0), B(&self.s.xr, 0), 0),
            };
            let args: [Arg; 10] = [
                B(wb, l.w).into(),
                B(wb, l.s).into(),
                x.into(),
                y.into(),
                (l.k as u32).into(),
                (l.o as u32).into(),
                (add as u32).into(),
                b.into(),
                u.into(),
                (r as u32).into(),
            ];
            return cmd.run("qmv4", &args, (l.o / 8, 1, 1), (64, 1, 1));
        }
        // T >= 4: the LoRA up projection is added in each matmul's store (u rows of R floats, member i at i * r);
        // T = 2, 3: by one lora_b pass after the group (qmvr's loop slows down with it inside)
        let post = t > 3;
        for (i, (l, y)) in ls.iter().enumerate() {
            let lora = group.filter(|_| post).map(|(g, st)| (B(&st.lora.buf, g.b[i]), B(&st.u, i * g.r * 4), g.r, g.rows()));
            self.qmm(cmd, wb, *l, x, *y, t, add, lora);
        }
        if let Some((g, st)) = group.filter(|_| !post) {
            let lb = &st.lora.buf;
            let mut args: Vec<Arg> = Vec::with_capacity(12);
            let mut os = [0u32; 3];
            for i in 0..3 {
                let j = i.min(ls.len() - 1);
                args.push(B(lb, g.b[j]).into());
                args.push(ls[j].1.into());
                if i < ls.len() {
                    os[i] = ls[i].0.o as u32;
                }
            }
            args.push(B(&st.u, 0).into());
            for o in os {
                args.push(o.into());
            }
            args.push((g.r as u32).into());
            args.push((g.rows() as u32).into());
            let total: u32 = os.iter().sum();
            cmd.run("lora_b", &args, ((total as usize).div_ceil(256), t, 1), (256, 1, 1));
        }
    }

    /// T > 1 matmul y (+)= x W^T, plus the LoRA up projection (B, u, r, row stride of u) in the store.
    #[allow(clippy::too_many_arguments)]
    fn qmm(&self, cmd: &mut Cmd, wb: &Buffer, l: Lin, x: B, y: B, t: usize, add: bool, lora: Option<(B, B, usize, usize)>) {
        assert!(t > 1, "t == 1 goes through qmm_lora");
        let (lb, lu, lr, ur) = lora.unwrap_or((B(&self.s.xr, 0), B(&self.s.xr, 0), 0, 0));
        let args = |c: usize, n: usize| -> [Arg; 12] {
            [
                B(wb, l.w).into(),
                B(wb, l.s).into(),
                B(x.0, x.1 + c * l.k * 4).into(),
                B(y.0, y.1 + c * l.o * 4).into(),
                (l.k as u32).into(),
                (l.o as u32).into(),
                (add as u32).into(),
                (n as u32).into(),
                lb.into(),
                B(lu.0, lu.1 + c * ur * 4).into(),
                (lr as u32).into(),
                (ur as u32).into(),
            ]
        };
        if t <= 3 {
            // qmv4's math per row (same results as a decode step); the T copies share each weight read
            cmd.run("qmvr", &args(0, t)[..8], (l.o / 8, 1, 1), (64 * t, 1, 1));
        } else if t <= 16 {
            // verify batches: MMA with X as the A operand, 8 rows per pass
            for c in (0..t).step_by(8) {
                cmd.run("qmvx", &args(c, (t - c).min(8)), (l.o / 32, 1, 1), (128, 1, 1));
            }
        } else {
            cmd.run("qmm3", &args(0, t), (l.o / 64, t.div_ceil(32), 1), (128, 1, 1));
        }
    }

    /// Single-token matvecs sharing input `x` in one dispatch, with the LoRA epilogue of `group`.
    /// `ab`: GDN a/b gate projection (Wa, Wb, normed input, output, NV) computed by extra threadgroups.
    fn qmm_multi(&self, cmd: &mut Cmd, wb: &Buffer, ls: &[(Lin, B)], x: B, group: Option<(&Group, &LoraState)>, ab: Option<[Arg; 5]>) {
        let mut args: Vec<Arg> = Vec::with_capacity(18);
        for i in 0..3 {
            let (l, y) = ls[i.min(ls.len() - 1)];
            args.push(B(wb, l.w).into());
            args.push(B(wb, l.s).into());
            args.push(y.into());
        }
        let o = |i: usize| if i < ls.len() { ls[i].0.o } else { 0 };
        let total: usize = ls.iter().map(|(l, _)| l.o).sum();
        args.push(x.into());
        args.push((ls[0].0.k as u32).into());
        args.push((o(0) as u32).into());
        args.push((o(1) as u32).into());
        match group {
            Some((g, st)) => {
                for i in 0..3 {
                    args.push(B(&st.lora.buf, g.b[i.min(g.b.len() - 1)]).into());
                }
                args.push(B(&st.u, 0).into());
                args.push((g.r as u32).into());
            }
            None => {
                for _ in 0..4 {
                    args.push(B(&self.s.xr, 0).into());
                }
                args.push(0u32.into());
            }
        }
        let dummy = || -> Arg { B(&self.s.xr, 0).into() };
        let a = ab.unwrap_or([dummy(), dummy(), dummy(), dummy(), 0u32.into()]);
        let extra = match a[4] {
            Arg::U32(v) => v as usize,
            _ => unreachable!(),
        };
        args.extend(a);
        args.push((total as u32).into());
        // NV extra threadgroups of 2 simdgroups: the 2 NV a/b rows
        cmd.run("qmv3", &args, (total / 8 + extra, 1, 1), (64, 1, 1));
    }

    pub fn encode_embed(&self, cmd: &mut Cmd, wb: &Buffer, t: usize) {
        self.encode_embed_from(cmd, wb, t, B(&self.s.ids, 0));
    }

    pub fn encode_embed_from(&self, cmd: &mut Cmd, wb: &Buffer, t: usize, ids: B) {
        let s = &self.s;
        let e = self.embed;
        let d = self.cfg.hidden;
        cmd.run(
            "embed",
            &[
                B(wb, e.w).into(),
                B(wb, e.s).into(),
                B(wb, e.signs).into(),
                ids.into(),
                B(&s.h, 0).into(),
                (d as u32).into(),
            ],
            (t, d.div_ceil(CHUNK), 1),
            (128, 1, 1),
        );
    }

    /// Encodes the layers for `t` rows already in `h`. `verify` keeps GDN inputs per layer and writes
    /// candidate states to scratch so `accept` can commit any prefix.
    pub fn encode_layers(&self, cmd: &mut Cmd, wb: &Buffer, t: usize, verify: bool) {
        self.encode_layers_at(cmd, wb, t, verify, B(&self.s.pos, 0));
    }

    pub fn encode_layers_at(&self, cmd: &mut Cmd, wb: &Buffer, t: usize, verify: bool, pos: B) {
        assert!(t <= TMAX && self.state.len + t <= self.ctx);
        assert!(!verify || t <= VMAX);
        let c = &self.cfg;
        let (d, ff, conv, gw, aw, kvw) = (c.hidden, c.intermediate, c.conv_dim(), c.gdn_width(), c.attn_width(), c.kv_width());
        let (nk, nv) = (c.lin_k_heads, c.lin_v_heads);
        let (gdn_state, conv_state) = (c.gdn_state(), c.conv_state());
        let s = &self.s;
        let st = &self.state;
        let off = st.len as u32;
        let f = 4usize;
        // LoRA A rows per threadgroup: one for decode latency, four for prefill (one norm per four rows)
        let rpt = if t == 1 { 1 } else { 4 };
        for (li, l) in self.layers.iter().enumerate() {
            let first = match &l.mixer {
                Mixer::Gdn { qkv, .. } => qkv,
                Mixer::Attn { q, .. } => q,
            };
            let is_gdn = matches!(l.mixer, Mixer::Gdn { .. });
            let g: [_; 4] = std::array::from_fn(|i| self.group(li, i));
            let (la, ln) = self.rot_lora(g[0], rpt);
            let [a0, a1, a2] = la;
            cmd.run(
                "norm_rotate",
                &[
                    B(&s.h, 0).into(),
                    B(wb, l.in_norm).into(),
                    B(wb, first.signs).into(),
                    B(&s.xn, 0).into(),
                    B(&s.xr, 0).into(),
                    (d as u32).into(),
                    (is_gdn as u32).into(),
                    a0,
                    a1,
                    a2,
                ],
                (t, d.div_ceil(CHUNK) + ln, 1),
                (128, 1, 1),
            );
            match &l.mixer {
                Mixer::Gdn {
                    qkv,
                    z,
                    out,
                    a,
                    b,
                    alog,
                    dtb,
                    conv: cw,
                    norm,
                    idx,
                } => {
                    let (qkv_b, qkvc_b, ab_b) = if verify {
                        (
                            B(&s.v_qkv, idx * VMAX * conv * f),
                            B(&s.v_qkvc, idx * VMAX * conv * f),
                            B(&s.v_ab, idx * VMAX * 2 * nv * f),
                        )
                    } else {
                        (B(&s.a1, 0), B(&s.qkvc, 0), B(&s.ab, 0))
                    };
                    let conv_in = B(&st.conv, idx * conv_state * f);
                    let gdn_in = B(&st.gdn, idx * gdn_state * f);
                    let (conv_out, gdn_out) = if verify {
                        (B(&s.v_conv, idx * conv_state * f), B(&s.v_gdn, idx * gdn_state * f))
                    } else {
                        (conv_in, gdn_in)
                    };
                    // decode: the a/b gate projection rides along in the qkv/z matvec dispatch
                    let fused = t == 1 && !verify;
                    let ab_args: [Arg; 5] = [B(wb, *a).into(), B(wb, *b).into(), B(&s.xn, 0).into(), ab_b.into(), (nv as u32).into()];
                    self.qmm_lora(cmd, wb, &[(*qkv, qkv_b), (*z, B(&s.a2, 0))], B(&s.xr, 0), t, false, g[0], fused.then_some(ab_args));
                    if !fused {
                        cmd.run(
                            "gdn_ab",
                            &[
                                B(wb, *a).into(),
                                B(wb, *b).into(),
                                B(&s.xn, 0).into(),
                                ab_b.into(),
                                (nv as u32).into(),
                                (d as u32).into(),
                            ],
                            (2 * nv / 4, t, 1),
                            (128, 1, 1),
                        );
                    }
                    cmd.run(
                        "gdn_conv",
                        &[
                            qkv_b.into(),
                            conv_in.into(),
                            B(wb, *cw).into(),
                            qkvc_b.into(),
                            (fused as u32).into(),
                            (conv as u32).into(),
                            (nk as u32).into(),
                        ],
                        (t, conv / 128, 1),
                        (128, 1, 1),
                    );
                    if !fused {
                        cmd.run(
                            "conv_state",
                            &[qkv_b.into(), conv_in.into(), conv_out.into(), (t as u32).into(), (conv as u32).into()],
                            (conv.div_ceil(128), 1, 1),
                            (128, 1, 1),
                        );
                    }
                    cmd.run(
                        "gdn_recurrent",
                        &[
                            qkvc_b.into(),
                            ab_b.into(),
                            B(wb, *alog).into(),
                            B(wb, *dtb).into(),
                            gdn_in.into(),
                            gdn_out.into(),
                            B(&s.y, 0).into(),
                            (t as u32).into(),
                            (nk as u32).into(),
                            (nv as u32).into(),
                        ],
                        (nv, 32, 1),
                        (32, 4, 1),
                    );
                    let ([a0, a1, a2], ln) = self.rot_lora(g[1], rpt);
                    cmd.run(
                        "gated_norm_rotate",
                        &[
                            B(&s.y, 0).into(),
                            B(&s.a2, 0).into(),
                            B(wb, *norm).into(),
                            B(wb, out.signs).into(),
                            B(&s.xr, 0).into(),
                            (gw as u32).into(),
                            a0,
                            a1,
                            a2,
                        ],
                        (t, gw / CHUNK + ln, 1),
                        (128, 1, 1),
                    );
                    self.qmm_lora(cmd, wb, &[(*out, B(&s.h, 0))], B(&s.xr, 0), t, true, g[1], None);
                }
                Mixer::Attn {
                    q,
                    k,
                    v,
                    o,
                    qn,
                    kn,
                    idx,
                } => {
                    let outs = [B(&s.a1, 0), B(&s.kp, 0), B(&s.vp, 0)];
                    self.qmm_lora(cmd, wb, &[(*q, outs[0]), (*k, outs[1]), (*v, outs[2])], B(&s.xr, 0), t, false, g[0], None);
                    let kc = B(&st.kc, idx * self.ctx * kvw * f);
                    let vc = B(&st.vc, idx * self.ctx * kvw * f);
                    cmd.run(
                        "attn_prep",
                        &[
                            B(&s.a1, 0).into(),
                            B(&s.kp, 0).into(),
                            B(&s.vp, 0).into(),
                            B(wb, *qn).into(),
                            B(wb, *kn).into(),
                            pos.into(),
                            B(&self.inv_freq, 0).into(),
                            B(&s.q, 0).into(),
                            kc.into(),
                            vc.into(),
                            (t as u32).into(),
                            off.into(),
                            (c.heads as u32).into(),
                            (c.kv_heads as u32).into(),
                            (c.rotary_dims as u32).into(),
                            (c.mrope[1] as u32).into(),
                            (c.mrope[2] as u32).into(),
                        ],
                        (t, c.heads + 2 * c.kv_heads, 1),
                        (256, 1, 1),
                    );
                    // decode: 64-key chunks for parallelism; prefill: 512-key chunks to bound the partials
                    let n = st.len + t;
                    let chunk = if t == 1 { 64 } else { 512 };
                    let ns = n.div_ceil(chunk);
                    cmd.run(
                        "attn_split",
                        &[
                            B(&s.q, 0).into(),
                            kc.into(),
                            vc.into(),
                            B(&s.apart, 0).into(),
                            off.into(),
                            (chunk as u32).into(),
                            (ns as u32).into(),
                            (c.heads as u32).into(),
                            (c.kv_heads as u32).into(),
                        ],
                        (c.kv_heads, ns, t),
                        (256, 1, 1),
                    );
                    cmd.run(
                        "attn_combine",
                        &[B(&s.apart, 0).into(), B(&s.y, 0).into(), (ns as u32).into(), (c.heads as u32).into()],
                        (c.heads, t, 1),
                        (256, 1, 1),
                    );
                    let ([a0, a1, a2], ln) = self.rot_lora(g[1], rpt);
                    cmd.run(
                        "gate_rotate",
                        &[
                            B(&s.y, 0).into(),
                            B(&s.a1, 0).into(),
                            B(wb, o.signs).into(),
                            B(&s.xr, 0).into(),
                            (aw as u32).into(),
                            a0,
                            a1,
                            a2,
                        ],
                        (t, aw.div_ceil(CHUNK) + ln, 1),
                        (128, 1, 1),
                    );
                    self.qmm_lora(cmd, wb, &[(*o, B(&s.h, 0))], B(&s.xr, 0), t, true, g[1], None);
                }
            }
            let ([a0, a1, a2], ln) = self.rot_lora(g[2], rpt);
            cmd.run(
                "norm_rotate",
                &[
                    B(&s.h, 0).into(),
                    B(wb, l.post_norm).into(),
                    B(wb, l.gate.signs).into(),
                    B(&s.xn, 0).into(),
                    B(&s.xr, 0).into(),
                    (d as u32).into(),
                    0u32.into(),
                    a0,
                    a1,
                    a2,
                ],
                (t, d.div_ceil(CHUNK) + ln, 1),
                (128, 1, 1),
            );
            self.qmm_lora(cmd, wb, &[(l.gate, B(&s.a1, 0)), (l.up, B(&s.a2, 0))], B(&s.xr, 0), t, false, g[2], None);
            let ([a0, a1, a2], ln) = self.rot_lora(g[3], rpt);
            cmd.run(
                "swiglu_rotate",
                &[
                    B(&s.a1, 0).into(),
                    B(&s.a2, 0).into(),
                    B(wb, l.down.signs).into(),
                    B(&s.xr, 0).into(),
                    (ff as u32).into(),
                    a0,
                    a1,
                    a2,
                ],
                (t, ff.div_ceil(CHUNK) + ln, 1),
                (128, 1, 1),
            );
            self.qmm_lora(cmd, wb, &[(l.down, B(&s.h, 0))], B(&s.xr, 0), t, true, g[3], None);
        }
    }

    /// Final norm, tied head and argmax for rows [first, first + n) of `h`.
    pub fn encode_head(&self, cmd: &mut Cmd, wb: &Buffer, first: usize, n: usize) {
        self.encode_head_to(cmd, wb, first, n, B(&self.s.out, 0));
    }

    pub fn encode_head_to(&self, cmd: &mut Cmd, wb: &Buffer, first: usize, n: usize, out: B) {
        assert!(n <= self.logit_rows);
        let s = &self.s;
        let d = self.cfg.hidden;
        cmd.run(
            "norm_rotate",
            &[
                B(&s.h, first * d * 4).into(),
                B(wb, self.final_norm).into(),
                B(wb, self.embed.signs).into(),
                B(&s.xn, 0).into(),
                B(&s.xr, 0).into(),
                (d as u32).into(),
                0u32.into(),
                B(&s.xr, 0).into(),
                B(&s.xr, 0).into(),
                0u32.into(),
            ],
            (n, d.div_ceil(CHUNK), 1),
            (128, 1, 1),
        );
        self.qmm_lora(cmd, wb, &[(self.embed, B(&s.logits, 0))], B(&s.xr, 0), n, false, None, None);
        cmd.run(
            "argmax1",
            &[
                B(&s.logits, 0).into(),
                B(&s.pv, 0).into(),
                B(&s.pi, 0).into(),
                (self.cfg.vocab as u32).into(),
            ],
            (ARGMAX_GROUPS, n, 1),
            (256, 1, 1),
        );
        cmd.run(
            "argmax2",
            &[
                B(&s.pv, 0).into(),
                B(&s.pi, 0).into(),
                out.into(),
                (ARGMAX_GROUPS as u32).into(),
            ],
            (1, 1, 1),
            (n, 1, 1),
        );
    }

    pub fn argmax(&self, n: usize) -> Vec<u32> {
        let p = contents::<u32>(&self.s.out);
        (0..n).map(|i| unsafe { *p.add(i) }).collect()
    }

    pub fn logits(&self, row: usize) -> &[f32] {
        let v = self.cfg.vocab;
        unsafe { std::slice::from_raw_parts(contents::<f32>(&self.s.logits).add(row * v), v) }
    }

    /// Encodes one decode step that reads its token from ring slot `k % 2` and writes the next token to
    /// slot `(k + 1) % 2`, so steps can be queued without a host round trip.
    pub fn encode_step(&mut self, cmd: &mut Cmd, wb: &Buffer, k: usize) {
        let p = self.state.pos;
        let dp = contents::<i32>(&self.s.dpos);
        for r in 0..3 {
            unsafe { *dp.add((k % 2) * 3 + r) = p };
        }
        let (a, b) = ((k % 2) * 4, ((k + 1) % 2) * 4);
        self.encode_embed_from(cmd, wb, 1, B(&self.s.dtok, a));
        self.encode_layers_at(cmd, wb, 1, false, B(&self.s.dpos, (k % 2) * 12));
        self.encode_head_to(cmd, wb, 0, 1, B(&self.s.dtok, b));
        self.state.len += 1;
        self.state.pos += 1;
    }

    pub fn ring(&self, k: usize) -> u32 {
        unsafe { *contents::<u32>(&self.s.dtok).add(k % 2) }
    }

    pub fn set_ring(&self, k: usize, id: u32) {
        unsafe { *contents::<u32>(&self.s.dtok).add(k % 2) = id };
    }

    /// After a verify forward of `t` rows, commit the first `n` (1..=t) to the state.
    pub fn encode_accept(&mut self, cmd: &mut Cmd, wb: &Buffer, t: usize, n: usize) {
        let s = &self.s;
        let st = &self.state;
        let c = &self.cfg;
        let (conv, nk, nv) = (c.conv_dim(), c.lin_k_heads, c.lin_v_heads);
        let (gdn_state, conv_state) = (c.gdn_state(), c.conv_state());
        let f = 4usize;
        for l in &self.layers {
            if let Mixer::Gdn { alog, dtb, idx, .. } = &l.mixer {
                let cb = B(&st.conv, idx * conv_state * f);
                let gdn = B(&st.gdn, idx * gdn_state * f);
                if n == t {
                    cmd.copy(B(&s.v_conv, idx * conv_state * f), cb, conv_state * f);
                    cmd.copy(B(&s.v_gdn, idx * gdn_state * f), gdn, gdn_state * f);
                } else {
                    let qkv = B(&s.v_qkv, idx * VMAX * conv * f);
                    cmd.run(
                        "conv_state",
                        &[qkv.into(), cb.into(), cb.into(), (n as u32).into(), (conv as u32).into()],
                        (conv.div_ceil(128), 1, 1),
                        (128, 1, 1),
                    );
                    cmd.run(
                        "gdn_recurrent",
                        &[
                            B(&s.v_qkvc, idx * VMAX * conv * f).into(),
                            B(&s.v_ab, idx * VMAX * 2 * nv * f).into(),
                            B(wb, *alog).into(),
                            B(wb, *dtb).into(),
                            gdn.into(),
                            gdn.into(),
                            B(&s.y, 0).into(),
                            (n as u32).into(),
                            (nk as u32).into(),
                            (nv as u32).into(),
                        ],
                        (nv, 32, 1),
                        (32, 4, 1),
                    );
                }
            }
        }
        self.state.len += n;
        self.state.pos += n as i32;
    }

    /// GPU copy of the whole state (used length of the KV cache only).
    pub fn encode_copy_state(&self, cmd: &mut Cmd, from: &State, to: &State) {
        let f = 4usize;
        let c = &self.cfg;
        cmd.copy(B(&from.gdn, 0), B(&to.gdn, 0), self.n_gdn * c.gdn_state() * f);
        cmd.copy(B(&from.conv, 0), B(&to.conv, 0), self.n_gdn * c.conv_state() * f);
        let kvw = c.kv_width();
        for i in 0..self.n_attn {
            let o = i * self.ctx * kvw * f;
            let n = from.len * kvw * f;
            if n > 0 {
                cmd.copy(B(&from.kc, o), B(&to.kc, o), n);
                cmd.copy(B(&from.vc, o), B(&to.vc, o), n);
            }
        }
    }
}

impl Text {
    /// Microbenchmark of the matvec kernels on the head and an MLP matrix (GB/s of weight bytes).
    pub fn bench_qmv(
        &self,
        gpu: &Gpu,
        wb: &Buffer,
        kernel: &str,
        rows_per_tg: usize,
        threads: usize,
    ) {
        let (qkv, z, out) = match &self.layers[0].mixer {
            Mixer::Gdn { qkv, z, out, .. } => (*qkv, *z, *out),
            _ => unreachable!(),
        };
        let q = match &self.layers[3].mixer {
            Mixer::Attn { q, .. } => *q,
            _ => unreachable!(),
        };
        for (name, l) in [
            ("head", self.embed),
            ("qkv", qkv),
            ("z", z),
            ("out", out),
            ("q", q),
            ("gate", self.layers[0].gate),
            ("down", self.layers[0].down),
            (
                "kproj",
                match &self.layers[3].mixer {
                    Mixer::Attn { k, .. } => *k,
                    _ => unreachable!(),
                },
            ),
        ] {
            if name == "head" && std::env::var_os("BONSAI_KGY").is_some() {
                continue;
            }
            let bytes = l.o * l.k / 4 + l.o * l.k / 128 * 2;
            let reps = (2_000_000_000 / bytes).clamp(5, 2000);
            let run = |n: usize| {
                let mut cmd = gpu.begin();
                for _ in 0..n {
                    let kt: u32 = std::env::var("BONSAI_KT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let mut args: Vec<Arg> = vec![
                        B(wb, l.w).into(),
                        B(wb, l.s).into(),
                        B(&self.s.xr, 0).into(),
                        B(&self.s.logits, 0).into(),
                        (l.k as u32).into(),
                        (l.o as u32).into(),
                        0u32.into(),
                    ];
                    if kt > 0 {
                        args.extend([kt.into(), B(&self.s.xr, 0).into(), B(&self.s.xr, 0).into(), 0u32.into(), 0u32.into()]);
                    } else {
                        args.extend([B(&self.s.xr, 0).into(), B(&self.s.xr, 0).into(), 0u32.into()]);
                    }
                    let gy: usize = std::env::var("BONSAI_KGY").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
                    cmd.run(kernel, &args, (l.o / rows_per_tg, gy, 1), (threads, 1, 1));
                }
                let t = std::time::Instant::now();
                cmd.finish();
                t.elapsed().as_secs_f64()
            };
            run(3);
            let t = std::time::Instant::now();
            run(reps);
            let dt = t.elapsed().as_secs_f64() / reps as f64;
            eprintln!(
                "{kernel} {name:6} {}x{} {:7.1} us  {:6.1} GB/s",
                l.o,
                l.k,
                dt * 1e6,
                bytes as f64 / dt / 1e9
            );
        }
    }
}
