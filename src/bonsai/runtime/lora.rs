//! LoRA adapters applied at load time next to the unchanged ternary base:
//! y = base(x) + (x @ A.T) @ B.T on each packed linear's ordinary (unrotated) input, fp16 A and B.
//! Accepts bonsai/train.py's export_coreml file (`model.layers.N.<mod>.lora_A` [r, in], `.lora_B` [out, r],
//! alpha / r already in B) and PEFT adapters (`...layers.N.<mod>.lora_A.weight`, alpha / r from the
//! adapter_config.json next to the file). A of linears that share an input are stacked into one matrix.

use super::gpu::{self, Buffer, Gpu};
use super::model::Text;
use half::f16;
use std::collections::HashMap;
use std::path::Path;

/// Linears sharing one input, all members present (a member the adapter lacks gets zero A and B).
/// Byte offsets into `Lora::buf`.
pub struct Group {
    /// Stacked A, fp16 [members * r, k].
    pub a: usize,
    pub k: usize,
    pub r: usize,
    /// Per member: B fp16 [rows of that linear, r].
    pub b: Vec<usize>,
}

impl Group {
    pub fn rows(&self) -> usize {
        self.b.len() * self.r
    }
}

pub struct Lora {
    pub buf: Buffer,
    /// Per layer: mixer input, mixer output, MLP input, MLP down (Text::linears order).
    pub groups: Vec<[Option<Group>; 4]>,
    pub max_rows: usize,
    pub rank: usize,
    pub linears: usize,
}

struct T {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn read_safetensors(path: &Path) -> Result<HashMap<String, T>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 8 {
        return Err("lora: not a safetensors file".into());
    }
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(b.get(8..8 + n).ok_or("lora: bad header")?).map_err(|e| e.to_string())?;
    let body = &b[8 + n..];
    let mut out = HashMap::new();
    for (k, v) in h.as_object().ok_or("lora: bad header")? {
        if k == "__metadata__" {
            continue;
        }
        let o = &v["data_offsets"];
        let (s, e) = (o[0].as_u64().unwrap_or(0) as usize, o[1].as_u64().unwrap_or(0) as usize);
        let raw = body.get(s..e).ok_or(format!("lora: {k} out of range"))?;
        let data: Vec<f32> = match v["dtype"].as_str().unwrap_or("") {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "F16" => raw.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            d => return Err(format!("lora: {k}: unsupported dtype {d}")),
        };
        let shape = v["shape"].as_array().ok_or("lora: shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
        out.insert(k.clone(), T { shape, data });
    }
    Ok(out)
}

/// alpha / r from a PEFT adapter_config.json.
fn peft_scaling(dir: &Path) -> Result<f32, String> {
    let p = dir.join("adapter_config.json");
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&p).map_err(|e| format!("lora: PEFT keys need {}: {e}", p.display()))?)
            .map_err(|e| e.to_string())?;
    for k in ["rank_pattern", "alpha_pattern"] {
        if j[k].as_object().is_some_and(|o| !o.is_empty()) {
            return Err(format!("lora: {k} is not supported"));
        }
    }
    let r = j["r"].as_f64().ok_or("lora: adapter_config r")?;
    let alpha = j["lora_alpha"].as_f64().ok_or("lora: adapter_config lora_alpha")?;
    Ok(if j["use_rslora"].as_bool() == Some(true) { alpha / r.sqrt() } else { alpha / r } as f32)
}

/// Loads `path` for `text`'s layers, B multiplied by `scale`.
pub fn load(path: &Path, scale: f32, text: &Text, gpu: &Gpu) -> Result<Lora, String> {
    let tensors = read_safetensors(path)?;
    let peft = tensors.keys().any(|k| k.ends_with(".weight"));
    let fold = if peft { peft_scaling(path.parent().unwrap_or(Path::new(".")))? } else { 1.0 } * scale;
    // "layers.N.<mod>" -> (A, B)
    let mut ab: HashMap<String, [Option<&T>; 2]> = HashMap::new();
    for (k, t) in &tensors {
        let k2 = k.strip_suffix(".weight").unwrap_or(k);
        let (stem, which) = match (k2.strip_suffix(".lora_A"), k2.strip_suffix(".lora_B")) {
            (Some(s), _) => (s, 0),
            (_, Some(s)) => (s, 1),
            _ => return Err(format!("lora: unexpected tensor {k}")),
        };
        let i = stem.find("layers.").ok_or(format!("lora: {k} is not a layer linear"))?;
        ab.entry(stem[i..].to_string()).or_default()[which] = Some(t);
    }
    let mut halfs: Vec<u16> = Vec::new();
    let mut push = |v: &mut dyn Iterator<Item = f32>| -> usize {
        let off = halfs.len() * 2;
        halfs.extend(v.map(|x| f16::from_f32(x).to_bits()));
        while halfs.len() % 8 != 0 {
            halfs.push(0);
        }
        off
    };
    let (mut used, mut rank, mut max_rows) = (0, 0, 0);
    let mut groups = Vec::new();
    for layer in 0..text.n_layers() {
        let lin = text.linears(layer);
        let mut gs: [Option<Group>; 4] = Default::default();
        for (gi, members) in lin.iter().enumerate() {
            let mut present = Vec::new();
            for (name, l) in members.iter() {
                let key = format!("layers.{layer}.{name}");
                let Some(e) = ab.get(&key) else {
                    present.push((None, l.o, l.k));
                    continue;
                };
                let (Some(a), Some(b)) = (e[0], e[1]) else { return Err(format!("lora: {key} needs lora_A and lora_B")) };
                let r = a.shape[0];
                if a.shape != [r, l.k] || b.shape != [l.o, r] {
                    return Err(format!("lora: {key}: A {:?} B {:?}, linear is {}x{}", a.shape, b.shape, l.o, l.k));
                }
                if rank != 0 && r != rank {
                    return Err("lora: mixed ranks are not supported".into());
                }
                rank = r;
                present.push((Some((a, b)), l.o, l.k));
                used += 1;
            }
            if present.iter().all(|p| p.0.is_none()) {
                continue;
            }
            let (r, k) = (rank, present[0].2);
            let a = push(&mut present.iter().flat_map(|p| match p.0 {
                Some((a, _)) => a.data.clone(),
                None => vec![0.0; r * k],
            }));
            let b = present
                .iter()
                .map(|p| match p.0 {
                    Some((_, b)) => push(&mut b.data.iter().map(|x| x * fold)),
                    None => push(&mut std::iter::repeat_n(0.0, p.1 * r)),
                })
                .collect();
            let g = Group { a, k, r, b };
            max_rows = max_rows.max(g.rows());
            gs[gi] = Some(g);
        }
        groups.push(gs);
    }
    if used != ab.len() {
        let known: Vec<String> = (0..text.n_layers())
            .flat_map(|l| text.linears(l).into_iter().flatten().map(move |(n, _)| format!("layers.{l}.{n}")))
            .collect();
        let bad: Vec<_> = ab.keys().filter(|k| !known.contains(k)).take(3).collect();
        return Err(format!("lora: tensors for unsupported modules, e.g. {bad:?}"));
    }
    if used == 0 {
        return Err("lora: no adapted linears".into());
    }
    let buf = gpu.buffer(halfs.len() * 2);
    let bytes: Vec<u8> = halfs.iter().flat_map(|h| h.to_le_bytes()).collect();
    gpu::write(&buf, 0, &bytes);
    Ok(Lora { buf, groups, max_rows, rank, linears: used })
}
