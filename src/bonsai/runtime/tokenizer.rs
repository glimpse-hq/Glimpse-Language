//! Byte-level BPE for the Qwen3.5 tokenizer.json (vocab, merges, added tokens). No NFC normalization:
//! input is assumed to be NFC already, which macOS and Windows text input is in practice.

use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    merges: Vec<(u64, u32, u32)>, // sorted (left << 32 | right, rank, merged id)
    byte_id: [u32; 256],
    blob: Vec<u8>,
    offs: Vec<u32>,
    added: Vec<(String, u32)>, // longest first
    split: regex::Regex,
}

// tokenizer.json pre-tokenizer pattern with `\s+(?!\S)|\s+` folded into `\s+`; the lookahead is applied after.
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

fn byte_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        map[*b as usize] = char::from_u32(c).unwrap();
    }
    map
}

impl Tokenizer {
    /// Loads `cache` if it was built from this tokenizer.json (size and mtime match), otherwise parses
    /// the JSON and writes the cache (best effort).
    pub fn load(json: &Path, cache: &Path) -> Result<Tokenizer, String> {
        let meta = std::fs::metadata(json).map_err(|e| format!("{}: {e}", json.display()))?;
        let stamp = meta.len()
            ^ meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos() as u64);
        if let Ok(b) = std::fs::read(cache) {
            if let Some(t) = Self::from_cache(&b, stamp) {
                return Ok(t);
            }
        }
        let t = Self::from_json(json)?;
        let _ = std::fs::write(cache, t.to_cache(stamp));
        Ok(t)
    }

    fn from_json(path: &Path) -> Result<Tokenizer, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let j: serde_json::Value = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
        let b2u = byte_unicode();
        let mut u2b = HashMap::new();
        for (b, c) in b2u.iter().enumerate() {
            u2b.insert(*c, b as u8);
        }
        let to_bytes = |s: &str| -> Vec<u8> { s.chars().map(|c| u2b[&c]).collect() };
        let model = &j["model"];
        let vmap = model["vocab"].as_object().ok_or("vocab")?;
        let mut vocab = HashMap::with_capacity(vmap.len());
        let mut id_bytes = vec![Vec::new(); 248320];
        for (k, v) in vmap {
            let id = v.as_u64().unwrap() as u32;
            let b = to_bytes(k);
            id_bytes[id as usize] = b.clone();
            vocab.insert(b, id);
        }
        let mut merges = Vec::new();
        for (r, m) in model["merges"]
            .as_array()
            .ok_or("merges")?
            .iter()
            .enumerate()
        {
            let (a, b) = match m {
                serde_json::Value::String(s) => {
                    let (a, b) = s.split_once(' ').ok_or("merge")?;
                    (a.to_string(), b.to_string())
                }
                serde_json::Value::Array(p) => {
                    (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())
                }
                _ => return Err("merge format".into()),
            };
            let (ab, bb) = (to_bytes(&a), to_bytes(&b));
            let mut m = ab.clone();
            m.extend_from_slice(&bb);
            if let (Some(&x), Some(&y), Some(&z)) = (vocab.get(&ab), vocab.get(&bb), vocab.get(&m))
            {
                merges.push((((x as u64) << 32) | y as u64, r as u32, z));
            }
        }
        merges.sort_unstable_by_key(|m| m.0);
        let mut byte_id = [0u32; 256];
        for b in 0..256 {
            byte_id[b] = vocab[&vec![b as u8]];
        }
        let mut added = Vec::new();
        for t in j["added_tokens"].as_array().ok_or("added_tokens")? {
            let s = t["content"].as_str().unwrap().to_string();
            let id = t["id"].as_u64().unwrap() as u32;
            if id_bytes.len() <= id as usize {
                id_bytes.resize(id as usize + 1, Vec::new());
            }
            id_bytes[id as usize] = s.as_bytes().to_vec();
            added.push((s, id));
        }
        added.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let mut blob = Vec::new();
        let mut offs = vec![0u32];
        for b in &id_bytes {
            blob.extend_from_slice(b);
            offs.push(blob.len() as u32);
        }
        Ok(Tokenizer {
            merges,
            byte_id,
            blob,
            offs,
            added,
            split: regex::Regex::new(PATTERN).unwrap(),
        })
    }

    fn to_cache(&self, stamp: u64) -> Vec<u8> {
        let mut o = Vec::new();
        let u = |o: &mut Vec<u8>, v: u32| o.extend_from_slice(&v.to_le_bytes());
        o.extend_from_slice(b"BTK1");
        o.extend_from_slice(&stamp.to_le_bytes());
        u(&mut o, self.offs.len() as u32);
        u(&mut o, self.blob.len() as u32);
        u(&mut o, self.merges.len() as u32);
        u(&mut o, self.added.len() as u32);
        for &x in &self.offs {
            u(&mut o, x);
        }
        o.extend_from_slice(&self.blob);
        for &(k, r, z) in &self.merges {
            o.extend_from_slice(&k.to_le_bytes());
            u(&mut o, r);
            u(&mut o, z);
        }
        for &b in &self.byte_id {
            u(&mut o, b);
        }
        for (s, id) in &self.added {
            u(&mut o, *id);
            u(&mut o, s.len() as u32);
            o.extend_from_slice(s.as_bytes());
        }
        o
    }

    fn from_cache(b: &[u8], stamp: u64) -> Option<Tokenizer> {
        if b.len() < 28
            || &b[..4] != b"BTK1"
            || u64::from_le_bytes(b[4..12].try_into().ok()?) != stamp
        {
            return None;
        }
        let mut p = 12;
        let mut u = || {
            let v = u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?);
            p += 4;
            Some(v)
        };
        let (no, nb, nm, na) = (u()? as usize, u()? as usize, u()? as usize, u()? as usize);
        let mut p = 28;
        let rd = |p: &mut usize, n: usize| -> Option<&[u8]> {
            let s = b.get(*p..*p + n)?;
            *p += n;
            Some(s)
        };
        let offs = rd(&mut p, no * 4)?
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let blob = rd(&mut p, nb)?.to_vec();
        let merges = rd(&mut p, nm * 16)?
            .chunks_exact(16)
            .map(|c| {
                (
                    u64::from_le_bytes(c[..8].try_into().unwrap()),
                    u32::from_le_bytes(c[8..12].try_into().unwrap()),
                    u32::from_le_bytes(c[12..].try_into().unwrap()),
                )
            })
            .collect();
        let mut byte_id = [0u32; 256];
        for (i, c) in rd(&mut p, 1024)?.chunks_exact(4).enumerate() {
            byte_id[i] = u32::from_le_bytes(c.try_into().unwrap());
        }
        let mut added = Vec::with_capacity(na);
        for _ in 0..na {
            let id = u32::from_le_bytes(rd(&mut p, 4)?.try_into().ok()?);
            let n = u32::from_le_bytes(rd(&mut p, 4)?.try_into().ok()?) as usize;
            added.push((String::from_utf8(rd(&mut p, n)?.to_vec()).ok()?, id));
        }
        Some(Tokenizer {
            merges,
            byte_id,
            blob,
            offs,
            added,
            split: regex::Regex::new(PATTERN).unwrap(),
        })
    }

    fn merge(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let k = ((a as u64) << 32) | b as u64;
        self.merges
            .binary_search_by_key(&k, |m| m.0)
            .ok()
            .map(|i| (self.merges[i].1, self.merges[i].2))
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // earliest added token occurrence (longest wins at the same position)
            let mut best: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.added {
                if let Some(p) = rest.find(s.as_str()) {
                    if best.map_or(true, |(bp, bs, _)| {
                        p < bp || (p == bp && s.len() > bs.len())
                    }) {
                        best = Some((p, s, *id));
                    }
                }
            }
            match best {
                Some((p, s, id)) => {
                    self.encode_plain(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + s.len()..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let mut i = 0;
        while i < text.len() {
            let m = self.split.find_at(text, i).unwrap();
            let (a, mut b) = (m.start(), m.end());
            let piece = &text[a..b];
            // `\s+(?!\S)`: a whitespace run followed by non-space gives up its last char.
            if b < text.len()
                && piece.chars().all(char::is_whitespace)
                && !piece.ends_with(['\r', '\n'])
            {
                let last = piece.chars().last().unwrap().len_utf8();
                if piece.len() > last {
                    b -= last;
                }
            }
            self.bpe(&text.as_bytes()[a..b], out);
            i = b;
        }
    }

    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = bytes.iter().map(|&b| self.byte_id[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some((r, z)) = self.merge(syms[i], syms[i + 1]) {
                    if best.map_or(true, |(br, _, _)| r < br) {
                        best = Some((r, i, z));
                    }
                }
            }
            match best {
                Some((_, i, z)) => {
                    syms[i] = z;
                    syms.remove(i + 1);
                }
                None => break,
            }
        }
        out.extend(syms);
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        &self.blob[self.offs[i] as usize..self.offs[i + 1] as usize]
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let b: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.token_bytes(i).to_vec())
            .collect();
        String::from_utf8_lossy(&b).into_owned()
    }
}

/// Streams token bytes out as valid UTF-8 pieces.
#[derive(Default)]
pub struct Utf8Stream(Vec<u8>);
impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.0.extend_from_slice(bytes);
        match std::str::from_utf8(&self.0) {
            Ok(s) => {
                let s = s.to_string();
                self.0.clear();
                s
            }
            Err(e) => {
                let n = e.valid_up_to();
                let s = String::from_utf8_lossy(&self.0[..n]).into_owned();
                self.0.drain(..n);
                if e.error_len().is_some() {
                    self.0.clear();
                }
                s
            }
        }
    }
}
