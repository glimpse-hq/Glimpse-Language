//! Prompt-lookup speculation, ported from the Metal runtime (runtime/mod.rs: `lookup`, `Policy`,
//! `normalize_draft`, `draft_source`). Keep the two in step.

use super::super::tokenizer::Tokenizer;

/// Longest draft. 4 beat the Metal runtime's 7 here: verify rows cost more on Vulkan, and each extra
/// rollback snapshot of the recurrent state costs device memory and decode time (50-word cleanup 183 vs
/// 225 ms on an RTX 4000 Ada).
pub const DRAFT_MAX: usize = 4;
const NGRAM_MAX: usize = 3;
const NGRAM_MIN: usize = 1;

/// Draft source: the request's source tokens plus a matching key per word token.
pub struct Source {
    pub ids: Vec<u32>,
    /// (key, index into ids) of the word tokens
    words: Vec<(u64, usize)>,
}

/// Matching key of a token: bit 0 set for word tokens (the only ones matched), bit 1 set when the token
/// contains a line break.
pub fn token_key(bytes: &[u8]) -> u64 {
    let nl = if bytes.contains(&b'\n') { 2 } else { 0 };
    let b = bytes.strip_prefix(b" ").unwrap_or(bytes);
    if b.iter()
        .all(|c| c.is_ascii_punctuation() || c.is_ascii_whitespace())
    {
        return nl;
    }
    let h = b.iter().fold(0xcbf29ce484222325u64, |h, c| {
        (h ^ c.to_ascii_lowercase() as u64).wrapping_mul(0x100000001b3)
    });
    (h & !3) | 1 | nl
}

/// How far behind the cursor a lookup match may start (self-corrections repeat a few words).
const BACK: usize = 24;

/// Drafts up to `k` source tokens that followed an occurrence of the output's last n word keys (n = 3 down
/// to 1); the first occurrence ending at or after `cursor`, else the first one. Returns the draft and the
/// source position it starts at.
pub fn lookup(
    src: &Source,
    seq: &[u32],
    keys: &[u64],
    k: usize,
    cursor: usize,
) -> (Vec<u32>, usize) {
    if k == 0 {
        return (Vec::new(), cursor);
    }
    let mut pat = Vec::with_capacity(NGRAM_MAX);
    for &id in seq.iter().rev().take(4 * NGRAM_MAX) {
        if pat.len() == NGRAM_MAX {
            break;
        }
        let key = keys.get(id as usize).copied().unwrap_or(0);
        if key & 1 != 0 {
            pat.push(key);
        }
    }
    pat.reverse();
    let w = &src.words;
    for n in (NGRAM_MIN..=NGRAM_MAX.min(pat.len())).rev() {
        let p = &pat[pat.len() - n..];
        let mut hit = None;
        for i in 0..w.len().saturating_sub(n - 1) {
            if w[i..i + n].iter().zip(p).all(|(a, b)| a.0 == *b) {
                let j = w[i + n - 1].1 + 1;
                // the output moves forward through the source: a match far behind it is a different spot
                if j < src.ids.len() && j + BACK >= cursor && (hit.is_none() || j >= cursor) {
                    hit = Some(j);
                }
                if j >= cursor {
                    break;
                }
            }
        }
        if let Some(j) = hit {
            return (src.ids[j..(j + k).min(src.ids.len())].to_vec(), j);
        }
    }
    (Vec::new(), cursor)
}

const SOURCE_TAGS: [(&str, &str); 2] = [
    ("<transcript>\n", "\n</transcript>"),
    ("<text>\n", "\n</text>"),
];
const CAPITALIZED: [&[u8]; 18] = [
    b"monday",
    b"tuesday",
    b"wednesday",
    b"thursday",
    b"friday",
    b"saturday",
    b"sunday",
    b"january",
    b"february",
    b"march",
    b"april",
    b"june",
    b"july",
    b"august",
    b"september",
    b"october",
    b"november",
    b"december",
];
const CONTRACTIONS: [(&str, &str); 22] = [
    ("dont", "don't"),
    ("doesnt", "doesn't"),
    ("didnt", "didn't"),
    ("isnt", "isn't"),
    ("wasnt", "wasn't"),
    ("arent", "aren't"),
    ("werent", "weren't"),
    ("wont", "won't"),
    ("cant", "can't"),
    ("couldnt", "couldn't"),
    ("wouldnt", "wouldn't"),
    ("shouldnt", "shouldn't"),
    ("havent", "haven't"),
    ("hasnt", "hasn't"),
    ("im", "I'm"),
    ("ive", "I've"),
    ("youre", "you're"),
    ("theyre", "they're"),
    ("thats", "that's"),
    ("whats", "what's"),
    ("theres", "there's"),
    ("lets", "let's"),
];
const FILLERS: [&str; 12] = [
    " um", " uh", " Um", " Uh", " umm", " uhm", " erm", " hmm", "um", "uh", "Um", "Uh",
];

/// What a cleanup does to the words of a lowercase transcript, applied to a draft word by word.
pub fn normalize_draft(tok: &Tokenizer, draft: &mut Vec<u32>, last: u32) {
    let k = draft.len();
    let mut out: Vec<u32> = Vec::with_capacity(k + 4);
    let mut prev_end = tok
        .token_bytes(last)
        .trim_ascii_end()
        .last()
        .is_some_and(|c| b".?!".contains(c));
    let mut i = 0;
    while i < draft.len() && out.len() < k {
        let mut j = i + 1;
        while j < draft.len()
            && !tok
                .token_bytes(draft[j])
                .first()
                .is_some_and(|c| !c.is_ascii_alphabetic() && *c != b'\'')
        {
            j += 1;
        }
        let w: Vec<u8> = draft[i..j]
            .iter()
            .flat_map(|&d| tok.token_bytes(d).to_vec())
            .collect();
        let mut v = w.clone();
        if v.len() > 1 && v[0] == b' ' && v[1..].iter().all(|c| c.is_ascii_lowercase()) {
            let bare = &w[1..];
            if let Some((_, to)) = CONTRACTIONS
                .iter()
                .find(|(from, _)| from.as_bytes() == bare)
            {
                v = format!(" {to}").into_bytes();
            } else if bare == b"i" || prev_end || CAPITALIZED.contains(&bare) {
                v[1] = v[1].to_ascii_uppercase();
            }
        }
        match (v != w).then(|| String::from_utf8(v).ok()).flatten() {
            Some(text) => out.extend(tok.encode(&text)),
            None => out.extend_from_slice(&draft[i..j]),
        }
        prev_end = w
            .trim_ascii_end()
            .last()
            .is_some_and(|c| b".?!".contains(c));
        i = j;
    }
    out.truncate(k);
    *draft = out;
}

/// The tokens inside the first SOURCE_TAGS block of the user turn (all of them without one), fillers removed.
pub fn draft_source(tok: &Tokenizer, user: &[u32], keys: &[u64]) -> Source {
    let mut text = Vec::new();
    let mut ends = Vec::with_capacity(user.len());
    for &id in user {
        text.extend_from_slice(tok.token_bytes(id));
        ends.push(text.len());
    }
    let find = |pat: &str, from: usize| {
        text.get(from..)?
            .windows(pat.len())
            .position(|w| w == pat.as_bytes())
            .map(|i| i + from)
    };
    let (lo, hi) = SOURCE_TAGS
        .iter()
        .find_map(|(open, close)| {
            let a = find(open, 0)? + open.len();
            Some((a, find(close, a)?))
        })
        .unwrap_or((0, text.len()));
    let fillers: Vec<u32> = FILLERS
        .iter()
        .filter_map(|f| match tok.encode(f).as_slice() {
            [id] => Some(*id),
            _ => None,
        })
        .collect();
    let ids: Vec<u32> = (0..user.len())
        .filter(|&i| {
            ends[i] > lo && ends[i] - tok.token_bytes(user[i]).len() >= lo && ends[i] <= hi
        })
        .map(|i| user[i])
        .filter(|id| !fillers.contains(id))
        .collect();
    let words = ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| Some((keys.get(id as usize).copied().filter(|&k| k & 1 != 0)?, i)))
        .collect();
    Source { ids, words }
}

/// Expected gain needed to keep speculating, and the higher bar for switching back from plain decoding.
pub const KEEP_GAIN: f64 = 1.05;
pub const ENTER_GAIN: f64 = 1.15;
/// After this many verify rounds, a request whose rounds returned fewer than KEEP_GAIN tokens per step of
/// table cost decodes plainly to the end.
pub const GIVE_UP_ROUNDS: usize = 6;

/// Adaptive draft length: each round picks the k with the best expected tokens per unit of verify cost.
pub struct Policy {
    cost: &'static [f64; 8],
    kmax: usize,
    acc: f64,
    rej: f64,
}

impl Policy {
    const PRIOR_ACC: f64 = 3.4;
    const PRIOR_REJ: f64 = 0.6;
    pub const DECAY: f64 = 0.85;

    pub fn new(cost: &'static [f64; 8], kmax: usize) -> Policy {
        Policy {
            cost,
            kmax: kmax.min(7),
            acc: 0.0,
            rej: 0.0,
        }
    }

    fn p(&self) -> f64 {
        (self.acc + Self::PRIOR_ACC) / (self.acc + self.rej + Self::PRIOR_ACC + Self::PRIOR_REJ)
    }

    /// Best draft length up to `limit`; 0 if none beats `min_gain`.
    pub fn choose(&self, limit: usize, min_gain: f64) -> usize {
        let p = self.p();
        let mut best = (0, 1.0);
        for k in 1..=self.kmax.min(limit) {
            let g = (1.0 - p.powi(k as i32 + 1)) / (1.0 - p) / self.cost[k];
            if g > best.1 && g > min_gain {
                best = (k, g);
            }
        }
        best.0
    }

    /// A round (real or shadow) drafted k tokens and m were accepted.
    pub fn update(&mut self, k: usize, m: usize) {
        self.acc = self.acc * Self::DECAY + m as f64;
        self.rej = self.rej * Self::DECAY + (m < k) as u32 as f64;
    }
}
