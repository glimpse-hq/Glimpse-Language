//! Runs a set of requests and writes one JSON line per request (output text and timings), for parity and
//! speed checks across platforms.
//!
//! cargo run --release --example bonsai_suite --features bonsai -- <model dir> <cases.json> <out.jsonl>
//!     [--adapter FILE] [--no-spec] [--repeat N]
//!
//! cases.json: [{"id": "...", "system": "...", "user": "..."}]

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use glimpse_language::bonsai::{Bonsai, GenerateOptions, availability};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(cases), Some(out)) = (args.next(), args.next(), args.next()) else {
        eprintln!(
            "usage: bonsai_suite <model dir> <cases.json> <out.jsonl> [--adapter FILE] [--no-spec] [--repeat N]"
        );
        std::process::exit(2);
    };
    let (mut adapter, mut speculate, mut repeat) = (None, true, 1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--adapter" => adapter = args.next().map(PathBuf::from),
            "--no-spec" => speculate = false,
            "--repeat" => repeat = args.next().and_then(|n| n.parse().ok()).unwrap_or(1),
            other => panic!("unknown argument {other}"),
        }
    }
    eprintln!("availability: {:?}", availability());

    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(&cases).expect("read cases")).expect("parse cases");
    let started = Instant::now();
    let model = Bonsai::load(PathBuf::from(&dir).as_path(), adapter.as_deref()).expect("load");
    eprintln!("load: {} ms", started.elapsed().as_millis());

    let options = GenerateOptions {
        max_tokens: 600,
        speculate,
        ..Default::default()
    };
    let mut file = std::fs::File::create(&out).expect("create output");
    let mut total = 0.0;
    for case in &cases {
        let (system, user) = (
            case["system"].as_str().unwrap(),
            case["user"].as_str().unwrap(),
        );
        for run in 0..repeat {
            let t = Instant::now();
            let g = model
                .generate(system, user, &options, None)
                .expect("generate");
            let wall = t.elapsed().as_secs_f64() * 1000.0;
            total += wall;
            let s = &g.stats;
            let line = serde_json::json!({
                "id": case["id"], "run": run, "text": g.text, "stop": format!("{:?}", g.stop),
                "wall_ms": wall, "prompt_tokens": s.prompt_tokens, "cached_tokens": s.cached_tokens,
                "generated_tokens": s.generated_tokens, "prefill_ms": s.prefill.as_secs_f64() * 1000.0,
                "first_token_ms": s.first_token.as_secs_f64() * 1000.0, "decode_ms": s.decode.as_secs_f64() * 1000.0,
                "drafted": s.drafted, "accepted": s.accepted,
            });
            writeln!(file, "{line}").unwrap();
        }
    }
    eprintln!("{} requests, {:.0} ms total", cases.len() * repeat, total);
}
