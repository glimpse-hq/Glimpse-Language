//! Golden check: greedy output must equal PrismML's MLX runtime on the reference prompts in
//! tests/golden (2B and 4B). Needs a model directory:
//!
//! BONSAI_MODEL_DIR=/path/to/mlx-2bit cargo test --release --features bonsai --test bonsai_golden -- --ignored

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use glimpse_language::bonsai::{Bonsai, BonsaiErrorKind, GenerateOptions, ModelSize, StopReason};

const ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Splits a golden prompt back into system prompt and user text.
fn split_prompt(prompt: &str) -> (&str, &str) {
    let body = prompt.strip_suffix(ASSISTANT).expect("assistant header");
    match body.strip_prefix("<|im_start|>system\n") {
        Some(rest) => rest
            .split_once("<|im_end|>\n<|im_start|>user\n")
            .expect("user turn"),
        None => (
            "",
            body.strip_prefix("<|im_start|>user\n").expect("user turn"),
        ),
    }
}

#[test]
#[ignore = "needs BONSAI_MODEL_DIR"]
fn matches_mlx_golden_outputs() {
    let dir = PathBuf::from(std::env::var("BONSAI_MODEL_DIR").expect("set BONSAI_MODEL_DIR"));
    // Saved prompt states from earlier runs would make the cold runs warm.
    let saved = dir.join("prefix-cache");
    let _ = std::fs::remove_dir_all(&saved);
    let model = Bonsai::load(&dir, None).expect("load");
    let set = match model.info().size {
        ModelSize::B2 => "2b",
        ModelSize::B4 => "4b",
    };
    let golden = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(set);
    let mut cases: Vec<_> = std::fs::read_dir(&golden)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    cases.sort();
    assert!(!cases.is_empty());

    let mut failures = Vec::new();
    for case in &cases {
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(case).unwrap()).unwrap();
        let name = case.file_stem().unwrap().to_string_lossy();
        let (system, user) = split_prompt(json["prompt_text"].as_str().unwrap());
        let expected = json["generated_text_until_eos"].as_str().unwrap().trim();
        // Cold, then from the cached system prompt, then without speculation.
        for (run, speculate) in [("cold", true), ("cached", true), ("plain", false)] {
            let options = GenerateOptions {
                max_tokens: 200,
                speculate,
                ..Default::default()
            };
            let out = model.generate(system, user, &options, None).unwrap();
            if out.text != expected || out.stop != StopReason::Eos {
                failures.push(format!("{name} ({run}): {:?} {:?}", out.stop, out.text));
            }
            if run == "cached" && !system.is_empty() {
                assert!(
                    out.stats.cached_tokens > 0,
                    "{name}: system prompt not cached"
                );
            }
        }
    }

    // Idle release keeps working output and the cached prompt states.
    model.release_idle();
    assert!(model.is_idle());
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&cases[0]).unwrap()).unwrap();
    let (system, user) = split_prompt(json["prompt_text"].as_str().unwrap());
    let expected = json["generated_text_until_eos"].as_str().unwrap().trim();
    let out = model
        .generate(system, user, &GenerateOptions::default(), None)
        .unwrap();
    assert!(!model.is_idle());
    assert_eq!(out.text, expected);

    // A new load restores the saved prompt states (written in the background).
    let saved_any = || {
        std::fs::read_dir(&saved)
            .is_ok_and(|d| d.flatten().any(|e| e.path().extension().is_some_and(|x| x == "prefix")))
    };
    for _ in 0..100 {
        if saved_any() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(saved_any(), "no prompt state saved");
    for case in &cases {
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(case).unwrap()).unwrap();
        let (system, user) = split_prompt(json["prompt_text"].as_str().unwrap());
        let expected = json["generated_text_until_eos"].as_str().unwrap().trim();
        let fresh = Bonsai::load(&dir, None).expect("load");
        let out = fresh
            .generate(system, user, &GenerateOptions::default(), None)
            .unwrap();
        if out.text != expected {
            failures.push(format!("{} (saved state): {:?}", case.display(), out.text));
        }
    }

    let cancel = AtomicBool::new(true);
    let err = model
        .generate(system, user, &GenerateOptions::default(), Some(&cancel))
        .unwrap_err();
    assert_eq!(err.kind, BonsaiErrorKind::Cancelled);

    let long = "word ".repeat(3000);
    let err = model
        .generate(system, &long, &GenerateOptions::default(), None)
        .unwrap_err();
    assert_eq!(err.kind, BonsaiErrorKind::ContextTooLarge);

    assert!(failures.is_empty(), "mismatches:\n{}", failures.join("\n"));
}
