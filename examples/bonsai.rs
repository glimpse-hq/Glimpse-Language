use std::path::PathBuf;

use glimpse_language::bonsai::{Bonsai, GenerateOptions, availability};

const SYSTEM: &str = "Clean up raw speech-to-text output. Fix punctuation and capitalization, \
remove filler words and false starts, keep the speaker's wording. Return only the cleaned text.";

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next().map(PathBuf::from) else {
        eprintln!("usage: bonsai <model dir> [adapter.safetensors]");
        std::process::exit(2);
    };
    let adapter = args.next().map(PathBuf::from);
    println!("bonsai availability: {:?}", availability());

    let model = match Bonsai::load(&dir, adapter.as_deref()) {
        Ok(model) => model,
        Err(err) => {
            println!("load failed ({:?}): {err}", err.kind);
            return;
        }
    };
    println!("model: {}", model.info().size.label());
    model.prewarm(SYSTEM).expect("prewarm");

    let raw = "um so i was thinking uh maybe we should we should move the the launch to thursday \
               because uh the build isnt ready and and jen said the uh the release notes arent done either";
    match model.generate(SYSTEM, raw, &GenerateOptions::default(), None) {
        Ok(out) => {
            println!("before: {raw}");
            println!("after:  {}", out.text);
            println!("{:?} {:?}", out.stop, out.stats);
        }
        Err(err) => println!("failed ({:?}): {err}", err.kind),
    }
}
