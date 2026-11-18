use std::time::Duration;

use glimpse_language::apple::{apple_availability, apple_generate, apple_supports_language};

const INSTRUCTIONS: &str = "Clean up raw speech-to-text output. Fix punctuation and capitalization, \
remove filler words and false starts, keep the speaker's wording. Return only the cleaned text.";

fn main() {
    println!("apple availability: {:?}", apple_availability());
    for language in ["en", "no", "nb", "nn", "pt-BR"] {
        println!("supports {language}: {}", apple_supports_language(language));
    }

    let raw = "um so i was thinking uh maybe we should we should move the the launch to thursday \
               because uh the build isnt ready and and jen said the uh the release notes arent done either";
    match apple_generate(INSTRUCTIONS, raw, 0.0, None, Duration::from_secs(60)) {
        Ok(cleaned) => {
            println!("before: {raw}");
            println!("after:  {cleaned}");
        }
        Err(err) => println!("failed ({:?}): {err}", err.kind),
    }
}
