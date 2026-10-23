use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let macos_arm = target_os == "macos" && target_arch == "aarch64";

    let apple = macos_arm && feature("APPLE");
    let bonsai_metal = macos_arm && feature("BONSAI");
    let bonsai_ggml = target_os == "windows" && feature("BONSAI");

    // Named cfgs so source files can gate on capabilities instead of
    // repeating feature and target predicates.
    emit_cfg("apple_model", apple);
    emit_cfg("bonsai_metal", bonsai_metal);
    emit_cfg("bonsai_ggml", bonsai_ggml);
    emit_cfg("bonsai_backend", bonsai_metal || bonsai_ggml);

    if apple {
        println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
        // Weak-link so binaries still launch on macOS < 26, where the
        // framework is absent. The shim guards every call with #available.
        println!("cargo::rustc-link-arg=-Wl,-weak_framework,FoundationModels");
        compile_swift_shim("apple");
    }

    if bonsai_metal {
        compile_metal_kernels();
    }

    #[cfg(feature = "bonsai")]
    if bonsai_ggml {
        compile_llama();
    }
}

/// Builds libllama from vendor/llama.cpp plus our C glue into one static library. ggml itself comes from
/// transcribe-cpp-sys (vendor/ggml holds its headers), so the process links a single ggml.
#[cfg(feature = "bonsai")]
fn compile_llama() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let src = root.join("vendor/llama.cpp/src");
    let glue = root.join("src/bonsai/ggml");
    for dir in [
        "vendor/llama.cpp",
        "vendor/ggml",
        "src/bonsai/ggml/glue.cpp",
        "src/bonsai/ggml/compat.cpp",
        "src/bonsai/ggml/compat.h",
    ] {
        println!("cargo::rerun-if-changed={dir}");
    }

    let cpp_files = |dir: &std::path::Path| -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "cpp"))
            .collect();
        files.sort();
        files
    };
    // Model graphs in batches of 16 per translation unit, like llama.cpp's own unity build.
    let mut units = cpp_files(&src);
    for (i, batch) in cpp_files(&src.join("models")).chunks(16).enumerate() {
        let unit = out_dir.join(format!("llama_models_{i}.cpp"));
        let body: String = batch
            .iter()
            .map(|f| {
                format!(
                    "#include \"{}\"\n",
                    f.display().to_string().replace('\\', "/")
                )
            })
            .collect();
        std::fs::write(&unit, body).unwrap();
        units.push(unit);
    }
    units.push(glue.join("glue.cpp"));
    units.push(glue.join("compat.cpp"));

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .files(&units)
        .include(root.join("vendor/llama.cpp/include"))
        .include(&src)
        .include(src.join("models"))
        .include(root.join("vendor/ggml/include"))
        .include(&glue)
        .define("_CRT_SECURE_NO_WARNINGS", None)
        .flag("/utf-8")
        .flag("/bigobj")
        .flag(format!("/FI{}", glue.join("compat.h").display()))
        // Optimized even in debug builds; an unoptimized libllama decodes several times slower.
        .opt_level(2)
        .debug(false)
        .warnings(false)
        .compile("bonsai_llama");
}

fn feature(name: &str) -> bool {
    std::env::var_os(format!("CARGO_FEATURE_{name}")).is_some()
}

fn emit_cfg(name: &str, enabled: bool) {
    println!("cargo::rustc-check-cfg=cfg({name})");
    if enabled {
        println!("cargo::rustc-cfg={name}");
    }
}

/// Builds swift/<name>.swift into a static lib with a macOS 14 deployment
/// target, so newer APIs stay weak behind #available.
fn compile_swift_shim(name: &str) {
    let source = format!("swift/{name}.swift");
    println!("cargo::rerun-if-changed={source}");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib_name = format!("glimpse_language_{name}");
    let lib_path = out_dir.join(format!("lib{lib_name}.a"));

    let status = Command::new("swiftc")
        .args([
            "-O",
            "-parse-as-library",
            "-module-name",
            &lib_name,
            "-emit-library",
            "-static",
            "-target",
            "arm64-apple-macosx14.0",
            &source,
            "-o",
        ])
        .arg(&lib_path)
        .status()
        .expect("failed to run swiftc; the apple feature needs Xcode command line tools");
    assert!(status.success(), "swiftc failed for {source}");

    println!("cargo::rustc-link-search=native={}", out_dir.display());
    println!("cargo::rustc-link-lib=static={lib_name}");
    println!("cargo::rustc-link-lib=framework=Foundation");
    println!("cargo::rustc-link-search=native=/usr/lib/swift");
}

/// Compiles the Bonsai kernels to a metallib that the runtime embeds, so the
/// app never compiles shaders. Needs the Metal toolchain (see README).
fn compile_metal_kernels() {
    let source = "src/bonsai/runtime/kernels.metal";
    println!("cargo::rerun-if-changed={source}");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let air = out_dir.join("kernels.air");
    let lib = out_dir.join("kernels.metallib");

    let found = Command::new("xcrun")
        .args(["-sdk", "macosx", "-f", "metal"])
        .output()
        .is_ok_and(|out| out.status.success());
    if !found {
        panic!(
            "the bonsai feature needs the Metal toolchain to compile its kernels, and `xcrun -sdk macosx -f metal` \
             found none. Install Xcode, then run `xcodebuild -downloadComponent MetalToolchain` (Xcode 26 ships it \
             as a separate component)."
        );
    }

    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-O3",
            "-std=metal3.0",
            "-mmacosx-version-min=14.0",
            "-c",
            source,
            "-o",
        ])
        .arg(&air)
        .status()
        .expect("failed to run xcrun metal");
    assert!(status.success(), "metal failed for {source}");

    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&lib)
        .status()
        .expect("failed to run xcrun metallib");
    assert!(status.success(), "metallib failed for {source}");
}
