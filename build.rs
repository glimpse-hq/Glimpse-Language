use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let macos_arm = target_os == "macos" && target_arch == "aarch64";

    let apple = macos_arm && feature("APPLE");
    let bonsai_metal = macos_arm && feature("BONSAI");
    let bonsai_vulkan = target_os == "windows" && feature("BONSAI");

    // Named cfgs so source files can gate on capabilities instead of
    // repeating feature and target predicates.
    emit_cfg("apple_model", apple);
    emit_cfg("bonsai_metal", bonsai_metal);
    emit_cfg("bonsai_vulkan", bonsai_vulkan);
    emit_cfg("bonsai_backend", bonsai_metal || bonsai_vulkan);

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

    if bonsai_vulkan {
        compile_vulkan_kernels();
    }
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

/// Compiles src/bonsai/runtime/kernels/*.comp to SPIR-V with glslc (Vulkan SDK) and writes a table the runtime
/// embeds, so the app never compiles shaders.
fn compile_vulkan_kernels() {
    let dir = PathBuf::from("src/bonsai/runtime/kernels");
    println!("cargo::rerun-if-changed={}", dir.display());
    println!("cargo::rerun-if-env-changed=VULKAN_SDK");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let glslc = std::env::var_os("VULKAN_SDK")
        .map(|sdk| PathBuf::from(sdk).join("Bin").join("glslc.exe"))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("glslc"));
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "comp").then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    let mut table = String::from("pub static SPV: &[(&str, &[u8])] = &[\n");
    let mut all = Vec::new();
    for name in &names {
        let src = dir.join(format!("{name}.comp"));
        let spv = out_dir.join(format!("{name}.spv"));
        let status = Command::new(&glslc)
            .args(["--target-env=vulkan1.3", "-O", "-o"])
            .arg(&spv)
            .arg(&src)
            .status()
            .unwrap_or_else(|e| {
                panic!("the bonsai feature needs glslc from the Vulkan SDK to compile its kernels ({e}); set VULKAN_SDK")
            });
        assert!(status.success(), "glslc failed for {}", src.display());
        all.extend(std::fs::read(&spv).unwrap());
        table.push_str(&format!("    (\"{name}\", include_bytes!(r\"{}\")),\n", spv.display()));
    }
    table.push_str("];\n");
    let all_path = out_dir.join("kernels.spvall");
    std::fs::write(&all_path, all).unwrap();
    table.push_str(&format!("pub static ALL: &[u8] = include_bytes!(r\"{}\");\n", all_path.display()));
    std::fs::write(out_dir.join("kernels_spv.rs"), table).unwrap();
}
