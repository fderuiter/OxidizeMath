//! Unit tests for dynamic `#![no_std]` crate discovery, feature legs, and CI parity.

use super::*;
use std::env;
use std::path::Path;

#[test]
fn discovers_all_workspace_no_std_crates_and_feature_capabilities() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let crates = discover_no_std_crates(&ws_root).expect("discover workspace #![no_std] crates");
    let names: Vec<&str> = crates.iter().map(|c| c.package_name.as_str()).collect();

    for expected in [
        "math_diagnostics",
        "math_memory",
        "math_rng",
        "math_scalars",
        "verified_engine",
        "engine_linalg",
        "engine_opt",
        "math_commons",
        "engine_diffeq",
        "engine_pde",
    ] {
        assert!(
            names.contains(&expected),
            "Expected `{expected}` to be discovered, got {names:?}"
        );
    }
    for excluded in ["oxidize_core", "pure_math", "domain_physics", "xtask"] {
        assert!(
            !names.contains(&excluded),
            "Did not expect std crate `{excluded}` in #![no_std] set"
        );
    }

    let rng = crates
        .iter()
        .find(|c| c.package_name == "math_rng")
        .unwrap();
    assert!(rng.has_alloc && rng.has_serde && rng.has_std && !rng.default_has_std);

    let mc = crates
        .iter()
        .find(|c| c.package_name == "math_commons")
        .unwrap();
    assert!(mc.has_alloc && mc.has_serde && mc.has_std && mc.default_has_std);

    let ve = crates
        .iter()
        .find(|c| c.package_name == "verified_engine")
        .unwrap();
    assert!(!ve.has_alloc && !ve.has_serde && ve.has_std && ve.default_has_std);
}

#[test]
fn dynamically_enrolls_new_no_std_crates_in_synthetic_workspace() {
    let temp = env::temp_dir().join(format!("xtask_no_std_synth_{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp);

    fs::create_dir_all(temp.join("crates/00_foundation/new_kernel/src")).unwrap();
    fs::create_dir_all(temp.join("crates/01_pure/pure_std/src")).unwrap();
    fs::create_dir_all(temp.join("crates/00_foundation/macro_helper/src")).unwrap();

    fs::write(
        temp.join("Cargo.toml"),
        "[workspace]\nmembers = [\n  \"crates/00_foundation/new_kernel\",\n  \"crates/01_pure/pure_std\",\n  \"crates/00_foundation/macro_helper\",\n]\n",
    )
    .unwrap();

    fs::write(
        temp.join("crates/00_foundation/new_kernel/Cargo.toml"),
        "[package]\nname = \"new_kernel\"\nversion = \"0.1.0\"\n\n[features]\ndefault = []\nserde = [\n  \"dep:serde\",\n]\nstd = [\"serde?/std\"]\n",
    )
    .unwrap();
    fs::write(
        temp.join("crates/00_foundation/new_kernel/src/lib.rs"),
        "//! New bare-metal kernel.\n#![cfg_attr(not(feature = \"std\"), no_std)]\npub fn step(x: f64) -> f64 { x }\n",
    )
    .unwrap();

    fs::write(
        temp.join("crates/01_pure/pure_std/Cargo.toml"),
        "[package]\nname = \"pure_std\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        temp.join("crates/01_pure/pure_std/src/lib.rs"),
        "//! Mentions #![no_std] in a doc comment only.\npub fn host_only() {}\n",
    )
    .unwrap();

    fs::write(
        temp.join("crates/00_foundation/macro_helper/Cargo.toml"),
        "[package]\nname = \"macro_helper\"\nversion = \"0.1.0\"\n[lib]\nproc-macro = true\n",
    )
    .unwrap();
    fs::write(
        temp.join("crates/00_foundation/macro_helper/src/lib.rs"),
        "#![no_std]\n",
    )
    .unwrap();

    let discovered = discover_no_std_crates(&temp).expect("discover synthetic workspace");
    let _ = fs::remove_dir_all(&temp);

    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].package_name, "new_kernel");
    assert!(!discovered[0].has_alloc);
    assert!(discovered[0].has_serde);
    assert!(discovered[0].has_std);
    assert!(!discovered[0].default_has_std);

    let legs = feature_legs_for_crate(&discovered[0]);
    let labels: Vec<&str> = legs.iter().map(|l| l.label).collect();
    assert_eq!(labels, vec!["default", "no-default-features", "serde"]);
    assert!(!legs[0].bare_metal_compatible);
    assert!(legs[1].bare_metal_compatible);
    assert!(legs[2].bare_metal_compatible);
}

#[test]
fn detects_std_and_alloc_feature_leaks_in_cargo_tree_output() {
    let clean = "\
math_rng v0.4.0 (/ws/crates/00_foundation/math_rng) [serde]
serde v1.0.229 [derive,serde_derive]
serde_core v1.0.229 [result]
serde_derive v1.0.229 (proc-macro) [default,std]
rand_chacha v0.3.1 []
";
    assert!(find_serde_feature_leaks(clean).is_empty());

    let leaky = "\
math_commons v0.4.0 (/ws/crates/02_engines/math_commons) [alloc,serde]
serde v1.0.229 [alloc,derive,std]
";
    let leaks = find_serde_feature_leaks(leaky);
    assert_eq!(leaks.len(), 3);
    assert!(
        leaks
            .iter()
            .any(|l| l.contains("math_commons") && l.contains("`alloc`"))
    );
    assert!(
        leaks
            .iter()
            .any(|l| l.contains("serde v1.0.229") && l.contains("`alloc`"))
    );
    assert!(
        leaks
            .iter()
            .any(|l| l.contains("serde v1.0.229") && l.contains("`std`"))
    );
}

#[test]
fn toolchain_and_ci_preflight_include_bare_metal_no_std_gate() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let toolchain =
        fs::read_to_string(ws_root.join("rust-toolchain.toml")).expect("read rust-toolchain.toml");
    assert!(
        toolchain.contains(BARE_METAL_TARGET),
        "rust-toolchain.toml must include {BARE_METAL_TARGET}"
    );

    let ci_yaml = fs::read_to_string(ws_root.join(".github/workflows/ci.yml"))
        .expect("read .github/workflows/ci.yml")
        .replace("\r\n", "\n");
    let preflight_start = ci_yaml.find("  pre-flight:\n").expect("pre-flight job");
    let build_start = ci_yaml
        .find("  build-and-test:\n")
        .expect("build-and-test job");
    let preflight_block = &ci_yaml[preflight_start..build_start];
    assert!(
        preflight_block.contains("targets: thumbv7em-none-eabihf"),
        "pre-flight Rust setup must install thumbv7em-none-eabihf"
    );
    assert!(
        preflight_block.contains("cargo run -p xtask -- check-no-std"),
        "pre-flight job must invoke `cargo run -p xtask -- check-no-std`"
    );
}
