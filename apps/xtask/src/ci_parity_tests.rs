use crate::gates::{CARGO_DOC_ARGS, HOST_CLIPPY_ARGS, WASM_CLIPPY_ARGS, workspace_wasm};
use crate::setup;
use crate::smoke::CHECK_BENCHES_ARGS;
use std::fs;
use std::path::Path;

#[test]
fn test_native_bench_and_smoke_ci_parity() {
    assert_eq!(
        CHECK_BENCHES_ARGS,
        &["test", "--workspace", "--benches", "--all-features"]
    );
    let ci_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml");
    let ci_yaml = fs::read_to_string(&ci_path)
        .expect("failed to read ci.yml")
        .replace("\r\n", "\n");
    assert!(ci_yaml.contains(
        "- name: Check Workspace Benchmarks (Native)\n        if: matrix.target != 'wasm32-unknown-unknown'\n        run: cargo run -p xtask -- check-benches"
    ));
    assert!(ci_yaml.contains(
        "- name: Smoke Test Registered Examples (Native)\n        if: matrix.target != 'wasm32-unknown-unknown'\n        run: cargo run -p xtask -- smoke-examples"
    ));
}

#[test]
fn test_wasm_clippy_args_and_ci_parity() {
    assert_eq!(
        HOST_CLIPPY_ARGS,
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ]
    );
    assert_eq!(
        WASM_CLIPPY_ARGS,
        &[
            "clippy",
            "--workspace",
            "--lib",
            "--bins",
            "--tests",
            "--target",
            "wasm32-unknown-unknown",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ]
    );
    assert!(!WASM_CLIPPY_ARGS.contains(&"--all-targets"));
    assert!(!WASM_CLIPPY_ARGS.contains(&"--benches"));

    let ci_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml");
    let ci_yaml = fs::read_to_string(&ci_path)
        .expect("failed to read ci.yml")
        .replace("\r\n", "\n");
    let expected_host_cmd = format!("cargo {}", HOST_CLIPPY_ARGS.join(" "));
    assert!(ci_yaml.contains("name: Run Clippy (Native)"));
    assert!(ci_yaml.contains(&expected_host_cmd));
    let expected_wasm_cmd = format!("cargo {}", WASM_CLIPPY_ARGS.join(" "));
    assert!(ci_yaml.contains("name: Run Clippy (WASM)"));
    assert!(ci_yaml.contains(&expected_wasm_cmd));
    let expected_doc_cmd = format!("cargo {}", CARGO_DOC_ARGS.join(" "));
    assert!(ci_yaml.contains("name: Check Workspace Documentation"));
    assert!(ci_yaml.contains(&expected_doc_cmd));
    assert!(ci_yaml.contains(
        "- name: Run Workspace Tests (Native)\n        if: runner.os != 'Linux' && matrix.target != 'wasm32-unknown-unknown'\n        run: cargo test --workspace --lib --tests --all-features"
    ));

    for preflight_cmd in [
        "cargo run -p unified_verification --release -- check-profiles",
        "cargo run -p unified_verification --release -- check-duplicates",
        "cargo run -p unified_verification --release -- check-api-drift",
    ] {
        assert!(
            ci_yaml.contains(preflight_cmd),
            "ci.yml missing pre-flight command: {preflight_cmd}"
        );
    }
}

#[test]
fn test_wasm_test_and_smoke_ci_parity() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let discovered = workspace_wasm::discover_wasm_crates(&ws_root)
        .expect("must discover wasm32 workspace crates");
    let invocations = workspace_wasm::build_wasm_cargo_test_invocations(&discovered);
    assert_eq!(invocations.len(), 2);
    assert!(invocations[0].contains(&"oxidize_core".to_string()));
    assert!(invocations[0].contains(&"egui_plot".to_string()));
    assert!(invocations[0].contains(&"--lib".to_string()));
    assert_eq!(
        invocations[1],
        vec![
            "test",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            "math_explorer_gui",
            "--bins",
        ]
    );
    let ci_path = ws_root.join(".github/workflows/ci.yml");
    let ci_yaml = fs::read_to_string(&ci_path)
        .expect("failed to read ci.yml")
        .replace("\r\n", "\n");
    let build_pos = ci_yaml
        .find("trunk build --release")
        .expect("ci.yml must build web app with trunk");
    let smoke_pos = ci_yaml
        .find("cargo run -p xtask -- test-wasm --skip-build")
        .expect("ci.yml must run xtask test-wasm gate");
    let upload_pos = ci_yaml
        .find("name: Upload Web Dist Artifact")
        .expect("ci.yml must upload wasm_web_dist artifact");
    assert!(smoke_pos > build_pos && smoke_pos < upload_pos);
}

#[test]
fn test_setup_browser_prerequisite_checks_and_remediation() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root_toml = fs::read_to_string(ws_root.join("Cargo.toml")).unwrap();
    let expected = setup::parse_expected_wasm_bindgen_version(&root_toml).unwrap();
    assert_eq!(expected, "0.2.130");

    let ok_wb = setup::check_wasm_bindgen_version(Ok("wasm-bindgen 0.2.130".into()), &expected);
    assert!(ok_wb.failure.is_none());

    let mismatch_wb =
        setup::check_wasm_bindgen_version(Ok("wasm-bindgen 0.2.100".into()), &expected);
    let (ok, report) = setup::format_probe_report(&[
        mismatch_wb,
        setup::check_presence_probe("trunk", Err("not found".into()), "cargo binstall trunk"),
    ]);
    assert!(!ok);
    assert!(report.contains("0.2.130"));
    assert!(report.contains("Fix:"));
}
