use super::*;
use crate::gates::{
    CARGO_DOC_ARGS, HOST_CLIPPY_ARGS, WASM_CLIPPY_ARGS, unified_verification_bin, workspace_wasm,
};
use crate::markdown_links::{collect_md_files, extract_md_links};

fn local_hooks_path(dir: &Path) -> Option<String> {
    let out = Command::new("git")
        .current_dir(dir)
        .args(["config", "--local", "--get", "core.hooksPath"])
        .output()
        .expect("failed to execute git config");
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[test]
fn test_git_hooks_path_configuration() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let before = local_hooks_path(&ws_root);

    let temp_dir = env::temp_dir().join(format!("xtask_hook_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(temp_dir.join(".githooks")).expect("failed to create temp .githooks");
    let hook_file = temp_dir.join(".githooks/pre-commit");
    fs::write(&hook_file, "#!/bin/sh\nexit 0\n").expect("failed to write temp hook");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&hook_file).unwrap().permissions();
        perms.set_mode(0o644);
        fs::set_permissions(&hook_file, perms).unwrap();
    }

    let init_status = Command::new("git")
        .current_dir(&temp_dir)
        .arg("init")
        .status()
        .expect("failed to execute git init");
    assert!(init_status.success());

    configure_git_hooks_in_dir(&temp_dir);

    assert_eq!(local_hooks_path(&temp_dir).as_deref(), Some(".githooks"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&hook_file).unwrap().permissions().mode();
        assert_ne!(mode & 0o111, 0, "Temp hook should be made executable");
    }

    let _ = fs::remove_dir_all(&temp_dir);
    assert_eq!(
        local_hooks_path(&ws_root),
        before,
        "Workspace core.hooksPath must not be mutated by tests"
    );
}

#[test]
fn test_unified_verification_uses_dev_profile() {
    let bin = unified_verification_bin();
    let components: Vec<String> = bin
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    assert!(
        components.iter().any(|c| c == "debug"),
        "Expected debug profile path, got {}",
        bin.display()
    );
    assert!(
        !components.iter().any(|c| c == "release"),
        "Did not expect release profile path, got {}",
        bin.display()
    );
}

#[test]
#[cfg(unix)]
fn test_pre_commit_hook_executable() {
    use std::os::unix::fs::PermissionsExt;
    let hook_path = Path::new(".githooks/pre-commit");
    if hook_path.exists() {
        let metadata = fs::metadata(hook_path).expect("failed to read metadata");
        assert_ne!(
            metadata.permissions().mode() & 0o111,
            0,
            "Hook should be executable"
        );
    }
}

#[test]
fn test_xtask_helpers() {
    print_help();
    assert_eq!(extract_md_links("[d](README.md)"), vec!["README.md"]);
    let mut md = Vec::new();
    collect_md_files(Path::new("."), &mut md);
    assert!(!md.is_empty());
    check_markdown_links();
}

#[test]
fn test_verification_gates() {
    let output = Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "--help"])
        .output()
        .expect("failed to execute xtask --help");
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    for cmd in [
        "check-api-drift",
        "check-duplicates",
        "check-profiles",
        "check-public-debug",
        "lint-workflows",
        "check-docs",
        "check-cargo-docs",
        "traceability",
        "lint-all",
        "lint-staged",
        "check-entropy",
        "check-vulnerabilities",
        "compile-papers",
        "test-policy-audit",
        "test-wasm",
        "verify-records",
        "verify-suite",
        "regenerate-baseline",
    ] {
        assert!(help.contains(cmd), "Missing help entry for {cmd}");
    }
    assert!(!help.contains("Tectonic"));
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

#[test]
fn test_ci_scoped_change_path_filters() {
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ci_yaml = fs::read_to_string(ws_root.join(".github/workflows/ci.yml"))
        .expect("failed to read ci.yml")
        .replace("\r\n", "\n");
    let root_toml =
        fs::read_to_string(ws_root.join("Cargo.toml")).expect("failed to read Cargo.toml");
    let (step, blocks) = crate::ci_filter_tests::extract_detect_changes_step(&ci_yaml);
    crate::ci_filter_tests::assert_detect_changes_invariants(&ci_yaml, &step);
    crate::ci_filter_tests::assert_workspace_members_parity(&blocks, &root_toml);
    crate::ci_filter_tests::assert_scoped_triggers_and_skips(&blocks);
}
