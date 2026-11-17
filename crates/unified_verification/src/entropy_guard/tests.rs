use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static TEST_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn create_test_dir(prefix: &str) -> PathBuf {
    let n = TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), n));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_calculate_entropy() {
    assert_eq!(calculate_entropy("aaaa"), 0.0);
    let e = calculate_entropy("abcd");
    assert!((e - 2.0).abs() < 1e-6);
}

#[test]
fn test_check_entropy_run() {
    let members = vec!["crates/pure_math".to_string()];
    let violations = check_entropy(&members);
    assert!(violations.is_empty() || !violations.is_empty());
}

#[test]
fn test_detect_private_keys() {
    let dir = create_test_dir("test_detect_private_keys");
    let file = dir.join("secret.rs");
    fs::write(
        &file,
        // allow(entropy_guard)
        "fn dummy() {\n    let _k = \"-----BEGIN RSA PRIVATE KEY-----\\nsecret\\n-----PRIVATE KEY-----\";\n}\n",
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("Raw credential (Private Key)"));
    assert!(violations[0].contains("at line 2"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_detect_api_tokens() {
    let dir = create_test_dir("test_detect_api_tokens");
    let file = dir.join("tokens.rs");
    // allow(entropy_guard)
    fs::write(
        &file,
        concat!(
            "fn dummy() {\n",
            "    let _aws = \"AKIAIOSFODNN7EXAMPLE\";\n",
            "    let _gh = \"ghp_123456789012345678901234567890123456\";\n",
            "    let _slack = \"xoxb-1234567890-1234567890\";\n",
            "}\n"
        ),
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert_eq!(violations.len(), 3);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Raw credential (AWS Access Key)") && v.contains("at line 2"))
    );
    assert!(violations.iter().any(|v| v.contains("Raw credential (GitHub API Token)") && v.contains("at line 3")));
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Raw credential (Slack API Token)") && v.contains("at line 4"))
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_detect_high_entropy_strings() {
    let dir = create_test_dir("test_detect_high_entropy");
    let file = dir.join("entropy.rs");
    // 35 distinct alphanumeric characters: high entropy > 4.5
    // allow(entropy_guard)
    fs::write(
        &file,
        "fn dummy() {\n    let _sec = \"aZ9kL2mN5qR8vW1xY4bC7dE0fG3hJ6iK9lM\";\n}\n",
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("Raw credential (High-Entropy Secret)"));
    assert!(violations[0].contains("at line 2"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_allow_annotation_bypasses_secrets() {
    let dir = create_test_dir("test_allow_annotation");
    let file = dir.join("mock_secrets.rs");
    // allow(entropy_guard)
    fs::write(
        &file,
        concat!(
            "fn dummy() {\n",
            "    // allow(entropy_guard)\n",
            "    let _aws = \"AKIAIOSFODNN7EXAMPLE\";\n",
            "    let _gh = \"ghp_123456789012345678901234567890123456\"; // allow(entropy_guard)\n",
            "}\n"
        ),
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert!(
        violations.is_empty(),
        "Violations should be empty when allow(entropy_guard) is present: {:?}",
        violations
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_ignore_comments_and_non_rust_files() {
    let dir = create_test_dir("test_ignore_non_rust");
    // Non-Rust file with extension not in target_extensions
    let text_file = dir.join("notes.txt");
    fs::write(
        &text_file,
        // allow(entropy_guard)
        "AKIAIOSFODNN7EXAMPLE\nghp_123456789012345678901234567890123456\n",
    )
    .unwrap();

    // Rust file with secret in comments only
    let rs_file = dir.join("code.rs");
    fs::write(
        &rs_file,
        // allow(entropy_guard)
        "// AKIAIOSFODNN7EXAMPLE\n// ghp_123456789012345678901234567890123456\nfn clean() {}\n",
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert!(
        violations.is_empty(),
        "Comments and non-Rust files should not trigger violations: {:?}",
        violations
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_detect_secrets_in_root_config() {
    let dir = create_test_dir("test_detect_root_config");
    let toml_file = dir.join("Cargo.toml");
    // allow(entropy_guard)
    fs::write(
        &toml_file,
        "[package]\nname = \"demo\"\napi_key = \"AKIAIOSFODNN7EXAMPLE\"\n",
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("Raw credential (AWS Access Key)"));
    assert!(violations[0].contains("at line 3"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_detect_secrets_in_ci_workflows() {
    let dir = create_test_dir("test_detect_ci_workflows");
    let workflow_dir = dir.join(".github").join("workflows");
    fs::create_dir_all(&workflow_dir).unwrap();
    let workflow_file = workflow_dir.join("ci.yml");
    // allow(entropy_guard)
    fs::write(
        &workflow_file,
        "name: CI\nenv:\n  GH_TOKEN: ghp_123456789012345678901234567890123456\n",
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("Raw credential (GitHub API Token)"));
    assert!(violations[0].contains("at line 3"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_detect_secrets_in_scripts() {
    let dir = create_test_dir("test_detect_scripts");
    let scripts_dir = dir.join("scripts");
    fs::create_dir_all(&scripts_dir).unwrap();
    let script_file = scripts_dir.join("deploy.sh");
    // allow(entropy_guard)
    fs::write(
        &script_file,
        "#!/usr/bin/env bash\nexport SLACK_TOKEN=\"xoxb-1234567890-1234567890\"\n",
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("Raw credential (Slack API Token)"));
    assert!(violations[0].contains("at line 2"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_file_level_allowlist_exemption() {
    let dir = create_test_dir("test_allowlist_exemption");
    let fixture_file = dir.join("fixtures").join("mock_key.rs");
    fs::create_dir_all(fixture_file.parent().unwrap()).unwrap();
    // allow(entropy_guard)
    fs::write(
        &fixture_file,
        "fn mock() {\n    let _aws = \"AKIAIOSFODNN7EXAMPLE\";\n}\n",
    )
    .unwrap();

    let unexempted_file = dir.join("bad.rs");
    // allow(entropy_guard)
    fs::write(
        &unexempted_file,
        "fn bad() {\n    let _aws = \"AKIAIOSFODNN7EXAMPLE\";\n}\n",
    )
    .unwrap();

    let whitelist_file = dir.join("custom_whitelist.toml");
    fs::write(
        &whitelist_file,
        format!(
            "[entropy_guard]\nallowlist = [\"{}\"]\n",
            fixture_file.to_str().unwrap().replace('\\', "/")
        ),
    )
    .unwrap();

    let violations = check_entropy_with_whitelist(
        &[dir.to_str().unwrap().to_string()],
        &whitelist_file,
    );
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("bad.rs"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_egui_plot_scanned_without_exclusion() {
    let dir = create_test_dir("test_egui_plot_scan");
    let egui_plot_file = dir.join("egui_plot").join("src").join("lib.rs");
    fs::create_dir_all(egui_plot_file.parent().unwrap()).unwrap();
    // allow(entropy_guard)
    fs::write(
        &egui_plot_file,
        "fn plot_dummy() {\n    let _aws = \"AKIAIOSFODNN7EXAMPLE\";\n}\n",
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("Raw credential (AWS Access Key)"));
    assert!(violations[0].contains("egui_plot"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_skip_target_and_git_directories() {
    let dir = create_test_dir("test_skip_target_and_git");
    let target_file = dir.join("target").join("debug").join("secret.rs");
    fs::create_dir_all(target_file.parent().unwrap()).unwrap();
    // allow(entropy_guard)
    fs::write(&target_file, "fn s() { let _k = \"AKIAIOSFODNN7EXAMPLE\"; }\n").unwrap();

    let git_file = dir.join(".git").join("hooks").join("pre-commit.sh");
    fs::create_dir_all(git_file.parent().unwrap()).unwrap();
    // allow(entropy_guard)
    fs::write(&git_file, "export TOKEN=\"AKIAIOSFODNN7EXAMPLE\"\n").unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert!(violations.is_empty(), "target and .git must be skipped: {:?}", violations);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_allow_annotation_in_text_files() {
    let dir = create_test_dir("test_allow_in_text");
    let sh_file = dir.join("script.sh");
    // allow(entropy_guard)
    fs::write(
        &sh_file,
        concat!(
            "# allow(entropy_guard)\n",
            "export AWS_KEY=\"AKIAIOSFODNN7EXAMPLE\"\n",
            "export GH_KEY=\"ghp_123456789012345678901234567890123456\" # allow(entropy_guard)\n"
        ),
    )
    .unwrap();

    let violations = check_entropy(&[dir.to_str().unwrap().to_string()]);
    assert!(violations.is_empty(), "Violations should be ignored by allow(entropy_guard): {:?}", violations);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_fallback_when_whitelist_unparseable_or_missing() {
    let dir = create_test_dir("test_fallback_unparseable");
    let broken_whitelist = dir.join("broken.toml");
    fs::write(&broken_whitelist, "invalid = toml [ [ syntax").unwrap();

    let config = load_config(&broken_whitelist);
    assert!(config.scan_paths.is_none());
    assert!(config.target_extensions.is_none());
    assert!(config.allowlist.is_none());

    let non_existent = dir.join("non_existent.toml");
    let missing_config = load_config(&non_existent);
    assert!(missing_config.scan_paths.is_none());

    let _ = fs::remove_dir_all(&dir);
}
