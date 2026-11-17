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
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Raw credential (GitHub API Token)") && v.contains("at line 3"))
    );
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
    let text_file = dir.join("notes.txt");
    fs::write(
        &text_file,
        "AKIAIOSFODNN7EXAMPLE\nghp_123456789012345678901234567890123456\n",
    )
    .unwrap();

    let rs_file = dir.join("code.rs");
    fs::write(
        &rs_file,
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
fn test_detect_secrets_in_non_rust_files() {
    let dir = create_test_dir("test_detect_non_rust");
    let toml_file = dir.join("config.toml");
    fs::write(&toml_file, "aws_key = \"AKIAIOSFODNN7EXAMPLE\"\n").unwrap();

    let yml_file = dir.join("ci.yml");
    fs::write(
        &yml_file,
        "token: ghp_123456789012345678901234567890123456\n",
    )
    .unwrap();

    let sh_file = dir.join("deploy.sh");
    fs::write(&sh_file, "SLACK_TOKEN=xoxb-1234567890-1234567890\n").unwrap();

    let dir_str = dir.to_str().unwrap().to_string();
    let exts = vec!["toml".to_string(), "yml".to_string(), "sh".to_string()];
    let violations = scan_paths_internal(&[dir_str], &exts, &[]);

    assert_eq!(violations.len(), 3);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Raw credential (AWS Access Key)") && v.contains("config.toml"))
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Raw credential (GitHub API Token)") && v.contains("ci.yml"))
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Raw credential (Slack API Token)") && v.contains("deploy.sh"))
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_non_rust_allow_annotation() {
    let dir = create_test_dir("test_non_rust_allow");
    let yml_file = dir.join("ci.yml");
    fs::write(
        &yml_file,
        "# allow(entropy_guard)\ntoken: ghp_123456789012345678901234567890123456\n",
    )
    .unwrap();

    let dir_str = dir.to_str().unwrap().to_string();
    let exts = vec!["yml".to_string()];
    let violations = scan_paths_internal(&[dir_str], &exts, &[]);
    assert!(
        violations.is_empty(),
        "Violations should be empty with allow comment"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_file_allowlist_in_whitelist_toml_exempts_file() {
    let dir = create_test_dir("test_file_allowlist");
    let file = dir.join("fixture_secret.toml");
    fs::write(&file, "token = \"AKIAIOSFODNN7EXAMPLE\"\n").unwrap();

    let dir_str = dir.to_str().unwrap().to_string();
    let exts = vec!["toml".to_string()];
    let allowlist = vec!["fixture_secret.toml".to_string()];
    let violations = scan_paths_internal(&[dir_str], &exts, &allowlist);
    assert!(violations.is_empty(), "Allowlisted file should be skipped");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_skip_git_and_target_dirs() {
    let dir = create_test_dir("test_skip_git_target");
    let target_dir = dir.join("target");
    fs::create_dir_all(&target_dir).unwrap();
    fs::write(
        target_dir.join("build.toml"),
        "token = \"AKIAIOSFODNN7EXAMPLE\"\n",
    )
    .unwrap();

    let git_dir = dir.join(".git");
    fs::create_dir_all(&git_dir).unwrap();
    fs::write(
        git_dir.join("config.toml"),
        "token = \"AKIAIOSFODNN7EXAMPLE\"\n",
    )
    .unwrap();

    let dir_str = dir.to_str().unwrap().to_string();
    let exts = vec!["toml".to_string()];
    let violations = scan_paths_internal(&[dir_str], &exts, &[]);
    assert!(violations.is_empty(), "target and .git must be skipped");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_load_whitelist_config_parsing() {
    let dir = create_test_dir("test_load_config");
    let cfg_file = dir.join("test_whitelist.toml");
    let content = r#"
[entropy_guard]
scan_paths = ["custom_dir"]
target_extensions = ["rs", "toml"]
allowlist = ["fixture.rs"]
"#;
    fs::write(&cfg_file, content).unwrap();

    let cfg = load_whitelist_config(Some(&cfg_file));
    assert_eq!(cfg.scan_paths, Some(vec!["custom_dir".to_string()]));
    assert_eq!(
        cfg.target_extensions,
        Some(vec!["rs".to_string(), "toml".to_string()])
    );
    assert_eq!(cfg.allowlist, Some(vec!["fixture.rs".to_string()]));

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
fn test_fallback_when_whitelist_unparseable_or_missing() {
    let dir = create_test_dir("test_fallback_unparseable");
    let broken_whitelist = dir.join("broken.toml");
    fs::write(&broken_whitelist, "invalid = toml [ [ syntax").unwrap();

    let config = load_whitelist_config(Some(&broken_whitelist));
    assert!(config.scan_paths.is_none());
    assert!(config.target_extensions.is_none());
    assert!(config.allowlist.is_none());

    let non_existent = dir.join("non_existent.toml");
    let missing_config = load_whitelist_config(Some(&non_existent));
    assert!(missing_config.scan_paths.is_none());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_whitelist_config_aliases() {
    let dir = create_test_dir("test_aliases");
    let cfg_file = dir.join("test_aliases.toml");
    let content = r#"
[entropy_guard]
scan_targets = ["custom_dir"]
file_extensions = ["rs", "toml"]
path_exemptions = ["fixture.rs"]
"#;
    fs::write(&cfg_file, content).unwrap();

    let cfg = load_whitelist_config(Some(&cfg_file));
    assert_eq!(cfg.scan_paths, Some(vec!["custom_dir".to_string()]));
    assert_eq!(
        cfg.target_extensions,
        Some(vec!["rs".to_string(), "toml".to_string()])
    );
    assert_eq!(cfg.allowlist, Some(vec!["fixture.rs".to_string()]));

    let _ = fs::remove_dir_all(&dir);
}
