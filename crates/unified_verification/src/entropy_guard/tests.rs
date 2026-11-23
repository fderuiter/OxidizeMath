use super::*;
use crate::utils::test_utils::{CurrentDirGuard, TempTestDir};
use std::path::PathBuf;

mod config;

fn write_bytes_fixture(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&path, body).unwrap();
    path
}

fn create_test_dir(prefix: &str) -> TempTestDir {
    TempTestDir::new(prefix)
}

fn write_fixture(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&path, body).unwrap();
    path
}

fn scan_dir(dir: &Path) -> Vec<String> {
    check_entropy(&[dir.to_str().unwrap().to_string()])
}

fn has_match(violations: &[String], a: &str, b: &str) -> bool {
    violations.iter().any(|v| v.contains(a) && v.contains(b))
}

#[test]
fn test_calculate_entropy() {
    assert_eq!(calculate_entropy("aaaa"), 0.0);
    assert!((calculate_entropy("abcd") - 2.0).abs() < 1e-6);
}

#[test]
fn test_check_entropy_run() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let violations = check_entropy(&["crates/pure_math".to_string()]);
    assert!(violations.is_empty(), "Unexpected: {violations:?}");
}

#[test]
fn test_detect_private_keys() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_detect_private_keys");
    write_fixture(
        &dir,
        "secret.rs",
        "fn dummy() {\n    let _k = \"-----BEGIN RSA PRIVATE KEY-----\\nsecret\\n-----PRIVATE KEY-----\";\n}\n",
    );
    let violations = scan_dir(&dir);
    assert_eq!(violations.len(), 1);
    assert!(has_match(
        &violations,
        "Raw credential (Private Key)",
        "at line 2"
    ));
}

#[test]
fn test_detect_api_tokens() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_detect_api_tokens");
    write_fixture(
        &dir,
        "tokens.rs",
        concat!(
            "fn dummy() {\n",
            "    let _aws = \"AKIAIOSFODNN7EXAMPLE\";\n",
            "    let _gh = \"ghp_123456789012345678901234567890123456\";\n",
            "    let _slack = \"xoxb-1234567890-1234567890\";\n",
            "}\n"
        ),
    );
    let violations = scan_dir(&dir);
    assert_eq!(violations.len(), 3);
    assert!(has_match(&violations, "AWS Access Key", "at line 2"));
    assert!(has_match(&violations, "GitHub API Token", "at line 3"));
    assert!(has_match(&violations, "Slack API Token", "at line 4"));
}

#[test]
fn test_detect_high_entropy_strings() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_detect_high_entropy");
    write_fixture(
        &dir,
        "entropy.rs",
        "fn dummy() {\n    let _sec = \"aZ9kL2mN5qR8vW1xY4bC7dE0fG3hJ6iK9lM\";\n}\n",
    );
    let violations = scan_dir(&dir);
    assert_eq!(violations.len(), 1);
    assert!(has_match(&violations, "High-Entropy Secret", "at line 2"));
}

#[test]
fn test_inline_allow_annotation_does_not_bypass_secrets_or_ast() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_inline_allow_rejected");
    write_fixture(
        &dir,
        "mock_secrets.rs",
        concat!(
            "fn dummy() {\n",
            "    // allow(entropy_guard)\n",
            "    let _aws = \"AKIAIOSFODNN7EXAMPLE\";\n",
            "    let _gh = \"ghp_123456789012345678901234567890123456\"; // allow(entropy_guard)\n",
            "    let _r = rand::random::<u64>(); // allow(entropy_guard)\n",
            "}\n"
        ),
    );
    let violations = scan_dir(&dir);
    assert_eq!(
        violations.len(),
        3,
        "Inline bypass succeeded: {violations:?}"
    );
    assert!(has_match(&violations, "AWS Access Key", "at line 3"));
    assert!(has_match(&violations, "GitHub API Token", "at line 4"));
    assert!(has_match(&violations, "random()", "at line 5"));
}

#[test]
fn test_ignore_comments_and_non_rust_files() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_ignore_non_rust");
    write_fixture(
        &dir,
        "notes.txt",
        "AKIAIOSFODNN7EXAMPLE\nghp_123456789012345678901234567890123456\n",
    );
    write_fixture(
        &dir,
        "code.rs",
        "// AKIAIOSFODNN7EXAMPLE\n// ghp_123456789012345678901234567890123456\nfn clean() {}\n",
    );
    let violations = scan_dir(&dir);
    assert!(violations.is_empty(), "Unexpected: {violations:?}");
}

#[test]
fn test_detect_secrets_in_non_rust_files() {
    let dir = create_test_dir("test_detect_non_rust");
    write_fixture(&dir, "config.toml", "aws_key = \"AKIAIOSFODNN7EXAMPLE\"\n");
    write_fixture(
        &dir,
        "ci.yml",
        "token: ghp_123456789012345678901234567890123456\n",
    );
    write_fixture(
        &dir,
        "deploy.sh",
        "SLACK_TOKEN=xoxb-1234567890-1234567890\n",
    );

    let exts = vec!["toml".to_string(), "yml".to_string(), "sh".to_string()];
    let violations = scan_paths_internal(&[dir.to_str().unwrap().to_string()], &exts, &[]);
    assert_eq!(violations.len(), 3);
    assert!(has_match(&violations, "AWS Access Key", "config.toml"));
    assert!(has_match(&violations, "GitHub API Token", "ci.yml"));
    assert!(has_match(&violations, "Slack API Token", "deploy.sh"));
}

#[test]
fn test_non_rust_inline_allow_annotation_rejected() {
    let dir = create_test_dir("test_non_rust_allow");
    write_fixture(
        &dir,
        "ci.yml",
        "# allow(entropy_guard)\ntoken: ghp_123456789012345678901234567890123456\n",
    );
    let violations =
        scan_paths_internal(&[dir.to_str().unwrap().to_string()], &["yml".into()], &[]);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("Raw credential (GitHub API Token)"));
}

#[test]
fn test_detect_high_entropy_in_config_and_web_files() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_web_and_config_entropy");
    write_fixture(
        &dir,
        "settings.toml",
        "api_secret = \"aZ9kL2mN5qR8+vW1xY4bC7dE0fG3hJ6iK9lM==\"\n",
    );
    write_fixture(
        &dir,
        "deploy.yml",
        "signing_secret: aZ9kL2mN5qR8_vW1xY4bC7dE0fG3hJ6iK9lM-9\n",
    );
    write_fixture(
        &dir,
        "app.js",
        "const CLIENT_SECRET = 'pQ7wE3rT6yU9iO2aS5dF8gH1jK4lZ0xC3vB+nM=';\n",
    );
    write_fixture(
        &dir,
        "index.html",
        "<meta name=\"api-token\" content=\"mN8bV4cX1zL7kJ3hG9fD2sA6pO0iU5yT_rE-wQ\">\n",
    );

    let violations = scan_dir(&dir);
    assert_eq!(violations.len(), 4, "Expected 4 violations: {violations:?}");
    for file in ["settings.toml", "deploy.yml", "app.js", "index.html"] {
        assert!(has_match(&violations, "High-Entropy Secret", file));
    }
}

#[test]
fn test_ignore_hex_shas_sri_digests_and_paths_urls() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_ignore_hex_sri_paths");
    write_fixture(
        &dir,
        "workflow.yml",
        concat!(
            "uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683\n",
            "sha256: E3b0C44298fc1C149AFBF4c8996fb92427AE41e4649B934ca495991B7852b855\n",
            "url: https://cdn.example.com/Area51/ProjX9/ConfZ8/ModQ7/RunW6/bundle.js\n",
            "path: crates/Area51/ProjX9/ConfZ8/ModQ7/RunW6/BinV5\n",
        ),
    );
    write_fixture(
        &dir,
        "index.html",
        concat!(
            "<script src=\"/assets/Area51ProjX9ConfZ8ModQ7RunW6BinV5.js\" ",
            "integrity=\"sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU= ",
            "sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC ",
            "sha512-z4PhNX7vuL3xVChQ1m2AB9Yg5AULVxXcg/SpIdNs6c5H0NE8XYXysP+DGNKHfuwvY7kxvUdBeoGlODJ6+SfaPg==\"></script>\n",
        ),
    );
    write_fixture(
        &dir,
        "hashes.rs",
        concat!(
            "pub const HEX: &str = \"0xE3b0C44298fc1C149AFBF4c8996fb92427AE41e4649B934ca495991B7852b855\";\n",
            "pub const SRI: &str = \"sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC\";\n",
            "pub const REL_PATH: &str = \"crates/Area51/ProjX9/ConfZ8/ModQ7/RunW6\";\n",
        ),
    );

    let violations = scan_dir(&dir);
    assert!(violations.is_empty(), "False positives: {violations:?}");
}

#[test]
fn test_ast_terminal_segment_matching_and_deterministic_identifiers() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_ast_terminal_segments");
    let clean_rs = write_fixture(
        &dir,
        "deterministic.rs",
        concat!(
            "fn random_walk_1d_with_rng() -> u64 { 42 }\n",
            "fn custom_thread_rng_seed() -> u64 { 7 }\n",
            "struct CustomSystemTime;\n",
            "impl CustomSystemTime { fn now() -> u64 { 0 } }\n",
            "mod randomization { #[derive(Default)] pub struct RandomizationTool; }\n",
            "pub fn run() {\n",
            "    let _ = (random_walk_1d_with_rng(), custom_thread_rng_seed(), CustomSystemTime::now());\n",
            "    let _ = randomization::RandomizationTool::default();\n",
            "}\n",
        ),
    );
    assert!(check_entropy(&[clean_rs.to_str().unwrap().to_string()]).is_empty());

    let bad_rs = write_fixture(
        &dir,
        "unseeded.rs",
        concat!(
            "use rand::random as rnd;\n",
            "pub fn bad() {\n",
            "    let _a: u64 = rnd();\n",
            "    let _b = rand::thread_rng();\n",
            "    let _c = std::time::SystemTime::now();\n",
            "}\n",
        ),
    );
    let violations = check_entropy(&[bad_rs.to_str().unwrap().to_string()]);
    assert_eq!(violations.len(), 3, "Expected 3 violations: {violations:?}");
    assert!(has_match(&violations, "'random()'", "at line 3"));
    assert!(has_match(&violations, "'thread_rng()'", "at line 4"));
    assert!(has_match(&violations, "'SystemTime::now()'", "at line 5"));
}

#[test]
fn test_file_allowlist_in_whitelist_toml_exempts_file() {
    let dir = create_test_dir("test_file_allowlist");
    write_fixture(
        &dir,
        "fixture_secret.toml",
        "token = \"AKIAIOSFODNN7EXAMPLE\"\n",
    );
    let violations = scan_paths_internal(
        &[dir.to_str().unwrap().to_string()],
        &["toml".into()],
        &["fixture_secret.toml".into()],
    );
    assert!(violations.is_empty(), "Allowlisted file should be skipped");
}

#[test]
fn test_skip_git_and_target_dirs() {
    let dir = create_test_dir("test_skip_git_target");
    write_fixture(
        &dir,
        "target/build.toml",
        "token = \"AKIAIOSFODNN7EXAMPLE\"\n",
    );
    write_fixture(
        &dir,
        ".git/config.toml",
        "token = \"AKIAIOSFODNN7EXAMPLE\"\n",
    );
    let violations =
        scan_paths_internal(&[dir.to_str().unwrap().to_string()], &["toml".into()], &[]);
    assert!(violations.is_empty(), "target and .git must be skipped");
}

#[test]
fn test_egui_plot_scanned_without_exclusion() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_egui_plot_scan");
    write_fixture(
        &dir,
        "egui_plot/src/lib.rs",
        "fn plot_dummy() {\n    let _aws = \"AKIAIOSFODNN7EXAMPLE\";\n}\n",
    );
    let violations = scan_dir(&dir);
    assert_eq!(violations.len(), 1);
    assert!(has_match(&violations, "AWS Access Key", "egui_plot"));
}

#[test]
fn test_detect_secrets_in_macro_token_streams() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_macro_secrets");
    write_fixture(
        &dir,
        "macros.rs",
        concat!(
            "pub fn leak() -> Vec<String> {\n",
            "    let a = format!(\"key={}\", \"AKIAIOSFODNN7EXAMPLE\");\n",
            "    println!(\"{}\", r#\"ghp_123456789012345678901234567890123456\"#);\n",
            "    vec![a, String::from({ concat!(\"xoxb-1234567890-1234567890\") })]\n",
            "}\n",
        ),
    );
    let violations = scan_dir(&dir);
    assert_eq!(
        violations.len(),
        3,
        "Expected 3 macro secret hits: {violations:?}"
    );
    assert!(has_match(&violations, "AWS Access Key", "at line 2"));
    assert!(has_match(&violations, "GitHub API Token", "at line 3"));
    assert!(has_match(&violations, "Slack API Token", "at line 4"));
}

#[test]
fn test_detect_macro_entropy_calls_and_aliases() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_macro_entropy_calls");
    write_fixture(
        &dir,
        "macro_unseeded.rs",
        concat!(
            "use rand::random as rnd;\n",
            "use rand::thread_rng as trng;\n",
            "use std::time::SystemTime as SysTime;\n",
            "pub fn bad_macros() -> String {\n",
            "    let _v = vec![rand::random::<u64>(), rnd::<Option<Vec<u8>>>().is_some() as u64];\n",
            "    assert_eq!(format!(\"{:?}\", trng()), \"\");\n",
            "    format!(\"{:?}\", SysTime::now().duration_since(std::time::UNIX_EPOCH))\n",
            "}\n",
        ),
    );
    let violations = scan_dir(&dir);
    assert_eq!(
        violations.len(),
        4,
        "Expected 4 macro entropy hits: {violations:?}"
    );
    assert_eq!(
        violations
            .iter()
            .filter(|v| v.contains("'random()'") && v.contains("at line 5"))
            .count(),
        2
    );
    assert!(has_match(&violations, "'thread_rng()'", "at line 6"));
    assert!(has_match(&violations, "'SystemTime::now()'", "at line 7"));
}

#[test]
fn test_macro_ignores_method_calls_and_deterministic_paths() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_macro_clean_calls");
    write_fixture(
        &dir,
        "macro_clean.rs",
        concat!(
            "struct CustomSystemTime;\n",
            "impl CustomSystemTime { fn now<T>() -> u64 { 0 } }\n",
            "struct SeededRng;\n",
            "impl SeededRng { fn random<T>(&self) -> u64 { 1 } fn thread_rng(&self) -> u64 { 2 } fn now(&self) -> u64 { 3 } }\n",
            "fn random_walk_1d_with_rng() -> u64 { 42 }\n",
            "fn det<F>() -> u64 { 9 }\n",
            "macro_rules! def_fn { () => { fn random() -> u64 { 0 } }; }\n",
            "pub fn clean() -> String {\n",
            "    let rng = SeededRng;\n",
            "    format!(\"{} {} {} {} {} {}\", CustomSystemTime::now::<u8>(), CustomSystemTime::now(), random_walk_1d_with_rng(), rng.random::<u64>(), rng.thread_rng() + rng.now(), det::<fn() -> u64>())\n",
            "}\n",
        ),
    );
    let violations = scan_dir(&dir);
    assert!(
        violations.is_empty(),
        "Unexpected false positives: {violations:?}"
    );
}

#[test]
fn test_scan_githooks_dotfiles_and_extensionless_files() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_dotfiles_extensionless");
    write_fixture(
        &dir,
        ".githooks/pre-commit",
        "#!/bin/sh\nGH_TOKEN=ghp_123456789012345678901234567890123456\n",
    );
    write_fixture(&dir, ".env", "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n");
    write_fixture(
        &dir,
        ".env.local",
        "SLACK_TOKEN=xoxb-1234567890-1234567890\n",
    );
    write_fixture(
        &dir,
        "Dockerfile",
        "FROM scratch\nENV API_SECRET=aZ9kL2mN5qR8_vW1xY4bC7dE0fG3hJ6iK9lM-9\n",
    );
    let violations = scan_dir(&dir);
    assert_eq!(violations.len(), 4, "Expected 4 violations: {violations:?}");
    assert!(has_match(&violations, "GitHub API Token", "pre-commit"));
    assert!(has_match(&violations, "AWS Access Key", ".env"));
    assert!(has_match(&violations, "Slack API Token", ".env.local"));
    assert!(has_match(&violations, "High-Entropy Secret", "Dockerfile"));
}

#[test]
fn test_skip_license_files_and_binary_assets() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_skip_license_and_binaries");
    for lic in ["LICENSE", "NOTICE", "COPYING", "LICENSE-MIT"] {
        write_fixture(
            &dir,
            lic,
            "AKIAIOSFODNN7EXAMPLE aZ9kL2mN5qR8_vW1xY4bC7dE0fG3hJ6iK9lM-9\n",
        );
    }
    write_bytes_fixture(&dir, "icon.png", b"\x89PNG\r\n\x1a\nAKIAIOSFODNN7EXAMPLE");
    write_bytes_fixture(&dir, "elf_blob", b"\x7fELF\x00AKIAIOSFODNN7EXAMPLE\x00");
    write_bytes_fixture(&dir, ".bin_secret", b"\xff\xfeAKIAIOSFODNN7EXAMPLE");

    let violations = scan_dir(&dir);
    assert!(
        violations.is_empty(),
        "Expected clean skip, got: {violations:?}"
    );
}
