use super::*;

#[test]
fn test_load_whitelist_config_parsing_and_sibling_sections() {
    let dir = create_test_dir("test_load_config");
    let cfg_file = write_fixture(
        &dir,
        "test_whitelist.toml",
        concat!(
            "[doc_verification]\nallowlist = [\"crates/diagnostics\"]\n",
            "[entropy_guard]\nscan_paths = [\"custom_dir\"]\n",
            "target_extensions = [\"rs\", \"toml\", \"js\", \"html\"]\nallowlist = [\"fixture.rs\"]\n",
            "[[dead_code]]\nfile = \"crates/demo/src/lib.rs\"\n",
            "[[workflow_linter.exceptions]]\nworkflow = \".github/workflows/ci.yml\"\n",
            "job = \"deploy-pages\"\nallow_write = [\"pages\"]\n",
        ),
    );
    let cfg = load_whitelist_config(Some(&cfg_file)).expect("valid whitelist should parse");
    assert_eq!(cfg.scan_paths, Some(vec!["custom_dir".into()]));
    assert_eq!(
        cfg.target_extensions,
        Some(vec!["rs".into(), "toml".into(), "js".into(), "html".into()])
    );
    assert_eq!(cfg.allowlist, Some(vec!["fixture.rs".into()]));
}

#[test]
fn test_whitelist_fails_closed_on_invalid_or_unknown_fields() {
    let dir = create_test_dir("test_fail_closed_toml");
    let broken = write_fixture(&dir, "broken.toml", "invalid = toml [ [ syntax");
    assert!(load_whitelist_config(Some(&broken)).is_err());

    let inner = write_fixture(&dir, "i.toml", "[entropy_guard]\ntypo = [\"a.rs\"]\n");
    assert!(load_whitelist_config(Some(&inner)).is_err());

    let outer = write_fixture(&dir, "o.toml", "[entropy_gaurd]\nallowlist = [\"a.rs\"]\n");
    assert!(load_whitelist_config(Some(&outer)).is_err());

    assert!(load_whitelist_config(Some(&dir.join("non_existent.toml"))).is_err());
}

#[test]
fn test_check_entropy_fails_closed_and_allows_missing_default_whitelist() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(std::env::current_dir().unwrap());

    let root = create_test_dir("test_cwd_whitelist");
    let nested = root.join("a/b/c");
    write_fixture(&nested, "src/clean.rs", "pub fn ok() {}\n");
    std::env::set_current_dir(&nested).unwrap();

    assert!(check_entropy(&["src".into()]).is_empty());

    write_fixture(
        &nested,
        "verification_whitelist.toml",
        "[entropy_guard\nbad =",
    );
    let err_violations = check_entropy(&["src".into()]);
    assert_eq!(err_violations.len(), 1);
    assert!(err_violations[0].contains("Failed to parse"));

    write_fixture(
        &nested,
        "verification_whitelist.toml",
        "[entropy_guard]\nbad_key = 1\n",
    );
    let unknown_violations = check_entropy(&["src".into()]);
    assert_eq!(unknown_violations.len(), 1);
    assert!(unknown_violations[0].contains("Failed to parse"));
}

#[test]
fn test_whitelist_config_aliases() {
    let dir = create_test_dir("test_aliases");
    let cfg_file = write_fixture(
        &dir,
        "test_aliases.toml",
        "[entropy_guard]\nscan_targets = [\"custom_dir\"]\nfile_extensions = [\"rs\", \"toml\"]\npath_exemptions = [\"fixture.rs\"]\n",
    );
    let cfg = load_whitelist_config(Some(&cfg_file)).expect("aliases should parse");
    assert_eq!(cfg.scan_paths, Some(vec!["custom_dir".into()]));
    assert_eq!(
        cfg.target_extensions,
        Some(vec!["rs".into(), "toml".into()])
    );
    assert_eq!(cfg.allowlist, Some(vec!["fixture.rs".into()]));
}

#[test]
fn test_workspace_whitelist_and_default_scan_paths_parity() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(std::env::current_dir().unwrap());
    let root = crate::manifest_cache::workspace_root_dir();
    if root != Path::new(".") {
        std::env::set_current_dir(root).unwrap();
    }

    for required in [".githooks", ".env", "Dockerfile"] {
        assert!(DEFAULT_SCAN_PATHS.contains(&required));
    }

    let cfg = load_whitelist_config(None).expect("workspace whitelist should parse");
    let expected: Vec<String> = DEFAULT_SCAN_PATHS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert_eq!(cfg.scan_paths, Some(expected));
    let allowlist = cfg.allowlist.unwrap_or_default();
    assert!(!allowlist.iter().any(|p| p.contains("vulnerabilities.rs")));
    assert!(
        allowlist
            .iter()
            .any(|p| p == "unified_verification:api_baseline.json")
    );

    let violations = check_entropy(&[]);
    assert!(
        violations.is_empty(),
        "Workspace entropy violations: {violations:?}"
    );
}

#[test]
fn test_ascii_and_unicode_entropy_scores_and_integrity_digest_case() {
    assert_eq!(calculate_entropy(""), 0.0);
    assert_eq!(calculate_entropy("zzzzzzzz"), 0.0);
    assert!((calculate_entropy("01234567") - 3.0).abs() < 1e-6);
    let ascii_token = ["AbCdEfGh", "01234567", "IjKlMnOp", "89+/=_-z"].concat();
    let expected_shannon = {
        let mut counts = std::collections::HashMap::new();
        for ch in ascii_token.chars() {
            *counts.entry(ch).or_insert(0usize) += 1;
        }
        let len = ascii_token.chars().count() as f64;
        counts.values().fold(0.0, |acc, &c| {
            let p = c as f64 / len;
            acc - p * p.log2()
        })
    };
    assert!((calculate_entropy(&ascii_token) - expected_shannon).abs() < 1e-6);

    assert!((calculate_entropy("αβγδ") - 2.0).abs() < 1e-6);
    assert!((calculate_entropy("ααββ") - 1.0).abs() < 1e-6);

    assert!(is_integrity_digest("sha256-abcdef"));
    assert!(is_integrity_digest("SHA384-ABCDEF"));
    assert!(is_integrity_digest("Sha512-012345"));
    assert!(!is_integrity_digest("md5-012345"));
}

#[cfg(unix)]
#[test]
fn test_symlink_deduplication_and_allowlisted_symlink_skip() {
    use std::os::unix::fs::symlink;

    let dir = create_test_dir("test_symlink_dedup");
    let mock_secret = [
        "aZ9kL2mN5",
        "qR8_vW1xY",
        "4bC7dE0fG",
        "3hJ6iK9lM",
        "pQ2sT5uV8",
    ]
    .concat();
    let real_file = write_fixture(
        &dir,
        "sub/real.toml",
        &format!("token = \"{mock_secret}\"\n"),
    );
    let link_file = dir.join("sub/alias_link.toml");
    symlink(&real_file, &link_file).unwrap();

    let targets = vec![dir.to_str().unwrap().to_string()];
    let exts = vec!["toml".to_string()];
    let violations = scan_paths_internal(&targets, &exts, &[]);
    assert_eq!(
        violations.len(),
        1,
        "Symlink and target pointing to same canonical file must be scanned once: {violations:?}"
    );

    let allowed = vec!["real.toml".to_string()];
    let skipped = scan_paths_internal(&targets, &exts, &allowed);
    assert!(
        skipped.is_empty(),
        "Allowlisted canonical target must also skip symlink alias: {skipped:?}"
    );
}
