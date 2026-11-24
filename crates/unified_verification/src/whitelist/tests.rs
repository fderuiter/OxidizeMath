use super::*;
use crate::utils::test_utils::TempTestDir;

#[test]
fn test_resolve_whitelist_path_respects_candidate_order() {
    let dir = TempTestDir::new("whitelist_candidates");
    let missing = dir.join("missing.toml");
    let first = dir.join("first.toml");
    let second = dir.join("second.toml");
    fs::write(&second, "").unwrap();
    assert_eq!(
        resolve_whitelist_path(&[&missing, &first, &second]),
        Some(second.clone())
    );

    fs::write(&first, "").unwrap();
    assert_eq!(
        resolve_whitelist_path(&[&missing, &first, &second]),
        Some(first)
    );
    assert_eq!(resolve_whitelist_path(&[&missing]), None);
}

#[test]
fn test_strict_and_fallback_loaders_preserve_aliases() {
    let dir = TempTestDir::new("whitelist_loaders");
    let valid = dir.join("valid.toml");
    fs::write(
        &valid,
        concat!(
            "[adr_verification]\nallowlist = [\"docs/adr/0002-portable-build-paths.md\"]\n",
            "[doc_verification]\nallowlist = [\"crates/demo\"]\n",
            "[entropy_guard]\nscan_targets = [\"crates\"]\n",
            "file_extensions = [\"rs\"]\npath_exemptions = [\"allow.rs\"]\n",
            "[missing_docs]\nvendored_crates = [\"egui_plot\"]\n",
            "[missing_docs.files]\n\"crates/demo/src/lib.rs\" = 2\n",
            "[math_doc_contract.files]\n\"crates/pure_math/src/lib.rs\" = 3\n",
            "[[workflow_linter.allowed_job_permissions]]\n",
            "workflow = \".github/workflows/ci.yml\"\njob = \"deploy\"\nwrite_scopes = [\"pages\"]\n",
            "[[dead_code]]\nfile = \"crates/demo/src/lib.rs\"\nidentifier = \"Item\"\n",
            "[[unused_dependencies]]\ncrate = \"crates/demo\"\ndependency = \"getrandom\"\nreason = \"wasm js feature\"\n",
        ),
    )
    .unwrap();

    let parsed = load_whitelist_strict(Some(&valid), &[] as &[&Path]).unwrap();
    assert_eq!(
        parsed.adr_verification.unwrap().allowlist,
        Some(vec!["docs/adr/0002-portable-build-paths.md".to_string()])
    );
    assert_eq!(
        parsed.doc_verification.unwrap().allowlist,
        Some(vec!["crates/demo".to_string()])
    );
    assert_eq!(
        parsed.entropy_guard.unwrap().allowlist,
        Some(vec!["allow.rs".to_string()])
    );
    assert_eq!(
        parsed.workflow_linter.unwrap().exceptions[0].allow_write,
        vec!["pages".to_string()]
    );
    assert_eq!(parsed.dead_code.len(), 1);
    assert_eq!(parsed.unused_dependencies.len(), 1);
    let md_cfg = parsed.missing_docs.unwrap();
    assert_eq!(md_cfg.vendored_crates, vec!["egui_plot".to_string()]);
    assert_eq!(md_cfg.files.get("crates/demo/src/lib.rs"), Some(&2));
}

#[test]
fn test_strict_and_fallback_loaders_reject_malformed_inputs() {
    let dir = TempTestDir::new("whitelist_loader_errors");
    let zero_cap = dir.join("zero_cap.toml");
    fs::write(
        &zero_cap,
        "[missing_docs.files]\n\"crates/demo/src/lib.rs\" = 0\n",
    )
    .unwrap();
    assert!(load_whitelist_strict(Some(&zero_cap), &[] as &[&Path]).is_err());

    let empty_reason = dir.join("empty_reason.toml");
    fs::write(
        &empty_reason,
        "[[unused_dependencies]]\ncrate = \"crates/demo\"\ndependency = \"getrandom\"\nreason = \"   \"\n",
    )
    .unwrap();
    assert!(load_whitelist_strict(Some(&empty_reason), &[] as &[&Path]).is_err());

    let invalid = dir.join("invalid.toml");
    fs::write(&invalid, "[broken").unwrap();
    assert!(load_whitelist_strict(Some(&invalid), &[] as &[&Path]).is_err());
    let fallback = load_whitelist_or_default(Some(&invalid), &[] as &[&Path]);
    assert!(fallback.doc_verification.is_none());
    assert!(fallback.dead_code.is_empty());
}

#[test]
fn test_math_doc_contract_whitelist_validation() {
    let dir = TempTestDir::new("whitelist_math_doc_contract");
    let valid = dir.join("valid_mdc.toml");
    fs::write(
        &valid,
        "[math_doc_contract.files]\n\"crates/pure_math/src/lib.rs\" = 3\n",
    )
    .unwrap();
    let parsed = load_whitelist_strict(Some(&valid), &[] as &[&Path]).unwrap();
    let mdc_cfg = parsed.math_doc_contract.unwrap();
    assert_eq!(mdc_cfg.files.get("crates/pure_math/src/lib.rs"), Some(&3));

    let zero_mdc_cap = dir.join("zero_mdc_cap.toml");
    fs::write(
        &zero_mdc_cap,
        "[math_doc_contract.files]\n\"crates/pure_math/src/lib.rs\" = 0\n",
    )
    .unwrap();
    assert!(load_whitelist_strict(Some(&zero_mdc_cap), &[] as &[&Path]).is_err());
}

#[test]
fn test_path_matches_lossy_per_gate_flags() {
    let entropy_opts = PathMatchOptions {
        trim_dot_slash: true,
        require_non_empty: false,
        match_file_name: true,
    };
    assert!(path_matches_lossy(
        Path::new("./crates/foo/bar.rs/"),
        "bar.rs",
        entropy_opts
    ));

    let workflow_opts = PathMatchOptions {
        trim_dot_slash: true,
        require_non_empty: true,
        match_file_name: false,
    };
    assert!(!path_matches_lossy(
        Path::new(".github/workflows/ci.yml"),
        "",
        workflow_opts
    ));
    assert!(!path_matches_lossy(
        Path::new(".github/workflows/ci.yml"),
        "./",
        workflow_opts
    ));
    assert!(!path_matches_lossy(
        Path::new(".github/workflows/ci.yml/"),
        "ci.yml",
        workflow_opts
    ));
    assert!(path_matches_lossy(
        Path::new("./.github/workflows/ci.yml"),
        ".github\\workflows\\ci.yml",
        workflow_opts
    ));

    let policy_opts = PathMatchOptions {
        trim_dot_slash: false,
        require_non_empty: false,
        match_file_name: false,
    };
    assert!(!path_matches_lossy(
        Path::new("crates/foo/lib.rs"),
        "./crates/foo/lib.rs",
        policy_opts
    ));
    assert!(path_matches_lossy(
        Path::new("root/crates/foo/lib.rs"),
        "crates\\foo\\lib.rs",
        policy_opts
    ));
}

#[test]
#[cfg(unix)]
fn test_path_matches_lossy_preserves_non_utf8_prefix_segments() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let bad_dir = Path::new(OsStr::from_bytes(b"/tmp/non_utf8_\xff_dir"));
    let full_path = bad_dir.join("crates/demo/src/lib.rs");
    let opts = PathMatchOptions {
        trim_dot_slash: true,
        require_non_empty: true,
        match_file_name: false,
    };
    assert!(path_matches_lossy(
        &full_path,
        "crates/demo/src/lib.rs",
        opts
    ));
}

#[test]
fn test_crate_prefix_for_file_supports_nested_multi_segment_crates() {
    assert_eq!(
        debt::crate_prefix_for_file("math_explorer_gui/src/tabs/tests/mod.rs"),
        "math_explorer_gui"
    );
    assert_eq!(
        debt::crate_prefix_for_file("crates/pure_math/src/pure_math/mod.rs"),
        "crates/pure_math"
    );
    assert_eq!(
        debt::crate_prefix_for_file("crates/02_engines/engine_linalg/src/lu.rs"),
        "crates/02_engines/engine_linalg"
    );
    assert_eq!(
        debt::crate_prefix_for_file("crates/02_engines/engine_linalg/src/tests.rs"),
        "crates/02_engines/engine_linalg"
    );
}
