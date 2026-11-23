use super::*;
use crate::utils::test_utils::{CurrentDirGuard, TempTestDir};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;

#[test]
fn test_select_touched_policy_members_and_global_triggers() {
    let members = vec![
        "crates/pure_math".to_string(),
        "crates/domain_physics".to_string(),
        "math_explorer_gui".to_string(),
    ];

    assert!(select_touched_policy_members(&["README.md".to_string()], &members).is_empty());

    assert_eq!(
        select_touched_policy_members(
            &[
                "crates/domain_physics/src/quantum.rs".to_string(),
                "docs/README.md".to_string()
            ],
            &members
        ),
        vec!["crates/domain_physics".to_string()]
    );

    assert_eq!(
        select_touched_policy_members(&["math_explorer_gui/Cargo.toml".to_string()], &members),
        vec!["math_explorer_gui".to_string()]
    );

    assert_eq!(
        select_touched_policy_members(&["verification_whitelist.toml".to_string()], &members),
        members
    );
    for root_cfg in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".clippy.toml",
        ".rustfmt.toml",
        ".cargo/config.toml",
    ] {
        assert_eq!(
            select_touched_policy_members(&[root_cfg.to_string()], &members),
            members,
            "Expected full member escalation for {root_cfg}"
        );
    }
}

#[test]
fn test_select_touched_policy_members_includes_markdown_tests_for_md_edits() {
    let full_members = vec![
        "crates/diagnostics".to_string(),
        "crates/domain_physics".to_string(),
        "crates/markdown_tests".to_string(),
        "math_explorer_gui".to_string(),
    ];
    assert_eq!(
        select_touched_policy_members(&["crates/diagnostics/README.md".to_string()], &full_members),
        vec![
            "crates/diagnostics".to_string(),
            "crates/markdown_tests".to_string()
        ]
    );
    assert_eq!(
        select_touched_policy_members(&["docs/README.md".to_string()], &full_members),
        vec!["crates/markdown_tests".to_string()]
    );
}

#[test]
fn test_run_batch_staged_for_paths_catches_profile_and_doc_failures() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let tmp = TempTestDir::new("batch_staged_profile_doc");
    let orig = env::current_dir().unwrap();
    let _guard = CurrentDirGuard(orig);

    env::set_current_dir(&*tmp).unwrap();
    fs::create_dir_all("crates/demo/src").unwrap();
    fs::write(
        "Cargo.toml",
        "[workspace]\nmembers = [\"crates/demo\"]\n\n[profile.release]\noverflow-checks = false\n",
    )
    .unwrap();
    fs::write(
        "crates/demo/Cargo.toml",
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(
        "crates/demo/src/lib.rs",
        "//! Demo crate.\npub fn ok() {}\n",
    )
    .unwrap();

    let staged = vec!["Cargo.toml".to_string()];
    let members = vec!["crates/demo".to_string()];
    let mut cache = AstCache::new();
    assert!(
        !run_batch_staged_for_paths(&staged, &staged, &members, &mut cache),
        "Disabled overflow-checks or missing crate README must fail batch-staged"
    );
}

#[test]
fn test_run_batch_staged_for_paths_catches_guide_gate_violations() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let tmp = TempTestDir::new("batch_staged_guide_gate");
    let orig = env::current_dir().unwrap();
    let _guard = CurrentDirGuard(orig);

    env::set_current_dir(&*tmp).unwrap();
    fs::create_dir_all("apps/xtask/src").unwrap();
    fs::write(
        "Cargo.toml",
        "[workspace]\nmembers = [\"apps/xtask\"]\n\n[profile.release]\noverflow-checks = true\n",
    )
    .unwrap();
    fs::write(
        "apps/xtask/Cargo.toml",
        "[package]\nname = \"xtask\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write("apps/xtask/README.md", "# xtask\n\nValid helper crate.\n").unwrap();
    fs::write(
        "apps/xtask/src/main.rs",
        "#![doc = include_str!(\"../README.md\")]\n#![cfg_attr(any(), verified(opt_out = \"tests\"))]\nfn main() {}\n",
    )
    .unwrap();
    fs::write(
        "apps/xtask/src/papers.rs",
        "#![cfg_attr(any(), verified(opt_out = \"tests\"))]\npub fn compile() { let _ = \"pdflatex\"; }\n",
    )
    .unwrap();
    fs::write("README.md", "Build academic papers with Tectonic 0.15.0.\n").unwrap();

    let staged = vec!["README.md".to_string()];
    let members = vec!["apps/xtask".to_string()];
    let mut cache = AstCache::new();
    assert!(
        !run_batch_staged_for_paths(&staged, &staged, &members, &mut cache),
        "Stale Tectonic guide instruction must fail batch-staged"
    );
}

#[test]
fn test_batch_ast_parses_each_unique_workspace_file_at_most_once() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let orig = env::current_dir().unwrap();
    let _guard = CurrentDirGuard(orig);
    let root = crate::manifest_cache::workspace_root_dir();
    if root != Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let (_, members) = crate::manifest_cache::discover_workspace();
    let mut cache = AstCache::new();

    assert!(crate::ast_visitor::run_clone_detector_with_cache(
        &mut cache
    ));
    let apis = crate::api_drift::extract_apis_in_dir_with_cache(Path::new("."), &mut cache);
    assert!(!apis.is_empty());
    assert!(run_traceability_with_cache(false, &mut cache));
    assert!(crate::entropy_guard::check_entropy_with_cache(&[], &mut cache).is_empty());
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    assert!(crate::policy_audit::run_policy_audit_with_cache(
        &member_refs,
        &mut cache
    ));

    assert!(cache.parse_calls() > 500);
    assert_eq!(cache.parse_calls(), cache.len());

    let canonical_files: HashSet<PathBuf> = cache
        .keys()
        .map(|k| fs::canonicalize(k).unwrap_or_else(|_| PathBuf::from(k)))
        .collect();
    assert_eq!(
        canonical_files.len(),
        cache.parse_calls(),
        "Every physical Rust file in the workspace must be parsed by syn::parse_file at most once"
    );
}

#[test]
fn test_batch_staged_enforces_adr_for_staged_core_changes() {
    let dir = TempTestDir::new("batch_staged_adr_gate");
    let run_git = |args: &[&str]| {
        let status = Command::new("git")
            .current_dir(&*dir)
            .args(args)
            .status()
            .unwrap();
        assert!(status.success(), "git {:?} failed", args);
    };
    run_git(&["init"]);
    run_git(&["branch", "-M", "main"]);
    run_git(&["config", "user.email", "test@example.com"]);
    run_git(&["config", "user.name", "Test"]);
    fs::write(dir.join("README.md"), "init\n").unwrap();
    run_git(&["add", "."]);
    run_git(&["commit", "-m", "init"]);

    fs::create_dir_all(dir.join("crates/pure_math/src")).unwrap();
    fs::write(
        dir.join("crates/pure_math/src/lib.rs"),
        "pub fn unrecorded() -> i32 { 1 }\n",
    )
    .unwrap();
    run_git(&["add", "crates/pure_math/src/lib.rs"]);

    let mut cache = AstCache::new();
    let staged_only_core = vec!["crates/pure_math/src/lib.rs".to_string()];
    assert!(!run_batch_staged_for_paths_in_dir(
        &dir,
        &staged_only_core,
        &[],
        &[],
        &mut cache,
    ));

    fs::create_dir_all(dir.join("docs/adr")).unwrap();
    fs::write(
        dir.join("docs/adr/0001-core-edit.md"),
        "# ADR 0001: Core Edit\n\n## Status\n\nAccepted\n",
    )
    .unwrap();
    run_git(&["add", "docs/adr/0001-core-edit.md"]);

    let staged_with_adr = vec![
        "crates/pure_math/src/lib.rs".to_string(),
        "docs/adr/0001-core-edit.md".to_string(),
    ];
    assert!(run_batch_staged_for_paths_in_dir(
        &dir,
        &staged_with_adr,
        &[],
        &[],
        &mut cache,
    ));
}

#[test]
fn test_batch_static_parses_each_member_manifest_and_checks_lints_once() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let orig = env::current_dir().unwrap();
    let _guard = CurrentDirGuard(orig);
    let root = crate::manifest_cache::workspace_root_dir();
    if root != Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let (_, members) = crate::manifest_cache::discover_workspace();
    let mut ast_cache = AstCache::new();
    let mut manifest_cache = crate::manifest_cache::ManifestCache::new();

    assert!(run_batch_static_with_caches(
        &[],
        &mut ast_cache,
        &mut manifest_cache,
    ));

    assert_eq!(manifest_cache.lint_check_calls(), members.len());
    assert_eq!(manifest_cache.parse_calls(), members.len() + 1);
    for member in &members {
        let manifest = Path::new(member).join("Cargo.toml");
        assert_eq!(manifest_cache.parse_count_for(&manifest), 1);
    }
}
