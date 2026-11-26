use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(target_arch = "wasm32"))]
static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(target_arch = "wasm32"))]
fn next_nonce() -> usize {
    TEST_COUNTER.fetch_add(1, Ordering::SeqCst)
}

#[test]
fn test_parse_workspace_members_valid() {
    let content = r#"
[workspace]
members = [
    "crates/foo", # inline comment
    "crates/bar",
    "apps/baz"
]
"#;
    let members = parse_workspace_members(content).unwrap();
    assert_eq!(members, vec!["crates/foo", "crates/bar", "apps/baz"]);
    let std_members = parse_workspace_members_std(content).unwrap();
    assert_eq!(std_members, members);
}

#[test]
fn test_parse_workspace_members_missing_section() {
    let content = r#"
[package]
name = "non_workspace"
version = "0.1.0"
"#;
    let err = parse_workspace_members(content).unwrap_err();
    assert_eq!(err, WorkspaceError::MissingWorkspaceSection);
    assert!(
        err.to_string()
            .contains("Could not find [workspace] section")
    );
    assert!(parse_workspace_members_std(content).is_err());
}

#[test]
fn test_parse_workspace_members_invalid_toml() {
    let content = "this is not valid toml = [[[";
    let err = parse_workspace_members(content).unwrap_err();
    assert!(matches!(err, WorkspaceError::InvalidManifestSyntax(_)));
    assert!(err.to_string().contains("Failed to parse Cargo.toml"));
}

#[test]
fn test_parse_oxidize_metadata_fields() {
    let manifest = r#"
[package]
name = "domain_physics"
version.workspace = true
edition = "2024"

[package.metadata.oxidize]
tier = "03_domains"
release_role = "release"
facade_role = "domain"
facade_module = "physics"
traceability = true
math_contract = true
"#;
    let parsed = parse_oxidize_metadata(manifest).unwrap();
    assert_eq!(parsed.tier, Some(TierKind::Domains03));
    assert_eq!(parsed.release_role, Some(ReleaseRole::Release));
    assert_eq!(parsed.facade_role, FacadeRole::Domain);
    assert_eq!(parsed.facade_module.as_deref(), Some("physics"));
    assert_eq!(parsed.traceability, Some(true));
    assert_eq!(parsed.math_contract, Some(true));
    assert_eq!(parse_oxidize_metadata_std(manifest), parsed);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_get_workspace_members_from_paths_primary() {
    let nonce = next_nonce();
    let dir =
        std::env::temp_dir().join(format!("test_ws_primary_{}_{}", std::process::id(), nonce));
    fs::create_dir_all(&dir).unwrap();

    let primary_path = dir.join("Cargo.toml");
    let fallback_path = dir.join("fallback_Cargo.toml");

    let primary_toml = r#"
[workspace]
members = ["crates/primary_a", "crates/primary_b"]
"#;
    fs::write(&primary_path, primary_toml).unwrap();

    let (is_root, members) = get_workspace_members_from_paths(&primary_path, &fallback_path);

    assert!(is_root);
    assert_eq!(members, vec!["crates/primary_a", "crates/primary_b"]);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_get_workspace_members_from_paths_fallback() {
    let nonce = next_nonce();
    let dir =
        std::env::temp_dir().join(format!("test_ws_fallback_{}_{}", std::process::id(), nonce));
    fs::create_dir_all(&dir).unwrap();

    let primary_path = dir.join("sub_crate").join("Cargo.toml");
    fs::create_dir_all(primary_path.parent().unwrap()).unwrap();

    let primary_toml = r#"
[package]
name = "sub_crate"
"#;
    fs::write(&primary_path, primary_toml).unwrap();

    let fallback_path = dir.join("Cargo.toml");
    let fallback_toml = r#"
[workspace]
members = ["sub_crate", "other_crate"]
"#;
    fs::write(&fallback_path, fallback_toml).unwrap();

    let (is_root, members) = get_workspace_members_from_paths(&primary_path, &fallback_path);

    assert!(!is_root);
    assert_eq!(members, vec!["sub_crate", "other_crate"]);

    let _ = fs::remove_dir_all(dir);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_find_workspace_root_from_nested_depths() {
    let nonce = next_nonce();
    let dir = std::env::temp_dir().join(format!(
        "test_ws_ancestors_{}_{}",
        std::process::id(),
        nonce
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\"gui\", \"crates/core\"]\n",
    )
    .unwrap();

    let depth1 = dir.join("gui");
    let depth3 = dir.join("crates/core/src");
    fs::create_dir_all(&depth1).unwrap();
    fs::create_dir_all(&depth3).unwrap();
    fs::write(depth1.join("Cargo.toml"), "[package]\nname = \"gui\"\n").unwrap();
    fs::write(
        dir.join("crates/core/Cargo.toml"),
        "[package]\nname = \"core\"\n",
    )
    .unwrap();

    assert_eq!(find_workspace_root_from(&dir), Some(dir.clone()));
    assert_eq!(find_workspace_root_from(&depth1), Some(dir.clone()));
    assert_eq!(find_workspace_root_from(&depth3), Some(dir.clone()));

    let isolated = std::env::temp_dir().join(format!(
        "test_ws_isolated_{}_{}",
        std::process::id(),
        next_nonce()
    ));
    fs::create_dir_all(&isolated).unwrap();
    assert_eq!(find_workspace_root_from(&isolated), None);

    let _ = fs::remove_dir_all(dir);
    let _ = fs::remove_dir_all(isolated);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_discover_wasm_crates_and_packages_cover_workspace() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = find_workspace_root_from(manifest_dir).expect("must find workspace root");
    let crates = discover_wasm_crates(&root).expect("must discover wasm crates");
    let names: Vec<&str> = crates.iter().map(|c| c.package_name.as_str()).collect();
    for required in [
        "oxidize_core",
        "egui_plot",
        "math_explorer_gui",
        "math_explorer",
        "math_commons",
        "pure_math",
        "domain_physics",
    ] {
        assert!(names.contains(&required), "missing wasm crate {required}");
    }
    for excluded in [
        "xtask",
        "unified_verification",
        "markdown_tests",
        "verified_engine_macros",
    ] {
        assert!(
            !names.contains(&excluded),
            "host-only crate {excluded} must not be in wasm discovery"
        );
    }

    let packages = discover_workspace_packages(&root).expect("must discover workspace packages");
    assert_eq!(packages, discover_workspace_packages_std(&root).unwrap());
    for pkg in &packages {
        assert!(
            pkg.effective_tier().is_some(),
            "missing effective tier for {}",
            pkg.package_name
        );
        assert!(
            pkg.effective_release_role().is_some(),
            "missing effective release_role for {}",
            pkg.package_name
        );
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ast-analysis"))]
#[test]
fn test_gui_snapshot_code_dirs_include_tier2_engines_for_domain_features() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = find_workspace_root_from(manifest_dir).expect("must find workspace root");

    for feat in ["ai", "applied", "biology", "physics"] {
        let enabled = std::collections::BTreeSet::from([feat.to_string()]);
        let dirs = compute_gui_snapshot_code_dirs(&root, "math_explorer_gui", &enabled)
            .expect("must compute GUI snapshot code dirs");
        assert!(
            dirs.iter().any(|d| d.ends_with("engine_diffeq/src")),
            "feature `{feat}` must transitively include engine_diffeq/src, got: {dirs:?}"
        );
        assert!(
            dirs.iter().any(|d| d.ends_with("engine_pde/src")),
            "feature `{feat}` must transitively include engine_pde/src, got: {dirs:?}"
        );
        assert!(
            dirs.iter().any(|d| d.ends_with("engine_linalg/src")),
            "feature `{feat}` must include engine_linalg/src, got: {dirs:?}"
        );
        assert!(
            !dirs.iter().any(|d| d.contains("verified_engine")),
            "proc-macro harness crates must remain excluded from GUI snapshot dirs: {dirs:?}"
        );
    }
}
