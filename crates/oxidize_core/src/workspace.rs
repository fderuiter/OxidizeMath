//! Workspace discovery and manifest parsing utilities.

#[path = "workspace_wasm.rs"]
mod workspace_wasm;

pub use workspace_wasm::{
    HOST_ONLY_WORKSPACE_PACKAGES, WasmCrateSpec, WasmTestTargetKind, WorkspaceError,
    build_wasm_cargo_test_invocations, discover_wasm_crates, find_workspace_root,
    parse_workspace_members_std,
};

use std::fs;
use std::path::Path;

/// Discovers workspace members from `Cargo.toml` manifests.
///
/// First checks `Cargo.toml` in the current directory. If it contains a `[workspace]`
/// definition, returns `(true, members)`.
///
/// Otherwise walks ancestor directories (`../Cargo.toml`, `../../Cargo.toml`, etc.)
/// to locate the workspace root manifest and returns `(false, members)`.
pub fn get_workspace_members() -> (bool, Vec<String>) {
    if let Some(members) = try_read_members(Path::new("Cargo.toml")) {
        return (true, members);
    }
    if let Ok(cwd) = std::env::current_dir()
        && let Some(root) = find_workspace_root(&cwd)
        && let Some(members) = try_read_members(&root.join("Cargo.toml"))
    {
        return (false, members);
    }
    get_workspace_members_from_paths(Path::new("Cargo.toml"), Path::new("../../Cargo.toml"))
}

fn try_read_members(path: &Path) -> Option<Vec<String>> {
    let content = fs::read_to_string(path).ok()?;
    if content.contains("[workspace]") {
        parse_workspace_members(&content).ok()
    } else {
        None
    }
}

/// Helper function to discover workspace members given primary and fallback manifest paths.
pub fn get_workspace_members_from_paths(primary: &Path, fallback: &Path) -> (bool, Vec<String>) {
    if let Some(members) = try_read_members(primary) {
        return (true, members);
    }

    if let Some(members) = try_read_members(fallback) {
        return (false, members);
    }

    (true, Vec::new())
}

/// Parses the `workspace.members` array from a TOML string.
pub fn parse_workspace_members(content: &str) -> Result<Vec<String>, WorkspaceError> {
    let table = content
        .parse::<toml::Table>()
        .map_err(|e| WorkspaceError::InvalidManifestSyntax(e.to_string()))?;

    let members = table
        .get("workspace")
        .and_then(|w| w.as_table())
        .ok_or(WorkspaceError::MissingWorkspaceSection)?
        .get("members")
        .and_then(|m| m.as_array())
        .ok_or(WorkspaceError::MissingMembersArray)?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    Ok(members)
}

#[cfg(test)]
mod tests {
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
    fn test_discover_wasm_crates_covers_workspace_targets() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = find_workspace_root(manifest_dir).expect("must find workspace root");
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
    }
}
