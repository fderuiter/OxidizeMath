//! Workspace discovery and manifest parsing utilities.

#[path = "workspace_wasm.rs"]
mod workspace_wasm;

pub use workspace_wasm::{
    HOST_ONLY_WORKSPACE_PACKAGES, WasmCrateSpec, WasmTestTargetKind,
    build_wasm_cargo_test_invocations, discover_wasm_crates, parse_workspace_members_std,
};

use std::fs;
use std::path::{Path, PathBuf};

const MAX_ANCESTOR_DEPTH: usize = 32;
const RELATIVE_ROOT_FALLBACKS: &[&str] = &["..", "../..", "../../..", "../../../.."];

/// Locates the workspace root directory by walking ancestor directories from `start`.
///
/// Checks up to 32 ancestor directories for a `Cargo.toml` containing a `[workspace]`
/// section with a valid `members` array.
///
/// # Example
/// ```
/// use oxidize_core::workspace::find_workspace_root_from;
/// use std::path::Path;
///
/// let _ = find_workspace_root_from(Path::new("."));
/// ```
pub fn find_workspace_root_from(start: &Path) -> Option<PathBuf> {
    let base = if start.is_file() {
        start.parent()?
    } else {
        start
    };
    for ancestor in base.ancestors().take(MAX_ANCESTOR_DEPTH) {
        if try_read_members(&ancestor.join("Cargo.toml")).is_some() {
            return Some(ancestor.to_path_buf());
        }
    }
    workspace_wasm::find_workspace_root(base)
}

/// Locates the workspace root directory relative to the current working directory.
///
/// Returns `Some(PathBuf::from("."))` when `./Cargo.toml` defines `[workspace]`,
/// otherwise searches ancestor directories of `std::env::current_dir()` (falling back
/// to bounded relative parent paths when `current_dir()` is unavailable).
///
/// # Example
/// ```
/// use oxidize_core::workspace::find_workspace_root;
///
/// let _ = find_workspace_root();
/// ```
pub fn find_workspace_root() -> Option<PathBuf> {
    if try_read_members(Path::new("Cargo.toml")).is_some() {
        return Some(PathBuf::from("."));
    }
    if let Ok(cwd) = std::env::current_dir()
        && let Some(root) = find_workspace_root_from(&cwd)
    {
        return Some(root);
    }
    for rel in RELATIVE_ROOT_FALLBACKS {
        let candidate = Path::new(rel);
        if try_read_members(&candidate.join("Cargo.toml")).is_some() {
            return Some(candidate.to_path_buf());
        }
    }
    None
}

/// Discovers workspace members from `Cargo.toml` manifests.
///
/// First checks `Cargo.toml` in the current directory. If it contains a `[workspace]`
/// definition, returns `(true, members)`.
///
/// Otherwise searches ancestor directories via [`find_workspace_root`] (handling
/// invocation from 1-level, 2-level, or nested crate subdirectories). If found there,
/// returns `(false, members)`.
///
/// The return tuple is `(is_root, members)`, where `is_root` is `true` if the manifest
/// was found in the current directory, and `false` if resolved via ancestor discovery.
pub fn get_workspace_members() -> (bool, Vec<String>) {
    if let Some(members) = try_read_members(Path::new("Cargo.toml")) {
        return (true, members);
    }
    if let Some(root) = find_workspace_root()
        && let Some(members) = try_read_members(&root.join("Cargo.toml"))
    {
        return (false, members);
    }
    (true, Vec::new())
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
pub fn parse_workspace_members(content: &str) -> Result<Vec<String>, String> {
    let table = content
        .parse::<toml::Table>()
        .map_err(|e| format!("Failed to parse Cargo.toml: {e}"))?;

    let members = table
        .get("workspace")
        .and_then(|w| w.as_table())
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .ok_or_else(|| "Could not find workspace.members array in Cargo.toml".to_string())?
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
        assert!(err.contains("Could not find workspace.members array"));
        assert!(parse_workspace_members_std(content).is_err());
    }

    #[test]
    fn test_parse_workspace_members_invalid_toml() {
        let content = "this is not valid toml = [[[";
        let err = parse_workspace_members(content).unwrap_err();
        assert!(err.contains("Failed to parse Cargo.toml"));
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
    fn test_discover_wasm_crates_covers_workspace_targets() {
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
    }
}
