//! Workspace discovery and manifest parsing utilities.

use std::fs;
use std::path::Path;

/// Discovers workspace members from Cargo.toml manifests.
///
/// First checks `Cargo.toml` in the current directory. If it contains a `[workspace]`
/// definition, returns `(true, members)`.
///
/// If `Cargo.toml` in the current directory does not contain `[workspace]`,
/// falls back to `../../Cargo.toml` (handling invocation from crate subdirectories).
/// If found there, returns `(false, members)`.
///
/// The return tuple is `(is_root, members)`, where `is_root` is `true` if the manifest
/// was found in the current directory, and `false` if resolved via relative fallback.
pub fn get_workspace_members() -> (bool, Vec<String>) {
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
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn next_nonce() -> usize {
        TEST_COUNTER.fetch_add(1, Ordering::SeqCst)
    }

    #[test]
    fn test_parse_workspace_members_valid() {
        let content = r#"
[workspace]
members = [
    "crates/foo",
    "crates/bar",
    "apps/baz"
]
"#;
        let members = parse_workspace_members(content).unwrap();
        assert_eq!(members, vec!["crates/foo", "crates/bar", "apps/baz"]);
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
    }

    #[test]
    fn test_parse_workspace_members_invalid_toml() {
        let content = "this is not valid toml = [[[";
        let err = parse_workspace_members(content).unwrap_err();
        assert!(err.contains("Failed to parse Cargo.toml"));
    }

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
}
