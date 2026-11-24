//! Workspace discovery and manifest parsing utilities.

#[path = "workspace_wasm.rs"]
mod workspace_wasm;

pub use workspace_wasm::{
    BROWSER_WASM_PACKAGES, HOST_ONLY_WORKSPACE_PACKAGES, WasmCrateSpec, WasmExecutionRuntime,
    WasmTestTargetKind, WorkspaceError, build_wasm_no_run_json_invocations, discover_wasm_crates,
    extract_wasm_test_executables_from_cargo_json, parse_workspace_members_std,
};

/// Constructs Cargo CLI argument vectors (`cargo test --target wasm32-unknown-unknown ...`)
/// covering all discovered library and binary crates.
#[must_use]
pub fn build_wasm_cargo_test_invocations(crates: &[WasmCrateSpec]) -> Vec<Vec<String>> {
    let refs: Vec<&WasmCrateSpec> = crates.iter().collect();
    workspace_wasm::build_cargo_test_group_invocations(&refs, false)
}

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
#[cfg(feature = "ast-analysis")]
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

#[cfg(not(feature = "ast-analysis"))]
fn validate_toml_bracket_balance(content: &str) -> Result<(), WorkspaceError> {
    let mut bracket_depth: i32 = 0;
    let mut in_double = false;
    let mut in_single = false;
    let mut in_comment = false;
    let mut escaped = false;

    for ch in content.chars() {
        if ch == '\n' {
            in_comment = false;
            escaped = false;
            continue;
        }
        if in_comment {
            continue;
        }
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_double {
            escaped = true;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
        } else if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if !in_double && !in_single {
            if ch == '#' {
                in_comment = true;
            } else if ch == '[' {
                bracket_depth += 1;
            } else if ch == ']' {
                bracket_depth -= 1;
                if bracket_depth < 0 {
                    return Err(WorkspaceError::InvalidManifestSyntax(
                        "unmatched ']'".to_string(),
                    ));
                }
            }
        }
    }

    if in_double || in_single || bracket_depth != 0 {
        return Err(WorkspaceError::InvalidManifestSyntax(
            "malformed TOML syntax".to_string(),
        ));
    }
    Ok(())
}

/// Parses the `workspace.members` array from a TOML string.
#[cfg(not(feature = "ast-analysis"))]
pub fn parse_workspace_members(content: &str) -> Result<Vec<String>, WorkspaceError> {
    validate_toml_bracket_balance(content)?;
    parse_workspace_members_std(content)
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

        let mut browser: Vec<&str> = crates
            .iter()
            .filter(|c| c.execution_runtime() == WasmExecutionRuntime::BrowserChromium)
            .map(|c| c.package_name.as_str())
            .collect();
        browser.sort_unstable();
        assert_eq!(
            browser,
            vec!["egui_plot", "math_explorer_gui", "oxidize_core"]
        );
        let node_crates: Vec<&str> = crates
            .iter()
            .filter(|c| c.execution_runtime() == WasmExecutionRuntime::NodeJs)
            .map(|c| c.package_name.as_str())
            .collect();
        assert_eq!(node_crates.len(), 16);
        assert!(node_crates.contains(&"math_explorer"));
        for new_engine in ["engine_linalg", "engine_diffeq", "engine_pde"] {
            assert!(
                node_crates.contains(&new_engine),
                "missing Node.js wasm engine crate {new_engine}"
            );
        }
        assert_eq!(
            build_wasm_no_run_json_invocations(&crates, WasmExecutionRuntime::BrowserChromium)
                .len(),
            2
        );
        assert_eq!(
            build_wasm_no_run_json_invocations(&crates, WasmExecutionRuntime::NodeJs).len(),
            1
        );
    }

    #[test]
    fn test_extract_wasm_test_executables_handles_spaces_and_filters_non_tests() {
        let sample = concat!(
            "{\"reason\":\"compiler-artifact\",\"profile\":{\"test\":false},\"executable\":\"/tmp/with space/non_test.wasm\"}\n",
            "{\"reason\":\"compiler-artifact\",\"profile\":{\"opt_level\":\"0\",\"test\":true},\"executable\":\"/tmp/work tree/target/wasm32-unknown-unknown/debug/deps/math_commons-0123456789abcdef.wasm\"}\n",
            "{\"reason\":\"build-finished\",\"success\":true}\n",
        );
        let paths = extract_wasm_test_executables_from_cargo_json(sample).unwrap();
        assert_eq!(
            paths,
            vec![std::path::PathBuf::from(
                "/tmp/work tree/target/wasm32-unknown-unknown/debug/deps/math_commons-0123456789abcdef.wasm"
            )]
        );
        assert_eq!(
            extract_wasm_test_executables_from_cargo_json(""),
            Err(WorkspaceError::NoWasmTestArtifacts)
        );
    }
}
