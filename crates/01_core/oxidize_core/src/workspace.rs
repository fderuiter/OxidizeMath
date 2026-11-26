//! Workspace discovery and manifest parsing utilities.

#[cfg(feature = "ast-analysis")]
#[path = "workspace_graph.rs"]
mod workspace_graph;
#[path = "workspace_meta.rs"]
mod workspace_meta;
#[path = "workspace_wasm.rs"]
mod workspace_wasm;

#[cfg(feature = "ast-analysis")]
pub use workspace_graph::{compute_gui_snapshot_code_dirs, resolve_transitive_workspace_deps};
pub use workspace_meta::{
    FacadeRole, OxidizeMetadata, ReleaseRole, TierKind, WorkspacePackageMeta,
    discover_workspace_packages_std, parse_oxidize_metadata_std, parse_package_name_std,
    parse_patched_crates_std,
};
pub use workspace_wasm::{
    HOST_ONLY_WORKSPACE_PACKAGES, WasmCrateSpec, WasmTestTargetKind, WorkspaceError,
    build_wasm_cargo_test_invocations, discover_wasm_crates, find_package_member_path_std,
    parse_workspace_members_std,
};

use std::fs;
use std::path::{Path, PathBuf};

const MAX_ANCESTOR_DEPTH: usize = 32;

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
        let dir = if ancestor.as_os_str().is_empty() {
            Path::new(".")
        } else {
            ancestor
        };
        if try_read_members(&dir.join("Cargo.toml")).is_some() {
            return Some(dir.to_path_buf());
        }
    }
    workspace_wasm::find_workspace_root(base)
}

/// Locates the workspace root directory relative to the current working directory.
///
/// Returns `Some(PathBuf::from("."))` when `./Cargo.toml` defines `[workspace]`,
/// otherwise searches ancestor directories of `std::env::current_dir()`.
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
    workspace_wasm::find_workspace_root(Path::new("."))
}

/// Discovers workspace members from `Cargo.toml` manifests.
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

/// Parses `[package.metadata.oxidize]` from a `Cargo.toml` manifest string.
#[cfg(feature = "ast-analysis")]
pub fn parse_oxidize_metadata(content: &str) -> Result<OxidizeMetadata, WorkspaceError> {
    let table = content
        .parse::<toml::Table>()
        .map_err(|e| WorkspaceError::InvalidManifestSyntax(e.to_string()))?;
    let Some(meta_table) = table
        .get("package")
        .and_then(|p| p.get("metadata"))
        .and_then(|m| m.get("oxidize"))
        .and_then(|o| o.as_table())
    else {
        return Ok(OxidizeMetadata::default());
    };
    let tier = meta_table
        .get("tier")
        .and_then(toml::Value::as_str)
        .and_then(TierKind::parse_str);
    let release_role = meta_table
        .get("release_role")
        .and_then(toml::Value::as_str)
        .and_then(ReleaseRole::parse_str);
    let facade_role = meta_table
        .get("facade_role")
        .and_then(toml::Value::as_str)
        .and_then(FacadeRole::parse_str)
        .unwrap_or_default();
    let facade_module = meta_table
        .get("facade_module")
        .and_then(toml::Value::as_str)
        .map(String::from);
    let traceability = meta_table
        .get("traceability")
        .and_then(toml::Value::as_bool);
    let traceability_subdir = meta_table
        .get("traceability_subdir")
        .and_then(toml::Value::as_str)
        .map(String::from);
    let math_contract = meta_table
        .get("math_contract")
        .and_then(toml::Value::as_bool);
    Ok(OxidizeMetadata {
        tier,
        release_role,
        facade_role,
        facade_module,
        traceability,
        traceability_subdir,
        math_contract,
    })
}

/// Parses `[package.metadata.oxidize]` from a `Cargo.toml` manifest string.
#[cfg(not(feature = "ast-analysis"))]
pub fn parse_oxidize_metadata(content: &str) -> Result<OxidizeMetadata, WorkspaceError> {
    validate_toml_bracket_balance(content)?;
    Ok(parse_oxidize_metadata_std(content))
}

/// Discovers all workspace member packages and their `[package.metadata.oxidize]` metadata.
pub fn discover_workspace_packages(
    workspace_root: &Path,
) -> Result<Vec<WorkspacePackageMeta>, WorkspaceError> {
    let root_manifest = workspace_root.join("Cargo.toml");
    let root_toml = fs::read_to_string(&root_manifest).map_err(|e| WorkspaceError::ManifestIo {
        path: root_manifest.display().to_string(),
        reason: e.to_string(),
    })?;
    let members = parse_workspace_members(&root_toml)?;
    let patched = parse_patched_crates_std(&root_toml);
    let mut pkgs = Vec::with_capacity(members.len());
    for member in members {
        let norm_member = member
            .replace('\\', "/")
            .trim_start_matches("./")
            .trim_end_matches('/')
            .to_string();
        let manifest_path = workspace_root.join(&norm_member).join("Cargo.toml");
        let manifest =
            fs::read_to_string(&manifest_path).map_err(|e| WorkspaceError::ManifestIo {
                path: manifest_path.display().to_string(),
                reason: e.to_string(),
            })?;
        let package_name = parse_package_name_std(&manifest).ok_or_else(|| {
            WorkspaceError::MissingPackageField {
                path: manifest_path.display().to_string(),
                field: "package.name",
            }
        })?;
        let metadata = parse_oxidize_metadata(&manifest)?;
        let is_vendored_patch = patched.iter().any(|p| p == &package_name);
        pkgs.push(WorkspacePackageMeta {
            member_path: norm_member,
            package_name,
            metadata,
            is_vendored_patch,
        });
    }
    Ok(pkgs)
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
mod tests;
