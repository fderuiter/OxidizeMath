use crate::manifest_cache::ManifestCache;
use crate::policy_audit::lint_policy::is_vendored_egui_plot;
use crate::release_tag::parse_release_version;
use regex::Regex;
use std::fs;
use std::path::Path;
use toml::Value;

const RELEASE_WORKSPACE_MEMBERS: &[&str] = &[
    "crates/math_commons",
    "crates/scientific_metadata",
    "crates/pure_math",
    "crates/domain_ai",
    "crates/domain_physics",
    "crates/domain_biology",
    "crates/domain_applied",
    "crates/domain_climate",
    "crates/domain_epidemiology",
    "math_explorer",
    "math_explorer_gui",
];

const INTERNAL_FOUNDATION_MEMBERS: &[&str] = &[
    "crates/federated_registry",
    "crates/diagnostics",
    "crates/oxidize_core",
    "crates/verified_engine",
    "crates/verified_engine_macros",
    "crates/markdown_tests",
    "apps/xtask",
    "crates/unified_verification",
];

struct WorkspaceVersionContext {
    workspace_version: Option<String>,
    has_egui_plot_patch: bool,
}

fn normalize_member_rel(member: &str) -> String {
    member
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_string()
}

fn is_release_member(member: &str) -> bool {
    let norm = normalize_member_rel(member);
    RELEASE_WORKSPACE_MEMBERS.contains(&norm.as_str())
}

fn is_internal_foundation_member(member: &str) -> bool {
    let norm = normalize_member_rel(member);
    INTERNAL_FOUNDATION_MEMBERS.contains(&norm.as_str())
}

fn has_workspace_version_inheritance(parsed: &Value) -> bool {
    parsed
        .get("package")
        .and_then(|p| p.get("version"))
        .and_then(Value::as_table)
        .is_some_and(|t| t.len() == 1 && t.get("workspace").and_then(Value::as_bool) == Some(true))
}

fn check_root_profile_and_version(
    root_path: &Path,
    workspace_members: &[&str],
    auto_fix: bool,
    manifest_cache: &mut ManifestCache,
) -> (bool, WorkspaceVersionContext) {
    let empty_ctx = WorkspaceVersionContext {
        workspace_version: None,
        has_egui_plot_patch: false,
    };
    let (root_content, parsed) = match manifest_cache.get_or_load(root_path) {
        Ok(cached) => (cached.content.clone(), cached.parsed.clone()),
        Err(err) => {
            eprintln!("{err}");
            return (false, empty_ctx);
        }
    };

    let mut passed =
        ensure_root_overflow_checks(root_path, &root_content, &parsed, auto_fix, manifest_cache);
    let has_egui_plot_patch = parsed
        .get("patch")
        .and_then(|p| p.get("crates-io"))
        .and_then(|c| c.get("egui_plot"))
        .is_some();

    let ws_pkg = parsed.get("workspace").and_then(|w| w.get("package"));
    let requires_ws_ver =
        ws_pkg.is_some() || workspace_members.iter().any(|m| is_release_member(m));
    let workspace_version = match ws_pkg
        .and_then(|p| p.get("version"))
        .and_then(Value::as_str)
    {
        Some(ver) => match parse_release_version(ver) {
            Ok(_) => Some(ver.to_string()),
            Err(err) => {
                eprintln!(
                    "[!] Invalid [workspace.package].version in {}: {err}",
                    root_path.display()
                );
                passed = false;
                None
            }
        },
        None if requires_ws_ver => {
            eprintln!(
                "[!] Profile mismatch: {} must define a valid [workspace.package].version string",
                root_path.display()
            );
            passed = false;
            None
        }
        None => None,
    };

    (
        passed,
        WorkspaceVersionContext {
            workspace_version,
            has_egui_plot_patch,
        },
    )
}

fn ensure_root_overflow_checks(
    root_path: &Path,
    root_content: &str,
    parsed: &Value,
    auto_fix: bool,
    manifest_cache: &mut ManifestCache,
) -> bool {
    let has_checks = parsed
        .get("profile")
        .and_then(|p| p.get("release"))
        .and_then(|r| r.get("overflow-checks"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if has_checks {
        println!(
            "[+] Verified workspace root profile: overflow-checks = true is strictly enforced."
        );
        return true;
    }
    if auto_fix {
        println!(
            "[+] Auto-fixing: Adding [profile.release] overflow-checks = true to workspace root Cargo.toml"
        );
        let mut updated = root_content.to_string();
        if !updated.ends_with('\n') {
            updated.push('\n');
        }
        updated.push_str("\n[profile.release]\noverflow-checks = true\n");
        fs::write(root_path, updated).unwrap();
        manifest_cache.invalidate(root_path);
        true
    } else {
        eprintln!(
            "[!] Profile mismatch: workspace root Cargo.toml is missing [profile.release] overflow-checks = true"
        );
        false
    }
}

fn check_readme_version_sync(
    root_manifest: &Path,
    workspace_members: &[&str],
    ws_version: &str,
) -> bool {
    let readme_path = root_manifest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("README.md");
    let require_readme = readme_path.exists()
        || workspace_members
            .iter()
            .any(|m| normalize_member_rel(m) == "math_explorer");
    if !require_readme {
        return true;
    }
    let Ok(content) = fs::read_to_string(&readme_path) else {
        eprintln!(
            "[!] Version sync failed: missing {} required for workspace version {}",
            readme_path.display(),
            ws_version
        );
        return false;
    };

    let badge_re =
        Regex::new(r"https://img\.shields\.io/badge/version-([0-9A-Za-z.\-]+)-blue\.svg").unwrap();
    let dep_re = Regex::new(r#"math_explorer\s*=\s*\{\s*version\s*=\s*"([^"]+)""#).unwrap();
    let escaped_ws = ws_version.replace('-', "--");

    let badges: Vec<&str> = badge_re
        .captures_iter(&content)
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();
    let deps: Vec<&str> = dep_re
        .captures_iter(&content)
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();

    let badges_ok =
        !badges.is_empty() && badges.iter().all(|b| *b == ws_version || *b == escaped_ws);
    let deps_ok = !deps.is_empty() && deps.iter().all(|d| *d == ws_version);
    if !badges_ok || !deps_ok {
        eprintln!(
            "[!] Version sync mismatch in {}: expected version badge and math_explorer snippet to match [workspace.package].version = \"{}\"",
            readme_path.display(),
            ws_version
        );
        return false;
    }
    true
}

fn has_no_packager_version_override(path: &Path, parsed: &Value) -> bool {
    if parsed
        .get("package")
        .and_then(|p| p.get("metadata"))
        .and_then(|m| m.get("packager"))
        .and_then(|pk| pk.get("version"))
        .is_some()
    {
        eprintln!(
            "[!] Version mismatch in {}: [package.metadata.packager].version must be omitted so cargo-packager inherits [package].version",
            path.display()
        );
        return false;
    }
    true
}

fn check_fallback_explicit_version(
    path: &Path,
    explicit_ver: Option<&str>,
    ctx: &WorkspaceVersionContext,
    packager_ok: bool,
) -> bool {
    match explicit_ver {
        Some(_) if ctx.workspace_version.is_none() => packager_ok,
        Some(other) => {
            eprintln!(
                "[!] Version mismatch in {}: hardcoded version = \"{other}\" is disallowed; use `version.workspace = true`",
                path.display()
            );
            false
        }
        None => {
            eprintln!(
                "[!] Version mismatch in {}: missing valid [package].version",
                path.display()
            );
            false
        }
    }
}

fn check_member_version_and_packager(
    member: &str,
    path: &Path,
    parsed: &Value,
    ctx: &WorkspaceVersionContext,
) -> bool {
    let packager_ok = has_no_packager_version_override(path, parsed);
    let inherits_ws = has_workspace_version_inheritance(parsed);
    let explicit_ver = parsed
        .get("package")
        .and_then(|p| p.get("version"))
        .and_then(Value::as_str);

    if inherits_ws && ctx.workspace_version.is_none() {
        eprintln!(
            "[!] Version mismatch in {}: version.workspace = true requires [workspace.package].version in root Cargo.toml",
            path.display()
        );
        return false;
    }
    if is_release_member(member) {
        if !inherits_ws {
            eprintln!(
                "[!] Version mismatch in {}: release crate must set `version.workspace = true`",
                path.display()
            );
            return false;
        }
        return packager_ok;
    }
    if is_vendored_egui_plot(member) {
        if explicit_ver.is_none() || (ctx.has_egui_plot_patch && explicit_ver != Some("0.34.1")) {
            eprintln!(
                "[!] Version mismatch in {}: vendored egui_plot must preserve explicit version = \"0.34.1\"",
                path.display()
            );
            return false;
        }
        return packager_ok;
    }
    if is_internal_foundation_member(member) {
        if explicit_ver != Some("0.1.0") {
            eprintln!(
                "[!] Version mismatch in {}: internal foundation/tooling crate must retain version = \"0.1.0\"",
                path.display()
            );
            return false;
        }
        return packager_ok;
    }
    if inherits_ws || explicit_ver == Some("0.1.0") {
        return packager_ok;
    }
    check_fallback_explicit_version(path, explicit_ver, ctx, packager_ok)
}

fn check_member_manifest(
    member: &str,
    auto_fix: bool,
    ctx: &WorkspaceVersionContext,
    manifest_cache: &mut ManifestCache,
) -> bool {
    let path = crate::policy_audit::lint_policy::resolve_member_dir(member).join("Cargo.toml");
    let (content, parsed) = match manifest_cache.get_or_load(&path) {
        Ok(cached) => (cached.content.clone(), cached.parsed.clone()),
        Err(err) => {
            eprintln!("{err}");
            return false;
        }
    };

    let mut passed = manifest_cache.check_manifest_lints_once(member, &path, &parsed);
    if !check_member_version_and_packager(member, &path, &parsed, ctx) {
        passed = false;
    }

    if parsed
        .get("profile")
        .and_then(|p| p.get("release"))
        .is_some()
    {
        if auto_fix {
            println!(
                "[+] Auto-fixing: Removing redundant [profile.release] from {}",
                path.display()
            );
            let new_content = remove_profile_release(&content);
            fs::write(&path, new_content).unwrap();
            manifest_cache.invalidate(&path);
        } else {
            eprintln!(
                "[!] Profile mismatch in {}: redundant [profile.release] found. Configuration must be consolidated in workspace root.",
                path.display()
            );
            passed = false;
        }
    }

    passed
}

pub fn check_profiles(workspace_members: &[&str], auto_fix: bool) -> bool {
    let mut manifest_cache = ManifestCache::new();
    check_profiles_with_cache(workspace_members, auto_fix, &mut manifest_cache)
}

pub(crate) fn check_profiles_with_cache(
    workspace_members: &[&str],
    auto_fix: bool,
    manifest_cache: &mut ManifestCache,
) -> bool {
    let root_manifest = crate::manifest_cache::resolve_workspace_path("Cargo.toml");
    let (mut passed, ctx) =
        check_root_profile_and_version(&root_manifest, workspace_members, auto_fix, manifest_cache);

    if let Some(ws_ver) = ctx.workspace_version.as_deref()
        && !check_readme_version_sync(&root_manifest, workspace_members, ws_ver)
    {
        passed = false;
    }

    for member in workspace_members {
        if !check_member_manifest(member, auto_fix, &ctx, manifest_cache) {
            passed = false;
        }
    }

    passed
}

fn remove_profile_release(content: &str) -> String {
    let mut out = String::new();
    let mut in_profile_release = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[profile.release]") {
            in_profile_release = true;
            continue;
        }
        if in_profile_release && trimmed.starts_with('[') {
            in_profile_release = false;
        }
        if !in_profile_release {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.trim_end().to_string() + "\n"
}

#[cfg(test)]
mod tests;
