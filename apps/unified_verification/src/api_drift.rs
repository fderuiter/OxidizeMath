use oxidize_core::traceability::{AstCache, MaturityRegistry, MaturityTier};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use traversal::traverse_crate_root_with_cache;

mod facade;
pub(crate) mod semver_lock;
pub(crate) mod traversal;
mod visitor;

#[cfg(test)]
mod semver_lock_tests;
#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ApiItem {
    pub signature: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deprecation_expires: Option<String>,
}

pub(crate) fn resolve_baseline_path(workspace_root: &Path) -> PathBuf {
    if let Some(uv_dir) = oxidize_core::workspace::find_package_member_path_std(
        workspace_root,
        "unified_verification",
    ) {
        return workspace_root.join(uv_dir).join("api_baseline.json");
    }
    let apps_candidate = workspace_root.join("apps/unified_verification/api_baseline.json");
    if apps_candidate.is_file() {
        return apps_candidate;
    }
    workspace_root.join("crates/unified_verification/api_baseline.json")
}

fn collect_crate_roots_fallback(workspace_root: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for top in ["oxidizemath", "math_explorer"] {
        let lib = workspace_root.join(top).join("src/lib.rs");
        if lib.is_file() {
            roots.push(lib);
        }
    }
    let crates_dir = workspace_root.join("crates");
    if let Ok(entries) = fs::read_dir(crates_dir) {
        let mut crate_dirs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        crate_dirs.sort();
        for path in crate_dirs {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name == "unified_verification" {
                continue;
            }
            let direct_lib = path.join("src/lib.rs");
            if direct_lib.is_file() {
                roots.push(direct_lib);
            } else if let Ok(nested_entries) = fs::read_dir(&path) {
                let mut nested_dirs: Vec<PathBuf> = nested_entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect();
                nested_dirs.sort();
                for nested in nested_dirs {
                    let nested_lib = nested.join("src/lib.rs");
                    if nested_lib.is_file() {
                        roots.push(nested_lib);
                    }
                }
            }
        }
    }
    roots
}

pub(crate) fn collect_crate_roots(workspace_root: &Path) -> Vec<PathBuf> {
    if let Ok(model) =
        crate::doc_audit::manifest_model::WorkspaceManifestModel::load(workspace_root)
    {
        let mut roots: Vec<PathBuf> = model
            .packages_by_name
            .values()
            .filter(|pkg| {
                pkg.has_lib
                    && pkg.effective_release_role
                        != Some(oxidize_core::workspace::ReleaseRole::Vendored)
                    && pkg.name != "unified_verification"
            })
            .map(|pkg| workspace_root.join(&pkg.member_dir).join("src/lib.rs"))
            .filter(|p| p.is_file())
            .collect();
        roots.sort();
        return roots;
    }
    collect_crate_roots_fallback(workspace_root)
}

pub fn extract_apis_in_dir(workspace_root: &Path) -> BTreeMap<String, Vec<ApiItem>> {
    let mut cache = AstCache::new();
    extract_apis_in_dir_with_cache(workspace_root, &mut cache)
}

pub fn extract_apis_in_dir_with_cache(
    workspace_root: &Path,
    cache: &mut AstCache,
) -> BTreeMap<String, Vec<ApiItem>> {
    let mut apis = BTreeMap::new();
    for crate_lib in collect_crate_roots(workspace_root) {
        traverse_crate_root_with_cache(&crate_lib, workspace_root, &mut apis, cache);
    }
    apis
}

#[cfg(test)]
pub fn extract_apis() -> BTreeMap<String, Vec<ApiItem>> {
    extract_apis_in_dir(&crate::manifest_cache::workspace_root_dir())
}

pub fn regenerate_baseline_with_args(args: &[String]) {
    let allow_override = args.iter().any(|a| a == "--allow-breaking-certified")
        || std::env::var("OXIDIZE_ALLOW_BREAKING_CERTIFIED")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let root = crate::manifest_cache::workspace_root_dir();
    if let Err(errors) = regenerate_baseline_in_dir(&root, allow_override) {
        for err in errors {
            eprintln!("{err}");
        }
        std::process::exit(1);
    }
}

pub fn regenerate_baseline_in_dir(
    workspace_root: &Path,
    allow_breaking_override: bool,
) -> Result<(), Vec<String>> {
    let apis = extract_apis_in_dir(workspace_root);
    let baseline_path = resolve_baseline_path(workspace_root);
    let baseline_apis: BTreeMap<String, Vec<ApiItem>> = fs::read_to_string(&baseline_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    semver_lock::check_regenerate_baseline_semver_lock(
        workspace_root,
        &apis,
        &baseline_apis,
        &today,
        allow_breaking_override,
    )?;
    if let Some(parent) = baseline_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(&apis).unwrap();
    fs::write(baseline_path, format!("{json}\n")).expect("Failed to write baseline");
    println!("API baseline regenerated successfully.");
    Ok(())
}

fn collect_expired_messages(
    path: &str,
    items: &[ApiItem],
    today: &str,
    out: &mut Vec<String>,
) -> bool {
    let mut expired = false;
    for item in items {
        if let Some(ref date) = item.deprecation_expires
            && date.as_str() <= today
        {
            out.push(format!(
                "API Expired: {} in {} (Expired: {})",
                item.signature, path, date
            ));
            out.push(
                "  This deprecated API has passed its expiration target and must be removed."
                    .to_string(),
            );
            expired = true;
        }
    }
    expired
}

fn compare_file_items(
    path: &str,
    items: &[ApiItem],
    base_items: &[ApiItem],
    tier: MaturityTier,
    today: &str,
    out: &mut Vec<String>,
) -> bool {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut bypassed = Vec::new();

    for item in items {
        if !base_items.iter().any(|b| b.signature == item.signature) {
            added.push(item.signature.as_str());
        }
    }
    for item in base_items {
        if !items.iter().any(|c| c.signature == item.signature) {
            if let Some(ref date) = item.deprecation_expires
                && date.as_str() <= today
            {
                bypassed.push(item.signature.as_str());
                continue;
            }
            removed.push(item.signature.as_str());
        }
    }

    let local_changed = !added.is_empty() || !removed.is_empty();
    if local_changed && tier == MaturityTier::Experimental {
        out.push(format!(
            "Experimental API evolution permitted in {path} (+{} / -{} signatures).",
            added.len(),
            removed.len()
        ));
        return false;
    }
    if local_changed {
        if tier == MaturityTier::Certified && !removed.is_empty() {
            out.push(format!(
                "SemVer Lock Violation [Certified] in {path} (breaking removal requires workspace version bump):"
            ));
        }
        out.push(format!("API Drift detected in {path}:"));
        for sig in added {
            out.push(format!("  + {sig}"));
        }
        for sig in removed {
            out.push(format!("  - {sig}"));
        }
    }
    for sig in bypassed {
        out.push(format!(
            "  ~ {sig} (Deleted, bypassed drift check due to expired deprecation)"
        ));
    }
    local_changed
}

fn check_removed_files(
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    registry: Option<&MaturityRegistry>,
    today: &str,
    out: &mut Vec<String>,
) -> bool {
    let mut changed = false;
    for (path, base_items) in baseline_apis {
        if current_apis.contains_key(path) {
            continue;
        }
        let tier = registry
            .map(|r| r.tier_for_repo_path(path))
            .unwrap_or(MaturityTier::Validated);
        if tier == MaturityTier::Experimental {
            out.push(format!(
                "Experimental API evolution permitted: file moved/removed in {path}."
            ));
            continue;
        }
        let mut all_bypassed = true;
        for item in base_items {
            if let Some(ref date) = item.deprecation_expires
                && date.as_str() <= today
            {
                out.push(format!(
                    "  ~ {} in {} (File deleted, bypassed due to expired deprecation)",
                    item.signature, path
                ));
                continue;
            }
            all_bypassed = false;
        }
        if !all_bypassed {
            if tier == MaturityTier::Certified {
                out.push(format!(
                    "SemVer Lock Violation [Certified]: Public file removed without SemVer bump: {path}"
                ));
            }
            out.push(format!("API Drift: File with public API removed: {path}"));
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
pub(crate) fn collect_api_drift_messages(
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    today: &str,
) -> (bool, Vec<String>) {
    collect_api_drift_messages_with_registry(current_apis, baseline_apis, None, today)
}

pub(crate) fn collect_api_drift_messages_with_registry(
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    registry: Option<&MaturityRegistry>,
    today: &str,
) -> (bool, Vec<String>) {
    let mut changed = false;
    let mut out = Vec::new();

    for (path, items) in current_apis {
        let tier = registry
            .map(|r| r.tier_for_repo_path(path))
            .unwrap_or(MaturityTier::Validated);
        if collect_expired_messages(path, items, today, &mut out) {
            changed = true;
        }
        if let Some(base_items) = baseline_apis.get(path) {
            if compare_file_items(path, items, base_items, tier, today, &mut out) {
                changed = true;
            }
        } else if tier == MaturityTier::Experimental {
            out.push(format!(
                "Experimental API evolution permitted: new file {path}."
            ));
        } else {
            out.push(format!("API Drift: New file with public API: {path}"));
            changed = true;
        }
    }

    if check_removed_files(current_apis, baseline_apis, registry, today, &mut out) {
        changed = true;
    }

    if changed {
        out.push("Public API drift detected!".to_string());
        out.push(
            "  Hint: If this public API change is intentional, run `cargo run -p xtask -- regenerate-baseline` to update `api_baseline.json` and add a corresponding Architecture Decision Record in `docs/adr/` per `AGENTS.md` (breaking `Certified` changes require a workspace SemVer bump in `Cargo.toml`)."
                .to_string(),
        );
        return (false, out);
    }

    out.push("Public API drift check passed.".to_string());
    (true, out)
}

#[cfg(test)]
pub fn check_api_drift_against(
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    today: &str,
) -> bool {
    let (passed, messages) = collect_api_drift_messages(current_apis, baseline_apis, today);
    for line in messages {
        println!("{line}");
    }
    passed
}

pub fn check_api_drift() -> bool {
    let mut cache = AstCache::new();
    check_api_drift_with_cache(&mut cache)
}

pub fn check_api_drift_with_cache(cache: &mut AstCache) -> bool {
    let root = crate::manifest_cache::workspace_root_dir();
    let current_apis = extract_apis_in_dir_with_cache(&root, cache);
    let facade_violations =
        facade::check_facade_completeness_in_dir_with_cache(&root, &current_apis, cache);
    if !facade_violations.is_empty() {
        println!("Facade export completeness check failed:");
        for v in &facade_violations {
            println!("  - {v}");
        }
    }
    let baseline_content =
        fs::read_to_string(resolve_baseline_path(&root)).unwrap_or_else(|_| "{}".to_string());
    let baseline_apis: BTreeMap<String, Vec<ApiItem>> =
        serde_json::from_str(&baseline_content).unwrap_or_default();
    let registry = root
        .join("traceability.toml")
        .is_file()
        .then(|| MaturityRegistry::load_or_embedded(Some(&root)));
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let (drift_ok, messages) = collect_api_drift_messages_with_registry(
        &current_apis,
        &baseline_apis,
        registry.as_ref(),
        &today,
    );
    for line in messages {
        println!("{line}");
    }
    drift_ok && facade_violations.is_empty()
}
