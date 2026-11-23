use oxidize_core::traceability::AstCache;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use traversal::traverse_crate_root_with_cache;

mod facade;
pub(crate) mod public_debug;
mod traversal;
mod visitor;

#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ApiItem {
    pub signature: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deprecation_expires: Option<String>,
}

const BASELINE_REL_PATH: &str = "crates/unified_verification/api_baseline.json";

fn collect_crate_roots(workspace_root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![workspace_root.join("math_explorer/src/lib.rs")];
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
            if name != "unified_verification" {
                roots.push(path.join("src/lib.rs"));
            }
        }
    }
    roots
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

pub fn extract_apis() -> BTreeMap<String, Vec<ApiItem>> {
    extract_apis_in_dir(&crate::manifest_cache::workspace_root_dir())
}

pub fn regenerate_baseline() {
    let root = crate::manifest_cache::workspace_root_dir();
    let apis = extract_apis();
    let json = serde_json::to_string_pretty(&apis).unwrap();
    fs::write(root.join(BASELINE_REL_PATH), format!("{json}\n")).expect("Failed to write baseline");
    println!("API baseline regenerated successfully.");
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
    if local_changed {
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
    today: &str,
    out: &mut Vec<String>,
) -> bool {
    let mut changed = false;
    for (path, base_items) in baseline_apis {
        if current_apis.contains_key(path) {
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
            out.push(format!("API Drift: File with public API removed: {path}"));
            changed = true;
        }
    }
    changed
}

pub(crate) fn collect_api_drift_messages(
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    today: &str,
) -> (bool, Vec<String>) {
    let mut changed = false;
    let mut out = Vec::new();

    for (path, items) in current_apis {
        if collect_expired_messages(path, items, today, &mut out) {
            changed = true;
        }
        if let Some(base_items) = baseline_apis.get(path) {
            if compare_file_items(path, items, base_items, today, &mut out) {
                changed = true;
            }
        } else {
            out.push(format!("API Drift: New file with public API: {path}"));
            changed = true;
        }
    }

    if check_removed_files(current_apis, baseline_apis, today, &mut out) {
        changed = true;
    }

    if changed {
        out.push("Public API drift detected!".to_string());
        out.push(
            "  Hint: If this public API change is intentional, run `cargo run -p xtask -- regenerate-baseline` to update `crates/unified_verification/api_baseline.json` and add a corresponding Architecture Decision Record in `docs/adr/` per `AGENTS.md`."
                .to_string(),
        );
        return (false, out);
    }

    out.push("Public API drift check passed.".to_string());
    (true, out)
}

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
    let debug_violations = {
        let (checked, violations) =
            public_debug::check_workspace_public_debug_with_cache(&root, cache);
        if violations.is_empty() {
            println!(
                "Public Debug coverage check passed ({checked} public structs/enums verified)."
            );
        } else {
            println!("Public Debug coverage check failed:");
            for v in &violations {
                println!("  - {v}");
            }
        }
        violations
    };
    let baseline_content =
        fs::read_to_string(root.join(BASELINE_REL_PATH)).unwrap_or_else(|_| "{}".to_string());
    let baseline_apis: BTreeMap<String, Vec<ApiItem>> =
        serde_json::from_str(&baseline_content).unwrap_or_default();
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let drift_ok = check_api_drift_against(&current_apis, &baseline_apis, &today);
    drift_ok && facade_violations.is_empty() && debug_violations.is_empty()
}
