use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use traversal::traverse_crate_root;

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

fn resolve_workspace_root() -> &'static Path {
    if !Path::new("crates").is_dir() && Path::new("../../crates").is_dir() {
        Path::new("../..")
    } else {
        Path::new(".")
    }
}

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
    let mut apis = BTreeMap::new();
    for crate_lib in collect_crate_roots(workspace_root) {
        traverse_crate_root(&crate_lib, workspace_root, &mut apis);
    }
    apis
}

pub fn extract_apis() -> BTreeMap<String, Vec<ApiItem>> {
    extract_apis_in_dir(resolve_workspace_root())
}

pub fn regenerate_baseline() {
    let root = resolve_workspace_root();
    let apis = extract_apis();
    let json = serde_json::to_string_pretty(&apis).unwrap();
    fs::write(root.join(BASELINE_REL_PATH), format!("{json}\n")).expect("Failed to write baseline");
    println!("API baseline regenerated successfully.");
}

fn check_expired_items(path: &str, items: &[ApiItem], today: &str) -> bool {
    let mut expired = false;
    for item in items {
        if let Some(ref date) = item.deprecation_expires
            && date.as_str() <= today
        {
            println!(
                "API Expired: {} in {} (Expired: {})",
                item.signature, path, date
            );
            println!("  This deprecated API has passed its expiration target and must be removed.");
            expired = true;
        }
    }
    expired
}

fn compare_file_items(path: &str, items: &[ApiItem], base_items: &[ApiItem], today: &str) -> bool {
    let mut local_changed = false;
    for item in items {
        if !base_items.iter().any(|b| b.signature == item.signature) {
            println!("  + {}", item.signature);
            local_changed = true;
        }
    }
    for item in base_items {
        if !items.iter().any(|c| c.signature == item.signature) {
            if let Some(ref date) = item.deprecation_expires
                && date.as_str() <= today
            {
                println!(
                    "  ~ {} (Deleted, bypassed drift check due to expired deprecation)",
                    item.signature
                );
                continue;
            }
            println!("  - {}", item.signature);
            local_changed = true;
        }
    }
    if local_changed {
        println!("API Drift detected in {}:", path);
    }
    local_changed
}

fn check_removed_files(
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    today: &str,
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
                println!(
                    "  ~ {} in {} (File deleted, bypassed due to expired deprecation)",
                    item.signature, path
                );
                continue;
            }
            all_bypassed = false;
        }
        if !all_bypassed {
            println!("API Drift: File with public API removed: {}", path);
            changed = true;
        }
    }
    changed
}

pub fn check_api_drift_against(
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    today: &str,
) -> bool {
    let mut changed = false;

    for (path, items) in current_apis {
        if check_expired_items(path, items, today) {
            changed = true;
        }
        if let Some(base_items) = baseline_apis.get(path) {
            if compare_file_items(path, items, base_items, today) {
                changed = true;
            }
        } else {
            println!("API Drift: New file with public API: {}", path);
            changed = true;
        }
    }

    if check_removed_files(current_apis, baseline_apis, today) {
        changed = true;
    }

    if changed {
        println!("Public API drift detected! Please regenerate the baseline and write an ADR.");
        return false;
    }

    println!("Public API drift check passed.");
    true
}

pub fn check_api_drift() -> bool {
    let root = resolve_workspace_root();
    let current_apis = extract_apis();
    let baseline_content =
        fs::read_to_string(root.join(BASELINE_REL_PATH)).unwrap_or_else(|_| "{}".to_string());
    let baseline_apis: BTreeMap<String, Vec<ApiItem>> =
        serde_json::from_str(&baseline_content).unwrap_or_default();
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    check_api_drift_against(&current_apis, &baseline_apis, &today)
}
