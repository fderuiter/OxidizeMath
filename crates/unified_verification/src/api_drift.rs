use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use syn::visit::Visit;
use walkdir::WalkDir;

#[derive(Serialize, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ApiItem {
    pub signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecation_expires: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ApiItemShim {
    Simple(String),
    Detailed {
        signature: String,
        deprecation_expires: Option<String>,
    },
}

impl From<ApiItemShim> for ApiItem {
    fn from(shim: ApiItemShim) -> Self {
        match shim {
            ApiItemShim::Simple(sig) => ApiItem {
                signature: sig,
                deprecation_expires: None,
            },
            ApiItemShim::Detailed {
                signature,
                deprecation_expires,
            } => ApiItem {
                signature,
                deprecation_expires,
            },
        }
    }
}

impl<'de> Deserialize<'de> for ApiItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let shim = ApiItemShim::deserialize(deserializer)?;
        Ok(ApiItem::from(shim))
    }
}

fn extract_expiration_date(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("deprecated") {
            let attr_str = quote::quote!(#attr).to_string();
            if let Some(pos) = attr_str.find("expires") {
                if let Some(date) = find_date_pattern(&attr_str[pos..]) {
                    return Some(date);
                }
            }
            if let Some(pos) = attr_str.find("expiry") {
                if let Some(date) = find_date_pattern(&attr_str[pos..]) {
                    return Some(date);
                }
            }
            if let Some(date) = find_date_pattern(&attr_str) {
                return Some(date);
            }
        }
    }
    None
}

fn find_date_pattern(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    for i in 0..=(bytes.len() - 10) {
        let is_digit = |b: u8| b.is_ascii_digit();
        if is_digit(bytes[i])
            && is_digit(bytes[i + 1])
            && is_digit(bytes[i + 2])
            && is_digit(bytes[i + 3])
            && bytes[i + 4] == b'-'
            && is_digit(bytes[i + 5])
            && is_digit(bytes[i + 6])
            && bytes[i + 7] == b'-'
            && is_digit(bytes[i + 8])
            && is_digit(bytes[i + 9])
        {
            return Some(s[i..i + 10].to_string());
        }
    }
    None
}

#[derive(Default)]
struct ApiVisitor {
    pub items: Vec<ApiItem>,
}

impl<'ast> Visit<'ast> for ApiVisitor {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        if matches!(i.vis, syn::Visibility::Public(_)) {
            let sig = &i.sig;
            self.items.push(ApiItem {
                signature: quote::quote!(pub #sig).to_string(),
                deprecation_expires: extract_expiration_date(&i.attrs),
            });
        }
        syn::visit::visit_item_fn(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        if matches!(i.vis, syn::Visibility::Public(_)) {
            let ident = &i.ident;
            let generics = &i.generics;
            self.items.push(ApiItem {
                signature: quote::quote!(pub struct #ident #generics).to_string(),
                deprecation_expires: extract_expiration_date(&i.attrs),
            });
        }
        syn::visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        if matches!(i.vis, syn::Visibility::Public(_)) {
            let ident = &i.ident;
            let generics = &i.generics;
            self.items.push(ApiItem {
                signature: quote::quote!(pub enum #ident #generics).to_string(),
                deprecation_expires: extract_expiration_date(&i.attrs),
            });
        }
        syn::visit::visit_item_enum(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        if matches!(i.vis, syn::Visibility::Public(_)) {
            let ident = &i.ident;
            let generics = &i.generics;
            self.items.push(ApiItem {
                signature: quote::quote!(pub trait #ident #generics).to_string(),
                deprecation_expires: extract_expiration_date(&i.attrs),
            });
        }
        syn::visit::visit_item_trait(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        if matches!(i.vis, syn::Visibility::Public(_)) {
            let ident = &i.ident;
            let generics = &i.generics;
            let ty = &i.ty;
            self.items.push(ApiItem {
                signature: quote::quote!(pub type #ident #generics = #ty).to_string(),
                deprecation_expires: extract_expiration_date(&i.attrs),
            });
        }
        syn::visit::visit_item_type(self, i);
    }
}

pub fn extract_apis() -> BTreeMap<String, Vec<ApiItem>> {
    let mut apis = BTreeMap::new();
    let mut dirs_to_scan = vec![PathBuf::from("math_explorer/src")];

    if let Ok(entries) = fs::read_dir("crates") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name != "unified_verification" {
                    dirs_to_scan.push(path.join("src"));
                }
            }
        }
    }

    for dir in dirs_to_scan {
        if !dir.exists() {
            continue;
        }
        for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file()
                && entry.path().extension().and_then(|s| s.to_str()) == Some("rs")
            {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if let Ok(ast) = syn::parse_file(&content) {
                        let mut visitor = ApiVisitor::default();
                        visitor.visit_file(&ast);
                        if !visitor.items.is_empty() {
                            let mut items = visitor.items;
                            items.sort();
                            let norm_path = entry.path().to_string_lossy().replace("\\", "/");
                            apis.insert(norm_path, items);
                        }
                    }
                }
            }
        }
    }
    apis
}

const BASELINE_PATH: &str = "crates/unified_verification/api_baseline.json";

pub fn regenerate_baseline() {
    let apis = extract_apis();
    let json = serde_json::to_string_pretty(&apis).unwrap();
    fs::write(BASELINE_PATH, json).expect("Failed to write baseline");
    println!("API baseline regenerated successfully.");
}

pub fn check_api_drift() -> bool {
    let current_apis = extract_apis();
    let baseline_content = fs::read_to_string(BASELINE_PATH).unwrap_or_else(|_| "{}".to_string());
    let baseline_apis: BTreeMap<String, Vec<ApiItem>> =
        serde_json::from_str(&baseline_content).unwrap_or_default();

    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let mut changed = false;

    // Check current APIs against baseline
    for (path, items) in &current_apis {
        // Find if any currently existing API is past its expiration
        for item in items {
            if let Some(ref date) = item.deprecation_expires {
                if date.as_str() <= today.as_str() {
                    println!(
                        "API Expired: {} in {} (Expired: {})",
                        item.signature, path, date
                    );
                    println!(
                        "  This deprecated API has passed its expiration target and must be removed."
                    );
                    changed = true;
                }
            }
        }

        if let Some(base_items) = baseline_apis.get(path) {
            // Check for added/removed
            let mut local_changed = false;
            for item in items {
                if !base_items.iter().any(|b| b.signature == item.signature) {
                    println!("  + {}", item.signature);
                    local_changed = true;
                }
            }
            for item in base_items {
                if !items.iter().any(|c| c.signature == item.signature) {
                    // Item was deleted. Check if it's a valid bypass.
                    if let Some(ref date) = item.deprecation_expires {
                        if date.as_str() <= today.as_str() {
                            println!(
                                "  ~ {} (Deleted, bypassed drift check due to expired deprecation)",
                                item.signature
                            );
                            continue; // Skip flagging as changed!
                        }
                    }
                    println!("  - {}", item.signature);
                    local_changed = true;
                }
            }
            if local_changed {
                println!("API Drift detected in {}:", path);
                changed = true;
            }
        } else {
            println!("API Drift: New file with public API: {}", path);
            changed = true;
        }
    }

    // Check files that were completely removed
    for (path, base_items) in &baseline_apis {
        if !current_apis.contains_key(path) {
            let mut all_bypassed = true;
            for item in base_items {
                if let Some(ref date) = item.deprecation_expires {
                    if date.as_str() <= today.as_str() {
                        // bypassed
                        println!(
                            "  ~ {} in {} (File deleted, bypassed due to expired deprecation)",
                            item.signature, path
                        );
                        continue;
                    }
                }
                all_bypassed = false;
            }
            if !all_bypassed {
                println!("API Drift: File with public API removed: {}", path);
                changed = true;
            }
        }
    }

    if changed {
        println!("Public API drift detected! Please regenerate the baseline and write an ADR.");
        return false;
    }

    println!("Public API drift check passed.");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_pattern_parsing() {
        assert_eq!(
            find_date_pattern("expires = 2026-12-31"),
            Some("2026-12-31".to_string())
        );
        assert_eq!(find_date_pattern("short"), None);
        assert_eq!(find_date_pattern("no-date-here-at-all"), None);
    }

    #[test]
    fn test_api_item_shim_deserialization() {
        let json_simple = r#""fn test_function()""#;
        let item: ApiItem = serde_json::from_str(json_simple).unwrap();
        assert_eq!(item.signature, "fn test_function()");
        assert_eq!(item.deprecation_expires, None);

        let json_detailed = r#"{
            "signature": "fn old_func()",
            "deprecation_expires": "2026-11-01"
        }"#;
        let item2: ApiItem = serde_json::from_str(json_detailed).unwrap();
        assert_eq!(item2.signature, "fn old_func()");
        assert_eq!(item2.deprecation_expires, Some("2026-11-01".to_string()));
    }

    #[test]
    fn test_check_api_drift_runs() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        // Run against repository baseline
        let passed = check_api_drift();
        assert!(
            passed,
            "Public API drift should match baseline in repository"
        );
    }
}
