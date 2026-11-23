use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Parsed Cargo manifest metadata for a single workspace member crate.
#[derive(Debug, Clone)]
pub struct PackageManifest {
    pub name: String,
    pub has_lib: bool,
    pub features: BTreeMap<String, Vec<String>>,
    pub examples: BTreeMap<String, Vec<String>>,
    pub benches: BTreeMap<String, Vec<String>>,
}

/// Aggregated workspace Cargo manifests for static CLI, TOML, and catalog validation.
#[derive(Debug, Clone, Default)]
pub struct WorkspaceManifestModel {
    pub packages_by_name: BTreeMap<String, PackageManifest>,
    pub packages_by_dir: BTreeMap<String, String>,
}

fn parse_string_array(val: Option<&toml::Value>) -> Vec<String> {
    val.and_then(toml::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(toml::Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_targets_table(doc: &toml::Value, key: &str) -> BTreeMap<String, Vec<String>> {
    let mut map = BTreeMap::new();
    let Some(entries) = doc.get(key).and_then(toml::Value::as_array) else {
        return map;
    };
    for entry in entries {
        if let Some(name) = entry.get("name").and_then(toml::Value::as_str) {
            let req = parse_string_array(entry.get("required-features"));
            map.insert(name.to_string(), req);
        }
    }
    map
}

fn parse_features_table(doc: &toml::Value) -> BTreeMap<String, Vec<String>> {
    let mut map = BTreeMap::new();
    let Some(table) = doc.get("features").and_then(toml::Value::as_table) else {
        return map;
    };
    for (feat_name, deps_val) in table {
        map.insert(feat_name.clone(), parse_string_array(Some(deps_val)));
    }
    map
}

fn parse_member_manifest(
    workspace_root: &Path,
    member_dir: &str,
) -> Result<PackageManifest, String> {
    let crate_root = workspace_root.join(member_dir);
    let manifest_path = crate_root.join("Cargo.toml");
    let raw = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: Failed to read manifest: {e}", manifest_path.display()))?;
    let doc: toml::Value = toml::from_str(&raw)
        .map_err(|e| format!("{}: Invalid Cargo.toml: {e}", manifest_path.display()))?;

    let fallback = member_dir.rsplit('/').next().unwrap_or(member_dir);
    let name = doc
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(toml::Value::as_str)
        .unwrap_or(fallback)
        .to_string();

    let has_lib = crate_root.join("src").join("lib.rs").is_file() || doc.get("lib").is_some();

    Ok(PackageManifest {
        name,
        has_lib,
        features: parse_features_table(&doc),
        examples: parse_targets_table(&doc, "example"),
        benches: parse_targets_table(&doc, "bench"),
    })
}

impl WorkspaceManifestModel {
    /// Loads workspace members from `<workspace_root>/Cargo.toml` and parses each member manifest.
    pub fn load(workspace_root: &Path) -> Result<Self, String> {
        let root_manifest_path = workspace_root.join("Cargo.toml");
        let root_raw = fs::read_to_string(&root_manifest_path).map_err(|e| {
            format!(
                "{}: Failed to read workspace Cargo.toml: {e}",
                root_manifest_path.display()
            )
        })?;
        let members = oxidize_core::workspace::parse_workspace_members(&root_raw).map_err(|e| {
            format!(
                "{}: Failed to parse workspace members: {e}",
                root_manifest_path.display()
            )
        })?;

        let mut model = Self::default();
        for member in members {
            let norm_dir = member.replace('\\', "/");
            let pkg = parse_member_manifest(workspace_root, &norm_dir)?;
            model.packages_by_dir.insert(norm_dir, pkg.name.clone());
            model.packages_by_name.insert(pkg.name.clone(), pkg);
        }
        Ok(model)
    }

    /// Resolves the transitive set of local package features enabled by Cargo feature flags.
    pub fn resolve_enabled_features(
        &self,
        pkg_name: &str,
        explicit_features: &[String],
        all_features: bool,
        no_default_features: bool,
    ) -> Result<BTreeSet<String>, String> {
        let pkg = self
            .packages_by_name
            .get(pkg_name)
            .ok_or_else(|| format!("Unknown workspace package `{pkg_name}`"))?;

        for feat in explicit_features {
            if !pkg.features.contains_key(feat) {
                return Err(format!(
                    "Package `{pkg_name}` does not define feature `{feat}`"
                ));
            }
        }

        let mut enabled = BTreeSet::new();
        if all_features {
            enabled.extend(pkg.features.keys().cloned());
            return Ok(enabled);
        }
        if !no_default_features && pkg.features.contains_key("default") {
            enabled.insert("default".to_string());
        }
        enabled.extend(explicit_features.iter().cloned());

        let max_steps = pkg.features.len().saturating_add(1);
        for _ in 0..max_steps {
            let snapshot: Vec<String> = enabled.iter().cloned().collect();
            let before_len = enabled.len();
            for active in snapshot {
                if let Some(deps) = pkg.features.get(&active) {
                    extend_local_feature_deps(&pkg.features, deps, &mut enabled);
                }
            }
            if enabled.len() == before_len {
                break;
            }
        }
        Ok(enabled)
    }

    /// Determines the single dominating domain feature covering `required_features` (e.g. `["applied", "pure_math"]` -> `"applied"`).
    pub fn primary_domain_feature(
        &self,
        pkg_name: &str,
        required_features: &[String],
    ) -> Option<String> {
        for candidate in required_features {
            let Ok(closure) = self.resolve_enabled_features(
                pkg_name,
                std::slice::from_ref(candidate),
                false,
                true,
            ) else {
                continue;
            };
            if required_features.iter().all(|req| closure.contains(req)) {
                return Some(candidate.clone());
            }
        }
        required_features.first().cloned()
    }
}

fn extend_local_feature_deps(
    defined_features: &BTreeMap<String, Vec<String>>,
    deps: &[String],
    enabled: &mut BTreeSet<String>,
) {
    for dep in deps {
        if dep.starts_with("dep:") || dep.contains('/') {
            continue;
        }
        if defined_features.contains_key(dep) {
            enabled.insert(dep.clone());
        }
    }
}
