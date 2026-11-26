use oxidize_core::workspace::{
    FacadeRole, OxidizeMetadata, ReleaseRole, TierKind, parse_oxidize_metadata,
    parse_patched_crates_std, parse_workspace_members,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Parsed Cargo manifest metadata for a single workspace member crate.
#[derive(Debug, Clone)]
pub struct PackageManifest {
    pub name: String,
    pub member_dir: String,
    pub has_lib: bool,
    pub is_proc_macro: bool,
    pub metadata: OxidizeMetadata,
    pub effective_tier: Option<TierKind>,
    pub effective_release_role: Option<ReleaseRole>,
    pub features: BTreeMap<String, Vec<String>>,
    pub examples: BTreeMap<String, Vec<String>>,
    pub benches: BTreeMap<String, Vec<String>>,
    pub normal_deps: BTreeSet<String>,
    pub dev_deps: BTreeSet<String>,
    pub build_deps: BTreeSet<String>,
}

/// Aggregated workspace Cargo manifests for static CLI, TOML, tier DAG, and catalog validation.
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

fn collect_section_dep_names(doc: &toml::Value, section: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let push_from = |tbl: &toml::map::Map<String, toml::Value>, set: &mut BTreeSet<String>| {
        for (key, val) in tbl {
            let pkg_name = val
                .as_table()
                .and_then(|t| t.get("package"))
                .and_then(toml::Value::as_str)
                .unwrap_or(key);
            set.insert(pkg_name.to_string());
        }
    };
    if let Some(tbl) = doc.get(section).and_then(toml::Value::as_table) {
        push_from(tbl, &mut out);
    }
    if let Some(targets) = doc.get("target").and_then(toml::Value::as_table) {
        for target_cfg in targets.values() {
            if let Some(tbl) = target_cfg.get(section).and_then(toml::Value::as_table) {
                push_from(tbl, &mut out);
            }
        }
    }
    out
}

fn parse_member_manifest(
    workspace_root: &Path,
    member_dir: &str,
    patched_crates: &[String],
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
    let is_proc_macro = doc
        .get("lib")
        .and_then(toml::Value::as_table)
        .and_then(|l| l.get("proc-macro").or_else(|| l.get("proc_macro")))
        .and_then(toml::Value::as_bool)
        .unwrap_or(false);

    let metadata = parse_oxidize_metadata(&raw).unwrap_or_default();
    let is_vendored_patch = patched_crates.iter().any(|p| p == &name);
    let ws_meta = oxidize_core::workspace::WorkspacePackageMeta {
        member_path: member_dir.to_string(),
        package_name: name.clone(),
        metadata: metadata.clone(),
        is_vendored_patch,
    };

    Ok(PackageManifest {
        name,
        member_dir: member_dir.to_string(),
        has_lib,
        is_proc_macro,
        effective_tier: ws_meta.effective_tier(),
        effective_release_role: ws_meta.effective_release_role(),
        metadata,
        features: parse_features_table(&doc),
        examples: parse_targets_table(&doc, "example"),
        benches: parse_targets_table(&doc, "bench"),
        normal_deps: collect_section_dep_names(&doc, "dependencies"),
        dev_deps: collect_section_dep_names(&doc, "dev-dependencies"),
        build_deps: collect_section_dep_names(&doc, "build-dependencies"),
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
        let members = parse_workspace_members(&root_raw).map_err(|e| {
            format!(
                "{}: Failed to parse workspace members: {e}",
                root_manifest_path.display()
            )
        })?;
        let patched = parse_patched_crates_std(&root_raw);

        let mut model = Self::default();
        for member in members {
            let norm_dir = member
                .replace('\\', "/")
                .trim_start_matches("./")
                .trim_end_matches('/')
                .to_string();
            let pkg = parse_member_manifest(workspace_root, &norm_dir, &patched)?;
            model.packages_by_dir.insert(norm_dir, pkg.name.clone());
            model.packages_by_name.insert(pkg.name.clone(), pkg);
        }
        Ok(model)
    }

    /// Returns the primary facade package (`FacadeRole::Facade`, preferring `oxidizemath`, falling back to `math_explorer`).
    pub fn primary_facade_pkg(&self) -> Option<&PackageManifest> {
        self.packages_by_name
            .get("oxidizemath")
            .filter(|p| p.metadata.facade_role == FacadeRole::Facade)
            .or_else(|| {
                self.packages_by_name
                    .values()
                    .find(|p| p.metadata.facade_role == FacadeRole::Facade)
            })
            .or_else(|| self.packages_by_name.get("math_explorer"))
    }

    /// Returns all facade packages (`FacadeRole::Facade` and `FacadeRole::CompatFacade`).
    pub fn facade_pkgs(&self) -> Vec<&PackageManifest> {
        self.packages_by_name
            .values()
            .filter(|p| {
                matches!(
                    p.metadata.facade_role,
                    FacadeRole::Facade | FacadeRole::CompatFacade
                )
            })
            .collect()
    }

    /// Returns the package owning CLI `[[example]]` targets (`oxidize-cli` or `math_explorer`).
    pub fn cli_examples_pkg(&self) -> Option<&PackageManifest> {
        self.packages_by_name
            .get("oxidize-cli")
            .filter(|p| !p.examples.is_empty())
            .or_else(|| {
                self.packages_by_name
                    .values()
                    .find(|p| !p.examples.is_empty())
            })
            .or_else(|| self.packages_by_name.get("oxidize-cli"))
            .or_else(|| self.packages_by_name.get("math_explorer"))
    }

    /// Returns all Tier 3 domain crates (`FacadeRole::Domain`).
    pub fn domain_pkgs(&self) -> Vec<&PackageManifest> {
        self.packages_by_name
            .values()
            .filter(|p| p.metadata.facade_role == FacadeRole::Domain)
            .collect()
    }

    /// Returns all Tier 2 engine crates (`FacadeRole::Engine`).
    pub fn engine_pkgs(&self) -> Vec<&PackageManifest> {
        self.packages_by_name
            .values()
            .filter(|p| p.metadata.facade_role == FacadeRole::Engine)
            .collect()
    }

    /// Returns all foundation/core crates re-exported on the facade (`FacadeRole::Foundation`).
    pub fn foundation_facade_pkgs(&self) -> Vec<&PackageManifest> {
        self.packages_by_name
            .values()
            .filter(|p| p.metadata.facade_role == FacadeRole::Foundation)
            .collect()
    }

    /// Returns all crates opted into the 4-part mathematical docstring contract.
    pub fn math_contract_pkgs(&self) -> Vec<&PackageManifest> {
        self.packages_by_name
            .values()
            .filter(|p| p.metadata.math_contract == Some(true))
            .collect()
    }

    /// Returns Tier 2 `#![no_std]` engine package names (`engine_*`) reachable from `pkg_name` without hopping across peer domain crates.
    pub fn transitive_engine_deps(&self, pkg_name: &str) -> BTreeSet<String> {
        let Some(start_pkg) = self.packages_by_name.get(pkg_name) else {
            return BTreeSet::new();
        };
        let mut visited = BTreeSet::new();
        let mut queue: Vec<String> = start_pkg
            .normal_deps
            .iter()
            .filter(|dep| {
                self.packages_by_name
                    .get(*dep)
                    .is_some_and(|p| p.metadata.facade_role == FacadeRole::Engine)
            })
            .cloned()
            .collect();
        let mut engines = BTreeSet::new();
        while let Some(curr) = queue.pop() {
            if !visited.insert(curr.clone()) {
                continue;
            }
            let Some(pkg) = self.packages_by_name.get(&curr) else {
                continue;
            };
            if pkg.metadata.facade_role == FacadeRole::Engine && curr != "math_commons" {
                engines.insert(curr.clone());
            }
            for dep in &pkg.normal_deps {
                if self
                    .packages_by_name
                    .get(dep)
                    .is_some_and(|p| p.metadata.facade_role == FacadeRole::Engine)
                {
                    queue.push(dep.clone());
                }
            }
        }
        engines
    }

    /// Audits acyclic 5-tier dependency ordering (`00_foundation <= ... <= 04_bindings <= apps`).
    pub fn audit_tier_dependencies(&self) -> Vec<String> {
        let mut violations = Vec::new();
        for pkg in self.packages_by_name.values() {
            if pkg.effective_release_role == Some(ReleaseRole::Vendored) {
                continue;
            }
            if let Some(dir_tier) = TierKind::from_member_path(&pkg.member_dir)
                && let Some(decl_tier) = pkg.metadata.tier
                && dir_tier != decl_tier
            {
                violations.push(format!(
                    "{}/Cargo.toml: [package.metadata.oxidize].tier `{}` conflicts with directory tier `{}`",
                    pkg.member_dir,
                    decl_tier.as_str(),
                    dir_tier.as_str()
                ));
            }
            let Some(pkg_tier) = pkg.effective_tier else {
                continue;
            };
            for (kind, deps) in [
                ("dependencies", &pkg.normal_deps),
                ("build-dependencies", &pkg.build_deps),
                ("dev-dependencies", &pkg.dev_deps),
            ] {
                for dep_name in deps {
                    if dep_name == &pkg.name {
                        continue;
                    }
                    let Some(dep_pkg) = self.packages_by_name.get(dep_name) else {
                        continue;
                    };
                    if dep_pkg.effective_release_role == Some(ReleaseRole::Vendored) {
                        continue;
                    }
                    if let Some(dep_tier) = dep_pkg.effective_tier
                        && dep_tier > pkg_tier
                    {
                        violations.push(format!(
                            "{}/Cargo.toml: Tier violation in [{kind}]: `{}` ({}) cannot depend on higher-tier `{}` ({})",
                            pkg.member_dir,
                            pkg.name,
                            pkg_tier.as_str(),
                            dep_pkg.name,
                            dep_tier.as_str()
                        ));
                    }
                }
            }
        }
        detect_normal_dep_cycles(&self.packages_by_name, &mut violations);
        violations
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

fn detect_normal_dep_cycles(
    packages: &BTreeMap<String, PackageManifest>,
    violations: &mut Vec<String>,
) {
    let mut state: BTreeMap<&str, u8> = BTreeMap::new();
    for name in packages.keys() {
        if state.get(name.as_str()).copied().unwrap_or(0) == 0 {
            let mut stack = Vec::new();
            dfs_cycle(name, packages, &mut state, &mut stack, violations);
        }
    }
}

fn dfs_cycle<'a>(
    curr: &'a str,
    packages: &'a BTreeMap<String, PackageManifest>,
    state: &mut BTreeMap<&'a str, u8>,
    stack: &mut Vec<&'a str>,
    violations: &mut Vec<String>,
) {
    state.insert(curr, 1);
    stack.push(curr);
    if let Some(pkg) = packages.get(curr)
        && pkg.effective_release_role != Some(ReleaseRole::Vendored)
    {
        for dep in &pkg.normal_deps {
            let Some(dep_pkg) = packages.get(dep) else {
                continue;
            };
            if dep_pkg.effective_release_role == Some(ReleaseRole::Vendored) {
                continue;
            }
            let dep_key = dep_pkg.name.as_str();
            match state.get(dep_key).copied().unwrap_or(0) {
                0 => dfs_cycle(dep_key, packages, state, stack, violations),
                1 => {
                    violations.push(format!(
                        "Workspace dependency cycle detected involving `{}` -> `{}` ({})",
                        curr,
                        dep_key,
                        stack.join(" -> ")
                    ));
                }
                _ => {}
            }
        }
    }
    stack.pop();
    state.insert(curr, 2);
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
