use crate::doc_audit::manifest_model::WorkspaceManifestModel;
use oxidize_core::traceability::MaturityRegistry;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const DEFAULT_REACHABLE_PKGS: &[&str] = &[
    "oxidizemath",
    "math_explorer",
    "math_diagnostics",
    "math_memory",
    "math_rng",
    "math_scalars",
    "diagnostics",
    "math_commons",
    "engine_linalg",
    "engine_opt",
    "engine_diffeq",
    "engine_pde",
    "pure_math",
    "domain_ai",
    "domain_applied",
    "domain_biology",
    "domain_climate",
    "domain_engineering",
    "domain_epidemiology",
    "domain_physics",
];

const DEFAULT_TIER2_ENGINES: &[&str] =
    &["engine_linalg", "engine_opt", "engine_diffeq", "engine_pde"];

pub(crate) struct FacadeContext {
    pub has_manifest: bool,
    pub facade_dir: String,
    pub facade_pkg_names: BTreeSet<String>,
    pub reachable_pkgs: BTreeSet<String>,
    pub tier2_engine_pkgs: BTreeSet<String>,
    pub engine_and_domain_dirs: Vec<(String, String)>,
    pub domain_mods: Vec<(String, String)>,
    pub umbrella_crates: Vec<(String, String)>,
    pub engine_deps_by_pkg: BTreeMap<String, BTreeSet<String>>,
}

impl FacadeContext {
    pub(crate) fn from_workspace(workspace_root: &Path) -> Option<Self> {
        if let Ok(model) = WorkspaceManifestModel::load(workspace_root) {
            return Self::from_manifest_model(workspace_root, &model);
        }
        Self::fallback_without_manifest(workspace_root)
    }

    fn from_manifest_model(workspace_root: &Path, model: &WorkspaceManifestModel) -> Option<Self> {
        let facade = model.primary_facade_pkg()?;
        if !workspace_root
            .join(&facade.member_dir)
            .join("src/lib.rs")
            .is_file()
        {
            return None;
        }
        let facade_pkg_names: BTreeSet<String> = model
            .facade_pkgs()
            .into_iter()
            .map(|p| p.name.clone())
            .collect();
        let mut reachable_pkgs = facade_pkg_names.clone();
        for p in model
            .foundation_facade_pkgs()
            .into_iter()
            .chain(model.engine_pkgs())
            .chain(model.domain_pkgs())
        {
            reachable_pkgs.insert(p.name.clone());
        }
        let tier2_engine_pkgs: BTreeSet<String> = model
            .engine_pkgs()
            .into_iter()
            .filter(|p| p.name != "math_commons")
            .map(|p| p.name.clone())
            .collect();
        let engine_and_domain_dirs: Vec<(String, String)> = model
            .engine_pkgs()
            .into_iter()
            .chain(model.domain_pkgs())
            .map(|p| (p.name.clone(), p.member_dir.clone()))
            .collect();
        let domain_mods = collect_domain_mods_from_model(model);
        let (umbrella_crates, engine_deps_by_pkg) = collect_umbrellas_from_model(model);
        Some(Self {
            has_manifest: true,
            facade_dir: facade.member_dir.clone(),
            facade_pkg_names,
            reachable_pkgs,
            tier2_engine_pkgs,
            engine_and_domain_dirs,
            domain_mods,
            umbrella_crates,
            engine_deps_by_pkg,
        })
    }

    fn fallback_without_manifest(workspace_root: &Path) -> Option<Self> {
        let facade_dir = ["oxidizemath", "math_explorer"]
            .into_iter()
            .find(|d| workspace_root.join(d).join("src/lib.rs").is_file())
            .map(str::to_string)?;
        let facade_pkg_names: BTreeSet<String> = ["oxidizemath", "math_explorer"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let reachable_pkgs: BTreeSet<String> = DEFAULT_REACHABLE_PKGS
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let tier2_engine_pkgs: BTreeSet<String> = DEFAULT_TIER2_ENGINES
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let (engine_and_domain_dirs, domain_mods, umbrella_crates) =
            scan_fallback_dirs(workspace_root, &tier2_engine_pkgs);
        let mut engine_deps_by_pkg = BTreeMap::new();
        for (pkg, _) in &umbrella_crates {
            engine_deps_by_pkg.insert(pkg.clone(), tier2_engine_pkgs.clone());
        }
        Some(Self {
            has_manifest: false,
            facade_dir,
            facade_pkg_names,
            reachable_pkgs,
            tier2_engine_pkgs,
            engine_and_domain_dirs,
            domain_mods,
            umbrella_crates,
            engine_deps_by_pkg,
        })
    }

    #[cfg(test)]
    pub(crate) fn default_for_sigs() -> Self {
        Self {
            has_manifest: false,
            facade_dir: "oxidizemath".to_string(),
            facade_pkg_names: ["oxidizemath".to_string(), "math_explorer".to_string()]
                .into_iter()
                .collect(),
            reachable_pkgs: DEFAULT_REACHABLE_PKGS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            tier2_engine_pkgs: DEFAULT_TIER2_ENGINES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            engine_and_domain_dirs: Vec::new(),
            domain_mods: Vec::new(),
            umbrella_crates: Vec::new(),
            engine_deps_by_pkg: BTreeMap::new(),
        }
    }
}

fn collect_domain_mods_from_model(model: &WorkspaceManifestModel) -> Vec<(String, String)> {
    model
        .domain_pkgs()
        .into_iter()
        .map(|p| {
            let m = p
                .metadata
                .facade_module
                .clone()
                .unwrap_or_else(|| p.name.trim_start_matches("domain_").to_string());
            (p.member_dir.clone(), m)
        })
        .collect()
}

type UmbrellaEngineDeps = (Vec<(String, String)>, BTreeMap<String, BTreeSet<String>>);

fn collect_umbrellas_from_model(model: &WorkspaceManifestModel) -> UmbrellaEngineDeps {
    let mut umbrella_crates = Vec::new();
    if let Some(mc) = model.packages_by_name.get("math_commons") {
        umbrella_crates.push((mc.name.clone(), mc.member_dir.clone()));
    }
    for p in model.domain_pkgs() {
        umbrella_crates.push((p.name.clone(), p.member_dir.clone()));
    }
    let mut engine_deps_by_pkg = BTreeMap::new();
    for (pkg_name, _) in &umbrella_crates {
        engine_deps_by_pkg.insert(pkg_name.clone(), model.transitive_engine_deps(pkg_name));
    }
    (umbrella_crates, engine_deps_by_pkg)
}

type FallbackScanResult = (
    Vec<(String, String)>,
    Vec<(String, String)>,
    Vec<(String, String)>,
);

fn scan_fallback_dirs(
    workspace_root: &Path,
    tier2_engine_pkgs: &BTreeSet<String>,
) -> FallbackScanResult {
    let mut engine_and_domain_dirs = Vec::new();
    let mut domain_mods = Vec::new();
    let mut umbrella_crates = Vec::new();
    for root in crate::api_drift::collect_crate_roots(workspace_root) {
        let Some(crate_dir) = root.parent().and_then(Path::parent) else {
            continue;
        };
        let pkg = crate_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let rel_dir = crate_dir
            .strip_prefix(workspace_root)
            .unwrap_or(crate_dir)
            .to_string_lossy()
            .replace('\\', "/");
        if tier2_engine_pkgs.contains(&pkg)
            || pkg == "math_commons"
            || pkg == "pure_math"
            || pkg.starts_with("domain_")
        {
            engine_and_domain_dirs.push((pkg.clone(), rel_dir.clone()));
        }
        if pkg == "pure_math" || pkg.starts_with("domain_") {
            let m = pkg.trim_start_matches("domain_").to_string();
            domain_mods.push((rel_dir.clone(), m));
            umbrella_crates.push((pkg, rel_dir));
        } else if pkg == "math_commons" {
            umbrella_crates.push((pkg, rel_dir));
        }
    }
    (engine_and_domain_dirs, domain_mods, umbrella_crates)
}

pub(crate) fn split_api_key(key: &str) -> Option<(&str, &str)> {
    let norm = key.trim_start_matches("./");
    if let Some((pkg, rel)) = norm.split_once(':')
        && !pkg.is_empty()
        && !pkg.contains('/')
    {
        return Some((pkg, rel));
    }
    let pkg = MaturityRegistry::crate_for_repo_path(norm)?;
    let src_idx = norm.find("/src/")?;
    Some((pkg, &norm[src_idx + 1..]))
}

pub(crate) fn is_facade_reachable_file(ctx: &FacadeContext, key: &str) -> bool {
    let Some((pkg, rel)) = split_api_key(key) else {
        return false;
    };
    if pkg == "oxidize_core" {
        return rel.starts_with("src/mesh/") || rel == "src/rng.rs";
    }
    ctx.reachable_pkgs.contains(pkg)
}

pub(crate) fn is_tier2_engine_file(ctx: &FacadeContext, key: &str) -> bool {
    split_api_key(key).is_some_and(|(pkg, _)| ctx.tier2_engine_pkgs.contains(pkg))
}

pub(crate) fn is_any_facade_pkg_file(ctx: &FacadeContext, key: &str) -> bool {
    split_api_key(key).is_some_and(|(pkg, _)| ctx.facade_pkg_names.contains(pkg))
}
