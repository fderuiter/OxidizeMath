//! Fail-closed workspace five-tier DAG and single-domain feature isolation gate (`check-dag`).
//!
//! Validates the strict downward runtime dependency hierarchy
//! `apps -> 04_bindings -> 03_domains -> 02_engines -> 01_pure -> 00_foundation`,
//! forbids peer `domain_* -> domain_*` edges and runtime-to-tooling edges, and verifies
//! that each single-domain Cargo feature on `oxidizemath`, `math_explorer`, and
//! `math_explorer_gui` activates at most one `domain_*` crate.

use crate::doc_audit::manifest_model::{PackageManifest, WorkspaceManifestModel};
use oxidize_core::workspace::{ReleaseRole, TierKind, resolve_transitive_workspace_deps};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const SINGLE_DOMAIN_FEATURES: &[&str] = &[
    "pure_math",
    "ai",
    "applied",
    "biology",
    "climate",
    "epidemiology",
    "physics",
];

const FEATURE_ISOLATED_PACKAGES: &[&str] = &["oxidizemath", "math_explorer", "math_explorer_gui"];

/// Logical runtime tier in the strict downward five-tier hierarchy (`04_bindings -> ... -> 00_foundation`), plus `apps`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DagTier {
    Foundation00,
    Pure01,
    Engines02,
    Domains03,
    Bindings04,
    Apps,
}

impl DagTier {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Foundation00 => "00_foundation",
            Self::Pure01 => "01_pure",
            Self::Engines02 => "02_engines",
            Self::Domains03 => "03_domains",
            Self::Bindings04 => "04_bindings",
            Self::Apps => "apps",
        }
    }
}

/// Classification of a workspace package for DAG enforcement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DagRole {
    Runtime(DagTier),
    Vendored,
    Tooling,
}

fn from_tier_kind(kind: TierKind) -> Option<DagRole> {
    match kind {
        TierKind::Foundation00 => Some(DagRole::Runtime(DagTier::Foundation00)),
        TierKind::Core01 => Some(DagRole::Runtime(DagTier::Pure01)),
        TierKind::Engines02 => Some(DagRole::Runtime(DagTier::Engines02)),
        TierKind::Domains03 => Some(DagRole::Runtime(DagTier::Domains03)),
        TierKind::Bindings04 => Some(DagRole::Runtime(DagTier::Bindings04)),
        TierKind::Apps => Some(DagRole::Runtime(DagTier::Apps)),
        TierKind::Vendored => Some(DagRole::Vendored),
    }
}

/// Resolves the logical [`DagRole`] for a workspace crate, failing closed (`None`) on unclassified crates.
#[must_use]
pub fn classify_package(pkg: &PackageManifest) -> Option<DagRole> {
    if matches!(
        pkg.name.as_str(),
        "xtask" | "unified_verification" | "markdown_tests"
    ) {
        return Some(DagRole::Tooling);
    }
    if pkg.effective_release_role == Some(ReleaseRole::Vendored)
        || pkg.effective_tier == Some(TierKind::Vendored)
    {
        return Some(DagRole::Vendored);
    }
    match pkg.name.as_str() {
        "math_commons" | "pure_math" | "oxidize_core" | "diagnostics" | "federated_registry" => {
            return Some(DagRole::Runtime(DagTier::Pure01));
        }
        "math_diagnostics"
        | "math_rng"
        | "math_scalars"
        | "scientific_metadata"
        | "verified_engine"
        | "verified_engine_macros" => return Some(DagRole::Runtime(DagTier::Foundation00)),
        "oxidizemath" | "math_explorer" | "scientific_verification" => {
            return Some(DagRole::Runtime(DagTier::Bindings04));
        }
        "oxidize-cli" | "math_explorer_gui" => return Some(DagRole::Runtime(DagTier::Apps)),
        _ => {}
    }
    if let Some(tier) = pkg
        .effective_tier
        .or_else(|| TierKind::from_member_path(&pkg.member_dir))
    {
        return from_tier_kind(tier);
    }
    if pkg.member_dir.starts_with("crates/01_pure/") {
        return Some(DagRole::Runtime(DagTier::Pure01));
    }
    if pkg.name.starts_with("domain_") {
        return Some(DagRole::Runtime(DagTier::Domains03));
    }
    if pkg.name.starts_with("engine_") {
        return Some(DagRole::Runtime(DagTier::Engines02));
    }
    None
}

fn check_single_edge(
    pkg: &PackageManifest,
    src_role: DagRole,
    dep_pkg: &PackageManifest,
    dst_role: DagRole,
    section: &str,
    violations: &mut Vec<String>,
) {
    match (src_role, dst_role) {
        (DagRole::Tooling, _) => {}
        (DagRole::Runtime(src_tier), DagRole::Tooling) => violations.push(format!(
            "[DAG] {}/Cargo.toml ([{section}]): runtime crate `{}` (source tier `{}`) cannot depend on developer tooling crate `{}` (target tier `developer_tooling`).\n  Remediation: Remove `{}` from `{}/Cargo.toml` or move verification logic into `unified_verification` / `xtask`.",
            pkg.member_dir, pkg.name, src_tier.as_str(), dep_pkg.name, dep_pkg.name, pkg.member_dir
        )),
        (DagRole::Vendored, DagRole::Tooling) => violations.push(format!(
            "[DAG] {}/Cargo.toml ([{section}]): vendored crate `{}` (source tier `vendored`) cannot depend on developer tooling crate `{}` (target tier `developer_tooling`).\n  Remediation: Remove `{}` from `{}/Cargo.toml`.",
            pkg.member_dir, pkg.name, dep_pkg.name, dep_pkg.name, pkg.member_dir
        )),
        (DagRole::Vendored, DagRole::Runtime(dst_tier)) if dst_tier > DagTier::Engines02 => {
            violations.push(format!(
                "[DAG] {}/Cargo.toml ([{section}]): vendored crate `{}` (source tier `vendored`) cannot depend upward on `{}` (target tier `{}`).\n  Remediation: Restrict vendored crate dependencies to `02_engines`, `01_pure`, or `00_foundation`.",
                pkg.member_dir, pkg.name, dep_pkg.name, dst_tier.as_str()
            ));
        }
        (DagRole::Runtime(src_tier), DagRole::Vendored) if src_tier < DagTier::Bindings04 => {
            violations.push(format!(
                "[DAG] {}/Cargo.toml ([{section}]): crate `{}` (source tier `{}`) cannot depend on vendored UI crate `{}` (target tier `vendored`).\n  Remediation: Keep vendored UI dependencies strictly inside `apps` or `04_bindings`.",
                pkg.member_dir, pkg.name, src_tier.as_str(), dep_pkg.name
            ));
        }
        (DagRole::Runtime(DagTier::Domains03), DagRole::Runtime(DagTier::Domains03)) => {
            violations.push(format!(
                "[DAG] {}/Cargo.toml ([{section}]): sideways peer-domain edge `{}` (source tier `03_domains`) -> `{}` (target tier `03_domains`) is forbidden.\n  Remediation: Promote shared mathematical kernels to `02_engines`, `01_pure` (`pure_math`/`oxidize_core`), or `00_foundation`, and compose cross-domain workflows in `04_bindings` or `apps`.",
                pkg.member_dir, pkg.name, dep_pkg.name
            ));
        }
        (DagRole::Runtime(src_tier), DagRole::Runtime(dst_tier)) if dst_tier > src_tier => {
            violations.push(format!(
                "[DAG] {}/Cargo.toml ([{section}]): upward tier edge `{}` (source tier `{}`) -> `{}` (target tier `{}`) violates `apps -> 04_bindings -> 03_domains -> 02_engines -> 01_pure -> 00_foundation`.\n  Remediation: Promote shared primitives downward to `{}` (or lower), or relocate higher-tier wrappers to `{}` and re-export through `oxidizemath`.",
                pkg.member_dir,
                pkg.name,
                src_tier.as_str(),
                dep_pkg.name,
                dst_tier.as_str(),
                src_tier.as_str(),
                dst_tier.as_str()
            ));
        }
        _ => {}
    }
}

fn audit_package_edges(
    model: &WorkspaceManifestModel,
    roles: &BTreeMap<String, DagRole>,
    violations: &mut Vec<String>,
) {
    for pkg in model.packages_by_name.values() {
        let Some(&src_role) = roles.get(&pkg.name) else {
            continue;
        };
        for (section, deps) in [
            ("dependencies", &pkg.normal_deps),
            ("build-dependencies", &pkg.build_deps),
            ("dev-dependencies", &pkg.dev_deps),
        ] {
            for dep_name in deps {
                if dep_name == &pkg.name {
                    continue;
                }
                let Some(dep_pkg) = model.packages_by_name.get(dep_name) else {
                    continue;
                };
                let Some(&dst_role) = roles.get(dep_name) else {
                    continue;
                };
                check_single_edge(pkg, src_role, dep_pkg, dst_role, section, violations);
            }
        }
    }
}

fn detect_cycles<'a>(
    curr: &'a str,
    packages: &'a BTreeMap<String, PackageManifest>,
    state: &mut BTreeMap<&'a str, u8>,
    stack: &mut Vec<&'a str>,
    violations: &mut Vec<String>,
) {
    state.insert(curr, 1);
    stack.push(curr);
    if let Some(pkg) = packages.get(curr) {
        for dep in &pkg.normal_deps {
            if dep == curr {
                continue;
            }
            let Some(dep_pkg) = packages.get(dep) else {
                continue;
            };
            let key = dep_pkg.name.as_str();
            match state.get(key).copied().unwrap_or(0) {
                0 => detect_cycles(key, packages, state, stack, violations),
                1 => violations.push(format!(
                    "[DAG] Workspace dependency cycle detected: {} -> {key}.\n  Remediation: Break the cyclic normal dependency edge in `{}/Cargo.toml`.",
                    stack.join(" -> "),
                    pkg.member_dir
                )),
                _ => {}
            }
        }
    }
    stack.pop();
    state.insert(curr, 2);
}

fn collect_candidate_domain_features(model: &WorkspaceManifestModel) -> BTreeSet<String> {
    let mut feats: BTreeSet<String> = SINGLE_DOMAIN_FEATURES
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    for pkg in model.packages_by_name.values() {
        if let Some(fm) = pkg.metadata.facade_module.as_deref() {
            feats.insert(fm.to_string());
        }
    }
    feats
}

fn audit_single_feature_closure(
    workspace_root: &Path,
    pkg: &PackageManifest,
    feat: &str,
    roles: &BTreeMap<String, DagRole>,
    violations: &mut Vec<String>,
) {
    let enabled = BTreeSet::from([feat.to_string()]);
    let reachable = match resolve_transitive_workspace_deps(workspace_root, &pkg.name, &enabled) {
        Ok(list) => list,
        Err(err) => {
            violations.push(format!(
                "[DAG] {}/Cargo.toml: failed to resolve transitive feature closure for `{}` (`--features {feat}`): {err}.\n  Remediation: Fix invalid feature or optional dependency syntax in `Cargo.toml`.",
                pkg.member_dir, pkg.name
            ));
            return;
        }
    };
    let domain_crates: Vec<String> = reachable
        .into_iter()
        .map(|m| m.package_name)
        .filter(|name| roles.get(name) == Some(&DagRole::Runtime(DagTier::Domains03)))
        .collect();

    if feat == "pure_math" && !domain_crates.is_empty() {
        violations.push(format!(
            "[DAG] {}/Cargo.toml: `--no-default-features --features pure_math` on `{}` activates `03_domains` crate(s) [{}], expected 0.\n  Remediation: Remove `domain_*` feature chains from `pure_math` in `{}/Cargo.toml`.",
            pkg.member_dir, pkg.name, domain_crates.join(", "), pkg.member_dir
        ));
        return;
    }
    if domain_crates.len() > 1 {
        violations.push(format!(
            "[DAG] {}/Cargo.toml: `--no-default-features --features {feat}` on `{}` activates {} `domain_*` crates ([{}]); at most 1 is allowed.\n  Remediation: Remove cross-domain feature chains from `{}/Cargo.toml` and guard cross-domain items with `#[cfg(all(feature = \"{feat}\", feature = \"...\"))]`.",
            pkg.member_dir,
            pkg.name,
            domain_crates.len(),
            domain_crates.join(", "),
            pkg.member_dir
        ));
        return;
    }
    if feat != "pure_math"
        && let Some(only_dom) = domain_crates.first()
        && only_dom != &format!("domain_{feat}")
    {
        violations.push(format!(
            "[DAG] {}/Cargo.toml: `--no-default-features --features {feat}` on `{}` activates unrelated domain crate `{only_dom}` (expected `domain_{feat}`).\n  Remediation: Wire feature `{feat}` to `domain_{feat}` instead of `{only_dom}`.",
            pkg.member_dir, pkg.name
        ));
    }
}

/// Evaluates all tier DAG, peer-domain, runtime-to-tooling, and single-domain feature closure rules in `workspace_root`.
#[must_use]
pub fn collect_dag_violations(workspace_root: &Path) -> Vec<String> {
    let model = match WorkspaceManifestModel::load(workspace_root) {
        Ok(m) => m,
        Err(err) => {
            return vec![format!(
                "[DAG] Fail-closed manifest load error in {}: {err}\n  Remediation: Ensure workspace and member `Cargo.toml` files exist and contain valid TOML.",
                workspace_root.display()
            )];
        }
    };
    if model.packages_by_name.is_empty() {
        return vec![format!(
            "[DAG] Fail-closed empty workspace in {}: zero member crates discovered.\n  Remediation: Populate `[workspace].members` in root `Cargo.toml`.",
            workspace_root.display()
        )];
    }

    let mut violations = Vec::new();
    let mut roles = BTreeMap::new();
    for pkg in model.packages_by_name.values() {
        match classify_package(pkg) {
            Some(role) => {
                roles.insert(pkg.name.clone(), role);
            }
            None => violations.push(format!(
                "[DAG] {}/Cargo.toml: workspace crate `{}` has no recognized tier classification.\n  Remediation: Declare `[package.metadata.oxidize] tier = \"00_foundation\" | \"01_core\" | \"02_engines\" | \"03_domains\" | \"04_bindings\" | \"apps\"` in `{}/Cargo.toml`.",
                pkg.member_dir, pkg.name, pkg.member_dir
            )),
        }
    }

    audit_package_edges(&model, &roles, &mut violations);

    let mut cycle_state = BTreeMap::new();
    for name in model.packages_by_name.keys() {
        if cycle_state.get(name.as_str()).copied().unwrap_or(0) == 0 {
            let mut stack = Vec::new();
            detect_cycles(
                name,
                &model.packages_by_name,
                &mut cycle_state,
                &mut stack,
                &mut violations,
            );
        }
    }

    let candidate_feats = collect_candidate_domain_features(&model);
    for pkg_name in FEATURE_ISOLATED_PACKAGES {
        let Some(pkg) = model.packages_by_name.get(*pkg_name) else {
            continue;
        };
        for feat in &candidate_feats {
            if pkg.features.contains_key(feat) {
                audit_single_feature_closure(workspace_root, pkg, feat, &roles, &mut violations);
            }
        }
    }

    violations
}

/// Runs the fail-closed tier DAG and feature-closure check against `workspace_root`.
pub fn check_dag_in_dir(workspace_root: &Path) -> bool {
    let violations = collect_dag_violations(workspace_root);
    if violations.is_empty() {
        println!(
            "[+] Verified strict 5-tier DAG (`04_bindings -> 03_domains -> 02_engines -> 01_pure -> 00_foundation`) and isolated single-domain feature closures."
        );
        return true;
    }
    for violation in &violations {
        eprintln!("[!] {violation}");
    }
    false
}

/// CLI entrypoint for `unified_verification check-dag`.
pub fn run_check_dag_cli() -> bool {
    let _root_guard = crate::manifest_cache::WorkspaceRootGuard::enter();
    check_dag_in_dir(Path::new("."))
}

#[cfg(test)]
#[path = "dag_gate_tests.rs"]
mod tests;
