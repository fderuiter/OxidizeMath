//! Transitive Cargo workspace feature and dependency graph resolution.

use crate::path_utils;
use crate::workspace::{
    ReleaseRole, WorkspaceError, WorkspacePackageMeta, discover_workspace_packages,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct WorkspaceDepEdge {
    dep_key: String,
    package_name: String,
    optional: bool,
    default_features: bool,
    features: Vec<String>,
}

#[derive(Clone, Debug)]
struct PackageGraphNode {
    meta: WorkspacePackageMeta,
    is_proc_macro: bool,
    features: BTreeMap<String, Vec<String>>,
    deps: Vec<WorkspaceDepEdge>,
}

fn collect_dep_table(
    table: &toml::Table,
    workspace_pkg_names: &BTreeSet<String>,
    out: &mut Vec<WorkspaceDepEdge>,
) {
    for (dep_key, val) in table {
        let pkg_name = val
            .as_table()
            .and_then(|t| t.get("package"))
            .and_then(toml::Value::as_str)
            .unwrap_or(dep_key);
        if !workspace_pkg_names.contains(pkg_name) {
            continue;
        }
        let optional = val
            .as_table()
            .and_then(|t| t.get("optional"))
            .and_then(toml::Value::as_bool)
            .unwrap_or(false);
        let default_features = val
            .as_table()
            .and_then(|t| {
                t.get("default-features")
                    .or_else(|| t.get("default_features"))
            })
            .and_then(toml::Value::as_bool)
            .unwrap_or(true);
        let features = val
            .as_table()
            .and_then(|t| t.get("features"))
            .and_then(toml::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        out.push(WorkspaceDepEdge {
            dep_key: dep_key.clone(),
            package_name: pkg_name.to_string(),
            optional,
            default_features,
            features,
        });
    }
}

fn parse_package_graph_nodes(
    workspace_root: &Path,
) -> Result<BTreeMap<String, PackageGraphNode>, WorkspaceError> {
    let packages = discover_workspace_packages(workspace_root)?;
    let pkg_names: BTreeSet<String> = packages.iter().map(|p| p.package_name.clone()).collect();
    let mut nodes = BTreeMap::new();

    for meta in packages {
        let manifest_path = workspace_root.join(&meta.member_path).join("Cargo.toml");
        let raw = fs::read_to_string(&manifest_path).map_err(|e| WorkspaceError::ManifestIo {
            path: manifest_path.display().to_string(),
            reason: e.to_string(),
        })?;
        let table = raw
            .parse::<toml::Table>()
            .map_err(|e| WorkspaceError::InvalidManifestSyntax(e.to_string()))?;
        let is_proc_macro = table
            .get("lib")
            .and_then(toml::Value::as_table)
            .and_then(|l| l.get("proc-macro").or_else(|| l.get("proc_macro")))
            .and_then(toml::Value::as_bool)
            .unwrap_or(false);

        let mut features = BTreeMap::new();
        if let Some(ft) = table.get("features").and_then(toml::Value::as_table) {
            for (k, v) in ft {
                let items = v
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                features.insert(k.clone(), items);
            }
        }

        let mut deps = Vec::new();
        if let Some(dt) = table.get("dependencies").and_then(toml::Value::as_table) {
            collect_dep_table(dt, &pkg_names, &mut deps);
        }
        if let Some(targets) = table.get("target").and_then(toml::Value::as_table) {
            for target_val in targets.values() {
                if let Some(dt) = target_val
                    .as_table()
                    .and_then(|t| t.get("dependencies"))
                    .and_then(toml::Value::as_table)
                {
                    collect_dep_table(dt, &pkg_names, &mut deps);
                }
            }
        }

        nodes.insert(
            meta.package_name.clone(),
            PackageGraphNode {
                meta,
                is_proc_macro,
                features,
                deps,
            },
        );
    }
    Ok(nodes)
}

#[derive(Default)]
struct ReachabilityState {
    active_pkgs: BTreeSet<String>,
    active_opt_deps: BTreeMap<String, BTreeSet<String>>,
    active_features: BTreeMap<String, BTreeSet<String>>,
}

fn normalize_feat_key(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace('-', "_")
}

fn match_feature_name(node: &PackageGraphNode, raw: &str) -> Option<String> {
    if node.features.contains_key(raw) {
        return Some(raw.to_string());
    }
    let norm = normalize_feat_key(raw);
    node.features
        .keys()
        .find(|k| normalize_feat_key(k) == norm)
        .cloned()
}

fn match_optional_dep_key(node: &PackageGraphNode, raw: &str) -> Option<String> {
    let norm = normalize_feat_key(raw);
    node.deps
        .iter()
        .find(|d| d.optional && (d.dep_key == raw || normalize_feat_key(&d.dep_key) == norm))
        .map(|d| d.dep_key.clone())
}

fn step_reachability(
    nodes: &BTreeMap<String, PackageGraphNode>,
    state: &mut ReachabilityState,
) -> bool {
    let mut changed = false;
    let active_snapshot: Vec<String> = state.active_pkgs.iter().cloned().collect();

    for pkg_name in active_snapshot {
        let Some(node) = nodes.get(&pkg_name) else {
            continue;
        };
        let enabled_opt = state
            .active_opt_deps
            .get(&pkg_name)
            .cloned()
            .unwrap_or_default();
        for dep in &node.deps {
            if !dep.optional || enabled_opt.contains(&dep.dep_key) {
                if state.active_pkgs.insert(dep.package_name.clone()) {
                    changed = true;
                }
                if dep.default_features
                    && nodes
                        .get(&dep.package_name)
                        .is_some_and(|dn| dn.features.contains_key("default"))
                    && state
                        .active_features
                        .entry(dep.package_name.clone())
                        .or_default()
                        .insert("default".to_string())
                {
                    changed = true;
                }
                for df in &dep.features {
                    if let Some(dn) = nodes.get(&dep.package_name)
                        && let Some(fk) = match_feature_name(dn, df)
                        && state
                            .active_features
                            .entry(dep.package_name.clone())
                            .or_default()
                            .insert(fk)
                    {
                        changed = true;
                    }
                }
            }
        }

        let pkg_feats: Vec<String> = state
            .active_features
            .get(&pkg_name)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        for feat_name in pkg_feats {
            let Some(tokens) = node.features.get(&feat_name) else {
                continue;
            };
            for tok in tokens {
                changed |= apply_feature_token(nodes, node, &pkg_name, tok, state);
            }
        }
    }
    changed
}

fn apply_feature_token(
    nodes: &BTreeMap<String, PackageGraphNode>,
    node: &PackageGraphNode,
    pkg_name: &str,
    tok: &str,
    state: &mut ReachabilityState,
) -> bool {
    let mut changed = false;
    if let Some(dep_key) = tok.strip_prefix("dep:") {
        if state
            .active_opt_deps
            .entry(pkg_name.to_string())
            .or_default()
            .insert(dep_key.to_string())
        {
            changed = true;
        }
        return changed;
    }
    if let Some((dep_part, sub_feat)) = tok.split_once('/') {
        return apply_dep_feature_slash(nodes, node, pkg_name, dep_part, sub_feat, state);
    }
    if let Some(fk) = match_feature_name(node, tok)
        && state
            .active_features
            .entry(pkg_name.to_string())
            .or_default()
            .insert(fk)
    {
        changed = true;
    }
    if let Some(dk) = match_optional_dep_key(node, tok)
        && state
            .active_opt_deps
            .entry(pkg_name.to_string())
            .or_default()
            .insert(dk)
    {
        changed = true;
    }
    changed
}

fn apply_dep_feature_slash(
    nodes: &BTreeMap<String, PackageGraphNode>,
    node: &PackageGraphNode,
    pkg_name: &str,
    dep_part: &str,
    sub_feat: &str,
    state: &mut ReachabilityState,
) -> bool {
    let mut changed = false;
    let (dep_key, weak) = match dep_part.strip_suffix('?') {
        Some(k) => (k, true),
        None => (dep_part, false),
    };
    if !weak
        && state
            .active_opt_deps
            .entry(pkg_name.to_string())
            .or_default()
            .insert(dep_key.to_string())
    {
        changed = true;
    }
    let dep_enabled = node.deps.iter().any(|d| {
        d.dep_key == dep_key
            && (!d.optional
                || state
                    .active_opt_deps
                    .get(pkg_name)
                    .is_some_and(|s| s.contains(dep_key)))
    });
    if dep_enabled {
        for dep in node.deps.iter().filter(|d| d.dep_key == dep_key) {
            if let Some(dn) = nodes.get(&dep.package_name)
                && let Some(fk) = match_feature_name(dn, sub_feat)
                && state
                    .active_features
                    .entry(dep.package_name.clone())
                    .or_default()
                    .insert(fk)
            {
                changed = true;
            }
        }
    }
    changed
}

/// Resolves all workspace packages transitively reachable from `root_package` given `enabled_features`.
pub fn resolve_transitive_workspace_deps(
    workspace_root: &Path,
    root_package: &str,
    enabled_features: &BTreeSet<String>,
) -> Result<Vec<WorkspacePackageMeta>, WorkspaceError> {
    let nodes = parse_package_graph_nodes(workspace_root)?;
    let mut state = ReachabilityState::default();
    state.active_pkgs.insert(root_package.to_string());
    if let Some(root_node) = nodes.get(root_package) {
        for raw_feat in enabled_features {
            if let Some(fk) = match_feature_name(root_node, raw_feat) {
                state
                    .active_features
                    .entry(root_package.to_string())
                    .or_default()
                    .insert(fk);
            }
            if let Some(dk) = match_optional_dep_key(root_node, raw_feat) {
                state
                    .active_opt_deps
                    .entry(root_package.to_string())
                    .or_default()
                    .insert(dk);
            }
        }
    }
    while step_reachability(&nodes, &mut state) {}

    let mut out = Vec::new();
    for pkg_name in &state.active_pkgs {
        if let Some(node) = nodes.get(pkg_name) {
            out.push(node.meta.clone());
        }
    }
    Ok(out)
}

/// Computes the sorted workspace-relative source directories watched and scanned by `gui_package`
/// (`math_explorer_gui`) under `enabled_features`.
///
/// Includes `gui_package`'s configured `traceability_subdir` alongside `<member_path>/src` for every
/// transitively enabled first-party non-proc-macro workspace dependency.
pub fn compute_gui_snapshot_code_dirs(
    workspace_root: &Path,
    gui_package: &str,
    enabled_features: &BTreeSet<String>,
) -> Result<Vec<String>, WorkspaceError> {
    let nodes = parse_package_graph_nodes(workspace_root)?;
    let proc_macro_pkgs: BTreeSet<String> = nodes
        .values()
        .filter(|n| n.is_proc_macro)
        .map(|n| n.meta.package_name.clone())
        .collect();
    let reachable =
        resolve_transitive_workspace_deps(workspace_root, gui_package, enabled_features)?;

    let mut dirs = BTreeSet::new();
    for meta in reachable {
        if meta.effective_release_role() == Some(ReleaseRole::Vendored) {
            continue;
        }
        let Some(node) = nodes.get(&meta.package_name) else {
            continue;
        };
        let is_macro_wrapper = node
            .deps
            .iter()
            .any(|d| proc_macro_pkgs.contains(&d.package_name));
        if node.is_proc_macro || is_macro_wrapper {
            continue;
        }
        let subdir = if meta.package_name == gui_package {
            meta.traceability_subdir()
        } else {
            "src"
        };
        let rel = PathBuf::from(&meta.member_path).join(subdir);
        dirs.insert(path_utils::normalize_path(&rel));
    }
    Ok(dirs.into_iter().collect())
}
