//! Single-pass `syn` AST gate enforcing the 4-part mathematical docstring contract
//! across public mathematical models and solvers in all `8` mathematical crates.

use super::math_contracts_parser::{
    ContractPatterns, FileAuditEnv, has_embed_theory, verify_item_contract,
};
use crate::api_drift::traversal::{
    ExportFilter, child_module_dir_for_file, collect_facade_exports, has_cfg_test,
    resolve_child_mod_file,
};
use crate::ast_helpers::impl_base_ident;
use crate::manifest_cache::resolve_workspace_path_from;
use oxidize_core::traceability::AstCache;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Canonical workspace directories of the `8` mathematical crates governed by the 4-part contract.
pub const MATH_CONTRACT_CRATES: &[&str] = &[
    "crates/math_commons",
    "crates/pure_math",
    "crates/domain_ai",
    "crates/domain_applied",
    "crates/domain_biology",
    "crates/domain_climate",
    "crates/domain_epidemiology",
    "crates/domain_physics",
];

const MAX_MODULES_PER_CRATE: usize = 10_000;
const MAX_INLINE_MODULES: usize = 2_048;

const EXCLUDED_TYPE_SUFFIXES: &[&str] = &[
    "Error", "Builder", "Config", "State", "Stats", "Result", "Outcome",
];

const EXPLICIT_MATH_TYPES: &[&str] = &[
    "SGD",
    "Adam",
    "L1RegularizedLeastSquares",
    "KalmanFilter",
    "StandardSolverAdapter",
    "TuringSolverStrategy",
    "Cera",
    "Euler",
    "Heun",
    "RungeKutta4",
    "StormerVerlet",
    "Yoshida4",
    "GaussLegendre4",
    "GreenFunctionSolver1D",
    "FusedStencilStepper",
];

/// Audits all `8` mathematical crates present in `members` using the shared single-pass `AstCache`.
pub fn audit_math_contracts_with_cache(
    workspace_root: &Path,
    members: &[String],
    cache: &mut AstCache,
) -> Vec<String> {
    let patterns = ContractPatterns::new();
    let mut violations = Vec::new();
    for member in members {
        let trimmed = member.trim_end_matches('/');
        if !MATH_CONTRACT_CRATES.contains(&trimmed) {
            continue;
        }
        let crate_dir = resolve_workspace_path_from(workspace_root, trimmed);
        let crate_lib = crate_dir.join("src").join("lib.rs");
        if crate_lib.is_file() {
            audit_math_crate(
                workspace_root,
                &crate_dir,
                &crate_lib,
                &patterns,
                cache,
                &mut violations,
            );
        }
    }
    violations
}

fn audit_math_crate(
    workspace_root: &Path,
    crate_dir: &Path,
    crate_lib: &Path,
    patterns: &ContractPatterns,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let mut visited: BTreeMap<PathBuf, ExportFilter> = BTreeMap::new();
    let mut worklist = vec![crate_lib.to_path_buf()];
    visited.insert(crate_lib.to_path_buf(), ExportFilter::AllPublic);
    for _ in 0..MAX_MODULES_PER_CRATE {
        let Some(file_path) = worklist.pop() else {
            break;
        };
        let Some(active_filter) = visited.get(&file_path).cloned() else {
            continue;
        };
        let mod_dir = child_module_dir_for_file(&file_path);
        let rs_dir = file_path.parent().unwrap_or(crate_dir);
        let rel = file_path.strip_prefix(workspace_root).unwrap_or(&file_path);
        let display_file = rel.to_string_lossy().replace('\\', "/");
        let env = FileAuditEnv {
            workspace_root,
            crate_dir,
            rs_dir,
            display_file: &display_file,
            patterns,
        };
        let child_files = {
            let Some(ast) = cache
                .get_or_load_fs(&file_path)
                .and_then(|e| e.ast.as_ref())
            else {
                continue;
            };
            audit_module_ast(ast, &active_filter, &mod_dir, &env, violations)
        };
        for (child_path, child_filter) in child_files {
            let enqueue = if let Some(prev) = visited.get_mut(&child_path) {
                prev.merge(child_filter)
            } else {
                visited.insert(child_path.clone(), child_filter);
                true
            };
            if enqueue {
                worklist.push(child_path);
            }
        }
    }
}

fn audit_module_ast(
    file: &syn::File,
    root_filter: &ExportFilter,
    root_mod_dir: &Path,
    env: &FileAuditEnv<'_>,
    violations: &mut Vec<String>,
) -> Vec<(PathBuf, ExportFilter)> {
    if has_cfg_test(&file.attrs) || has_doc_hidden(&file.attrs) {
        return Vec::new();
    }
    let mut child_files = Vec::new();
    let mut stack: Vec<(&[syn::Item], ExportFilter, PathBuf)> =
        vec![(&file.items, root_filter.clone(), root_mod_dir.to_path_buf())];
    for _ in 0..MAX_INLINE_MODULES {
        let Some((items, filter, mod_dir)) = stack.pop() else {
            break;
        };
        let facades = collect_facade_exports(items, &filter);
        for item in items {
            if let syn::Item::Mod(item_mod) = item {
                enqueue_mod(
                    item_mod,
                    &filter,
                    &facades,
                    &mod_dir,
                    &mut stack,
                    &mut child_files,
                );
            } else {
                audit_item(item, &filter, env, violations);
            }
        }
    }
    child_files
}

fn enqueue_mod<'a>(
    item_mod: &'a syn::ItemMod,
    parent_filter: &ExportFilter,
    facades: &BTreeMap<String, ExportFilter>,
    mod_dir: &Path,
    stack: &mut Vec<(&'a [syn::Item], ExportFilter, PathBuf)>,
    child_files: &mut Vec<(PathBuf, ExportFilter)>,
) {
    if has_cfg_test(&item_mod.attrs) || has_doc_hidden(&item_mod.attrs) {
        return;
    }
    let mod_name = item_mod.ident.to_string();
    let is_pub =
        matches!(item_mod.vis, syn::Visibility::Public(_)) && parent_filter.allows_ident(&mod_name);
    let mut child_filter = is_pub.then_some(ExportFilter::AllPublic);
    if let Some(reexport) = facades.get(&mod_name).cloned() {
        if let Some(existing) = &mut child_filter {
            existing.merge(reexport);
        } else {
            child_filter = Some(reexport);
        }
    }
    let Some(active_child) = child_filter else {
        return;
    };
    if let Some((_, inline_items)) = &item_mod.content {
        stack.push((inline_items, active_child, mod_dir.join(&mod_name)));
    } else if let Some(path) = resolve_child_mod_file(mod_dir, &mod_name, &item_mod.attrs) {
        child_files.push((path, active_child));
    }
}

fn audit_item(
    item: &syn::Item,
    filter: &ExportFilter,
    env: &FileAuditEnv<'_>,
    violations: &mut Vec<String>,
) {
    match item {
        syn::Item::Struct(s) if is_public_visible(&s.vis, &s.attrs, &s.ident, filter) => {
            if is_math_type_target(&s.ident.to_string(), &s.attrs) {
                verify_item_contract(
                    &s.attrs,
                    s.ident.span().start().line,
                    &format!("struct {}", s.ident),
                    env,
                    violations,
                );
            }
        }
        syn::Item::Enum(e) if is_public_visible(&e.vis, &e.attrs, &e.ident, filter) => {
            if is_math_type_target(&e.ident.to_string(), &e.attrs) {
                verify_item_contract(
                    &e.attrs,
                    e.ident.span().start().line,
                    &format!("enum {}", e.ident),
                    env,
                    violations,
                );
            }
        }
        syn::Item::Trait(t) if is_public_visible(&t.vis, &t.attrs, &t.ident, filter) => {
            if is_math_type_target(&t.ident.to_string(), &t.attrs) {
                verify_item_contract(
                    &t.attrs,
                    t.ident.span().start().line,
                    &format!("trait {}", t.ident),
                    env,
                    violations,
                );
            }
        }
        syn::Item::Fn(f) if is_public_visible(&f.vis, &f.attrs, &f.sig.ident, filter) => {
            if is_math_fn_target(&f.sig.ident.to_string(), None, &f.attrs) {
                verify_item_contract(
                    &f.attrs,
                    f.sig.ident.span().start().line,
                    &format!("fn {}", f.sig.ident),
                    env,
                    violations,
                );
            }
        }
        syn::Item::Impl(imp)
            if imp.trait_.is_none() && !has_cfg_test(&imp.attrs) && !has_doc_hidden(&imp.attrs) =>
        {
            audit_inherent_impl(imp, filter, env, violations);
        }
        _ => {}
    }
}

fn audit_inherent_impl(
    imp: &syn::ItemImpl,
    filter: &ExportFilter,
    env: &FileAuditEnv<'_>,
    violations: &mut Vec<String>,
) {
    let Some(self_ident) = impl_base_ident(&imp.self_ty) else {
        return;
    };
    if !filter.allows_ident(&self_ident) {
        return;
    }
    for impl_item in &imp.items {
        let syn::ImplItem::Fn(m) = impl_item else {
            continue;
        };
        if !matches!(m.vis, syn::Visibility::Public(_))
            || has_cfg_test(&m.attrs)
            || has_doc_hidden(&m.attrs)
        {
            continue;
        }
        let fn_name = m.sig.ident.to_string();
        if is_math_fn_target(&fn_name, Some(&self_ident), &m.attrs) {
            verify_item_contract(
                &m.attrs,
                m.sig.ident.span().start().line,
                &format!("fn {self_ident}::{fn_name}"),
                env,
                violations,
            );
        }
    }
}

fn is_public_visible(
    vis: &syn::Visibility,
    attrs: &[syn::Attribute],
    ident: &syn::Ident,
    filter: &ExportFilter,
) -> bool {
    matches!(vis, syn::Visibility::Public(_))
        && !has_cfg_test(attrs)
        && !has_doc_hidden(attrs)
        && filter.allows_ident(&ident.to_string())
}

fn is_math_type_target(ident: &str, attrs: &[syn::Attribute]) -> bool {
    if ident == "SolverExt"
        || EXCLUDED_TYPE_SUFFIXES
            .iter()
            .any(|suffix| ident.ends_with(suffix))
    {
        return false;
    }
    ident.ends_with("Solver")
        || ident.ends_with("Model")
        || ident.ends_with("Optimizer")
        || ident.ends_with("Integrator")
        || EXPLICIT_MATH_TYPES.contains(&ident)
        || has_derive_theory(attrs)
}

fn is_math_fn_target(fn_name: &str, self_ident: Option<&str>, attrs: &[syn::Attribute]) -> bool {
    if has_embed_theory(attrs) {
        return true;
    }
    match self_ident {
        None => {
            fn_name.starts_with("solve_")
                || fn_name.starts_with("levenberg_marquardt")
                || fn_name.starts_with("fit_model")
        }
        Some("Integer") => matches!(fn_name, "pow_mod" | "invert"),
        Some(_) => false,
    }
}

fn has_doc_hidden(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("doc")
            && let Ok(nested) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            && nested
                .iter()
                .any(|m| matches!(m, syn::Meta::Path(p) if p.is_ident("hidden")))
        {
            return true;
        }
    }
    false
}

fn has_derive_theory(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("derive")
            && let Ok(nested) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )
            && nested
                .iter()
                .any(|p| p.segments.last().is_some_and(|s| s.ident == "Theory"))
        {
            return true;
        }
    }
    false
}
