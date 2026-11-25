use super::ApiItem;
use super::{
    ParsedSignatureMap, is_facade_reachable_file, is_tier0_foundation_file, is_tier2_engine_file,
    parse_exported_idents_in_file,
};
use oxidize_core::traceability::AstCache;
use std::collections::BTreeMap;
use std::path::Path;

#[path = "facade_umbrella.rs"]
mod facade_umbrella;

pub(super) const ERROR_CRATE_FILES: &[&str] = &[
    "crates/00_foundation/math_rng/src/error.rs",
    "crates/02_engines/engine_linalg/src/error.rs",
    "crates/engine_diffeq/src/error.rs",
    "crates/engine_pde/src/error.rs",
    "crates/math_commons/src/error.rs",
    "crates/pure_math/src/error.rs",
    "crates/domain_ai/src/error.rs",
    "crates/domain_applied/src/error.rs",
    "crates/domain_biology/src/error.rs",
    "crates/domain_climate/src/error.rs",
    "crates/domain_epidemiology/src/error.rs",
    "crates/domain_physics/src/error.rs",
];

pub(super) const DOMAIN_INNER_MODS: &[&str] = &[
    "crates/pure_math/src/pure_math/mod.rs",
    "crates/domain_ai/src/ai/mod.rs",
    "crates/domain_applied/src/applied/mod.rs",
    "crates/domain_biology/src/biology/mod.rs",
    "crates/domain_climate/src/climate/mod.rs",
    "crates/domain_epidemiology/src/epidemiology/mod.rs",
    "crates/domain_physics/src/physics/mod.rs",
];

fn is_crate_error_module_path(rel_path: &str) -> bool {
    rel_path.starts_with("crates/")
        && (rel_path.ends_with("/src/error.rs") || rel_path.contains("/src/error/"))
}

fn is_reachable_error_ident(ident: &str, rel_path: &str) -> bool {
    ident.ends_with("Error") || ident == "Error" || is_crate_error_module_path(rel_path)
}

pub(super) fn extract_defined_error_idents(file: &syn::File, rel_path: &str) -> Vec<String> {
    let mut found = Vec::new();
    for item in &file.items {
        let candidate = match item {
            syn::Item::Enum(e) => Some(e.ident.to_string()),
            syn::Item::Struct(s) => Some(s.ident.to_string()),
            _ => None,
        };
        if let Some(name) = candidate
            && is_reachable_error_ident(&name, rel_path)
        {
            found.push(name);
        }
    }
    found
}

fn audit_reachable_submodule_errors(
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    facade_errors: &std::collections::BTreeSet<String>,
    violations: &mut Vec<String>,
) {
    let mut definitions_by_name: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (path, items) in apis {
        let in_scope = is_facade_reachable_file(path)
            || is_tier0_foundation_file(path)
            || is_tier2_engine_file(path);
        if !in_scope || path.starts_with("math_explorer/src/") {
            continue;
        }
        for item in items {
            if item.signature.starts_with("pub mod __doctest_facade") {
                continue;
            }
            let Some(Some(file)) = parsed_sigs.get(item.signature.as_str()) else {
                continue;
            };
            for err_name in extract_defined_error_idents(file, path) {
                definitions_by_name
                    .entry(err_name.clone())
                    .or_default()
                    .push(format!("{path} (`{}`)", item.signature));
                if !facade_errors.contains(&err_name) {
                    violations.push(format!(
                        "Facade error gap: reachable error type `{err_name}` in `{path}` is not re-exported by `math_explorer/src/error.rs`"
                    ));
                }
            }
        }
    }

    for (err_name, locations) in definitions_by_name {
        if locations.len() > 1 {
            violations.push(format!(
                "Duplicate reachable error name `{err_name}` defined across multiple modules: {}",
                locations.join(", ")
            ));
        }
    }
}

pub(super) fn check_crate_error_reexports(
    workspace_root: &Path,
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let facade_errors =
        parse_exported_idents_in_file(&workspace_root.join("math_explorer/src/error.rs"), cache);
    for rel in ERROR_CRATE_FILES {
        let path = workspace_root.join(rel);
        let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) else {
            continue;
        };
        for item in &ast.items {
            let ident = match item {
                syn::Item::Enum(e) if matches!(e.vis, syn::Visibility::Public(_)) => Some(&e.ident),
                syn::Item::Struct(s) if matches!(s.vis, syn::Visibility::Public(_)) => {
                    Some(&s.ident)
                }
                _ => None,
            };
            if let Some(id) = ident
                && !facade_errors.contains(&id.to_string())
            {
                violations.push(format!(
                    "Facade error gap: `pub {id}` in `{rel}` is not re-exported by `math_explorer/src/error.rs`"
                ));
            }
        }
    }
    for rel in DOMAIN_INNER_MODS {
        if !parse_exported_idents_in_file(&workspace_root.join(rel), cache).contains("error") {
            violations.push(format!(
                "Facade domain error gap: `{rel}` must publicly expose `error` (`pub use crate::error;`)"
            ));
        }
    }
    audit_reachable_submodule_errors(apis, parsed_sigs, &facade_errors, violations);
    facade_umbrella::audit_crate_umbrella_conversions(
        workspace_root,
        apis,
        parsed_sigs,
        cache,
        violations,
    );
}
