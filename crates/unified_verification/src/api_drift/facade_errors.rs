use super::ApiItem;
use super::{is_facade_reachable_file, parse_exported_idents_in_file, parse_signature_ast};
use oxidize_core::traceability::AstCache;
use std::collections::BTreeMap;
use std::path::Path;

pub(super) const ERROR_CRATE_FILES: &[&str] = &[
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

fn extract_defined_error_idents(sig: &str, rel_path: &str) -> Vec<String> {
    let Some(file) = parse_signature_ast(sig) else {
        return Vec::new();
    };
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
    facade_errors: &std::collections::BTreeSet<String>,
    violations: &mut Vec<String>,
) {
    let mut definitions_by_name: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (path, items) in apis {
        if !is_facade_reachable_file(path) || path.starts_with("math_explorer/src/") {
            continue;
        }
        for item in items {
            if item.signature.starts_with("pub mod __doctest_facade") {
                continue;
            }
            for err_name in extract_defined_error_idents(&item.signature, path) {
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
    audit_reachable_submodule_errors(apis, &facade_errors, violations);
}
