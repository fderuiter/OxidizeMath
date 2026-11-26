use super::ApiItem;
use super::{
    FacadeContext, ParsedSignatureMap, is_any_facade_pkg_file, is_facade_reachable_file,
    is_tier2_engine_file, parse_exported_idents_in_file, split_api_key,
};
use oxidize_core::traceability::AstCache;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[path = "facade_umbrella.rs"]
mod facade_umbrella;

fn is_crate_error_module_path(ctx: &FacadeContext, key: &str) -> bool {
    let Some((pkg, rel)) = split_api_key(key) else {
        return false;
    };
    !ctx.facade_pkg_names.contains(pkg) && (rel == "src/error.rs" || rel.starts_with("src/error/"))
}

fn is_reachable_error_ident(ctx: &FacadeContext, ident: &str, key: &str) -> bool {
    ident.ends_with("Error") || ident == "Error" || is_crate_error_module_path(ctx, key)
}

pub(super) fn extract_defined_error_idents(
    ctx: &FacadeContext,
    file: &syn::File,
    key: &str,
) -> Vec<String> {
    let mut found = Vec::new();
    for item in &file.items {
        let candidate = match item {
            syn::Item::Enum(e) => Some(e.ident.to_string()),
            syn::Item::Struct(s) => Some(s.ident.to_string()),
            _ => None,
        };
        if let Some(name) = candidate
            && is_reachable_error_ident(ctx, &name, key)
        {
            found.push(name);
        }
    }
    found
}

fn audit_reachable_submodule_errors(
    ctx: &FacadeContext,
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    facade_errors: &BTreeSet<String>,
    violations: &mut Vec<String>,
) {
    let facade_err_rel = format!("{}/src/error.rs", ctx.facade_dir);
    let mut definitions_by_name: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (path, items) in apis {
        let in_scope = is_facade_reachable_file(ctx, path) || is_tier2_engine_file(ctx, path);
        if !in_scope || is_any_facade_pkg_file(ctx, path) {
            continue;
        }
        for item in items {
            if item.signature.starts_with("pub mod __doctest_facade") {
                continue;
            }
            let Some(Some(file)) = parsed_sigs.get(item.signature.as_str()) else {
                continue;
            };
            for err_name in extract_defined_error_idents(ctx, file, path) {
                definitions_by_name
                    .entry(err_name.clone())
                    .or_default()
                    .push(format!("{path} (`{}`)", item.signature));
                if !facade_errors.contains(&err_name) {
                    violations.push(format!(
                        "Facade error gap: reachable error type `{err_name}` in `{path}` is not re-exported by `{facade_err_rel}`"
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
    ctx: &FacadeContext,
    workspace_root: &Path,
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let facade_err_rel = format!("{}/src/error.rs", ctx.facade_dir);
    let facade_errors = parse_exported_idents_in_file(&workspace_root.join(&facade_err_rel), cache);
    for (_, member_dir) in &ctx.engine_and_domain_dirs {
        let rel = format!("{member_dir}/src/error.rs");
        let path = workspace_root.join(&rel);
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
                    "Facade error gap: `pub {id}` in `{rel}` is not re-exported by `{facade_err_rel}`"
                ));
            }
        }
    }
    for (member_dir, mod_name) in &ctx.domain_mods {
        let rel = format!("{member_dir}/src/{mod_name}/mod.rs");
        if !parse_exported_idents_in_file(&workspace_root.join(&rel), cache).contains("error") {
            violations.push(format!(
                "Facade domain error gap: `{rel}` must publicly expose `error` (`pub use crate::error;`)"
            ));
        }
    }
    audit_reachable_submodule_errors(ctx, apis, parsed_sigs, &facade_errors, violations);
    facade_umbrella::audit_crate_umbrella_conversions(
        ctx,
        workspace_root,
        apis,
        parsed_sigs,
        cache,
        violations,
    );
}
