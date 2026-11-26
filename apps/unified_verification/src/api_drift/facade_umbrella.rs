use super::extract_defined_error_idents;
use crate::api_drift::ApiItem;
use crate::api_drift::facade::{
    FacadeContext, ParsedSignatureMap, is_any_facade_pkg_file, is_facade_reachable_file,
    is_tier2_engine_file, split_api_key,
};
use oxidize_core::traceability::AstCache;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::visit::Visit;

fn last_type_ident(ty: &syn::Type) -> Option<String> {
    if let syn::Type::Path(tp) = ty {
        tp.path.segments.last().map(|s| s.ident.to_string())
    } else {
        None
    }
}

fn extract_from_impl_source(imp: &syn::ItemImpl, target_ident: &str) -> Option<String> {
    if last_type_ident(&imp.self_ty).as_deref() != Some(target_ident) {
        return None;
    }
    let (_, trait_path, _) = imp.trait_.as_ref()?;
    let last_seg = trait_path.segments.last()?;
    if last_seg.ident != "From" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        syn::GenericArgument::Type(ty) => last_type_ident(ty),
        _ => None,
    })
}

fn collect_macro_from_sources(tokens: proc_macro2::TokenStream, out: &mut BTreeSet<String>) {
    let items: Vec<proc_macro2::TokenTree> = tokens.into_iter().collect();
    for idx in 0..items.len() {
        if let proc_macro2::TokenTree::Group(group) = &items[idx] {
            collect_macro_from_sources(group.stream(), out);
        }
        if idx + 2 < items.len()
            && let (
                proc_macro2::TokenTree::Ident(err_id),
                proc_macro2::TokenTree::Punct(eq),
                proc_macro2::TokenTree::Punct(gt),
            ) = (&items[idx], &items[idx + 1], &items[idx + 2])
            && eq.as_char() == '='
            && gt.as_char() == '>'
        {
            out.insert(err_id.to_string());
        }
    }
}

fn discover_umbrella_file_and_ident(
    workspace_root: &Path,
    member_dir: &str,
    cache: &mut AstCache,
) -> Option<(String, String)> {
    for suffix in ["src/error/umbrella.rs", "src/error.rs"] {
        let rel = format!("{member_dir}/{suffix}");
        let full = workspace_root.join(&rel);
        if !full.is_file() {
            continue;
        }
        let ast = cache.get_or_load_fs(&full).and_then(|e| e.ast.as_ref())?;
        for item in &ast.items {
            if let syn::Item::Enum(en) = item
                && matches!(en.vis, syn::Visibility::Public(_))
                && en.ident.to_string().ends_with("Error")
            {
                return Some((rel, en.ident.to_string()));
            }
        }
    }
    None
}

fn parse_supported_from_idents(
    path: &Path,
    umbrella_ident: &str,
    cache: &mut AstCache,
) -> BTreeSet<String> {
    let mut supported = BTreeSet::new();
    let Some(ast) = cache.get_or_load_fs(path).and_then(|e| e.ast.as_ref()) else {
        return supported;
    };
    for item in &ast.items {
        match item {
            syn::Item::Enum(en) if en.ident == umbrella_ident => {
                for variant in &en.variants {
                    for field in &variant.fields {
                        let has_from = field.attrs.iter().any(|a| a.path().is_ident("from"));
                        if has_from && let Some(src) = last_type_ident(&field.ty) {
                            supported.insert(src);
                        }
                    }
                }
            }
            syn::Item::Impl(imp) => {
                if let Some(src) = extract_from_impl_source(imp, umbrella_ident) {
                    supported.insert(src);
                }
            }
            syn::Item::Macro(mac) if mac.mac.path.is_ident("impl_error_from") => {
                collect_macro_from_sources(mac.mac.tokens.clone(), &mut supported);
            }
            _ => {}
        }
    }
    supported
}

#[derive(Default)]
struct ResultErrorVisitor {
    errors: Vec<String>,
}

impl<'ast> Visit<'ast> for ResultErrorVisitor {
    fn visit_path_segment(&mut self, seg: &'ast syn::PathSegment) {
        if seg.ident == "MathResult" {
            self.errors.push("MathError".to_string());
        } else if seg.ident == "Result"
            && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
            && args.args.len() >= 2
            && let Some(syn::GenericArgument::Type(err_ty)) = args.args.iter().nth(1)
            && let Some(id) = last_type_ident(err_ty)
            && id.ends_with("Error")
        {
            self.errors.push(id);
        }
        syn::visit::visit_path_segment(self, seg);
    }
}

fn collect_defined_errors_matching(
    ctx: &FacadeContext,
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    pred: impl Fn(&str) -> bool,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (path, items) in apis {
        if !pred(path) {
            continue;
        }
        for item in items {
            if item.signature.starts_with("pub mod __doctest_facade") {
                continue;
            }
            let Some(Some(file)) = parsed_sigs.get(item.signature.as_str()) else {
                continue;
            };
            for name in extract_defined_error_idents(ctx, file, path) {
                if name.ends_with("Error") {
                    out.entry(name).or_insert_with(|| path.clone());
                }
            }
        }
    }
    out
}

fn collect_signature_result_errors_for_pkg(
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    pkg_name: &str,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (path, items) in apis {
        if split_api_key(path).map(|(p, _)| p) != Some(pkg_name) {
            continue;
        }
        for item in items {
            if item.signature.starts_with("pub mod __doctest_facade") {
                continue;
            }
            let Some(Some(file)) = parsed_sigs.get(item.signature.as_str()) else {
                continue;
            };
            let mut visitor = ResultErrorVisitor::default();
            visitor.visit_file(file);
            for err_id in visitor.errors {
                out.entry(err_id)
                    .or_insert_with(|| format!("`{path}` (`{}`)", item.signature));
            }
        }
    }
    out
}

pub(super) fn audit_crate_umbrella_conversions(
    ctx: &FacadeContext,
    workspace_root: &Path,
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let tier2_errors =
        collect_defined_errors_matching(ctx, apis, parsed_sigs, |p| is_tier2_engine_file(ctx, p));

    for (pkg_name, member_dir) in &ctx.umbrella_crates {
        let Some((umbrella_rel, umbrella_ident)) =
            discover_umbrella_file_and_ident(workspace_root, member_dir, cache)
        else {
            continue;
        };
        let umbrella_path = workspace_root.join(&umbrella_rel);
        let supported = parse_supported_from_idents(&umbrella_path, &umbrella_ident, cache);
        let allowed_engines = ctx.engine_deps_by_pkg.get(pkg_name);
        let mut required: BTreeMap<String, String> = tier2_errors
            .iter()
            .filter(|(_, p)| {
                split_api_key(p)
                    .and_then(|(eng_pkg, _)| allowed_engines.map(|set| set.contains(eng_pkg)))
                    .unwrap_or(false)
            })
            .map(|(e, p)| (e.clone(), format!("Tier 2 engine `{p}`")))
            .collect();
        for (err, p) in collect_defined_errors_matching(ctx, apis, parsed_sigs, |p| {
            split_api_key(p).map(|(k, _)| k) == Some(pkg_name.as_str())
        }) {
            required
                .entry(err)
                .or_insert_with(|| format!("crate error in `{p}`"));
        }
        for (err, origin) in collect_signature_result_errors_for_pkg(apis, parsed_sigs, pkg_name) {
            required.entry(err).or_insert(origin);
        }
        required.remove(&umbrella_ident);

        for (err_name, origin) in required {
            if !supported.contains(&err_name) {
                violations.push(format!(
                    "Umbrella conversion gap: `{umbrella_ident}` in `{umbrella_rel}` lacks `From<{err_name}>` required by {origin}"
                ));
            }
        }
    }

    audit_facade_conversions_file(ctx, workspace_root, apis, parsed_sigs, cache, violations);
}

fn audit_facade_conversions_file(
    ctx: &FacadeContext,
    workspace_root: &Path,
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let facade_conv_rel = format!("{}/src/error/conversions.rs", ctx.facade_dir);
    let facade_conv_path = workspace_root.join(&facade_conv_rel);
    if !facade_conv_path.is_file() {
        return;
    }
    let supported = parse_supported_from_idents(&facade_conv_path, "Error", cache);
    let all_reachable = collect_defined_errors_matching(ctx, apis, parsed_sigs, |p| {
        (is_facade_reachable_file(ctx, p) || is_tier2_engine_file(ctx, p))
            && !is_any_facade_pkg_file(ctx, p)
    });
    for (err_name, path) in all_reachable {
        if !supported.contains(&err_name) {
            violations.push(format!(
                "Umbrella conversion gap: `Error` in `{facade_conv_rel}` lacks `From<{err_name}>` for `{path}`"
            ));
        }
    }
}
