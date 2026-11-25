use super::extract_defined_error_idents;
use crate::api_drift::ApiItem;
use crate::api_drift::facade::{
    ParsedSignatureMap, is_facade_reachable_file, is_tier0_foundation_file, is_tier2_engine_file,
};
use oxidize_core::traceability::AstCache;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use syn::visit::Visit;

struct CrateUmbrellaSpec {
    crate_prefix: &'static str,
    umbrella_ident: &'static str,
    umbrella_file: &'static str,
}

const CRATE_UMBRELLA_SPECS: &[CrateUmbrellaSpec] = &[
    CrateUmbrellaSpec {
        crate_prefix: "crates/math_commons/src/",
        umbrella_ident: "MathCommonsError",
        umbrella_file: "crates/math_commons/src/error/umbrella.rs",
    },
    CrateUmbrellaSpec {
        crate_prefix: "crates/pure_math/src/",
        umbrella_ident: "PureMathError",
        umbrella_file: "crates/pure_math/src/error/umbrella.rs",
    },
    CrateUmbrellaSpec {
        crate_prefix: "crates/domain_ai/src/",
        umbrella_ident: "AIError",
        umbrella_file: "crates/domain_ai/src/error.rs",
    },
    CrateUmbrellaSpec {
        crate_prefix: "crates/domain_applied/src/",
        umbrella_ident: "AppliedError",
        umbrella_file: "crates/domain_applied/src/error/umbrella.rs",
    },
    CrateUmbrellaSpec {
        crate_prefix: "crates/domain_biology/src/",
        umbrella_ident: "BiologyError",
        umbrella_file: "crates/domain_biology/src/error/umbrella.rs",
    },
    CrateUmbrellaSpec {
        crate_prefix: "crates/domain_climate/src/",
        umbrella_ident: "ClimateError",
        umbrella_file: "crates/domain_climate/src/error.rs",
    },
    CrateUmbrellaSpec {
        crate_prefix: "crates/domain_epidemiology/src/",
        umbrella_ident: "EpidemiologyDomainError",
        umbrella_file: "crates/domain_epidemiology/src/error/umbrella.rs",
    },
    CrateUmbrellaSpec {
        crate_prefix: "crates/domain_physics/src/",
        umbrella_ident: "PhysicsError",
        umbrella_file: "crates/domain_physics/src/error/umbrella.rs",
    },
];

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

fn resolve_child_mod_path(
    parent_file: &Path,
    item_mod: &syn::ItemMod,
) -> Option<std::path::PathBuf> {
    if let Some(attr_path) = item_mod
        .attrs
        .iter()
        .find(|a| a.path().is_ident("path"))
        .and_then(|a| match &a.meta {
            syn::Meta::NameValue(nv) => match &nv.value {
                syn::Expr::Lit(el) => match &el.lit {
                    syn::Lit::Str(s) => Some(s.value()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        })
    {
        return parent_file.parent().map(|p| p.join(attr_path));
    }
    if parent_file.file_name().and_then(|n| n.to_str()) == Some("mod.rs") {
        parent_file
            .parent()
            .map(|p| p.join(format!("{}.rs", item_mod.ident)))
    } else {
        Some(
            parent_file
                .with_extension("")
                .join(format!("{}.rs", item_mod.ident)),
        )
    }
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
    let mut child_paths = Vec::new();
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
            syn::Item::Mod(item_mod) if item_mod.content.is_none() => {
                if let Some(child) = resolve_child_mod_path(path, item_mod) {
                    child_paths.push(child);
                }
            }
            _ => {}
        }
    }
    for child in child_paths {
        if child.is_file() {
            supported.extend(parse_supported_from_idents(&child, umbrella_ident, cache));
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

fn collect_defined_errors_under_prefix(
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
            for name in extract_defined_error_idents(file, path) {
                if name.ends_with("Error") {
                    out.entry(name).or_insert_with(|| path.clone());
                }
            }
        }
    }
    out
}

fn collect_signature_result_errors(
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    crate_prefix: &str,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (path, items) in apis {
        if !path.starts_with(crate_prefix) {
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
    workspace_root: &Path,
    apis: &BTreeMap<String, Vec<ApiItem>>,
    parsed_sigs: &ParsedSignatureMap<'_>,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let tier0_errors =
        collect_defined_errors_under_prefix(apis, parsed_sigs, is_tier0_foundation_file);
    let tier2_errors = collect_defined_errors_under_prefix(apis, parsed_sigs, is_tier2_engine_file);

    for spec in CRATE_UMBRELLA_SPECS {
        let umbrella_path = workspace_root.join(spec.umbrella_file);
        if !umbrella_path.is_file() {
            continue;
        }
        let supported = parse_supported_from_idents(&umbrella_path, spec.umbrella_ident, cache);
        let mut required: BTreeMap<String, String> = tier0_errors
            .iter()
            .map(|(e, p)| (e.clone(), format!("Tier 0 foundation `{p}`")))
            .chain(
                tier2_errors
                    .iter()
                    .map(|(e, p)| (e.clone(), format!("Tier 2 engine `{p}`"))),
            )
            .collect();
        for (err, p) in collect_defined_errors_under_prefix(apis, parsed_sigs, |p| {
            p.starts_with(spec.crate_prefix)
        }) {
            required
                .entry(err)
                .or_insert_with(|| format!("crate error in `{p}`"));
        }
        for (err, origin) in collect_signature_result_errors(apis, parsed_sigs, spec.crate_prefix) {
            required.entry(err).or_insert(origin);
        }
        required.remove(spec.umbrella_ident);

        for (err_name, origin) in required {
            if !supported.contains(&err_name) {
                violations.push(format!(
                    "Umbrella conversion gap: `{}` in `{}` lacks `From<{err_name}>` required by {origin}",
                    spec.umbrella_ident, spec.umbrella_file
                ));
            }
        }
    }

    let facade_conv_rel = "math_explorer/src/error/conversions.rs";
    let facade_conv_path = workspace_root.join(facade_conv_rel);
    if facade_conv_path.is_file() {
        let supported = parse_supported_from_idents(&facade_conv_path, "Error", cache);
        let all_reachable = collect_defined_errors_under_prefix(apis, parsed_sigs, |p| {
            (is_facade_reachable_file(p) || is_tier0_foundation_file(p) || is_tier2_engine_file(p))
                && !p.starts_with("math_explorer/src/")
        });
        for (err_name, path) in all_reachable {
            if !supported.contains(&err_name) {
                violations.push(format!(
                    "Umbrella conversion gap: `Error` in `{facade_conv_rel}` lacks `From<{err_name}>` for `{path}`"
                ));
            }
        }
    }
}
