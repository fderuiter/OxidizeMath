use crate::api_drift::traversal::{
    ExportFilter, collect_facade_exports, has_cfg_test, resolve_child_mod_file,
};
use crate::ast_helpers::{MAX_TYPE_UNWRAP_STEPS, format_file_line, impl_base_ident};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::visit::Visit;

use super::TRAIT_CONTRACT_ALLOWLIST;

const PANIC_MACROS: &[&str] = &[
    "panic",
    "assert",
    "assert_eq",
    "assert_ne",
    "unreachable",
    "unimplemented",
    "todo",
];

const PANIC_METHODS: &[&str] = &["unwrap", "expect", "unwrap_err", "expect_err"];

#[derive(Clone, Debug)]
pub(super) enum FnOwner {
    Free,
    Inherent(String),
    Trait { trait_name: String, self_ty: String },
}

#[derive(Clone, Debug)]
pub(super) struct PanickingFnRecord {
    pub(super) rel_path: String,
    pub(super) line: usize,
    pub(super) name: String,
    pub(super) owner: FnOwner,
}

#[derive(Default)]
pub(super) struct CrateContractState {
    pub(super) free_fns: BTreeSet<String>,
    pub(super) inherent_methods: BTreeMap<String, BTreeSet<String>>,
    pub(super) panicking_fns: Vec<PanickingFnRecord>,
}

fn unwrap_type_wrappers(mut ty: &syn::Type) -> &syn::Type {
    for _ in 0..MAX_TYPE_UNWRAP_STEPS {
        match ty {
            syn::Type::Group(g) => ty = &g.elem,
            syn::Type::Paren(p) => ty = &p.elem,
            _ => break,
        }
    }
    ty
}

fn is_untyped_error_type(ty: &syn::Type) -> Option<&'static str> {
    match unwrap_type_wrappers(ty) {
        syn::Type::Tuple(tup) if tup.elems.is_empty() => Some("()"),
        syn::Type::Path(tp) if tp.qself.is_none() => {
            let last = tp.path.segments.last()?.ident.to_string();
            match last.as_str() {
                "String" => Some("String"),
                "str" => Some("str"),
                _ => None,
            }
        }
        syn::Type::Reference(r) => match unwrap_type_wrappers(&r.elem) {
            syn::Type::Path(tp) if tp.qself.is_none() => {
                let last = tp.path.segments.last()?.ident.to_string();
                match last.as_str() {
                    "str" => Some("&str"),
                    "String" => Some("&String"),
                    _ => None,
                }
            }
            _ => None,
        },
        _ => None,
    }
}

fn untyped_result_error_in_return(output: &syn::ReturnType) -> Option<&'static str> {
    let syn::ReturnType::Type(_, ret_ty) = output else {
        return None;
    };
    let syn::Type::Path(tp) = unwrap_type_wrappers(ret_ty) else {
        return None;
    };
    if tp.qself.is_some() {
        return None;
    }
    let seg = tp.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let type_args: Vec<&syn::Type> = args
        .args
        .iter()
        .filter_map(|arg| match arg {
            syn::GenericArgument::Type(t) => Some(t),
            _ => None,
        })
        .collect();
    if type_args.len() != 2 {
        return None;
    }
    is_untyped_error_type(type_args[1])
}

fn returns_result_like(output: &syn::ReturnType) -> bool {
    let syn::ReturnType::Type(_, ret_ty) = output else {
        return false;
    };
    let syn::Type::Path(tp) = unwrap_type_wrappers(ret_ty) else {
        return false;
    };
    tp.qself.is_none()
        && tp
            .path
            .segments
            .last()
            .is_some_and(|seg| seg.ident.to_string().ends_with("Result"))
}

fn is_proc_macro_attr(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("proc_macro")
            || attr.path().is_ident("proc_macro_attribute")
            || attr.path().is_ident("proc_macro_derive")
    })
}

#[derive(Default)]
struct PanicBodyVisitor {
    panics: bool,
}

impl<'ast> Visit<'ast> for PanicBodyVisitor {
    fn visit_item(&mut self, _: &'ast syn::Item) {}

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if let Some(seg) = mac.path.segments.last()
            && PANIC_MACROS.contains(&seg.ident.to_string().as_str())
        {
            self.panics = true;
            return;
        }
        syn::visit::visit_macro(self, mac);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if PANIC_METHODS.contains(&call.method.to_string().as_str()) {
            self.panics = true;
            return;
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}

fn body_contains_direct_panic(block: &syn::Block) -> bool {
    let mut visitor = PanicBodyVisitor::default();
    visitor.visit_block(block);
    visitor.panics
}

fn is_companion_name(name: &str) -> bool {
    name.starts_with("try_") || name.starts_with("checked_")
}

pub(super) fn has_matching_companion(name: &str, candidates: &BTreeSet<String>) -> bool {
    let try_name = format!("try_{name}");
    let checked_name = format!("checked_{name}");
    candidates.contains(&try_name) || candidates.contains(&checked_name)
}

pub(super) struct ModuleContractAnalyzer<'a> {
    pub(super) rel_path: &'a str,
    pub(super) check_companions: bool,
    pub(super) state: &'a mut CrateContractState,
    pub(super) violations: &'a mut Vec<String>,
}

impl ModuleContractAnalyzer<'_> {
    fn check_sig_result(&mut self, sig: &syn::Signature, context_prefix: &str) {
        if let Some(bad_err) = untyped_result_error_in_return(&sig.output) {
            let loc = format_file_line(self.rel_path, sig.fn_token.span.start().line);
            self.violations.push(format!(
                "[!] Untyped Result signature at {loc}: `{context_prefix}{}` returns `Result<_, {bad_err}>` (use a typed domain error enum)",
                sig.ident
            ));
        }
    }

    fn inspect_free_fn(&mut self, func: &syn::ItemFn, filter: &ExportFilter) {
        if !matches!(func.vis, syn::Visibility::Public(_))
            || has_cfg_test(&func.attrs)
            || is_proc_macro_attr(&func.attrs)
        {
            return;
        }
        let name = func.sig.ident.to_string();
        if !filter.allows_ident(&name) {
            return;
        }
        self.check_sig_result(&func.sig, "pub fn ");
        if returns_result_like(&func.sig.output) {
            self.state.free_fns.insert(name.clone());
        }
        if self.check_companions
            && !is_companion_name(&name)
            && !returns_result_like(&func.sig.output)
            && body_contains_direct_panic(&func.block)
        {
            self.state.panicking_fns.push(PanickingFnRecord {
                rel_path: self.rel_path.to_string(),
                line: func.sig.fn_token.span.start().line,
                name,
                owner: FnOwner::Free,
            });
        }
    }

    fn inspect_trait(&mut self, item_trait: &syn::ItemTrait, filter: &ExportFilter) {
        let trait_name = item_trait.ident.to_string();
        if !matches!(item_trait.vis, syn::Visibility::Public(_))
            || has_cfg_test(&item_trait.attrs)
            || !filter.allows_ident(&trait_name)
        {
            return;
        }
        let prefix = format!("trait {trait_name}::");
        for item in &item_trait.items {
            if let syn::TraitItem::Fn(m) = item
                && !has_cfg_test(&m.attrs)
            {
                self.check_sig_result(&m.sig, &prefix);
            }
        }
    }

    fn inspect_impl(&mut self, item_impl: &syn::ItemImpl) {
        if has_cfg_test(&item_impl.attrs) {
            return;
        }
        let Some(self_ty) = impl_base_ident(&item_impl.self_ty) else {
            return;
        };
        if let Some((_, trait_path, _)) = &item_impl.trait_ {
            self.inspect_trait_impl(item_impl, trait_path, &self_ty);
        } else {
            self.inspect_inherent_impl(item_impl, &self_ty);
        }
    }

    fn inspect_trait_impl(
        &mut self,
        item_impl: &syn::ItemImpl,
        trait_path: &syn::Path,
        self_ty: &str,
    ) {
        let Some(seg) = trait_path.segments.last() else {
            return;
        };
        let trait_name = seg.ident.to_string();
        if !self.check_companions
            || TRAIT_CONTRACT_ALLOWLIST
                .binary_search(&trait_name.as_str())
                .is_ok()
        {
            return;
        }
        for sub in &item_impl.items {
            if let syn::ImplItem::Fn(m) = sub
                && !has_cfg_test(&m.attrs)
                && !is_companion_name(&m.sig.ident.to_string())
                && !returns_result_like(&m.sig.output)
                && body_contains_direct_panic(&m.block)
            {
                self.state.panicking_fns.push(PanickingFnRecord {
                    rel_path: self.rel_path.to_string(),
                    line: m.sig.fn_token.span.start().line,
                    name: m.sig.ident.to_string(),
                    owner: FnOwner::Trait {
                        trait_name: trait_name.clone(),
                        self_ty: self_ty.to_string(),
                    },
                });
            }
        }
    }

    fn inspect_inherent_impl(&mut self, item_impl: &syn::ItemImpl, self_ty: &str) {
        let prefix = format!("{self_ty}::");
        for sub in &item_impl.items {
            let syn::ImplItem::Fn(m) = sub else {
                continue;
            };
            if !matches!(m.vis, syn::Visibility::Public(_)) || has_cfg_test(&m.attrs) {
                continue;
            }
            let name = m.sig.ident.to_string();
            self.check_sig_result(&m.sig, &prefix);
            if returns_result_like(&m.sig.output) {
                self.state
                    .inherent_methods
                    .entry(self_ty.to_string())
                    .or_default()
                    .insert(name.clone());
            }
            if self.check_companions
                && !is_companion_name(&name)
                && !returns_result_like(&m.sig.output)
                && body_contains_direct_panic(&m.block)
            {
                self.state.panicking_fns.push(PanickingFnRecord {
                    rel_path: self.rel_path.to_string(),
                    line: m.sig.fn_token.span.start().line,
                    name,
                    owner: FnOwner::Inherent(self_ty.to_string()),
                });
            }
        }
    }
}

pub(super) fn scan_module_items(
    items: &[syn::Item],
    filter: &ExportFilter,
    mod_dir: &Path,
    analyzer: &mut ModuleContractAnalyzer<'_>,
) -> Vec<(PathBuf, ExportFilter)> {
    let facades = collect_facade_exports(items, filter);
    let mut children = Vec::new();

    for item in items {
        match item {
            syn::Item::Fn(f) => analyzer.inspect_free_fn(f, filter),
            syn::Item::Trait(t) => analyzer.inspect_trait(t, filter),
            syn::Item::Impl(imp) => analyzer.inspect_impl(imp),
            syn::Item::Mod(m) => {
                if has_cfg_test(&m.attrs) {
                    continue;
                }
                let mod_name = m.ident.to_string();
                let is_pub =
                    matches!(m.vis, syn::Visibility::Public(_)) && filter.allows_ident(&mod_name);
                let eff_filter = if is_pub {
                    Some(ExportFilter::AllPublic)
                } else {
                    facades.get(&mod_name).cloned()
                };
                let Some(child_filter) = eff_filter else {
                    continue;
                };
                if let Some((_, inline_items)) = &m.content {
                    let inline_dir = mod_dir.join(&mod_name);
                    children.extend(scan_module_items(
                        inline_items,
                        &child_filter,
                        &inline_dir,
                        analyzer,
                    ));
                } else if let Some(child_file) =
                    resolve_child_mod_file(mod_dir, &mod_name, &m.attrs)
                {
                    children.push((child_file, child_filter));
                }
            }
            _ => {}
        }
    }
    children
}
