//! `syn::Meta` helpers for `#[cfg(test)]` filtering, `opt_out` attribute extraction,
//! and verification macro body parsing.

use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Block, Expr, ExprLit, ImplItem, ImplItemFn, Item, ItemFn, Lit, LitStr, Macro, Meta,
    MetaList, Token, TraitItem, TraitItemFn,
};

const MAX_ATTR_NESTING_DEPTH: usize = 8;

fn parse_nested_metas(list: &MetaList) -> Vec<Meta> {
    list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
        .map(|items| items.into_iter().collect())
        .unwrap_or_default()
}

fn meta_requires_cfg_test(meta: &Meta, depth: usize) -> bool {
    if depth == 0 {
        return false;
    }
    match meta {
        Meta::Path(path) => path.is_ident("test"),
        Meta::List(list) if list.path.is_ident("not") => false,
        Meta::List(list) if list.path.is_ident("all") => parse_nested_metas(list)
            .iter()
            .any(|nested| meta_requires_cfg_test(nested, depth - 1)),
        Meta::List(list) if list.path.is_ident("any") => {
            let nested = parse_nested_metas(list);
            !nested.is_empty()
                && nested
                    .iter()
                    .all(|item| meta_requires_cfg_test(item, depth - 1))
        }
        _ => false,
    }
}

/// Returns `true` when `attrs` contain a `cfg(...)` predicate that requires `test` to be active.
///
/// Items gated with `#[cfg(any(target_arch = "wasm32", test))]` return `false` because they
/// compile into production `wasm32` targets.
pub fn attrs_require_cfg_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        attr.meta
            .require_list()
            .ok()
            .and_then(|list| list.parse_args::<Meta>().ok())
            .is_some_and(|meta| meta_requires_cfg_test(&meta, MAX_ATTR_NESTING_DEPTH))
    })
}

fn meta_marks_test_fn(meta: &Meta, depth: usize) -> bool {
    if depth == 0 {
        return false;
    }
    match meta {
        Meta::List(list) if list.path.is_ident("cfg_attr") => parse_nested_metas(list)
            .into_iter()
            .skip(1)
            .any(|nested| meta_marks_test_fn(&nested, depth - 1)),
        Meta::Path(path) | Meta::List(MetaList { path, .. }) => {
            path.segments.last().is_some_and(|seg| {
                seg.ident == "test" || seg.ident == "wasm_bindgen_test" || seg.ident == "bench"
            })
        }
        _ => false,
    }
}

/// Returns `true` when `attrs` mark a function as an executable test or benchmark item.
pub fn attrs_mark_test_fn(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        matches!(attr.style, syn::AttrStyle::Outer)
            && meta_marks_test_fn(&attr.meta, MAX_ATTR_NESTING_DEPTH)
    })
}

pub(super) fn is_verified_attr_path(path: &syn::Path) -> bool {
    path.segments
        .last()
        .is_some_and(|seg| seg.ident == "verified")
}

fn is_embed_theory_attr_path(path: &syn::Path) -> bool {
    path.segments
        .last()
        .is_some_and(|seg| seg.ident == "embed_theory")
}

fn collect_opt_outs_from_verified_list(list: &MetaList, out: &mut Vec<String>) {
    let Ok(args) = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) else {
        return;
    };
    for arg in args {
        match arg {
            Meta::NameValue(nv) if nv.path.is_ident("opt_out") => {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Str(lit_str),
                    ..
                }) = &nv.value
                {
                    out.push(lit_str.value());
                } else {
                    out.push("<non_literal_opt_out>".to_string());
                }
            }
            Meta::Path(path) if path.is_ident("opt_out") => {
                out.push("<missing_opt_out_reason>".to_string());
            }
            Meta::List(sub) if sub.path.is_ident("opt_out") => {
                out.push("<malformed_opt_out>".to_string());
            }
            _ => {}
        }
    }
}

fn collect_opt_out_reasons_from_meta(meta: &Meta, depth: usize, out: &mut Vec<String>) {
    if depth == 0 {
        return;
    }
    let Meta::List(list) = meta else {
        return;
    };
    if list.path.is_ident("cfg_attr") {
        for nested in parse_nested_metas(list).into_iter().skip(1) {
            collect_opt_out_reasons_from_meta(&nested, depth - 1, out);
        }
    } else if is_verified_attr_path(&list.path) {
        collect_opt_outs_from_verified_list(list, out);
    }
}

/// Extracts all `verified(opt_out = "...")` reason strings from `attr`, including
/// conditional `cfg_attr(..., verified(opt_out = "..."))` forms, while ignoring `doc` comments.
pub fn extract_opt_out_reasons_from_attr(attr: &Attribute) -> Vec<String> {
    if attr.path().is_ident("doc") {
        return Vec::new();
    }
    let mut reasons = Vec::new();
    collect_opt_out_reasons_from_meta(&attr.meta, MAX_ATTR_NESTING_DEPTH, &mut reasons);
    reasons
}

pub(super) fn attr_has_opt_out(attr: &Attribute) -> bool {
    !extract_opt_out_reasons_from_attr(attr).is_empty()
}

fn extract_embed_theory_module(list: &MetaList) -> Option<String> {
    let raw = list
        .parse_args_with(|input: syn::parse::ParseStream<'_>| {
            let lit: LitStr = input.parse()?;
            let _ = input.parse::<proc_macro2::TokenStream>();
            Ok(lit.value())
        })
        .ok()?;
    let clean = raw
        .replace('\\', "/")
        .trim_start_matches("papers/")
        .trim_start_matches("spec:")
        .trim_start_matches("registry:")
        .trim_end_matches(".tex")
        .to_string();
    (!clean.is_empty()).then_some(clean)
}

pub(super) fn inspect_non_doc_verified_attr(attr: &Attribute) -> (bool, Option<String>) {
    if !matches!(attr.style, syn::AttrStyle::Outer) || attr_has_opt_out(attr) {
        return (false, None);
    }
    match &attr.meta {
        Meta::Path(path) if is_verified_attr_path(path) => (true, None),
        Meta::List(list) if is_verified_attr_path(&list.path) => (true, None),
        Meta::List(list) if is_embed_theory_attr_path(&list.path) => {
            (true, extract_embed_theory_module(list))
        }
        _ => (false, None),
    }
}

pub(super) fn parse_verification_macro(
    mac: &Macro,
    test_key: &str,
) -> Option<(String, Option<Block>)> {
    mac.parse_body_with(|input: syn::parse::ParseStream<'_>| {
        let mut module_name: Option<String> = None;
        let mut test_block: Option<Block> = None;
        while !input.is_empty() {
            if input.peek(syn::Ident) {
                let ident: syn::Ident = input.parse()?;
                if ident == "module" && input.peek(Token![=]) {
                    let _: Token![=] = input.parse()?;
                    if input.peek(LitStr) {
                        let lit: LitStr = input.parse()?;
                        if module_name.is_none() {
                            module_name = Some(lit.value());
                        }
                        continue;
                    }
                } else if ident == test_key && input.peek(Token![=]) {
                    let _: Token![=] = input.parse()?;
                    if input.peek(syn::token::Brace) {
                        test_block = Some(input.parse::<Block>()?);
                        continue;
                    }
                }
                continue;
            }
            input.step(|cursor| {
                cursor
                    .token_tree()
                    .map(|(_, next)| ((), next))
                    .ok_or_else(|| cursor.error("unexpected end of macro tokens"))
            })?;
        }
        Ok((module_name, test_block))
    })
    .ok()
    .and_then(|(module_name, test_block)| module_name.map(|m| (m, test_block)))
}

pub(super) fn item_attrs(node: &Item) -> &[Attribute] {
    match node {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

pub(super) fn impl_item_attrs(node: &ImplItem) -> &[Attribute] {
    match node {
        ImplItem::Const(i) => &i.attrs,
        ImplItem::Fn(i) => &i.attrs,
        ImplItem::Type(i) => &i.attrs,
        ImplItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

pub(super) fn trait_item_attrs(node: &TraitItem) -> &[Attribute] {
    match node {
        TraitItem::Const(i) => &i.attrs,
        TraitItem::Fn(i) => &i.attrs,
        TraitItem::Type(i) => &i.attrs,
        TraitItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

struct FileDebtAstVisitor {
    in_test_scope: bool,
    entries: Vec<String>,
}

impl FileDebtAstVisitor {
    fn check_legacy_fn(&mut self, attrs: &[Attribute], ident: &syn::Ident) {
        if self.in_test_scope || attrs_require_cfg_test(attrs) || attrs_mark_test_fn(attrs) {
            return;
        }
        let name = ident.to_string();
        if name.ends_with("_legacy") {
            self.entries.push(format!("Legacy function call: {name}"));
        }
    }
}

impl<'ast> Visit<'ast> for FileDebtAstVisitor {
    fn visit_attribute(&mut self, i: &'ast Attribute) {
        self.entries.extend(extract_opt_out_reasons_from_attr(i));
        visit::visit_attribute(self, i);
    }

    fn visit_item(&mut self, i: &'ast Item) {
        let prev = self.in_test_scope;
        if attrs_require_cfg_test(item_attrs(i)) {
            self.in_test_scope = true;
        }
        visit::visit_item(self, i);
        self.in_test_scope = prev;
    }

    fn visit_impl_item(&mut self, i: &'ast ImplItem) {
        let prev = self.in_test_scope;
        if attrs_require_cfg_test(impl_item_attrs(i)) {
            self.in_test_scope = true;
        }
        visit::visit_impl_item(self, i);
        self.in_test_scope = prev;
    }

    fn visit_trait_item(&mut self, i: &'ast TraitItem) {
        let prev = self.in_test_scope;
        if attrs_require_cfg_test(trait_item_attrs(i)) {
            self.in_test_scope = true;
        }
        visit::visit_trait_item(self, i);
        self.in_test_scope = prev;
    }

    fn visit_item_fn(&mut self, i: &'ast ItemFn) {
        self.check_legacy_fn(&i.attrs, &i.sig.ident);
        visit::visit_item_fn(self, i);
    }

    fn visit_impl_item_fn(&mut self, i: &'ast ImplItemFn) {
        self.check_legacy_fn(&i.attrs, &i.sig.ident);
        visit::visit_impl_item_fn(self, i);
    }

    fn visit_trait_item_fn(&mut self, i: &'ast TraitItemFn) {
        self.check_legacy_fn(&i.attrs, &i.sig.ident);
        visit::visit_trait_item_fn(self, i);
    }
}

/// Collects all AST `verified(opt_out = "...")` reasons (direct and `cfg_attr`) and non-test
/// `fn *_legacy` markers from a parsed `syn::File`, ignoring comments and string literals.
pub fn collect_ast_file_debt(ast: &syn::File) -> Vec<String> {
    let mut visitor = FileDebtAstVisitor {
        in_test_scope: attrs_require_cfg_test(&ast.attrs),
        entries: Vec::new(),
    };
    visitor.visit_file(ast);
    visitor.entries
}
