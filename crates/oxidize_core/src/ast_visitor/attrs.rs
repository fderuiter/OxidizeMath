use proc_macro2::{Delimiter, TokenStream, TokenTree};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    AttrStyle, Attribute, Block, Expr, ImplItemFn, ItemFn, Lit, Macro, Meta, Path, Token,
    TraitItemFn,
};

const MAX_ATTR_NESTING_DEPTH: usize = 16;

pub(super) struct AttrScanResult {
    pub opt_outs: Vec<String>,
    pub is_verified: bool,
    pub has_semantic: bool,
    pub modules_with_lines: Vec<(String, usize)>,
}

fn style_matches(style: &AttrStyle, allow_inner: bool, allow_outer: bool) -> bool {
    match style {
        AttrStyle::Inner(_) => allow_inner,
        AttrStyle::Outer => allow_outer,
    }
}

fn is_verified_path(path: &Path) -> bool {
    if path.segments.iter().any(|seg| !seg.arguments.is_empty()) {
        return false;
    }
    path.segments
        .last()
        .is_some_and(|seg| seg.ident == "verified")
}

fn is_embed_theory_path(path: &Path) -> bool {
    if path.segments.iter().any(|seg| !seg.arguments.is_empty()) {
        return false;
    }
    path.segments
        .last()
        .is_some_and(|seg| seg.ident == "embed_theory")
}

fn extract_opt_out_reason(meta: &Meta) -> Option<String> {
    match meta {
        Meta::NameValue(nv) if nv.path.is_ident("opt_out") => {
            if let Expr::Lit(expr_lit) = &nv.value
                && let Lit::Str(lit_str) = &expr_lit.lit
            {
                Some(lit_str.value())
            } else {
                Some("<non_literal_opt_out>".to_string())
            }
        }
        Meta::Path(p) if p.is_ident("opt_out") => Some("<missing_opt_out_reason>".to_string()),
        Meta::List(sub) if sub.path.is_ident("opt_out") => Some("<malformed_opt_out>".to_string()),
        _ => None,
    }
}

fn extract_embed_theory_module(list: &syn::MetaList) -> Option<String> {
    let raw = list
        .parse_args_with(|input: syn::parse::ParseStream<'_>| {
            let lit: syn::LitStr = input.parse()?;
            let _ = input.parse::<TokenStream>();
            Ok(lit.value())
        })
        .ok()?;
    let normalized = raw.replace('\\', "/");
    let clean = normalized
        .trim_start_matches("papers/")
        .trim_start_matches("spec:")
        .trim_start_matches("registry:")
        .trim_end_matches(".tex");
    (!clean.is_empty()).then(|| clean.to_string())
}

fn parse_meta_list(list: &syn::MetaList) -> Vec<Meta> {
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
        Meta::List(list) if list.path.is_ident("all") => parse_meta_list(list)
            .iter()
            .any(|nested| meta_requires_cfg_test(nested, depth - 1)),
        Meta::List(list) if list.path.is_ident("any") => {
            let nested = parse_meta_list(list);
            !nested.is_empty()
                && nested
                    .iter()
                    .all(|item| meta_requires_cfg_test(item, depth - 1))
        }
        _ => false,
    }
}

pub(super) fn attrs_require_cfg_test_filtered(
    attrs: &[Attribute],
    allow_inner: bool,
    allow_outer: bool,
) -> bool {
    attrs.iter().any(|attr| {
        if !style_matches(&attr.style, allow_inner, allow_outer) || !attr.path().is_ident("cfg") {
            return false;
        }
        attr.meta
            .require_list()
            .ok()
            .and_then(|list| list.parse_args::<Meta>().ok())
            .is_some_and(|meta| meta_requires_cfg_test(&meta, MAX_ATTR_NESTING_DEPTH))
    })
}

/// Returns `true` when `attrs` contain a `cfg(...)` predicate that requires `test` to be active.
///
/// Items gated with `#[cfg(any(target_arch = "wasm32", test))]` return `false` because they
/// compile into production `wasm32` targets.
pub fn attrs_require_cfg_test(attrs: &[Attribute]) -> bool {
    attrs_require_cfg_test_filtered(attrs, true, true)
}

/// Extracts all `verified(opt_out = "...")` reason strings from `attr`, including
/// conditional `cfg_attr(..., verified(opt_out = "..."))` forms, while ignoring `doc` comments.
pub fn extract_opt_out_reasons_from_attr(attr: &Attribute) -> Vec<String> {
    if attr.path().is_ident("doc") {
        return Vec::new();
    }
    let mut result = AttrScanResult {
        opt_outs: Vec::new(),
        is_verified: false,
        has_semantic: false,
        modules_with_lines: Vec::new(),
    };
    scan_meta_tree(attr.meta.clone(), attr.span().start().line, &mut result);
    result.opt_outs
}

pub(super) fn extract_opt_outs_filtered(
    attrs: &[Attribute],
    allow_inner: bool,
    allow_outer: bool,
) -> Vec<String> {
    let mut result = AttrScanResult {
        opt_outs: Vec::new(),
        is_verified: false,
        has_semantic: false,
        modules_with_lines: Vec::new(),
    };
    for attr in attrs {
        if !style_matches(&attr.style, allow_inner, allow_outer) || attr.path().is_ident("doc") {
            continue;
        }
        scan_meta_tree(attr.meta.clone(), attr.span().start().line, &mut result);
    }
    result.opt_outs
}

fn scan_meta_tree(root: Meta, attr_line: usize, result: &mut AttrScanResult) {
    let mut stack = vec![(root, true)];
    let mut safety = 0usize;
    while let Some((meta, allow_verified)) = stack.pop() {
        safety += 1;
        if safety > 256 {
            break;
        }
        match meta {
            Meta::List(list) if list.path.is_ident("cfg_attr") => {
                let mut items = parse_meta_list(&list).into_iter();
                let Some(pred) = items.next() else {
                    continue;
                };
                let dead_any = matches!(
                    &pred,
                    Meta::List(sub) if sub.path.is_ident("any") && parse_meta_list(sub).is_empty()
                );
                let child_allow = allow_verified
                    && !dead_any
                    && !meta_requires_cfg_test(&pred, MAX_ATTR_NESTING_DEPTH);
                stack.extend(items.map(|child| (child, child_allow)));
            }
            Meta::Path(path) if is_verified_path(&path) && allow_verified => {
                result.is_verified = true;
                result.has_semantic = true;
            }
            Meta::List(list) if is_verified_path(&list.path) => {
                let nested = parse_meta_list(&list);
                let mut found_opt_out = false;
                for item in &nested {
                    if let Some(reason) = extract_opt_out_reason(item) {
                        result.opt_outs.push(reason);
                        found_opt_out = true;
                    }
                }
                if !found_opt_out && allow_verified {
                    result.is_verified = true;
                    result.has_semantic = true;
                }
            }
            Meta::List(list) if is_embed_theory_path(&list.path) && allow_verified => {
                result.is_verified = true;
                result.has_semantic = true;
                if let Some(module_name) = extract_embed_theory_module(&list) {
                    result.modules_with_lines.push((module_name, attr_line));
                }
            }
            _ => {}
        }
    }
}

pub(super) fn scan_attributes(attrs: &[Attribute]) -> AttrScanResult {
    let mut result = AttrScanResult {
        opt_outs: Vec::new(),
        is_verified: false,
        has_semantic: false,
        modules_with_lines: Vec::new(),
    };

    for attr in attrs {
        let attr_line = attr.span().start().line;
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(nv) = &attr.meta
                && let Expr::Lit(expr_lit) = &nv.value
                && let Lit::Str(lit_str) = &expr_lit.lit
            {
                let extracted = crate::traceability::TraceabilityEngine::<
                    crate::vfs::DefaultVfs,
                >::extract_citations(&lit_str.value());
                if !extracted.is_empty() {
                    result.has_semantic = true;
                    for cite in extracted {
                        result.modules_with_lines.push((cite, attr_line));
                    }
                }
            }
            continue;
        }
        if matches!(attr.style, AttrStyle::Outer) {
            scan_meta_tree(attr.meta.clone(), attr_line, &mut result);
        }
    }

    result
}

/// Returns `true` when `attrs` mark a function as an executable test or benchmark item.
pub fn attrs_mark_test_fn(attrs: &[Attribute]) -> bool {
    if attrs_require_cfg_test_filtered(attrs, true, true) {
        return true;
    }
    for attr in attrs {
        if !matches!(attr.style, AttrStyle::Outer) {
            continue;
        }
        let mut stack = vec![attr.meta.clone()];
        let mut safety = 0usize;
        while let Some(meta) = stack.pop() {
            safety += 1;
            if safety > 256 {
                break;
            }
            if let Meta::List(list) = &meta
                && list.path.is_ident("cfg_attr")
            {
                for nested in parse_meta_list(list).into_iter().skip(1) {
                    stack.push(nested);
                }
                continue;
            }
            let path = meta.path();
            if let Some(last) = path.segments.last() {
                let name = last.ident.to_string();
                if name == "test"
                    || name == "wasm_bindgen_test"
                    || name == "bench"
                    || name.ends_with("_test")
                {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn is_assertion_macro_name(name: &str) -> bool {
    matches!(
        name,
        "assert"
            | "assert_eq"
            | "assert_ne"
            | "debug_assert"
            | "debug_assert_eq"
            | "debug_assert_ne"
            | "assert_relative_eq"
            | "assert_relative_ne"
            | "assert_abs_diff_eq"
            | "assert_abs_diff_ne"
            | "assert_ulps_eq"
            | "assert_ulps_ne"
    )
}

pub(super) struct AssertVisitor {
    pub count: usize,
}

impl<'ast> Visit<'ast> for AssertVisitor {
    fn visit_macro(&mut self, node: &'ast Macro) {
        if let Some(ident) = node.path.segments.last().map(|s| &s.ident)
            && is_assertion_macro_name(&ident.to_string())
        {
            self.count += 1;
        }
        visit::visit_macro(self, node);
    }

    fn visit_item_fn(&mut self, _node: &'ast ItemFn) {}

    fn visit_impl_item_fn(&mut self, _node: &'ast ImplItemFn) {}

    fn visit_trait_item_fn(&mut self, _node: &'ast TraitItemFn) {}
}

pub(super) fn count_block_assertions(block: &Block) -> usize {
    let mut visitor = AssertVisitor { count: 0 };
    visitor.visit_block(block);
    visitor.count
}

fn count_stream_assertions_fallback(stream: TokenStream) -> usize {
    let mut count = 0usize;
    let mut stack = vec![stream];
    while let Some(ts) = stack.pop() {
        let tokens: Vec<TokenTree> = ts.into_iter().collect();
        for idx in 0..tokens.len() {
            if let TokenTree::Group(g) = &tokens[idx] {
                stack.push(g.stream());
            } else if let TokenTree::Ident(ident) = &tokens[idx]
                && is_assertion_macro_name(&ident.to_string())
                && matches!(tokens.get(idx + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!')
                && matches!(tokens.get(idx + 2), Some(TokenTree::Group(_)))
            {
                count += 1;
            }
        }
    }
    count
}

fn count_top_level_semicolons(stream: TokenStream) -> usize {
    stream
        .into_iter()
        .filter(|tt| matches!(tt, TokenTree::Punct(p) if p.as_char() == ';'))
        .count()
}

pub(super) struct ParsedVerificationMacro {
    pub module_name: Option<String>,
    pub assert_count: usize,
}

pub(super) fn parse_verification_macro_tokens(tokens: &TokenStream) -> ParsedVerificationMacro {
    let items: Vec<TokenTree> = tokens.clone().into_iter().collect();
    let mut module_name = None;
    let mut assert_count = 0usize;
    let mut idx = 0usize;

    while idx + 2 < items.len() {
        if let TokenTree::Ident(key_ident) = &items[idx]
            && matches!(&items[idx + 1], TokenTree::Punct(p) if p.as_char() == '=')
        {
            let key = key_ident.to_string();
            if key == "module" {
                let lit_tt = items[idx + 2].clone();
                if let Ok(lit_str) = syn::parse2::<syn::LitStr>(TokenStream::from(lit_tt)) {
                    module_name = Some(lit_str.value());
                }
                idx += 3;
                continue;
            }
            if let TokenTree::Group(group) = &items[idx + 2]
                && group.delimiter() == Delimiter::Brace
            {
                if key == "test" || key == "stochastic_test" {
                    let group_ts = TokenStream::from(TokenTree::Group(group.clone()));
                    if let Ok(block) = syn::parse2::<Block>(group_ts) {
                        assert_count += count_block_assertions(&block);
                    } else {
                        assert_count += count_stream_assertions_fallback(group.stream());
                    }
                } else if key == "signatures" || key == "statistical_bounds" {
                    assert_count += count_top_level_semicolons(group.stream());
                }
                idx += 3;
                continue;
            }
        }
        idx += 1;
    }

    ParsedVerificationMacro {
        module_name,
        assert_count,
    }
}
