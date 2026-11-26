//! AST evidence extraction for `Experimental`, `Validated`, and `Certified` maturity tiers.

use super::vacuous::span_line;
use scientific_metadata::theory::MaturityTier;
use syn::Macro;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

/// Parsed structure of a `theory_verification!` macro invocation.
pub(super) struct TheoryVerificationInfo {
    pub has_non_empty_constants: bool,
    pub has_non_empty_test: bool,
}

fn parse_theory_verification_body(
    input: syn::parse::ParseStream<'_>,
) -> syn::Result<TheoryVerificationInfo> {
    let mut has_non_empty_constants = false;
    let mut has_non_empty_test = false;
    while !input.is_empty() {
        if input.peek(syn::Ident) {
            let ident: syn::Ident = input.parse()?;
            if ident == "constants" && input.peek(syn::Token![=]) {
                let _: syn::Token![=] = input.parse()?;
                if input.peek(syn::token::Brace) {
                    let content;
                    syn::braced!(content in input);
                    let tokens: proc_macro2::TokenStream = content.parse()?;
                    has_non_empty_constants = !tokens.is_empty();
                    continue;
                }
            } else if ident == "test" && input.peek(syn::Token![=]) {
                let _: syn::Token![=] = input.parse()?;
                if input.peek(syn::token::Brace) {
                    let content;
                    syn::braced!(content in input);
                    let tokens: proc_macro2::TokenStream = content.parse()?;
                    has_non_empty_test = !tokens.is_empty();
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
    Ok(TheoryVerificationInfo {
        has_non_empty_constants,
        has_non_empty_test,
    })
}

pub(super) fn inspect_theory_verification_macro(mac: &Macro) -> TheoryVerificationInfo {
    mac.parse_body_with(parse_theory_verification_body)
        .unwrap_or(TheoryVerificationInfo {
            has_non_empty_constants: false,
            has_non_empty_test: false,
        })
}

/// Extracts `#[theory(..., maturity = "...")]` declarations on structs or enums.
pub(super) fn extract_item_theory_maturity(
    item_name: &str,
    attrs: &[syn::Attribute],
) -> Option<(String, MaturityTier, usize)> {
    for attr in attrs {
        if !attr.path().is_ident("theory") {
            continue;
        }
        let mut found: Option<MaturityTier> = None;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("maturity") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                found = MaturityTier::parse_label(&lit.value());
            } else if meta.input.peek(syn::Token![=]) {
                let _: syn::Expr = meta.value()?.parse()?;
            }
            Ok(())
        });
        if let Some(tier) = found {
            return Some((item_name.to_string(), tier, span_line(attr.span())));
        }
    }
    None
}

fn extract_self_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
        syn::Type::Paren(p) => extract_self_type_name(&p.elem),
        syn::Type::Group(g) => extract_self_type_name(&g.elem),
        syn::Type::Reference(r) => extract_self_type_name(&r.elem),
        _ => None,
    }
}

#[derive(Default)]
struct MaturityVariantFinder {
    found: Option<MaturityTier>,
}

impl<'ast> Visit<'ast> for MaturityVariantFinder {
    fn visit_path(&mut self, node: &'ast syn::Path) {
        let len = node.segments.len();
        if len >= 2 && node.segments[len - 2].ident == "MaturityTier" {
            let variant = node.segments[len - 1].ident.to_string();
            if let Some(tier) = MaturityTier::parse_label(&variant) {
                self.found = Some(tier);
            }
        }
        visit::visit_path(self, node);
    }
}

/// Extracts explicit `MaturityTier::<Variant>` returns inside `fn maturity_tier(&self)`.
pub(super) fn extract_impl_maturity_method(
    self_ty: &syn::Type,
    method: &syn::ImplItemFn,
) -> Option<(String, MaturityTier, usize)> {
    if method.sig.ident != "maturity_tier" {
        return None;
    }
    let ty_name = extract_self_type_name(self_ty)?;
    let mut finder = MaturityVariantFinder::default();
    finder.visit_block(&method.block);
    let tier = finder.found?;
    Some((ty_name, tier, span_line(method.sig.ident.span())))
}

/// Appends any explicit `MaturityTier` declarations attached to `node` into `out`.
pub(super) fn collect_item_maturity_decls(
    node: &syn::Item,
    out: &mut Vec<(String, MaturityTier, usize)>,
) {
    match node {
        syn::Item::Struct(s) => {
            if let Some(decl) = extract_item_theory_maturity(&s.ident.to_string(), &s.attrs) {
                out.push(decl);
            }
        }
        syn::Item::Enum(e) => {
            if let Some(decl) = extract_item_theory_maturity(&e.ident.to_string(), &e.attrs) {
                out.push(decl);
            }
        }
        syn::Item::Impl(imp) if imp.trait_.is_some() => {
            for item in &imp.items {
                if let syn::ImplItem::Fn(m) = item
                    && let Some(decl) = extract_impl_maturity_method(&imp.self_ty, m)
                {
                    out.push(decl);
                }
            }
        }
        _ => {}
    }
}

/// Visitor scanning a file or function AST for zero-allocation, RNG seeding, golden fixtures, and assertions.
#[derive(Default)]
pub(super) struct FileEvidenceScanner {
    pub has_zero_alloc_proof: bool,
    pub has_seeded_rng: bool,
    pub has_unseeded_rng: bool,
    pub has_golden_fixture: bool,
    pub maturity_asserts: usize,
}

impl FileEvidenceScanner {
    fn inspect_ident(&mut self, ident: &str) {
        match ident {
            "VerifiedAllocator" | "run_verified" | "lock_allocations" => {
                self.has_zero_alloc_proof = true;
            }
            "from_seed" | "seed_from_u64" | "OxidizeRng" | "ChaCha8Rng" => {
                self.has_seeded_rng = true;
            }
            "thread_rng" => {
                self.has_unseeded_rng = true;
            }
            _ => {}
        }
    }
}

impl<'ast> Visit<'ast> for FileEvidenceScanner {
    fn visit_path(&mut self, node: &'ast syn::Path) {
        for seg in &node.segments {
            self.inspect_ident(&seg.ident.to_string());
        }
        visit::visit_path(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        self.inspect_ident(&node.method.to_string());
        visit::visit_expr_method_call(self, node);
    }

    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        self.has_golden_fixture = true;
        visit::visit_item_const(self, node);
    }

    fn visit_impl_item_const(&mut self, node: &'ast syn::ImplItemConst) {
        self.has_golden_fixture = true;
        visit::visit_impl_item_const(self, node);
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        if let Some(ident) = node.path.segments.last().map(|s| &s.ident) {
            let name = ident.to_string();
            match name.as_str() {
                "assert_eq" | "assert_relative_eq" | "assert_abs_diff_eq" => {
                    self.has_golden_fixture = true;
                    self.maturity_asserts += 1;
                }
                "assert" | "assert_ne" | "debug_assert" | "debug_assert_eq" | "debug_assert_ne" => {
                    if node.tokens.to_string().chars().any(|c| c.is_ascii_digit()) {
                        self.has_golden_fixture = true;
                    }
                    self.maturity_asserts += 1;
                }
                "include_str" | "include_bytes" => {
                    self.has_golden_fixture = true;
                }
                _ => {}
            }
        }
        visit::visit_macro(self, node);
    }
}
