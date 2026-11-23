use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{ImplItemFn, ItemFn, ItemMacro, Macro, TraitItemFn};

#[allow(missing_docs)]
pub struct AstVisitor {
    #[allow(missing_docs)]
    pub verified_modules: Vec<String>,
    #[allow(missing_docs)]
    pub module_tiers: HashMap<String, String>,
    #[allow(missing_docs)]
    pub module_spans: HashMap<String, usize>,
    #[allow(missing_docs)]
    pub has_vacuous_bypass: bool,
    #[allow(missing_docs)]
    pub vacuous_bypass_line: Option<usize>,
    #[allow(missing_docs)]
    pub first_unverified_fn: Option<(String, usize)>,
    #[allow(missing_docs)]
    pub total_funcs: usize,
    #[allow(missing_docs)]
    pub total_asserts: usize,
    #[allow(missing_docs)]
    pub verified_funcs: usize,
    #[allow(missing_docs)]
    pub verified_asserts: usize,
    #[allow(missing_docs)]
    pub semantic_integrity_funcs: usize,
    #[allow(missing_docs)]
    pub active_submodules: Vec<String>,
    #[allow(missing_docs)]
    pub opted_out: bool,
    file_opted_out: bool,
}

impl Default for AstVisitor {
    fn default() -> Self {
        Self::new()
    }
}

fn attr_has_opt_out(attr: &syn::Attribute) -> bool {
    if attr.path().is_ident("doc") {
        return false;
    }
    quote::quote!(#attr)
        .to_string()
        .replace(' ', "")
        .contains("opt_out")
}

fn item_attrs(node: &syn::Item) -> &[syn::Attribute] {
    match node {
        syn::Item::Struct(i) => &i.attrs,
        syn::Item::Enum(i) => &i.attrs,
        syn::Item::Union(i) => &i.attrs,
        syn::Item::Impl(i) => &i.attrs,
        syn::Item::Fn(i) => &i.attrs,
        syn::Item::Mod(i) => &i.attrs,
        syn::Item::Trait(i) => &i.attrs,
        syn::Item::Const(i) => &i.attrs,
        syn::Item::Static(i) => &i.attrs,
        syn::Item::Type(i) => &i.attrs,
        syn::Item::Macro(i) => &i.attrs,
        _ => &[],
    }
}

fn span_line(span: proc_macro2::Span) -> usize {
    span.start().line
}

fn is_zero_literal_expr(mut expr: &syn::Expr) -> bool {
    for _ in 0..16 {
        match expr {
            syn::Expr::Paren(p) => expr = &p.expr,
            syn::Expr::Group(g) => expr = &g.expr,
            syn::Expr::Cast(c) => expr = &c.expr,
            syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => expr = &u.expr,
            syn::Expr::Lit(syn::ExprLit { lit, .. }) => {
                return match lit {
                    syn::Lit::Int(li) => matches!(li.base10_parse::<u128>(), Ok(0)),
                    syn::Lit::Float(lf) => matches!(lf.base10_parse::<f64>(), Ok(v) if v == 0.0),
                    _ => false,
                };
            }
            _ => return false,
        }
    }
    false
}

fn is_zero_initializer_expr(expr: &syn::Expr) -> bool {
    if is_zero_literal_expr(expr) {
        return true;
    }
    match expr {
        syn::Expr::Repeat(r) => is_zero_literal_expr(&r.expr),
        syn::Expr::Array(a) => !a.elems.is_empty() && a.elems.iter().all(is_zero_literal_expr),
        syn::Expr::Tuple(t) => !t.elems.is_empty() && t.elems.iter().all(is_zero_literal_expr),
        _ => false,
    }
}

fn is_zero_closure_expr(expr: &syn::Expr) -> bool {
    matches!(expr, syn::Expr::Closure(c) if is_zero_initializer_expr(&c.body))
}

fn vec_macro_is_zero_init(mac: &Macro) -> bool {
    mac.parse_body_with(|input: syn::parse::ParseStream<'_>| {
        if input.is_empty() {
            return Ok(false);
        }
        let first: syn::Expr = input.parse()?;
        if input.peek(syn::Token![;]) {
            let _: syn::Token![;] = input.parse()?;
            let _: syn::Expr = input.parse()?;
            return Ok(is_zero_literal_expr(&first));
        }
        let mut all_zero = is_zero_literal_expr(&first);
        while input.peek(syn::Token![,]) {
            let _: syn::Token![,] = input.parse()?;
            if input.is_empty() {
                break;
            }
            let next: syn::Expr = input.parse()?;
            all_zero &= is_zero_literal_expr(&next);
        }
        Ok(all_zero)
    })
    .unwrap_or(false)
}

fn extract_stochastic_test_block(
    input: syn::parse::ParseStream<'_>,
) -> syn::Result<Option<syn::Block>> {
    let mut found = None;
    while !input.is_empty() {
        if input.peek(syn::Ident) {
            let ident: syn::Ident = input.parse()?;
            if ident == "stochastic_test" && input.peek(syn::Token![=]) {
                let _: syn::Token![=] = input.parse()?;
                if input.peek(syn::token::Brace) {
                    found = Some(input.parse::<syn::Block>()?);
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
    Ok(found)
}

struct StochasticZeroInitVisitor {
    bypass_line: Option<usize>,
}

impl StochasticZeroInitVisitor {
    fn flag(&mut self, line: usize) {
        if self.bypass_line.is_none() {
            self.bypass_line = Some(line);
        }
    }
}

impl<'ast> Visit<'ast> for StochasticZeroInitVisitor {
    fn visit_local(&mut self, i: &'ast syn::Local) {
        if let Some(init) = &i.init
            && is_zero_initializer_expr(&init.expr)
        {
            self.flag(span_line(init.expr.span()));
        }
        visit::visit_local(self, i);
    }

    fn visit_expr_repeat(&mut self, i: &'ast syn::ExprRepeat) {
        if is_zero_literal_expr(&i.expr) {
            self.flag(span_line(i.span()));
        }
        visit::visit_expr_repeat(self, i);
    }

    fn visit_expr_array(&mut self, i: &'ast syn::ExprArray) {
        if !i.elems.is_empty() && i.elems.iter().all(is_zero_literal_expr) {
            self.flag(span_line(i.span()));
        }
        visit::visit_expr_array(self, i);
    }

    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*i.func
            && let Some(seg) = p.path.segments.last()
        {
            let name = seg.ident.to_string();
            let is_zero_call = name == "zeros"
                || name == "zeros_like"
                || (name == "from_element" && i.args.last().is_some_and(is_zero_literal_expr))
                || (name == "from_fn" && i.args.last().is_some_and(is_zero_closure_expr));
            if is_zero_call {
                self.flag(span_line(i.span()));
            }
        }
        visit::visit_expr_call(self, i);
    }

    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        let name = i.method.to_string();
        if name == "zeros"
            || name == "zeros_like"
            || (name == "fill" && i.args.first().is_some_and(is_zero_literal_expr))
        {
            self.flag(span_line(i.span()));
        }
        visit::visit_expr_method_call(self, i);
    }

    fn visit_macro(&mut self, i: &'ast Macro) {
        if i.path.is_ident("vec") && vec_macro_is_zero_init(i) {
            self.flag(span_line(i.span()));
        }
        visit::visit_macro(self, i);
    }
}

fn find_stochastic_zero_init_line(mac: &Macro) -> Option<usize> {
    let block = mac
        .parse_body_with(extract_stochastic_test_block)
        .ok()
        .flatten()?;
    let mut visitor = StochasticZeroInitVisitor { bypass_line: None };
    visitor.visit_block(&block);
    visitor.bypass_line
}

impl AstVisitor {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            verified_modules: Vec::new(),
            module_tiers: HashMap::new(),
            module_spans: HashMap::new(),
            has_vacuous_bypass: false,
            vacuous_bypass_line: None,
            first_unverified_fn: None,
            total_funcs: 0,
            total_asserts: 0,
            verified_funcs: 0,
            verified_asserts: 0,
            semantic_integrity_funcs: 0,
            active_submodules: Vec::new(),
            opted_out: false,
            file_opted_out: false,
        }
    }

    fn record_module(&mut self, name: String, line: usize) {
        if line > 0 {
            self.module_spans.entry(name.clone()).or_insert(line);
        }
        self.verified_modules.push(name);
    }

    fn extract_embed_theory(&mut self, attr_str: &str, line: usize) {
        let s = attr_str.replace('\\', "/");
        if let Some((pos, end)) = s
            .find('"')
            .and_then(|p| s[p + 1..].find('"').map(|e| (p, e)))
        {
            let raw_path = &s[pos + 1..pos + 1 + end];
            let clean_name = raw_path
                .trim_start_matches("papers/")
                .trim_start_matches("spec:")
                .trim_start_matches("registry:")
                .trim_end_matches(".tex");
            if !clean_name.is_empty() {
                self.record_module(clean_name.to_string(), line);
            }
        }
    }

    fn process_fn_attrs(
        &mut self,
        attrs: &[syn::Attribute],
        fn_ident: &syn::Ident,
        assert_count: usize,
    ) {
        self.total_funcs += 1;

        let mut is_verified = false;
        let mut has_semantic = false;
        for attr in attrs {
            if attr_has_opt_out(attr) {
                continue;
            }
            let attr_line = span_line(attr.span());
            let doc_text = quote::quote!(#attr).to_string();
            let extracted =
                crate::traceability::TraceabilityEngine::<crate::vfs::DefaultVfs>::extract_citations(
                    &doc_text,
                );
            if !extracted.is_empty() {
                has_semantic = true;
                for cite in extracted {
                    self.record_module(cite, attr_line);
                }
            }
            if attr.path().is_ident("doc") {
                continue;
            }

            let attr_str = doc_text.replace(' ', "");

            if attr_str.contains("verified_engine::verified")
                || attr_str.contains("#[verified]")
                || attr_str.starts_with("#[verified(")
            {
                is_verified = true;
                has_semantic = true;
            }

            if attr_str.contains("embed_theory") {
                is_verified = true;
                has_semantic = true;
                self.extract_embed_theory(&attr_str, attr_line);
            }
        }

        if self.opted_out {
            is_verified = false;
        } else if !is_verified && self.first_unverified_fn.is_none() {
            self.first_unverified_fn = Some((fn_ident.to_string(), span_line(fn_ident.span())));
        }

        if is_verified {
            self.verified_funcs += 1;
            self.verified_asserts += assert_count;
        }
        if has_semantic {
            self.semantic_integrity_funcs += 1;
        }
        self.total_asserts += assert_count;
    }
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_file(&mut self, node: &'ast syn::File) {
        if node
            .attrs
            .iter()
            .any(|a| matches!(a.style, syn::AttrStyle::Inner(_)) && attr_has_opt_out(a))
        {
            self.file_opted_out = true;
            self.opted_out = true;
        }
        visit::visit_file(self, node);
        self.opted_out = self.file_opted_out;
    }

    fn visit_item(&mut self, node: &'ast syn::Item) {
        let prev = self.opted_out;
        let allow_inner = matches!(node, syn::Item::Mod(m) if m.content.is_some());
        if item_attrs(node).iter().any(|a| {
            (matches!(a.style, syn::AttrStyle::Outer) || allow_inner) && attr_has_opt_out(a)
        }) {
            self.opted_out = true;
        }
        visit::visit_item(self, node);
        self.opted_out = prev;
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if node.content.is_none() {
            self.active_submodules.push(node.ident.to_string());
        }
        visit::visit_item_mod(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast ItemMacro) {
        if let Some(ident) = node.mac.path.segments.last().map(|s| &s.ident) {
            let name = ident.to_string();
            let tier = if name == "theory_verification" {
                Some("Deterministic")
            } else if name == "stochastic_signature_verification" {
                Some("Stochastic")
            } else {
                None
            };

            if let Some(tier_name) = tier {
                let macro_line = span_line(node.span());
                if name == "stochastic_signature_verification"
                    && let Some(zero_line) = find_stochastic_zero_init_line(&node.mac)
                {
                    self.has_vacuous_bypass = true;
                    let recorded = if zero_line > 0 { zero_line } else { macro_line };
                    if self.vacuous_bypass_line.is_none() && recorded > 0 {
                        self.vacuous_bypass_line = Some(recorded);
                    }
                }

                let tokens = node.mac.tokens.to_string();
                if let Some(idx) = tokens.find("module = \"") {
                    let start = idx + 10;
                    if let Some(end) = tokens[start..].find('"') {
                        let module_name = &tokens[start..start + end];
                        self.record_module(module_name.to_string(), macro_line);
                        self.module_tiers
                            .insert(module_name.to_string(), tier_name.to_string());
                        self.semantic_integrity_funcs += 1;
                        self.total_funcs += 1;
                        self.verified_funcs += 1;
                        let assert_count = tokens.matches("assert").count().max(3);
                        self.total_asserts += assert_count;
                        self.verified_asserts += assert_count;
                    }
                }
            }
        }
        visit::visit_item_macro(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let prev = self.opted_out;
        if node.attrs.iter().any(attr_has_opt_out) {
            self.opted_out = true;
        }
        let mut assert_visitor = AssertVisitor { count: 0 };
        visit::visit_item_fn(&mut assert_visitor, node);
        self.process_fn_attrs(&node.attrs, &node.sig.ident, assert_visitor.count);
        visit::visit_item_fn(self, node);
        self.opted_out = prev;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        let prev = self.opted_out;
        if node.attrs.iter().any(attr_has_opt_out) {
            self.opted_out = true;
        }
        let mut assert_visitor = AssertVisitor { count: 0 };
        visit::visit_impl_item_fn(&mut assert_visitor, node);
        self.process_fn_attrs(&node.attrs, &node.sig.ident, assert_visitor.count);
        visit::visit_impl_item_fn(self, node);
        self.opted_out = prev;
    }

    fn visit_trait_item_fn(&mut self, node: &'ast TraitItemFn) {
        let prev = self.opted_out;
        if node.attrs.iter().any(attr_has_opt_out) {
            self.opted_out = true;
        }
        if node.default.is_some() {
            let mut assert_visitor = AssertVisitor { count: 0 };
            visit::visit_trait_item_fn(&mut assert_visitor, node);
            self.process_fn_attrs(&node.attrs, &node.sig.ident, assert_visitor.count);
        }
        visit::visit_trait_item_fn(self, node);
        self.opted_out = prev;
    }
}

struct AssertVisitor {
    count: usize,
}

impl<'ast> Visit<'ast> for AssertVisitor {
    fn visit_macro(&mut self, node: &'ast Macro) {
        if let Some(ident) = node.path.segments.last().map(|s| &s.ident) {
            let name = ident.to_string();
            if name == "assert"
                || name == "assert_eq"
                || name == "assert_ne"
                || name == "debug_assert"
                || name == "debug_assert_eq"
                || name == "debug_assert_ne"
            {
                self.count += 1;
            }
        }
        visit::visit_macro(self, node);
    }
}

#[cfg(test)]
mod tests;
