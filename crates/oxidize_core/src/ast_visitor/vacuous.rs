use syn::Macro;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

pub(super) fn span_line(span: proc_macro2::Span) -> usize {
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

pub(super) fn find_stochastic_zero_init_line(mac: &Macro) -> Option<usize> {
    let block = mac
        .parse_body_with(extract_stochastic_test_block)
        .ok()
        .flatten()?;
    let mut visitor = StochasticZeroInitVisitor { bypass_line: None };
    visitor.visit_block(&block);
    visitor.bypass_line
}
