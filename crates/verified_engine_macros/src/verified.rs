use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote};
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct VerifiedArgs {
    pub(crate) opt_out: bool,
    pub(crate) allow_recursion: bool,
}

fn parse_reason_value(meta: &syn::meta::ParseNestedMeta<'_>, key: &str) -> syn::Result<()> {
    if !meta.input.peek(syn::Token![=]) {
        return Err(meta.error(format!(
            "expected `{key} = \"reason\"` with a non-empty string literal"
        )));
    }
    let value = meta.value()?;
    let lit: syn::LitStr = value.parse()?;
    if lit.value().trim().is_empty() {
        return Err(syn::Error::new_spanned(
            &lit,
            format!("`{key}` reason must not be empty"),
        ));
    }
    Ok(())
}

pub(crate) fn parse_verified_args(attr: TokenStream) -> syn::Result<VerifiedArgs> {
    if attr.is_empty() {
        return Ok(VerifiedArgs::default());
    }
    let mut args = VerifiedArgs::default();
    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("opt_out") {
            if args.opt_out {
                return Err(meta.error("duplicate `opt_out` key in `#[verified]`"));
            }
            parse_reason_value(&meta, "opt_out")?;
            args.opt_out = true;
            Ok(())
        } else if meta.path.is_ident("allow_recursion") {
            if args.allow_recursion {
                return Err(meta.error("duplicate `allow_recursion` key in `#[verified]`"));
            }
            parse_reason_value(&meta, "allow_recursion")?;
            args.allow_recursion = true;
            Ok(())
        } else {
            let key = meta.path.to_token_stream().to_string();
            Err(meta.error(format!(
                "unknown `#[verified]` attribute key `{key}`; valid keys are `opt_out = \"reason\"` and `allow_recursion = \"reason\"`"
            )))
        }
    });
    syn::parse::Parser::parse2(parser, attr)?;
    Ok(args)
}

struct ExprCounter {
    arithmetic: usize,
    loads: usize,
}

impl<'ast> Visit<'ast> for ExprCounter {
    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        use syn::BinOp::*;
        if matches!(
            node.op,
            Add(_)
                | Sub(_)
                | Mul(_)
                | Div(_)
                | Rem(_)
                | BitXor(_)
                | BitAnd(_)
                | BitOr(_)
                | Shl(_)
                | Shr(_)
        ) {
            self.arithmetic += 1;
        }
        syn::visit::visit_expr_binary(self, node);
    }

    fn visit_expr_index(&mut self, node: &'ast syn::ExprIndex) {
        self.loads += 1;
        syn::visit::visit_expr_index(self, node);
    }

    fn visit_block(&mut self, _node: &'ast syn::Block) {
        // Nested blocks are instrumented by InjectorVisitor::visit_block_mut.
    }
}

struct InjectorVisitor {
    ve_path: TokenStream,
}

impl VisitMut for InjectorVisitor {
    fn visit_item_mut(&mut self, _item: &mut syn::Item) {
        // Nested items have their own scope and do not capture `__ve_active`.
    }

    fn visit_block_mut(&mut self, block: &mut syn::Block) {
        let ve_path = self.ve_path.clone();
        let mut new_stmts = Vec::new();
        for mut stmt in std::mem::take(&mut block.stmts) {
            let mut counter = ExprCounter {
                arithmetic: 0,
                loads: 0,
            };
            counter.visit_stmt(&stmt);

            let arith = counter.arithmetic as u64;
            let loads = counter.loads as u64;
            if arith > 0 && loads > 0 {
                new_stmts.push(syn::parse_quote! {
                    if __ve_active {
                        #ve_path::metrics::increment_arithmetic(#arith);
                        #ve_path::metrics::increment_memory_loads(#loads);
                    }
                });
            } else if arith > 0 {
                new_stmts.push(syn::parse_quote! {
                    if __ve_active {
                        #ve_path::metrics::increment_arithmetic(#arith);
                    }
                });
            } else if loads > 0 {
                new_stmts.push(syn::parse_quote! {
                    if __ve_active {
                        #ve_path::metrics::increment_memory_loads(#loads);
                    }
                });
            }

            syn::visit_mut::visit_stmt_mut(self, &mut stmt);
            new_stmts.push(stmt);
        }
        block.stmts = new_stmts;
    }
}

struct DeepAstVisitor {
    statements: usize,
    direct_recursion: Option<Span>,
    function_name: String,
}

impl DeepAstVisitor {
    fn is_direct_call_path(&self, expr_path: &syn::ExprPath) -> bool {
        if let Some(ref qself) = expr_path.qself {
            return qself.as_token.is_none()
                && matches!(&*qself.ty, syn::Type::Path(tp) if tp.qself.is_none() && tp.path.is_ident("Self"))
                && expr_path.path.is_ident(&self.function_name);
        }
        let segs = &expr_path.path.segments;
        (segs.len() == 1 && segs[0].ident == self.function_name)
            || (segs.len() == 2 && segs[0].ident == "Self" && segs[1].ident == self.function_name)
    }
}

impl<'ast> Visit<'ast> for DeepAstVisitor {
    fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
        let is_unnamed_const =
            matches!(node, syn::Stmt::Item(syn::Item::Const(c)) if c.ident == "_");
        if !is_unnamed_const {
            self.statements += 1;
        }
        syn::visit::visit_stmt(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if self.direct_recursion.is_none()
            && let syn::Expr::Path(ref expr_path) = *node.func
            && self.is_direct_call_path(expr_path)
        {
            self.direct_recursion = Some(node.span());
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if self.direct_recursion.is_none()
            && node.method == self.function_name
            && let syn::Expr::Path(ref expr_path) = *node.receiver
            && expr_path.qself.is_none()
            && expr_path.path.is_ident("self")
        {
            self.direct_recursion = Some(node.span());
        }
        syn::visit::visit_expr_method_call(self, node);
    }
}

fn verify_and_instrument_block(
    sig: &syn::Signature,
    block: &mut syn::Block,
    args: &VerifiedArgs,
    ve_path: &TokenStream,
) -> syn::Result<()> {
    let mut visitor = DeepAstVisitor {
        statements: 0,
        direct_recursion: None,
        function_name: sig.ident.to_string(),
    };
    visitor.visit_block(block);

    if !args.opt_out
        && !args.allow_recursion
        && let Some(span) = visitor.direct_recursion
    {
        return Err(syn::Error::new(
            span,
            "Direct recursion is prohibited in #[verified] functions (NASA Power of 10 Rule 1); use an iterative implementation, #[verified(allow_recursion = \"reason\")] for trait delegation, or #[verified(opt_out = \"reason\")]",
        ));
    }

    if !args.opt_out && visitor.statements > 60 {
        return Err(syn::Error::new_spanned(
            &sig.ident,
            format!(
                "Function `{}` exceeds the 60-statement limit (NASA Power of 10 Rule 4): found {} statements",
                sig.ident, visitor.statements
            ),
        ));
    }

    if sig.constness.is_none() {
        let mut injector = InjectorVisitor {
            ve_path: ve_path.clone(),
        };
        injector.visit_block_mut(block);
        let inner_block = &*block;
        *block = syn::parse_quote! {
            {
                let __ve_active = #ve_path::metrics::is_verification_enabled();
                if __ve_active {
                    #ve_path::metrics::increment_calls();
                }
                #inner_block
            }
        };
    }
    Ok(())
}

fn unsupported_item_error(item: TokenStream) -> syn::Error {
    if let Ok(parsed) = syn::parse2::<syn::Item>(item.clone()) {
        return match parsed {
            syn::Item::Struct(s) => syn::Error::new_spanned(
                s.struct_token,
                "`#[verified]` can only be applied to functions and methods, not `struct` items",
            ),
            syn::Item::Enum(e) => syn::Error::new_spanned(
                e.enum_token,
                "`#[verified]` can only be applied to functions and methods, not `enum` items",
            ),
            syn::Item::Union(u) => syn::Error::new_spanned(
                u.union_token,
                "`#[verified]` can only be applied to functions and methods, not `union` items",
            ),
            syn::Item::Mod(m) => syn::Error::new_spanned(
                m.mod_token,
                "`#[verified]` can only be applied to functions and methods, not `mod` items",
            ),
            syn::Item::Impl(i) => syn::Error::new_spanned(
                i.impl_token,
                "`#[verified]` can only be applied to functions and methods, not `impl` blocks; annotate individual methods instead",
            ),
            syn::Item::Trait(t) => syn::Error::new_spanned(
                t.trait_token,
                "`#[verified]` can only be applied to functions and methods, not `trait` blocks; annotate individual trait methods instead",
            ),
            syn::Item::Const(c) => syn::Error::new_spanned(
                c.const_token,
                "`#[verified]` can only be applied to functions and methods, not `const` items",
            ),
            syn::Item::Static(s) => syn::Error::new_spanned(
                s.static_token,
                "`#[verified]` can only be applied to functions and methods, not `static` items",
            ),
            syn::Item::Type(t) => syn::Error::new_spanned(
                t.type_token,
                "`#[verified]` can only be applied to functions and methods, not `type` aliases",
            ),
            other => syn::Error::new_spanned(
                other,
                "`#[verified]` can only be applied to functions and methods",
            ),
        };
    }
    syn::Error::new_spanned(
        item,
        "`#[verified]` can only be applied to functions and methods",
    )
}

pub(crate) fn verified_impl_with_path(
    attr: TokenStream,
    item: TokenStream,
    ve_path: &TokenStream,
) -> TokenStream {
    let args = match parse_verified_args(attr) {
        Ok(parsed) => parsed,
        Err(err) => return err.to_compile_error(),
    };

    if let Ok(mut item_fn) = syn::parse2::<syn::ItemFn>(item.clone()) {
        if let Err(err) =
            verify_and_instrument_block(&item_fn.sig, &mut item_fn.block, &args, ve_path)
        {
            return err.to_compile_error();
        }
        return quote! { #item_fn };
    }

    if let Ok(mut impl_fn) = syn::parse2::<syn::ImplItemFn>(item.clone()) {
        if let Err(err) =
            verify_and_instrument_block(&impl_fn.sig, &mut impl_fn.block, &args, ve_path)
        {
            return err.to_compile_error();
        }
        return quote! { #impl_fn };
    }

    if let Ok(mut trait_fn) = syn::parse2::<syn::TraitItemFn>(item.clone()) {
        if let Some(ref mut block) = trait_fn.default
            && let Err(err) = verify_and_instrument_block(&trait_fn.sig, block, &args, ve_path)
        {
            return err.to_compile_error();
        }
        return quote! { #trait_fn };
    }

    unsupported_item_error(item).to_compile_error()
}

pub(crate) fn verified_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let ve_path = crate::crate_paths::resolve_verified_engine_path();
    verified_impl_with_path(attr, item, &ve_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verified_args_strict_parsing() {
        assert_eq!(
            parse_verified_args(quote! {}).unwrap(),
            VerifiedArgs::default()
        );
        assert!(
            parse_verified_args(quote! { opt_out = "legacy" })
                .unwrap()
                .opt_out
        );
        assert!(
            parse_verified_args(quote! { allow_recursion = "trait forwarding" })
                .unwrap()
                .allow_recursion
        );
        assert!(parse_verified_args(quote! { "opt_out" }).is_err());
        assert!(parse_verified_args(quote! { not_opt_out = "x" }).is_err());
        assert!(parse_verified_args(quote! { opt_out = "" }).is_err());
        assert!(parse_verified_args(quote! { allow_recursion = "   " }).is_err());
    }

    #[test]
    fn test_const_fn_skips_metrics_but_enforces_static_rules() {
        let ve = quote! { ::verified_engine };
        let ok_out = verified_impl_with_path(
            quote! {},
            quote! { pub const fn double(x: i32) -> i32 { x + x } },
            &ve,
        )
        .to_string();
        assert!(!ok_out.contains("increment_calls"));
        assert!(!ok_out.contains("compile_error"));

        let rec_out = verified_impl_with_path(
            quote! {},
            quote! { pub const fn bad(n: u32) -> u32 { bad(n) } },
            &ve,
        )
        .to_string();
        assert!(rec_out.contains("compile_error"));
        assert!(!rec_out.contains("severity"));
    }

    #[test]
    fn test_trait_signature_and_container_rejection() {
        let ve = quote! { ::verified_engine };
        let sig_out =
            verified_impl_with_path(quote! {}, quote! { fn apply(&self, x: f64) -> f64; }, &ve)
                .to_string();
        assert!(!sig_out.contains("compile_error"));

        for bad in [
            quote! { struct S { a: i32 } },
            quote! { enum E { A } },
            quote! { mod m {} },
            quote! { impl S {} },
        ] {
            let err_out = verified_impl_with_path(quote! {}, bad, &ve).to_string();
            assert!(err_out.contains("compile_error"));
            assert!(!err_out.contains("severity"));
        }
    }
}
