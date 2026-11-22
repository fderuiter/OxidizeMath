#![doc = include_str!("../README.md")]
extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::ItemFn;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;

struct ExprCounter {
    arithmetic: usize,
    loads: usize,
}

impl<'ast> Visit<'ast> for ExprCounter {
    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        use syn::BinOp::*;
        match node.op {
            Add(_) | Sub(_) | Mul(_) | Div(_) | Rem(_) | BitXor(_) | BitAnd(_) | BitOr(_)
            | Shl(_) | Shr(_) => {
                self.arithmetic += 1;
            }
            _ => {}
        }
        syn::visit::visit_expr_binary(self, node);
    }

    fn visit_expr_index(&mut self, node: &'ast syn::ExprIndex) {
        self.loads += 1;
        syn::visit::visit_expr_index(self, node);
    }

    fn visit_block(&mut self, _node: &'ast syn::Block) {
        // Do not recurse into blocks; they will be handled by visit_block_mut
    }
}

struct InjectorVisitor;

impl VisitMut for InjectorVisitor {
    fn visit_item_mut(&mut self, _item: &mut syn::Item) {
        // Nested items have their own scope and do not capture `__ve_active`.
    }

    fn visit_block_mut(&mut self, block: &mut syn::Block) {
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
                        verified_engine::metrics::increment_arithmetic(#arith);
                        verified_engine::metrics::increment_memory_loads(#loads);
                    }
                });
            } else if arith > 0 {
                new_stmts.push(syn::parse_quote! {
                    if __ve_active {
                        verified_engine::metrics::increment_arithmetic(#arith);
                    }
                });
            } else if loads > 0 {
                new_stmts.push(syn::parse_quote! {
                    if __ve_active {
                        verified_engine::metrics::increment_memory_loads(#loads);
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
    direct_recursion: Option<proc_macro2::Span>,
    function_name: String,
}

impl<'ast> Visit<'ast> for DeepAstVisitor {
    fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
        self.statements += 1;
        syn::visit::visit_stmt(self, node);
    }

    #[allow(clippy::collapsible_if)]
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(ref expr_path) = *node.func {
            if expr_path.path.segments.len() == 1 {
                if let Some(ident) = expr_path.path.segments.last().map(|s| s.ident.to_string()) {
                    if ident == self.function_name {
                        if self.direct_recursion.is_none() {
                            self.direct_recursion = Some(syn::spanned::Spanned::span(node));
                        }
                    }
                }
            } else if expr_path.path.segments.len() == 2 {
                let first = expr_path.path.segments.first().unwrap().ident.to_string();
                let last = expr_path.path.segments.last().unwrap().ident.to_string();
                if first == "Self" && last == self.function_name {
                    if self.direct_recursion.is_none() {
                        self.direct_recursion = Some(syn::spanned::Spanned::span(node));
                    }
                }
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if node.method == self.function_name
            && let syn::Expr::Path(ref expr_path) = *node.receiver
            && expr_path.path.is_ident("self")
            && self.direct_recursion.is_none()
        {
            self.direct_recursion = Some(syn::spanned::Spanned::span(node));
        }
        syn::visit::visit_expr_method_call(self, node);
    }
}

#[proc_macro]
#[allow(missing_docs)]
pub fn runtime_violation(input: TokenStream) -> TokenStream {
    let msg = syn::parse_macro_input!(input as syn::LitStr);
    let msg_str = msg.value();
    let json = format!(
        r#"{{"severity":"FATAL","message":"{}","metadata":{{}},"thread_name":null}}"#,
        msg_str
    );
    let expanded = quote! {
        {
            eprintln!("{}", #json);
            std::process::abort();
        }
    };
    TokenStream::from(expanded)
}

mod latex_parser;
mod semantic_integrity;

mod embed_theory;

#[proc_macro_attribute]
#[allow(missing_docs)]
pub fn embed_theory(attr: TokenStream, item: TokenStream) -> TokenStream {
    embed_theory::embed_theory_impl(attr.into(), item.into()).into()
}

#[proc_macro_attribute]
#[allow(missing_docs)]
pub fn verified(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = match syn::parse::<ItemFn>(item.clone()) {
        Ok(f) => f,
        Err(_) => {
            let item_ts: proc_macro2::TokenStream = item.into();
            return quote! { #item_ts }.into();
        }
    };
    let attr_str = attr.to_string();

    let is_opt_out = attr_str.contains("opt_out");

    let mut visitor = DeepAstVisitor {
        statements: 0,
        direct_recursion: None,
        function_name: input_fn.sig.ident.to_string(),
    };

    visitor.visit_item_fn(&input_fn);

    if !is_opt_out {
        if let Some(span) = visitor.direct_recursion {
            let diag = r#"{"severity":"FATAL","message":"VIOLATION: Direct recursion is not allowed in high-integrity verified modules (NASA Power of 10 Rule 1). Use #[verified(opt_out = \"reason\")] to bypass.","metadata":{},"thread_name":null}"#.to_string();
            return syn::Error::new(span, diag).to_compile_error().into();
        }

        if visitor.statements > 60 {
            let diag = format!(
                r#"{{"severity":"FATAL","message":"VIOLATION: Function exceeds 60 statements (NASA Power of 10 Rule 4) - length: {} statements","metadata":{{}},"thread_name":null}}"#,
                visitor.statements
            );
            return syn::Error::new_spanned(&input_fn.sig.ident, diag)
                .to_compile_error()
                .into();
        }
    }

    let mut injector = InjectorVisitor;
    injector.visit_block_mut(&mut input_fn.block);

    let fn_block = &input_fn.block;
    let new_block: syn::Block = syn::parse_quote! {
        {
            let __ve_active = verified_engine::metrics::is_verification_enabled();
            if __ve_active {
                verified_engine::metrics::increment_calls();
            }
            #fn_block
        }
    };

    *input_fn.block = new_block;

    TokenStream::from(quote! { #input_fn })
}

#[proc_macro_derive(Theory, attributes(theory))]
#[allow(clippy::too_many_lines, clippy::collapsible_if)]
#[allow(missing_docs)]
pub fn derive_theory(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    let name = input.ident;
    let generics = input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let mut description = String::new();
    let mut phonetic = String::new();
    let mut citation = String::new();

    for attr in &input.attrs {
        if attr.path().is_ident("theory") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("description") {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    description = s.value();
                } else if meta.path.is_ident("phonetic") {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    phonetic = s.value();
                } else if meta.path.is_ident("citation") {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    citation = s.value();
                }
                Ok(())
            });
        }
    }

    // Fallback if phonetic is not explicitly provided
    if phonetic.is_empty() {
        phonetic.clone_from(&description);
    }

    let mut field_params = Vec::new();
    let mut get_params = Vec::new();
    let mut set_params = Vec::new();

    if let syn::Data::Struct(data) = &input.data {
        if let syn::Fields::Named(fields) = &data.fields {
            for field in &fields.named {
                let field_ident = field.ident.as_ref().unwrap();
                let field_name = field_ident.to_string();
                let mut min_val: Option<f64> = None;
                let mut max_val: Option<f64> = None;
                let mut step_val: Option<f64> = None;

                for attr in &field.attrs {
                    if attr.path().is_ident("theory") {
                        let _ = attr.parse_nested_meta(|meta| {
                            if meta.path.is_ident("min") {
                                let value = meta.value()?;
                                let lit: syn::LitFloat = value.parse()?;
                                min_val = Some(lit.base10_parse().unwrap());
                            } else if meta.path.is_ident("max") {
                                let value = meta.value()?;
                                let lit: syn::LitFloat = value.parse()?;
                                max_val = Some(lit.base10_parse().unwrap());
                            } else if meta.path.is_ident("step") {
                                let value = meta.value()?;
                                let lit: syn::LitFloat = value.parse()?;
                                step_val = Some(lit.base10_parse().unwrap());
                            }
                            Ok(())
                        });
                    }
                }

                if min_val.is_some() || max_val.is_some() || step_val.is_some() {
                    let min = min_val.unwrap_or(0.0);
                    let max = max_val.unwrap_or(1.0);
                    let step = step_val.unwrap_or(0.1);
                    field_params.push(quote::quote! {
                        map.insert(#field_name.to_string(), scientific_metadata::theory::ParameterConstraint {
                            min: #min,
                            max: #max,
                            step: #step,
                        });
                    });

                    get_params.push(quote::quote! {
                        #field_name => Some(self.#field_ident as f64),
                    });

                    set_params.push(quote::quote! {
                        #field_name => {
                            self.#field_ident = _value as _;
                        }
                    });
                }
            }
        }
    }

    let expanded = quote! {
        impl #impl_generics scientific_metadata::theory::TheoryDescribable for #name #ty_generics #where_clause {
            fn theory_description(&self) -> String {
                #description.to_string()
            }
            fn phonetic_description(&self) -> String {
                #phonetic.to_string()
            }
            fn theory_citation(&self) -> String {
                #citation.to_string()
            }
            fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
                let mut map = std::collections::HashMap::new();
                map.insert("default".to_string(), #description.to_string());
                map
            }
            fn theory_parameters(&self) -> std::collections::HashMap<String, scientific_metadata::theory::ParameterConstraint> {
                let mut map = std::collections::HashMap::new();
                #(#field_params)*
                map
            }
            fn get_parameter(&self, _name: &str) -> Option<f64> {
                match _name {
                    #(#get_params)*
                    _ => None,
                }
            }
            fn set_parameter(&mut self, _name: &str, _value: f64) {
                match _name {
                    #(#set_params)*
                    _ => {}
                }
            }
        }
    };

    TokenStream::from(expanded)
}
