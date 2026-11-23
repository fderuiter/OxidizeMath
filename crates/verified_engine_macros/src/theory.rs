use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote, quote_spanned};
use syn::parse::ParseStream;
use syn::spanned::Spanned;

fn push_error(errors: &mut Option<syn::Error>, err: syn::Error) {
    match errors {
        Some(existing) => existing.combine(err),
        None => *errors = Some(err),
    }
}

#[derive(Default)]
struct StructTheoryMeta {
    description: String,
    phonetic: String,
    citation: String,
    seen_description: bool,
    seen_phonetic: bool,
    seen_citation: bool,
}

fn parse_string_meta(
    meta: &syn::meta::ParseNestedMeta<'_>,
    seen: &mut bool,
    key_name: &str,
) -> syn::Result<String> {
    if *seen {
        return Err(meta.error(format!("duplicate `{key_name}` key in `#[theory]`")));
    }
    let value = meta.value()?;
    let lit: syn::LitStr = value.parse()?;
    *seen = true;
    Ok(lit.value())
}

fn parse_struct_attrs(
    attrs: &[syn::Attribute],
    errors: &mut Option<syn::Error>,
) -> StructTheoryMeta {
    let mut out = StructTheoryMeta::default();
    for attr in attrs {
        if !attr.path().is_ident("theory") {
            continue;
        }
        let res = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("description") {
                out.description =
                    parse_string_meta(&meta, &mut out.seen_description, "description")?;
                Ok(())
            } else if meta.path.is_ident("phonetic") {
                out.phonetic = parse_string_meta(&meta, &mut out.seen_phonetic, "phonetic")?;
                Ok(())
            } else if meta.path.is_ident("citation") {
                out.citation = parse_string_meta(&meta, &mut out.seen_citation, "citation")?;
                Ok(())
            } else {
                let key = meta.path.to_token_stream().to_string();
                Err(meta.error(format!(
                    "unknown struct-level `#[theory]` key `{key}`; valid keys are `description`, `phonetic`, `citation`"
                )))
            }
        });
        if let Err(err) = res {
            push_error(errors, err);
        }
    }
    if out.phonetic.is_empty() {
        out.phonetic.clone_from(&out.description);
    }
    out
}

#[derive(Default)]
struct RawFieldBounds {
    has_any_key: bool,
    has_invalid_value: bool,
    min: Option<(f64, Span)>,
    max: Option<(f64, Span)>,
    step: Option<(f64, Span)>,
    default: Option<(f64, Span)>,
}

fn eval_numeric_lit(
    lit: &syn::Lit,
    neg: bool,
    span: Span,
) -> Result<(f64, Span), (syn::Error, bool)> {
    let raw = match lit {
        syn::Lit::Float(lf) => lf.base10_parse::<f64>().map_err(|_| {
            (
                syn::Error::new(span, "invalid floating-point literal in `#[theory]` bound"),
                true,
            )
        })?,
        syn::Lit::Int(li) => li.base10_parse::<f64>().map_err(|_| {
            (
                syn::Error::new(span, "invalid integer literal in `#[theory]` bound"),
                true,
            )
        })?,
        _ => {
            return Err((
                syn::Error::new(
                    span,
                    "expected a finite integer or float literal for `#[theory]` bound",
                ),
                true,
            ));
        }
    };
    let signed = if neg { -raw } else { raw };
    if !signed.is_finite() {
        return Err((
            syn::Error::new(
                span,
                "parameter bound must be a finite number (literal overflows `f64`)",
            ),
            true,
        ));
    }
    Ok((signed, span))
}

fn parse_finite_bound_literal(input: ParseStream<'_>) -> Result<(f64, Span), (syn::Error, bool)> {
    let neg: Option<syn::Token![-]> = input.parse().map_err(|e| (e, false))?;
    if input.peek(syn::Lit) {
        let lit: syn::Lit = input.parse().map_err(|e| (e, false))?;
        let span = match neg {
            Some(minus) => minus.span.join(lit.span()).unwrap_or_else(|| lit.span()),
            None => lit.span(),
        };
        return eval_numeric_lit(&lit, neg.is_some(), span);
    }

    if let Ok(expr) = input.parse::<syn::Expr>() {
        let expr_span = match neg {
            Some(minus) => minus.span.join(expr.span()).unwrap_or_else(|| expr.span()),
            None => expr.span(),
        };
        return Err((
            syn::Error::new(
                expr_span,
                "expected a finite integer or float literal for `#[theory]` bound",
            ),
            true,
        ));
    }

    Err((
        input.error("expected a finite integer or float literal for `#[theory]` bound"),
        false,
    ))
}

fn record_bound_key(
    meta: &syn::meta::ParseNestedMeta<'_>,
    slot: &mut Option<(f64, Span)>,
    key_name: &str,
    has_any_key: &mut bool,
    has_invalid_value: &mut bool,
    errors: &mut Option<syn::Error>,
) -> syn::Result<()> {
    *has_any_key = true;
    if slot.is_some() {
        return Err(meta.error(format!("duplicate `{key_name}` key in `#[theory]`")));
    }
    let value = meta.value()?;
    match parse_finite_bound_literal(value) {
        Ok(parsed) => {
            *slot = Some(parsed);
            Ok(())
        }
        Err((err, consumed)) => {
            *has_invalid_value = true;
            if consumed {
                push_error(errors, err);
                Ok(())
            } else {
                Err(err)
            }
        }
    }
}

fn validate_field_bounds(
    bounds: RawFieldBounds,
    fallback_span: Span,
    errors: &mut Option<syn::Error>,
) -> Option<(f64, f64, f64, Option<f64>)> {
    if !bounds.has_any_key || bounds.has_invalid_value {
        return None;
    }

    let (min_val, min_span) = bounds.min.unwrap_or((0.0, fallback_span));
    let (max_val, max_span) = bounds.max.unwrap_or((1.0, fallback_span));
    let (step_val, step_span) = bounds.step.unwrap_or((0.1, fallback_span));

    if step_val <= 0.0 {
        push_error(
            errors,
            syn::Error::new(
                step_span,
                format!("`step` must be greater than 0.0, found {step_val}"),
            ),
        );
    }
    if min_val > max_val {
        let span = if bounds.min.is_some() {
            min_span
        } else {
            max_span
        };
        push_error(
            errors,
            syn::Error::new(
                span,
                format!("`min` ({min_val}) must not exceed `max` ({max_val})"),
            ),
        );
    } else if let Some((def_val, def_span)) = bounds.default
        && (def_val < min_val || def_val > max_val)
    {
        push_error(
            errors,
            syn::Error::new(
                def_span,
                format!("`default` ({def_val}) must lie within `[{min_val}, {max_val}]`"),
            ),
        );
    }
    Some((min_val, max_val, step_val, bounds.default.map(|(d, _)| d)))
}

fn parse_field_bounds(
    field: &syn::Field,
    errors: &mut Option<syn::Error>,
) -> Option<(f64, f64, f64, Option<f64>)> {
    let mut bounds = RawFieldBounds::default();
    for attr in &field.attrs {
        if !attr.path().is_ident("theory") {
            continue;
        }
        let res = attr.parse_nested_meta(|meta| {
            let (h_any, h_inv) = (&mut bounds.has_any_key, &mut bounds.has_invalid_value);
            if meta.path.is_ident("min") {
                record_bound_key(&meta, &mut bounds.min, "min", h_any, h_inv, errors)
            } else if meta.path.is_ident("max") {
                record_bound_key(&meta, &mut bounds.max, "max", h_any, h_inv, errors)
            } else if meta.path.is_ident("step") {
                record_bound_key(&meta, &mut bounds.step, "step", h_any, h_inv, errors)
            } else if meta.path.is_ident("default") {
                record_bound_key(&meta, &mut bounds.default, "default", h_any, h_inv, errors)
            } else {
                let key = meta.path.to_token_stream().to_string();
                Err(meta.error(format!(
                    "unknown field-level `#[theory]` key `{key}`; valid keys are `min`, `max`, `step`, `default`"
                )))
            }
        });
        if let Err(err) = res {
            push_error(errors, err);
        }
    }
    validate_field_bounds(bounds, field.span(), errors)
}

fn extract_named_fields(input: &syn::DeriveInput) -> syn::Result<Vec<&syn::Field>> {
    match &input.data {
        syn::Data::Enum(e) => Err(syn::Error::new_spanned(
            e.enum_token,
            "`#[derive(Theory)]` can only be derived for structs, not enums",
        )),
        syn::Data::Union(u) => Err(syn::Error::new_spanned(
            u.union_token,
            "`#[derive(Theory)]` can only be derived for structs, not unions",
        )),
        syn::Data::Struct(s) => match &s.fields {
            syn::Fields::Named(named) => Ok(named.named.iter().collect()),
            syn::Fields::Unit => Ok(Vec::new()),
            syn::Fields::Unnamed(unnamed) => Err(syn::Error::new_spanned(
                unnamed,
                "`#[derive(Theory)]` requires named fields and cannot be derived for tuple structs",
            )),
        },
    }
}

fn build_field_tokens(
    fields: &[&syn::Field],
    sm_path: &TokenStream,
    errors: &mut Option<syn::Error>,
) -> (Vec<TokenStream>, Vec<TokenStream>) {
    let mut type_checks = Vec::new();
    let mut field_params = Vec::new();

    for field in fields {
        let Some((min, max, step, default_opt)) = parse_field_bounds(field, errors) else {
            continue;
        };
        let Some(field_ident) = field.ident.as_ref() else {
            continue;
        };
        let field_name = field_ident.to_string();
        let field_ty = &field.ty;
        let ty_span = field_ty.span();
        let default_tok = match default_opt {
            Some(d) => quote!(::core::option::Option::Some(#d)),
            None => quote!(::core::option::Option::None),
        };

        type_checks.push(quote_spanned! {ty_span=>
            assert_theory_numeric::<#field_ty>(&self.#field_ident);
        });
        field_params.push(quote! {
            map.insert(
                #field_name.to_string(),
                #sm_path::theory::ParameterConstraint {
                    min: #min,
                    max: #max,
                    step: #step,
                    default: #default_tok,
                },
            );
        });
    }

    (type_checks, field_params)
}

fn build_theory_parameters_body(
    type_checks: &[TokenStream],
    field_params: &[TokenStream],
) -> TokenStream {
    if field_params.is_empty() {
        return quote! {
            ::std::collections::HashMap::new()
        };
    }
    let numeric_tys = [
        quote!(i8),
        quote!(i16),
        quote!(i32),
        quote!(i64),
        quote!(i128),
        quote!(isize),
        quote!(u8),
        quote!(u16),
        quote!(u32),
        quote!(u64),
        quote!(u128),
        quote!(usize),
        quote!(f32),
        quote!(f64),
    ];
    quote! {
        trait TheoryNumeric {}
        #(impl TheoryNumeric for #numeric_tys {})*
        fn assert_theory_numeric<TheoryNumericTarget: TheoryNumeric>(
            _: &TheoryNumericTarget,
        ) {}
        #(#type_checks)*
        let mut map = ::std::collections::HashMap::new();
        #(#field_params)*
        map
    }
}

pub(crate) fn derive_theory_impl_with_path(
    input: TokenStream,
    sm_path: &TokenStream,
) -> TokenStream {
    let input: syn::DeriveInput = match syn::parse2(input) {
        Ok(parsed) => parsed,
        Err(err) => return err.to_compile_error(),
    };

    let mut errors: Option<syn::Error> = None;
    let struct_meta = parse_struct_attrs(&input.attrs, &mut errors);

    let fields = match extract_named_fields(&input) {
        Ok(f) => f,
        Err(err) => {
            push_error(&mut errors, err);
            Vec::new()
        }
    };

    let (type_checks, field_params) = build_field_tokens(&fields, sm_path, &mut errors);

    if let Some(err) = errors {
        return err.to_compile_error();
    }

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let description = &struct_meta.description;
    let phonetic = &struct_meta.phonetic;
    let citation = &struct_meta.citation;
    let params_body = build_theory_parameters_body(&type_checks, &field_params);

    quote! {
        impl #impl_generics #sm_path::theory::TheoryDescribable for #name #ty_generics #where_clause {
            fn theory_description(&self) -> String {
                #description.to_string()
            }
            fn phonetic_description(&self) -> String {
                #phonetic.to_string()
            }
            fn theory_citation(&self) -> String {
                #citation.to_string()
            }
            fn available_descriptions(&self) -> ::std::collections::HashMap<String, String> {
                let mut map = ::std::collections::HashMap::new();
                map.insert("default".to_string(), #description.to_string());
                map
            }
            fn theory_parameters(&self) -> ::std::collections::HashMap<String, #sm_path::theory::ParameterConstraint> {
                #params_body
            }
        }
    }
}

pub(crate) fn derive_theory_impl(input: TokenStream) -> TokenStream {
    let sm_path = crate::crate_paths::resolve_scientific_metadata_path();
    derive_theory_impl_with_path(input, &sm_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accepts_integer_and_negative_bounds() {
        let sm = quote! { ::scientific_metadata };
        let expanded = derive_theory_impl_with_path(
            quote! {
                #[theory(description = "Grid", citation = "Ref")]
                struct GridModel {
                    #[theory(min = -10, max = 100, step = 5)]
                    offset: i32,
                }
            },
            &sm,
        )
        .to_string();
        assert!(!expanded.contains("compile_error"));
        assert!(expanded.contains("TheoryDescribable"));
        assert!(expanded.contains("TheoryNumeric"));
        assert!(!expanded.contains("unnecessary_cast"));
        assert!(!expanded.contains(" as "));
    }

    #[test]
    fn test_rejects_overflow_without_panic_and_combines_errors() {
        let sm = quote! { ::scientific_metadata };
        let expanded = derive_theory_impl_with_path(
            quote! {
                #[theory(typo_key = "bad")]
                struct BadModel {
                    #[theory(min = 10.0, max = 5.0, step = 0.0)]
                    a: f64,
                    #[theory(max = 1e999)]
                    b: f64,
                }
            },
            &sm,
        )
        .to_string();
        assert!(expanded.contains("valid keys are `description`, `phonetic`, `citation`"));
        assert!(expanded.contains("must not exceed `max`"));
        assert!(expanded.contains("`step` must be greater than 0.0"));
        assert!(expanded.contains("finite number"));
    }

    #[test]
    fn test_rejects_enum_union_and_tuple_struct() {
        let sm = quote! { ::scientific_metadata };
        for item in [
            quote! { enum E { A } },
            quote! { union U { f: u32 } },
            quote! { struct Tuple(f64); },
        ] {
            let out = derive_theory_impl_with_path(item, &sm).to_string();
            assert!(out.contains("compile_error"));
        }
    }
}
