use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote, quote_spanned};
use syn::parse::ParseStream;
use syn::spanned::Spanned;

pub(super) fn push_error(errors: &mut Option<syn::Error>, err: syn::Error) {
    match errors {
        Some(existing) => existing.combine(err),
        None => *errors = Some(err),
    }
}

#[derive(Default)]
struct RawFieldBounds {
    has_any_key: bool,
    has_invalid_value: bool,
    min: Option<(f64, Span)>,
    max: Option<(f64, Span)>,
    step: Option<(f64, Span)>,
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
) -> Option<(f64, f64, f64)> {
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
    }
    Some((min_val, max_val, step_val))
}

fn parse_field_bounds(
    field: &syn::Field,
    errors: &mut Option<syn::Error>,
) -> Option<(f64, f64, f64)> {
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
            } else {
                let key = meta.path.to_token_stream().to_string();
                Err(meta.error(format!(
                    "unknown field-level `#[theory]` key `{key}`; valid keys are `min`, `max`, `step`"
                )))
            }
        });
        if let Err(err) = res {
            push_error(errors, err);
        }
    }
    validate_field_bounds(bounds, field.span(), errors)
}

pub(super) fn build_field_tokens(
    fields: &[&syn::Field],
    sm_path: &TokenStream,
    errors: &mut Option<syn::Error>,
) -> (Vec<TokenStream>, Vec<TokenStream>) {
    let mut type_checks = Vec::new();
    let mut field_params = Vec::new();

    for field in fields {
        let Some((min, max, step)) = parse_field_bounds(field, errors) else {
            continue;
        };
        let Some(field_ident) = field.ident.as_ref() else {
            continue;
        };
        let field_name = field_ident.to_string();
        let field_ty = &field.ty;
        let ty_span = field_ty.span();

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
                },
            );
        });
    }

    (type_checks, field_params)
}

pub(super) fn build_theory_parameters_body(
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
