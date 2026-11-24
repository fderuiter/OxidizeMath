#[path = "theory_bounds.rs"]
mod theory_bounds;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use theory_bounds::{build_field_tokens, build_theory_parameters_body, push_error};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParsedMaturityTier {
    Experimental,
    Validated,
    Certified,
}

#[derive(Default)]
struct StructTheoryMeta {
    description: String,
    phonetic: String,
    citation: String,
    maturity: Option<ParsedMaturityTier>,
    seen_description: bool,
    seen_phonetic: bool,
    seen_citation: bool,
    seen_maturity: bool,
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

fn parse_maturity_meta(
    meta: &syn::meta::ParseNestedMeta<'_>,
    seen: &mut bool,
) -> syn::Result<ParsedMaturityTier> {
    if *seen {
        return Err(meta.error("duplicate `maturity` key in `#[theory]`"));
    }
    let value = meta.value()?;
    let lit: syn::LitStr = value.parse()?;
    *seen = true;
    match lit.value().trim() {
        "Experimental" => Ok(ParsedMaturityTier::Experimental),
        "Validated" => Ok(ParsedMaturityTier::Validated),
        "Certified" => Ok(ParsedMaturityTier::Certified),
        other => Err(syn::Error::new(
            lit.span(),
            format!(
                "invalid `maturity` tier `{other}` in `#[theory]`; expected `Experimental`, `Validated`, or `Certified`"
            ),
        )),
    }
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
            } else if meta.path.is_ident("maturity") {
                out.maturity = Some(parse_maturity_meta(&meta, &mut out.seen_maturity)?);
                Ok(())
            } else {
                let key = meta.path.to_token_stream().to_string();
                Err(meta.error(format!(
                    "unknown struct-level `#[theory]` key `{key}`; valid keys are `description`, `phonetic`, `citation`, `maturity`"
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

fn build_maturity_method(
    maturity: Option<ParsedMaturityTier>,
    sm_path: &TokenStream,
) -> TokenStream {
    let Some(tier) = maturity else {
        return TokenStream::new();
    };
    let variant = match tier {
        ParsedMaturityTier::Experimental => quote!(Experimental),
        ParsedMaturityTier::Validated => quote!(Validated),
        ParsedMaturityTier::Certified => quote!(Certified),
    };
    quote! {
        fn maturity_tier(&self) -> #sm_path::theory::MaturityTier {
            #sm_path::theory::MaturityTier::#variant
        }
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
    let maturity_method = build_maturity_method(struct_meta.maturity, sm_path);

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
            #maturity_method
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
    fn test_optional_maturity_attribute_and_invalid_tier_error() {
        let sm = quote! { ::scientific_metadata };
        let certified = derive_theory_impl_with_path(
            quote! {
                #[theory(description = "Solver", citation = "Ref", maturity = "Certified")]
                struct CertifiedModel {
                    pub tol: f64,
                }
            },
            &sm,
        )
        .to_string();
        assert!(!certified.contains("compile_error"));
        assert!(certified.contains("maturity_tier"));
        assert!(certified.contains("Certified"));

        let bad_tier = derive_theory_impl_with_path(
            quote! {
                #[theory(description = "Solver", maturity = "GoldPlated")]
                struct InvalidTierModel {
                    pub tol: f64,
                }
            },
            &sm,
        )
        .to_string();
        assert!(bad_tier.contains("compile_error"));
        assert!(bad_tier.contains("expected `Experimental`, `Validated`, or `Certified`"));
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
