#![doc = include_str!("../README.md")]
extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;

mod crate_paths;
mod embed_theory;
mod latex_parser;
mod semantic_integrity;
mod theory;
mod verified;

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

#[proc_macro_attribute]
#[allow(missing_docs)]
pub fn embed_theory(attr: TokenStream, item: TokenStream) -> TokenStream {
    embed_theory::embed_theory_impl(attr.into(), item.into()).into()
}

#[proc_macro_attribute]
#[allow(missing_docs)]
pub fn verified(attr: TokenStream, item: TokenStream) -> TokenStream {
    verified::verified_impl(attr.into(), item.into()).into()
}

#[proc_macro_derive(Theory, attributes(theory))]
#[allow(missing_docs)]
pub fn derive_theory(input: TokenStream) -> TokenStream {
    theory::derive_theory_impl(input.into()).into()
}
