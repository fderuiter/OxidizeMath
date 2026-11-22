use proc_macro2::TokenStream;
use quote::quote;
use std::fs;
use std::path::{Path, PathBuf};
use syn::{
    Ident, LitStr, Token,
    parse::{Parse, ParseStream},
};

const MAX_ANCESTOR_DEPTH: usize = 16;

pub struct EmbedTheoryArgs {
    pub file_path: LitStr,
    pub labels: Vec<LitStr>,
}

impl Parse for EmbedTheoryArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let file_path: LitStr = input.parse()?;
        if file_path.value().trim().is_empty() {
            return Err(syn::Error::new_spanned(
                &file_path,
                "LaTeX file path must not be empty",
            ));
        }

        let mut labels = Vec::new();
        while !input.is_empty() {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }

            let ident: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            if ident == "label" {
                let label_lit: LitStr = input.parse()?;
                if label_lit.value().trim().is_empty() {
                    return Err(syn::Error::new_spanned(
                        &label_lit,
                        "LaTeX label must not be empty",
                    ));
                }
                labels.push(label_lit);
            } else if ident == "labels" {
                return Err(syn::Error::new_spanned(
                    ident,
                    "Use multiple `label = \"...\"` arguments instead",
                ));
            } else {
                return Err(syn::Error::new_spanned(
                    ident,
                    "Unknown attribute argument, expected `label`",
                ));
            }
        }

        if labels.is_empty() {
            return Err(syn::Error::new_spanned(
                &file_path,
                "At least one `label = \"...\"` must be provided",
            ));
        }

        Ok(EmbedTheoryArgs { file_path, labels })
    }
}

fn strip_labels(text: &str) -> String {
    let mut result = String::new();
    let mut i = 0;
    let chars: Vec<char> = text.chars().collect();
    while i < chars.len() {
        let byte_offset: usize = chars[..i].iter().map(|c| c.len_utf8()).sum();
        if text[byte_offset..].starts_with("\\label{") {
            i += 7;
            let mut depth = 1;
            while i < chars.len() && depth > 0 {
                if chars[i] == '{' {
                    depth += 1;
                } else if chars[i] == '}' {
                    depth -= 1;
                }
                i += 1;
            }
            continue;
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

fn convert_environment(env_name: &str, inner_content: &str) -> String {
    let inner = strip_labels(inner_content).trim().to_string();
    match env_name {
        "equation" | "equation*" | "align" | "align*" | "eqnarray" | "eqnarray*"
        | "displaymath" => format!("$$\n{inner}\n$$"),
        "lemma" | "theorem" | "proof" | "definition" | "corollary" | "proposition" => {
            let title = env_name[0..1].to_uppercase() + &env_name[1..];
            format!("**{title}**\n\n{inner}")
        }
        _ => format!("{inner}\n"),
    }
}

fn extract_label(tex_content: &str, target_label: &str) -> Option<String> {
    let label_str = format!("\\label{{{target_label}}}");
    let label_pos = tex_content.find(&label_str)?;
    let before_label = &tex_content[..label_pos];

    let begin_pos = before_label.rfind("\\begin{")?;
    let env_start = &before_label[begin_pos..];
    let brace_end = env_start.find('}')?;
    let env_name = &env_start[7..brace_end];

    let end_str = format!("\\end{{{env_name}}}");
    let end_pos = tex_content[label_pos..].find(&end_str)?;
    let inner_start = begin_pos + 7 + env_name.len() + 1;
    let inner_end = label_pos + end_pos;
    Some(convert_environment(
        env_name,
        &tex_content[inner_start..inner_end],
    ))
}

fn check_shadow_mirroring(attrs: &[syn::Attribute]) -> syn::Result<()> {
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let syn::Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        let syn::Expr::Lit(expr_lit) = &nv.value else {
            continue;
        };
        let syn::Lit::Str(lit_str) = &expr_lit.lit else {
            continue;
        };
        let doc_val = lit_str.value();
        if doc_val.contains("$$")
            || doc_val.contains("\\begin{equation}")
            || doc_val.contains("\\begin{align}")
        {
            return Err(syn::Error::new_spanned(
                attr,
                "Manual docstring content for mathematical sections is strictly forbidden in items using #[embed_theory] to prevent shadow-mirroring.",
            ));
        }
    }
    Ok(())
}

fn has_workspace_table(dir: &Path) -> bool {
    fs::read_to_string(dir.join("Cargo.toml"))
        .map(|content| content.contains("[workspace]"))
        .unwrap_or(false)
}

pub(crate) fn resolve_tex_path_from_manifest(
    file_lit: &LitStr,
    manifest_dir: Option<&Path>,
) -> syn::Result<(PathBuf, TokenStream)> {
    let raw_path = file_lit.value();
    let norm_rel = raw_path.trim_start_matches("./").replace('\\', "/");

    if let Some(manifest) = manifest_dir {
        let direct = manifest.join(&norm_rel);
        if direct.is_file() {
            let expr = quote! { concat!(env!("CARGO_MANIFEST_DIR"), "/", #norm_rel) };
            return Ok((direct, expr));
        }

        let mut ancestor_fallback: Option<(PathBuf, String)> = None;
        let mut current = manifest;
        for depth in 1..=MAX_ANCESTOR_DEPTH {
            let Some(parent) = current.parent() else {
                break;
            };
            current = parent;
            let candidate = current.join(&norm_rel);
            let rel_from_manifest = format!("{}{norm_rel}", "../".repeat(depth));
            if candidate.is_file() {
                if has_workspace_table(current) {
                    let expr =
                        quote! { concat!(env!("CARGO_MANIFEST_DIR"), "/", #rel_from_manifest) };
                    return Ok((candidate, expr));
                }
                if ancestor_fallback.is_none() {
                    ancestor_fallback = Some((candidate, rel_from_manifest));
                }
            }
        }

        if let Some((path, rel_from_manifest)) = ancestor_fallback {
            let expr = quote! { concat!(env!("CARGO_MANIFEST_DIR"), "/", #rel_from_manifest) };
            return Ok((path, expr));
        }
    } else if let Ok(cwd) = std::env::current_dir() {
        let mut current: Option<&Path> = Some(cwd.as_path());
        for _ in 0..=MAX_ANCESTOR_DEPTH {
            let Some(dir) = current else {
                break;
            };
            let candidate = dir.join(&norm_rel);
            if candidate.is_file() {
                let slash = candidate.to_string_lossy().replace('\\', "/");
                return Ok((candidate, quote! { #slash }));
            }
            current = dir.parent();
        }
    }

    let direct_abs = PathBuf::from(&raw_path);
    if direct_abs.is_absolute() && direct_abs.is_file() {
        let slash = direct_abs.to_string_lossy().replace('\\', "/");
        return Ok((direct_abs, quote! { #slash }));
    }

    Err(syn::Error::new_spanned(
        file_lit,
        format!("LaTeX file not found: `{raw_path}`"),
    ))
}

fn extract_all_labels(
    tex_content: &str,
    labels: &[LitStr],
    file_display: &str,
) -> syn::Result<Vec<String>> {
    let mut docs = Vec::with_capacity(labels.len());
    for label_lit in labels {
        let label = label_lit.value();
        let Some(md) = extract_label(tex_content, &label) else {
            return Err(syn::Error::new_spanned(
                label_lit,
                format!(
                    "Label `{label}` not found or not in a supported environment in `{file_display}`"
                ),
            ));
        };
        docs.push(md);
    }
    Ok(docs)
}

fn check_block_integrity(
    ident: &Ident,
    block: &syn::Block,
    sections: &[String],
) -> syn::Result<()> {
    for section in sections {
        if let Err(err) = crate::semantic_integrity::verify_block_semantic_integrity(section, block)
        {
            return Err(syn::Error::new_spanned(
                ident,
                format!("Semantic Integrity Violation: {err}"),
            ));
        }
    }
    Ok(())
}

fn prepend_const_tracker(block: &mut syn::Block, include_expr: &TokenStream) {
    let tracker: syn::Stmt = syn::parse_quote! {
        const _: &str = include_str!(#include_expr);
    };
    block.stmts.insert(0, tracker);
}

fn expand_target_item(
    item: TokenStream,
    doc_str: &str,
    include_expr: &TokenStream,
    sections: &[String],
) -> syn::Result<TokenStream> {
    if let Ok(mut item_fn) = syn::parse2::<syn::ItemFn>(item.clone()) {
        check_shadow_mirroring(&item_fn.attrs)?;
        check_block_integrity(&item_fn.sig.ident, &item_fn.block, sections)?;
        prepend_const_tracker(&mut item_fn.block, include_expr);
        return Ok(quote! {
            #[doc = #doc_str]
            #item_fn
        });
    }

    if let Ok(mut impl_fn) = syn::parse2::<syn::ImplItemFn>(item.clone()) {
        check_shadow_mirroring(&impl_fn.attrs)?;
        check_block_integrity(&impl_fn.sig.ident, &impl_fn.block, sections)?;
        prepend_const_tracker(&mut impl_fn.block, include_expr);
        return Ok(quote! {
            #[doc = #doc_str]
            #impl_fn
        });
    }

    if let Ok(mut trait_fn) = syn::parse2::<syn::TraitItemFn>(item.clone()) {
        check_shadow_mirroring(&trait_fn.attrs)?;
        if let Some(ref mut block) = trait_fn.default {
            check_block_integrity(&trait_fn.sig.ident, block, sections)?;
            prepend_const_tracker(block, include_expr);
        } else {
            let where_clause = trait_fn.sig.generics.make_where_clause();
            where_clause.predicates.push(syn::parse_quote! {
                [(); { const _: &str = include_str!(#include_expr); 0 }]: Sized
            });
        }
        return Ok(quote! {
            #[doc = #doc_str]
            #trait_fn
        });
    }

    if let Ok(parsed_item) = syn::parse2::<syn::Item>(item.clone()) {
        let attrs: &[syn::Attribute] = match &parsed_item {
            syn::Item::Struct(s) => &s.attrs,
            syn::Item::Enum(e) => &e.attrs,
            syn::Item::Union(u) => &u.attrs,
            syn::Item::Mod(m) => &m.attrs,
            syn::Item::Impl(i) => &i.attrs,
            syn::Item::Trait(t) => &t.attrs,
            syn::Item::Type(t) => &t.attrs,
            syn::Item::Const(c) => &c.attrs,
            syn::Item::Static(s) => &s.attrs,
            _ => &[],
        };
        check_shadow_mirroring(attrs)?;
        return Ok(quote! {
            #[doc = #doc_str]
            #parsed_item
            const _: &str = include_str!(#include_expr);
        });
    }

    Err(syn::Error::new_spanned(
        item,
        "`#[embed_theory]` must be applied to a function, method, or item declaration",
    ))
}

pub fn embed_theory_impl_with_manifest(
    attr: TokenStream,
    item: TokenStream,
    manifest_dir: Option<&Path>,
) -> TokenStream {
    let args = match syn::parse2::<EmbedTheoryArgs>(attr) {
        Ok(parsed) => parsed,
        Err(err) => return err.to_compile_error(),
    };

    let (abs_path, include_expr) =
        match resolve_tex_path_from_manifest(&args.file_path, manifest_dir) {
            Ok(resolved) => resolved,
            Err(err) => return err.to_compile_error(),
        };

    let file_display = args.file_path.value();
    let tex_content = match fs::read_to_string(&abs_path) {
        Ok(content) => content,
        Err(err) => {
            return syn::Error::new_spanned(
                &args.file_path,
                format!("Failed to read `{file_display}`: {err}"),
            )
            .to_compile_error();
        }
    };

    let sections = match extract_all_labels(&tex_content, &args.labels, &file_display) {
        Ok(extracted) => extracted,
        Err(err) => return err.to_compile_error(),
    };
    let doc_str = sections.join("\n\n");

    match expand_target_item(item, &doc_str, &include_expr, &sections) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

pub fn embed_theory_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok().map(PathBuf::from);
    embed_theory_impl_with_manifest(attr, item, manifest_dir.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_NONCE: AtomicUsize = AtomicUsize::new(0);

    fn temp_workspace(prefix: &str) -> PathBuf {
        let id = TEST_NONCE.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "oxidize_embed_theory_{prefix}_{}_{id}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("papers")).unwrap();
        fs::write(root.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
        fs::write(
            root.join("papers/sample.tex"),
            "\\begin{equation}\\label{eq:add}\na + b\n\\end{equation}\n",
        )
        .unwrap();
        root
    }

    #[test]
    fn test_resolves_from_top_level_and_nested_manifest_dirs() {
        let ws = temp_workspace("depths");
        let top_crate = ws.join("math_explorer");
        let nested_crate = ws.join("crates/domain_physics");
        fs::create_dir_all(&top_crate).unwrap();
        fs::create_dir_all(&nested_crate).unwrap();

        let top_out = embed_theory_impl_with_manifest(
            quote! { "papers/sample.tex", label = "eq:add" },
            quote! { pub fn sum(a: f64, b: f64) -> f64 { a + b } },
            Some(&top_crate),
        )
        .to_string();
        assert!(!top_out.contains("compile_error"));
        assert!(top_out.contains("../papers/sample.tex"));
        assert!(!top_out.contains("allow"));
        assert!(!top_out.contains("_TRACK_THEORY_"));
        assert!(top_out.contains("const _ : & str"));

        let nested_out = embed_theory_impl_with_manifest(
            quote! { "papers/sample.tex", label = "eq:add" },
            quote! { pub fn sum(a: f64, b: f64) -> f64 { a + b } },
            Some(&nested_crate),
        )
        .to_string();
        assert!(!nested_out.contains("compile_error"));
        assert!(nested_out.contains("../../papers/sample.tex"));
        let _ = fs::remove_dir_all(&ws);
    }

    #[test]
    fn test_strip_labels_handles_unicode_safely() {
        let raw = "α + β \\label{eq:unicode} = γ";
        assert_eq!(strip_labels(raw), "α + β  = γ");
    }
}
