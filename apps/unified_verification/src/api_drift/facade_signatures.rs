use super::{ApiItem, FacadeContext, is_facade_reachable_file, is_tier2_engine_file};
use std::collections::{BTreeMap, HashMap};

pub(super) type ParsedSignatureMap<'a> = HashMap<&'a str, Option<syn::File>>;

fn peel_inline_mod_wrappers(mut sig: &str) -> &str {
    loop {
        let trimmed = sig.trim();
        let Some(rest) = trimmed.strip_prefix("pub mod ") else {
            return trimmed;
        };
        let Some(open_idx) = rest.find('{') else {
            return trimmed;
        };
        let mod_ident = rest[..open_idx].trim();
        if mod_ident.is_empty()
            || !mod_ident
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
            || !rest.ends_with('}')
        {
            return trimmed;
        }
        sig = rest[open_idx + 1..rest.len() - 1].trim();
    }
}

fn has_verbatim_item(file: &syn::File) -> bool {
    file.items.is_empty()
        || file.items.iter().any(|item| match item {
            syn::Item::Verbatim(_) => true,
            syn::Item::Impl(imp) => imp
                .items
                .iter()
                .any(|it| matches!(it, syn::ImplItem::Verbatim(_))),
            syn::Item::Trait(tr) => tr
                .items
                .iter()
                .any(|it| matches!(it, syn::TraitItem::Verbatim(_))),
            _ => false,
        })
}

fn is_const_value_sig(s: &str) -> bool {
    let Some(rest) = s
        .strip_prefix("pub const ")
        .or_else(|| s.strip_prefix("const "))
    else {
        return s.starts_with("pub static ");
    };
    !["fn ", "unsafe ", "async ", "extern "]
        .iter()
        .any(|p| rest.starts_with(p))
}

fn candidate_suffixes(s: &str, braced: bool) -> &'static [&'static str] {
    if braced {
        let body = s
            .find('{')
            .map(|i| s[i + 1..s.len() - 1].trim())
            .unwrap_or("");
        if body.is_empty() {
            &["", "; }", " = todo!(); }", " {} }"]
        } else if s.starts_with("impl") {
            if is_const_value_sig(body) {
                &[" = todo!(); }", "; }", " {} }", ""]
            } else if body.starts_with("pub type ") || body.starts_with("type ") {
                &["; }", " = todo!(); }", " {} }", ""]
            } else {
                &[" {} }", "; }", " = todo!(); }", ""]
            }
        } else if s.starts_with("pub trait ") || s.starts_with("pub unsafe trait ") {
            &["; }", " {} }", " = todo!(); }", ""]
        } else {
            &["", "; }", " = todo!(); }", " {} }"]
        }
    } else if s.starts_with("pub use ")
        || s.starts_with("pub mod ")
        || s.starts_with("pub type ")
        || s.starts_with("pub struct ")
    {
        &[";", " {}", " = todo!();", " { __f: () }"]
    } else if is_const_value_sig(s) {
        &[" = todo!();", ";", " {}", " { __f: () }"]
    } else if s.starts_with("pub union ") {
        &[" { __f: () }", " {}", ";", " = todo!();"]
    } else {
        &[" {}", ";", " = todo!();", " { __f: () }"]
    }
}

fn try_parse_candidate(stem: &str, suf: &str, parse_attempts: &mut usize) -> Option<syn::File> {
    *parse_attempts += 1;
    let parsed = if suf.is_empty() {
        syn::parse_file(stem)
    } else {
        let mut buf = String::with_capacity(stem.len() + suf.len());
        buf.push_str(stem);
        buf.push_str(suf);
        syn::parse_file(&buf)
    };
    parsed.ok().filter(|f| !has_verbatim_item(f))
}

pub(super) fn parse_signature_ast_counted(
    raw_sig: &str,
    parse_attempts: &mut usize,
) -> Option<syn::File> {
    let s = peel_inline_mod_wrappers(raw_sig);
    if let Some(without_close) = s.strip_suffix('}') {
        let trimmed_inner = without_close.trim_end();
        for &suf in candidate_suffixes(s, true) {
            let stem = if suf.is_empty() { s } else { trimmed_inner };
            if let Some(f) = try_parse_candidate(stem, suf, parse_attempts) {
                return Some(f);
            }
        }
    }
    for &suf in candidate_suffixes(s, false) {
        if let Some(f) = try_parse_candidate(s, suf, parse_attempts) {
            return Some(f);
        }
    }
    None
}

pub(super) fn build_parsed_signatures_counted<'a>(
    ctx: &FacadeContext,
    apis: &'a BTreeMap<String, Vec<ApiItem>>,
) -> (ParsedSignatureMap<'a>, usize) {
    let mut parsed = HashMap::new();
    let mut parse_attempts = 0usize;
    for (path, items) in apis {
        if !is_facade_reachable_file(ctx, path) && !is_tier2_engine_file(ctx, path) {
            continue;
        }
        for item in items {
            let sig = item.signature.as_str();
            if sig.starts_with("pub mod __doctest_facade") {
                continue;
            }
            parsed
                .entry(sig)
                .or_insert_with(|| parse_signature_ast_counted(sig, &mut parse_attempts));
        }
    }
    (parsed, parse_attempts)
}
