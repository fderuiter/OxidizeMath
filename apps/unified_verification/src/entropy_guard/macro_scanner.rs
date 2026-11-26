use super::{SecretScanner, classify_entropy_call, resolve_path};
use proc_macro2::{Delimiter, Spacing, TokenStream, TokenTree};
use std::collections::HashMap;

const MAX_MACRO_GROUPS: usize = 4096;

pub(super) fn scan_macro(
    mac: &syn::Macro,
    aliases: &HashMap<String, String>,
    scanner: &SecretScanner,
    entropy_violations: &mut Vec<(String, usize)>,
    secret_warnings: &mut Vec<(String, usize)>,
) {
    scan_token_stream(
        mac.tokens.clone(),
        aliases,
        scanner,
        entropy_violations,
        secret_warnings,
    );
}

fn scan_token_stream(
    root: TokenStream,
    aliases: &HashMap<String, String>,
    scanner: &SecretScanner,
    entropy_violations: &mut Vec<(String, usize)>,
    secret_warnings: &mut Vec<(String, usize)>,
) {
    let mut stack = vec![root];
    let mut visited_groups = 0usize;

    while let Some(stream) = stack.pop() {
        if visited_groups >= MAX_MACRO_GROUPS {
            break;
        }
        visited_groups += 1;

        let tokens: Vec<TokenTree> = stream.into_iter().collect();
        scan_literals_in_slice(&tokens, scanner, secret_warnings);
        scan_calls_in_slice(&tokens, aliases, entropy_violations);

        for token in tokens.into_iter().rev() {
            if let TokenTree::Group(group) = token {
                stack.push(group.stream());
            }
        }
    }
}

fn scan_literals_in_slice(
    tokens: &[TokenTree],
    scanner: &SecretScanner,
    secret_warnings: &mut Vec<(String, usize)>,
) {
    for token in tokens {
        if let TokenTree::Literal(lit) = token
            && let syn::Lit::Str(lit_str) = syn::Lit::new(lit.clone())
        {
            let value = lit_str.value();
            if let Some(kind) = scanner.classify(&value) {
                secret_warnings.push((kind.to_string(), lit_str.span().start().line));
            }
        }
    }
}

fn scan_calls_in_slice(
    tokens: &[TokenTree],
    aliases: &HashMap<String, String>,
    entropy_violations: &mut Vec<(String, usize)>,
) {
    let mut idx = 0usize;
    while idx < tokens.len() {
        let Some((path, line, end_idx)) = parse_path_at(tokens, idx) else {
            idx += 1;
            continue;
        };
        if !is_method_or_decl_prefix(tokens, idx) && is_paren_group_at(tokens, end_idx) {
            let resolved = resolve_path(&path, aliases);
            if let Some(violation) = classify_entropy_call(&resolved) {
                entropy_violations.push((violation.to_string(), line));
            }
        }
        idx = end_idx;
    }
}

fn is_paren_group_at(tokens: &[TokenTree], idx: usize) -> bool {
    idx < tokens.len()
        && matches!(&tokens[idx], TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis)
}

fn is_method_or_decl_prefix(tokens: &[TokenTree], start: usize) -> bool {
    if start == 0 {
        return false;
    }
    match &tokens[start - 1] {
        TokenTree::Ident(kw) => kw == "fn",
        TokenTree::Punct(p) if p.as_char() == '.' => {
            !(start >= 2
                && matches!(
                    &tokens[start - 2],
                    TokenTree::Punct(prev) if prev.as_char() == '.' && prev.spacing() == Spacing::Joint
                ))
        }
        _ => false,
    }
}

fn is_double_colon(tokens: &[TokenTree], idx: usize) -> bool {
    idx + 1 < tokens.len()
        && matches!(
            &tokens[idx],
            TokenTree::Punct(p1) if p1.as_char() == ':' && p1.spacing() == Spacing::Joint
        )
        && matches!(&tokens[idx + 1], TokenTree::Punct(p2) if p2.as_char() == ':')
}

fn is_arrow_gt(tokens: &[TokenTree], idx: usize) -> bool {
    idx > 0
        && matches!(
            &tokens[idx - 1],
            TokenTree::Punct(prev)
                if (prev.as_char() == '-' || prev.as_char() == '=')
                    && prev.spacing() == Spacing::Joint
        )
}

fn skip_turbofish(tokens: &[TokenTree], lt_idx: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut idx = lt_idx;
    while idx < tokens.len() {
        if let TokenTree::Punct(p) = &tokens[idx] {
            if p.as_char() == '<' {
                depth += 1;
            } else if p.as_char() == '>' && !is_arrow_gt(tokens, idx) {
                depth -= 1;
                if depth == 0 {
                    return Some(idx + 1);
                }
            }
        }
        idx += 1;
    }
    None
}

fn parse_path_at(tokens: &[TokenTree], start: usize) -> Option<(syn::Path, usize, usize)> {
    let (leading_colon, start_line, first_ident, mut cursor) = if is_double_colon(tokens, start) {
        let TokenTree::Ident(ident) = tokens.get(start + 2)? else {
            return None;
        };
        (true, tokens[start].span().start().line, ident, start + 3)
    } else if let TokenTree::Ident(ident) = tokens.get(start)? {
        (false, ident.span().start().line, ident, start + 1)
    } else {
        return None;
    };

    let mut segments = syn::punctuated::Punctuated::new();
    segments.push(syn::PathSegment::from(first_ident.clone()));

    while is_double_colon(tokens, cursor) {
        let after_colons = cursor + 2;
        match tokens.get(after_colons) {
            Some(TokenTree::Ident(next_ident)) => {
                segments.push(syn::PathSegment::from(next_ident.clone()));
                cursor = after_colons + 1;
            }
            Some(TokenTree::Punct(p)) if p.as_char() == '<' => {
                let Some(after_turbo) = skip_turbofish(tokens, after_colons) else {
                    break;
                };
                cursor = after_turbo;
            }
            _ => break,
        }
    }

    let path = syn::Path {
        leading_colon: leading_colon.then(<syn::Token![::]>::default),
        segments,
    };
    Some((path, start_line, cursor))
}
