//! Docstring resolution and 4-part Markdown/LaTeX/citation contract parser for `math_contracts`.

use crate::ast_helpers::format_file_line;
use crate::policy_audit::doctests::resolve_include_str_expr;
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

pub(super) struct ContractPatterns {
    citation_re: Regex,
    md_link_re: Regex,
}

impl ContractPatterns {
    pub(super) fn new() -> Self {
        Self {
            citation_re: Regex::new(
                r"(?i)(?:\b10\.\d{4,9}/[-._;()/:A-Z0-9]+|\bdoi\s*:|\bdoi\.org/|\barxiv\s*:\s*\d{4}\.\d{4,5}|\barxiv\.org/)",
            )
            .expect("Valid citation regex"),
            md_link_re: Regex::new(r"\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
                .expect("Valid markdown link regex"),
        }
    }
}

pub(super) struct FileAuditEnv<'a> {
    pub(super) workspace_root: &'a Path,
    pub(super) crate_dir: &'a Path,
    pub(super) rs_dir: &'a Path,
    pub(super) display_file: &'a str,
    pub(super) patterns: &'a ContractPatterns,
}

pub(super) fn has_embed_theory(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path()
            .segments
            .last()
            .is_some_and(|s| s.ident == "embed_theory")
    })
}

struct ResolvedDoc {
    markdown: String,
    source_dirs: Vec<PathBuf>,
    include_errors: Vec<String>,
}

fn resolve_item_doc(attrs: &[syn::Attribute], env: &FileAuditEnv<'_>, line: usize) -> ResolvedDoc {
    let mut lines = Vec::new();
    let mut source_dirs = vec![env.rs_dir.to_path_buf()];
    let mut include_errors = Vec::new();
    for attr in attrs {
        collect_doc_meta(
            &attr.meta,
            env,
            line,
            &mut lines,
            &mut source_dirs,
            &mut include_errors,
        );
    }
    ResolvedDoc {
        markdown: lines.join("\n"),
        source_dirs,
        include_errors,
    }
}

fn collect_doc_meta(
    meta: &syn::Meta,
    env: &FileAuditEnv<'_>,
    line: usize,
    lines: &mut Vec<String>,
    source_dirs: &mut Vec<PathBuf>,
    include_errors: &mut Vec<String>,
) {
    match meta {
        syn::Meta::NameValue(nv) if nv.path.is_ident("doc") => match &nv.value {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) => lines.push(s.value()),
            syn::Expr::Macro(em) if em.mac.path.is_ident("include_str") => {
                match resolve_include_str_expr(&em.mac, env.rs_dir, env.crate_dir) {
                    Some(inc_path) => match fs::read_to_string(&inc_path) {
                        Ok(text) => {
                            if let Some(parent) = inc_path.parent() {
                                source_dirs.push(parent.to_path_buf());
                            }
                            lines.push(text);
                        }
                        Err(err) => include_errors.push(format!(
                            "{}: unreadable #[doc = include_str!(...)] file `{}` ({err})",
                            format_file_line(env.display_file, line),
                            inc_path.display()
                        )),
                    },
                    None => include_errors.push(format!(
                        "{}: unresolvable #[doc = include_str!(...)] expression",
                        format_file_line(env.display_file, line)
                    )),
                }
            }
            _ => {}
        },
        syn::Meta::List(list) if list.path.is_ident("cfg_attr") => {
            if let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) {
                for nested_meta in nested.into_iter().skip(1) {
                    collect_doc_meta(
                        &nested_meta,
                        env,
                        line,
                        lines,
                        source_dirs,
                        &mut *include_errors,
                    );
                }
            }
        }
        _ => {}
    }
}

pub(super) fn verify_item_contract(
    attrs: &[syn::Attribute],
    line: usize,
    item_label: &str,
    env: &FileAuditEnv<'_>,
    violations: &mut Vec<String>,
) {
    let resolved = resolve_item_doc(attrs, env, line);
    violations.extend(resolved.include_errors);
    let prose = strip_fenced_code_blocks(&resolved.markdown);
    let mut missing = Vec::new();
    if !has_embed_theory(attrs) && !contains_latex_math(&prose) {
        missing.push("governing LaTeX equation (`$...$` / `$$...$$` or `#[embed_theory]`)");
    }
    if !has_valid_citation(
        &prose,
        &resolved.source_dirs,
        env,
        line,
        item_label,
        violations,
    ) {
        missing.push("peer-reviewed `DOI`/`arXiv` identifier or `docs/math_specs/` citation");
    }
    if !has_nonempty_section(&prose, "Stability") {
        missing.push("`# Stability` section");
    }
    if !has_nonempty_section(&prose, "Domain of Validity") {
        missing.push("`# Domain of Validity` section");
    }
    if !missing.is_empty() {
        violations.push(format!(
            "{}: Mathematical contract violation on `{item_label}`: missing {}",
            format_file_line(env.display_file, line),
            missing.join(", ")
        ));
    }
}

fn strip_fenced_code_blocks(markdown: &str) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn contains_latex_math(prose: &str) -> bool {
    let cleaned = strip_inline_backticks(prose);
    if let Some(first) = cleaned.find("$$")
        && let Some(rest) = cleaned.get(first + 2..)
        && let Some(end) = rest.find("$$")
        && !rest[..end].trim().is_empty()
    {
        return true;
    }
    for line in cleaned.lines() {
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'$'
                && let Some(end_rel) = line[i + 1..].find('$')
            {
                let inner = line[i + 1..i + 1 + end_rel].trim();
                if !inner.is_empty() && !inner.chars().all(|c| c.is_ascii_digit()) {
                    return true;
                }
                i += end_rel + 2;
                continue;
            }
            i += 1;
        }
    }
    false
}

fn strip_inline_backticks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_code = false;
    for ch in text.chars() {
        if ch == '`' {
            in_code = !in_code;
        } else if !in_code {
            out.push(ch);
        }
    }
    out
}

fn has_valid_citation(
    prose: &str,
    source_dirs: &[PathBuf],
    env: &FileAuditEnv<'_>,
    line: usize,
    item_label: &str,
    violations: &mut Vec<String>,
) -> bool {
    let mut has_spec_ref = false;
    for cap in env.patterns.md_link_re.captures_iter(prose) {
        let target = cap.get(1).map_or("", |m| m.as_str());
        let path_part = target.split('#').next().unwrap_or("");
        if path_part.is_empty()
            || path_part.starts_with("http://")
            || path_part.starts_with("https://")
            || path_part.contains("::")
        {
            continue;
        }
        let resolved_spec = resolve_math_spec_target(path_part, source_dirs, env.workspace_root);
        if path_part.contains("math_specs/") || resolved_spec.is_some() {
            if let Some(existing) = resolved_spec {
                if existing.is_file() {
                    has_spec_ref = true;
                }
            } else {
                violations.push(format!(
                    "{}: `{item_label}` links to non-existent math specification `{target}`",
                    format_file_line(env.display_file, line)
                ));
            }
        }
    }
    has_spec_ref || env.patterns.citation_re.is_match(prose)
}

fn resolve_math_spec_target(
    path_part: &str,
    source_dirs: &[PathBuf],
    workspace_root: &Path,
) -> Option<PathBuf> {
    for base in source_dirs {
        let candidate = base.join(path_part);
        if candidate.is_file() {
            let canon_str = fs::canonicalize(&candidate)
                .unwrap_or_else(|_| candidate.clone())
                .to_string_lossy()
                .replace('\\', "/");
            if canon_str.contains("/docs/math_specs/") {
                return Some(candidate);
            }
        }
    }
    let root_candidate = workspace_root.join(path_part.trim_start_matches('/'));
    if root_candidate.is_file()
        && path_part
            .trim_start_matches('/')
            .starts_with("docs/math_specs/")
    {
        return Some(root_candidate);
    }
    None
}

fn has_nonempty_section(prose: &str, heading_title: &str) -> bool {
    let mut in_target = false;
    for raw_line in prose.lines() {
        let trimmed = raw_line.trim();
        if let Some(title) = parse_markdown_heading(trimmed) {
            if in_target {
                return false;
            }
            if title.eq_ignore_ascii_case(heading_title) {
                in_target = true;
            }
            continue;
        }
        if in_target && !trimmed.is_empty() {
            return true;
        }
    }
    false
}

fn parse_markdown_heading(trimmed: &str) -> Option<&str> {
    let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
    if !(1..=3).contains(&hashes) {
        return None;
    }
    let rest = trimmed[hashes..].strip_prefix(' ')?.trim();
    Some(rest.trim_end_matches(':').trim())
}
