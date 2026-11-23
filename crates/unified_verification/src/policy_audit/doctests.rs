use super::dependencies::DependencyVisitor;
use proc_macro2::TokenStream;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

const COMPILED_RUSTDOC_MODIFIERS: &[&str] = &[
    "rust",
    "no_run",
    "should_panic",
    "compile_fail",
    "edition2015",
    "edition2018",
    "edition2021",
    "edition2024",
    "allow_fail",
    "test_harness",
    "standalone_crate",
];

const EXCLUDED_MD_DIRS: &[&str] = &[
    "target",
    ".git",
    "node_modules",
    ".cargo",
    ".rustup",
    "dist",
    "artifacts",
];

fn is_compiled_rustdoc_info(info: &str) -> bool {
    let tokens: Vec<&str> = info
        .split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return true;
    }
    if tokens
        .iter()
        .any(|t| *t == "ignore" || t.starts_with("ignore-"))
    {
        return false;
    }
    tokens
        .iter()
        .all(|t| COMPILED_RUSTDOC_MODIFIERS.contains(t))
}

fn parse_fence_delimiter(trimmed: &str) -> Option<(char, usize, &str)> {
    let ch = trimmed.chars().next()?;
    if ch != '`' && ch != '~' {
        return None;
    }
    let count = trimmed.chars().take_while(|&c| c == ch).count();
    if count < 3 {
        return None;
    }
    let rest = &trimmed[count..];
    if ch == '`' && rest.contains('`') {
        return None;
    }
    Some((ch, count, rest))
}

fn is_closing_fence(trimmed: &str, marker_char: char, min_len: usize) -> bool {
    let count = trimmed.chars().take_while(|&c| c == marker_char).count();
    count >= min_len && trimmed[count..].trim().is_empty()
}

fn unhide_rustdoc_line(line: &str) -> String {
    let leading_ws_bytes = line
        .bytes()
        .take_while(|b| *b == b' ' || *b == b'\t')
        .count();
    let trimmed = &line[leading_ws_bytes..];
    if trimmed == "#" {
        String::new()
    } else if let Some(rest) = trimmed.strip_prefix("# ") {
        rest.to_string()
    } else if let Some(rest) = trimmed.strip_prefix("##") {
        format!("{}#{}", &line[..leading_ws_bytes], rest)
    } else {
        line.to_string()
    }
}

pub(crate) fn extract_compiled_rustdoc_blocks(markdown: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut active_fence: Option<(char, usize, bool)> = None;
    let mut current_lines: Vec<String> = Vec::new();

    for raw_line in markdown.lines() {
        let trimmed = raw_line.trim_start();
        if let Some((ch, len, is_compiled)) = active_fence {
            if is_closing_fence(trimmed, ch, len) {
                if is_compiled {
                    blocks.push(current_lines.join("\n"));
                }
                current_lines.clear();
                active_fence = None;
            } else if is_compiled {
                current_lines.push(unhide_rustdoc_line(raw_line));
            }
        } else if let Some((ch, len, info)) = parse_fence_delimiter(trimmed) {
            active_fence = Some((ch, len, is_compiled_rustdoc_info(info.trim())));
            current_lines.clear();
        }
    }

    blocks
}

fn visit_doctest_snippet(snippet: &str, dep_visitor: &mut DependencyVisitor) {
    if snippet.trim().is_empty() {
        return;
    }
    if let Ok(ast) = syn::parse_file(snippet) {
        dep_visitor.visit_file(&ast);
        return;
    }
    let wrapped = format!("{{\n{snippet}\n}}");
    if let Ok(block) = syn::parse_str::<syn::Block>(&wrapped) {
        dep_visitor.visit_block(&block);
        return;
    }
    if let Ok(tokens) = snippet.parse::<TokenStream>() {
        dep_visitor.scan_token_stream(tokens);
    }
}

pub(crate) fn audit_markdown_text(markdown: &str, dep_visitor: &mut DependencyVisitor) {
    for block in extract_compiled_rustdoc_blocks(markdown) {
        visit_doctest_snippet(&block, dep_visitor);
    }
}

fn resolve_include_str_expr(mac: &syn::Macro, rs_dir: &Path, member_dir: &Path) -> Option<PathBuf> {
    if let Ok(lit) = syn::parse2::<syn::LitStr>(mac.tokens.clone()) {
        return Some(rs_dir.join(lit.value()));
    }
    let inner = syn::parse2::<syn::Macro>(mac.tokens.clone()).ok()?;
    if !inner.path.is_ident("concat") {
        return None;
    }
    let args = inner
        .parse_body_with(syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated)
        .ok()?;
    let mut uses_manifest_dir = false;
    let mut suffix = String::new();
    for arg in args {
        match arg {
            syn::Expr::Macro(em) if em.mac.path.is_ident("env") => {
                if let Ok(var) = syn::parse2::<syn::LitStr>(em.mac.tokens)
                    && var.value() == "CARGO_MANIFEST_DIR"
                {
                    uses_manifest_dir = true;
                }
            }
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) => suffix.push_str(&s.value()),
            _ => return None,
        }
    }
    uses_manifest_dir.then(|| member_dir.join(suffix.trim_start_matches('/')))
}

struct DocCollector<'a> {
    rs_dir: &'a Path,
    member_dir: &'a Path,
    doc_chunks: Vec<String>,
    included_files: Vec<PathBuf>,
}

impl<'a> DocCollector<'a> {
    fn collect_attrs(&mut self, attrs: &[syn::Attribute]) {
        let mut current_doc = Vec::new();
        for attr in attrs {
            self.extract_from_meta(&attr.meta, &mut current_doc);
        }
        if !current_doc.is_empty() {
            self.doc_chunks.push(current_doc.join("\n"));
        }
    }

    fn extract_from_meta(&mut self, meta: &syn::Meta, current_doc: &mut Vec<String>) {
        match meta {
            syn::Meta::NameValue(nv) if nv.path.is_ident("doc") => match &nv.value {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) => current_doc.push(s.value()),
                syn::Expr::Macro(em) if em.mac.path.is_ident("include_str") => {
                    if !current_doc.is_empty() {
                        self.doc_chunks.push(std::mem::take(current_doc).join("\n"));
                    }
                    if let Some(p) = resolve_include_str_expr(&em.mac, self.rs_dir, self.member_dir)
                    {
                        self.included_files.push(p);
                    }
                }
                _ => {}
            },
            syn::Meta::List(list) if list.path.is_ident("cfg_attr") => {
                if let Ok(nested) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) {
                    for item in nested.into_iter().skip(1) {
                        self.extract_from_meta(&item, current_doc);
                    }
                }
            }
            _ => {}
        }
    }
}

impl<'ast> Visit<'ast> for DocCollector<'_> {
    fn visit_file(&mut self, i: &'ast syn::File) {
        self.collect_attrs(&i.attrs);
        syn::visit::visit_file(self, i);
    }
    fn visit_item(&mut self, i: &'ast syn::Item) {
        let attrs = item_attrs(i);
        self.collect_attrs(attrs);
        syn::visit::visit_item(self, i);
    }
    fn visit_impl_item(&mut self, i: &'ast syn::ImplItem) {
        self.collect_attrs(impl_item_attrs(i));
        syn::visit::visit_impl_item(self, i);
    }
    fn visit_trait_item(&mut self, i: &'ast syn::TraitItem) {
        self.collect_attrs(trait_item_attrs(i));
        syn::visit::visit_trait_item(self, i);
    }
    fn visit_foreign_item(&mut self, i: &'ast syn::ForeignItem) {
        self.collect_attrs(foreign_item_attrs(i));
        syn::visit::visit_foreign_item(self, i);
    }
    fn visit_variant(&mut self, i: &'ast syn::Variant) {
        self.collect_attrs(&i.attrs);
        syn::visit::visit_variant(self, i);
    }
    fn visit_field(&mut self, i: &'ast syn::Field) {
        self.collect_attrs(&i.attrs);
        syn::visit::visit_field(self, i);
    }
}

fn item_attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Const(i) => &i.attrs,
        syn::Item::Enum(i) => &i.attrs,
        syn::Item::ExternCrate(i) => &i.attrs,
        syn::Item::Fn(i) => &i.attrs,
        syn::Item::ForeignMod(i) => &i.attrs,
        syn::Item::Impl(i) => &i.attrs,
        syn::Item::Macro(i) => &i.attrs,
        syn::Item::Mod(i) => &i.attrs,
        syn::Item::Static(i) => &i.attrs,
        syn::Item::Struct(i) => &i.attrs,
        syn::Item::Trait(i) => &i.attrs,
        syn::Item::TraitAlias(i) => &i.attrs,
        syn::Item::Type(i) => &i.attrs,
        syn::Item::Union(i) => &i.attrs,
        syn::Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

fn impl_item_attrs(item: &syn::ImplItem) -> &[syn::Attribute] {
    match item {
        syn::ImplItem::Const(i) => &i.attrs,
        syn::ImplItem::Fn(i) => &i.attrs,
        syn::ImplItem::Type(i) => &i.attrs,
        syn::ImplItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

fn trait_item_attrs(item: &syn::TraitItem) -> &[syn::Attribute] {
    match item {
        syn::TraitItem::Const(i) => &i.attrs,
        syn::TraitItem::Fn(i) => &i.attrs,
        syn::TraitItem::Type(i) => &i.attrs,
        syn::TraitItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

fn foreign_item_attrs(item: &syn::ForeignItem) -> &[syn::Attribute] {
    match item {
        syn::ForeignItem::Fn(i) => &i.attrs,
        syn::ForeignItem::Static(i) => &i.attrs,
        syn::ForeignItem::Type(i) => &i.attrs,
        syn::ForeignItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

pub(crate) fn audit_rs_doctests(
    ast: &syn::File,
    rs_path: &Path,
    member_dir: &Path,
    visited_md: &mut HashSet<PathBuf>,
    dep_visitor: &mut DependencyVisitor,
) {
    let rs_dir = rs_path.parent().unwrap_or(member_dir);
    let mut collector = DocCollector {
        rs_dir,
        member_dir,
        doc_chunks: Vec::new(),
        included_files: Vec::new(),
    };
    collector.visit_file(ast);

    for chunk in &collector.doc_chunks {
        audit_markdown_text(chunk, dep_visitor);
    }

    for md_path in collector.included_files {
        let key = fs::canonicalize(&md_path).unwrap_or_else(|_| md_path.clone());
        if visited_md.insert(key)
            && let Ok(content) = fs::read_to_string(&md_path)
        {
            audit_markdown_text(&content, dep_visitor);
        }
    }
}

pub(crate) fn is_markdown_tests_crate(member: &str, package_name: Option<&str>) -> bool {
    let norm = member.replace('\\', "/");
    let trimmed = norm.trim_end_matches('/');
    package_name == Some("markdown_tests")
        || trimmed == "crates/markdown_tests"
        || trimmed.ends_with("/crates/markdown_tests")
}

fn resolve_markdown_workspace_root(member_dir: &Path) -> PathBuf {
    let Some(parent) = member_dir.parent() else {
        return PathBuf::from(".");
    };
    let root = if parent.file_name().and_then(|s| s.to_str()) == Some("crates") {
        parent.parent().unwrap_or(parent)
    } else {
        parent
    };
    if root.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        root.to_path_buf()
    }
}

fn collect_workspace_markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default();
            if EXCLUDED_MD_DIRS.contains(&name.as_ref()) || name.starts_with("cargo-machete") {
                continue;
            }
            collect_workspace_markdown_files(&path, out);
        } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("md") {
            out.push(path);
        }
    }
}

pub(crate) fn audit_workspace_markdown_files(
    member_dir: &Path,
    visited_md: &mut HashSet<PathBuf>,
    dep_visitor: &mut DependencyVisitor,
) {
    let ws_root = resolve_markdown_workspace_root(member_dir);
    let mut md_files = Vec::new();
    collect_workspace_markdown_files(&ws_root, &mut md_files);
    for md_path in md_files {
        let key = fs::canonicalize(&md_path).unwrap_or_else(|_| md_path.clone());
        if visited_md.insert(key)
            && let Ok(content) = fs::read_to_string(&md_path)
        {
            audit_markdown_text(&content, dep_visitor);
        }
    }
}
