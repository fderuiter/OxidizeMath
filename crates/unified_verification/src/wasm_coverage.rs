use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, Field, ImplItem, ImplItemFn, Item, ItemFn, ItemMod, Lit, Meta, Token};

#[derive(Default)]
struct OutlineTestMod {
    ident: String,
    explicit_path: Option<String>,
}

#[derive(Default)]
struct FileWasmAstSummary {
    wasm_paths: usize,
    executable_wasm_tests: usize,
    outline_test_mods: Vec<OutlineTestMod>,
    orphaned_wasm_unit_tests: Vec<String>,
}

fn parse_nested_metas(list: &syn::MetaList) -> Vec<Meta> {
    list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
        .map(|items| items.into_iter().collect())
        .unwrap_or_default()
}

fn meta_mentions_test(meta: &Meta) -> bool {
    match meta {
        Meta::Path(path) => path.is_ident("test"),
        Meta::List(list) if list.path.is_ident("not") => false,
        Meta::List(list) if list.path.is_ident("all") || list.path.is_ident("any") => {
            parse_nested_metas(list).iter().any(meta_mentions_test)
        }
        _ => false,
    }
}

fn meta_requires_wasm32(meta: &Meta) -> bool {
    match meta {
        Meta::NameValue(nv) if nv.path.is_ident("target_arch") => {
            matches!(&nv.value, Expr::Lit(expr_lit) if matches!(&expr_lit.lit, Lit::Str(s) if s.value() == "wasm32"))
        }
        Meta::List(list) if list.path.is_ident("all") => {
            parse_nested_metas(list).iter().any(meta_requires_wasm32)
        }
        _ => false,
    }
}

fn attrs_require_wasm32(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .meta
                .require_list()
                .ok()
                .and_then(|list| list.parse_args::<Meta>().ok())
                .is_some_and(|meta| meta_requires_wasm32(&meta))
    })
}

fn attrs_mention_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        attr.meta
            .require_list()
            .ok()
            .and_then(|list| list.parse_args::<Meta>().ok())
            .is_some_and(|meta| meta_mentions_test(&meta))
    })
}

fn attrs_define_wasm_only_prod_path(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        attr.meta
            .require_list()
            .ok()
            .and_then(|list| list.parse_args::<Meta>().ok())
            .is_some_and(|meta| meta_requires_wasm32(&meta) && !meta_mentions_test(&meta))
    })
}

fn path_is_libtest_attr(path: &syn::Path) -> bool {
    let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    matches!(
        segs.iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        ["test"] | ["core", "prelude", "v1", "test"] | ["std", "prelude", "v1", "test"]
    )
}

fn meta_has_wasm_bindgen_test(meta: &Meta) -> bool {
    match meta {
        Meta::Path(path) => path
            .segments
            .last()
            .is_some_and(|seg| seg.ident == "wasm_bindgen_test"),
        Meta::List(list) if list.path.is_ident("cfg_attr") => parse_nested_metas(list)
            .into_iter()
            .skip(1)
            .any(|nested| meta_has_wasm_bindgen_test(&nested)),
        _ => false,
    }
}

fn attrs_have_wasm_bindgen_test(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|attr| meta_has_wasm_bindgen_test(&attr.meta))
}

fn attrs_have_orphaned_wasm32_libtest(attrs: &[Attribute], in_wasm32_scope: bool) -> bool {
    if attrs_have_wasm_bindgen_test(attrs) {
        return false;
    }
    let fn_requires_wasm = in_wasm32_scope || attrs_require_wasm32(attrs);
    let has_plain_test = attrs
        .iter()
        .any(|attr| matches!(&attr.meta, Meta::Path(path) if path_is_libtest_attr(path)));
    if fn_requires_wasm && has_plain_test {
        return true;
    }
    attrs.iter().any(|attr| {
        let Meta::List(list) = &attr.meta else {
            return false;
        };
        if !list.path.is_ident("cfg_attr") {
            return false;
        }
        let nested = parse_nested_metas(list);
        let Some((predicate, rest)) = nested.split_first() else {
            return false;
        };
        (in_wasm32_scope || meta_requires_wasm32(predicate))
            && rest
                .iter()
                .any(|m| matches!(m, Meta::Path(p) if path_is_libtest_attr(p)))
    })
}

fn extract_path_attr(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("path")
            && let Meta::NameValue(nv) = &attr.meta
            && let Expr::Lit(expr_lit) = &nv.value
            && let Lit::Str(lit_str) = &expr_lit.lit
        {
            return Some(lit_str.value());
        }
    }
    None
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

fn impl_item_attrs(item: &ImplItem) -> &[Attribute] {
    match item {
        ImplItem::Const(i) => &i.attrs,
        ImplItem::Fn(i) => &i.attrs,
        ImplItem::Type(i) => &i.attrs,
        ImplItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

fn expr_attrs(expr: &Expr) -> &[Attribute] {
    match expr {
        Expr::Block(e) => &e.attrs,
        Expr::Call(e) => &e.attrs,
        Expr::If(e) => &e.attrs,
        Expr::Match(e) => &e.attrs,
        Expr::MethodCall(e) => &e.attrs,
        _ => &[],
    }
}

struct WasmAstVisitor {
    in_test_scope: bool,
    in_wasm32_scope: bool,
    summary: FileWasmAstSummary,
}

impl<'ast> Visit<'ast> for WasmAstVisitor {
    fn visit_item(&mut self, node: &'ast Item) {
        if let Item::Mod(item_mod) = node {
            self.visit_mod_item(item_mod);
            return;
        }
        let attrs = item_attrs(node);
        let prev_test = self.in_test_scope;
        let prev_wasm = self.in_wasm32_scope;
        if attrs_mention_test(attrs) {
            self.in_test_scope = true;
        }
        if attrs_require_wasm32(attrs) {
            self.in_wasm32_scope = true;
        }
        if !self.in_test_scope && attrs_define_wasm_only_prod_path(attrs) {
            self.summary.wasm_paths += 1;
        }
        visit::visit_item(self, node);
        self.in_test_scope = prev_test;
        self.in_wasm32_scope = prev_wasm;
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if attrs_have_wasm_bindgen_test(&node.attrs) {
            self.summary.executable_wasm_tests += 1;
        } else if attrs_have_orphaned_wasm32_libtest(&node.attrs, self.in_wasm32_scope) {
            self.summary
                .orphaned_wasm_unit_tests
                .push(node.sig.ident.to_string());
        }
        visit::visit_item_fn(self, node);
    }

    fn visit_impl_item(&mut self, node: &'ast ImplItem) {
        let attrs = impl_item_attrs(node);
        let prev_test = self.in_test_scope;
        let prev_wasm = self.in_wasm32_scope;
        if attrs_mention_test(attrs) {
            self.in_test_scope = true;
        }
        if attrs_require_wasm32(attrs) {
            self.in_wasm32_scope = true;
        }
        if !self.in_test_scope && attrs_define_wasm_only_prod_path(attrs) {
            self.summary.wasm_paths += 1;
        }
        visit::visit_impl_item(self, node);
        self.in_test_scope = prev_test;
        self.in_wasm32_scope = prev_wasm;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        if attrs_have_wasm_bindgen_test(&node.attrs) {
            self.summary.executable_wasm_tests += 1;
        } else if attrs_have_orphaned_wasm32_libtest(&node.attrs, self.in_wasm32_scope) {
            self.summary
                .orphaned_wasm_unit_tests
                .push(node.sig.ident.to_string());
        }
        visit::visit_impl_item_fn(self, node);
    }

    fn visit_field(&mut self, node: &'ast Field) {
        if !self.in_test_scope && attrs_define_wasm_only_prod_path(&node.attrs) {
            self.summary.wasm_paths += 1;
        }
        visit::visit_field(self, node);
    }

    fn visit_expr(&mut self, node: &'ast Expr) {
        let attrs = expr_attrs(node);
        if attrs_mention_test(attrs) {
            let prev = self.in_test_scope;
            self.in_test_scope = true;
            visit::visit_expr(self, node);
            self.in_test_scope = prev;
            return;
        }
        if !self.in_test_scope && attrs_define_wasm_only_prod_path(attrs) {
            self.summary.wasm_paths += 1;
        }
        visit::visit_expr(self, node);
    }
}

impl WasmAstVisitor {
    fn visit_mod_item(&mut self, item_mod: &ItemMod) {
        let is_test_mod = self.in_test_scope || attrs_mention_test(&item_mod.attrs);
        let is_wasm_mod = self.in_wasm32_scope || attrs_require_wasm32(&item_mod.attrs);
        if !is_test_mod && attrs_define_wasm_only_prod_path(&item_mod.attrs) {
            self.summary.wasm_paths += 1;
        }
        if item_mod.content.is_none() && is_test_mod {
            self.summary.outline_test_mods.push(OutlineTestMod {
                ident: item_mod.ident.to_string(),
                explicit_path: extract_path_attr(&item_mod.attrs),
            });
            return;
        }
        let prev_test = self.in_test_scope;
        let prev_wasm = self.in_wasm32_scope;
        self.in_test_scope = is_test_mod;
        self.in_wasm32_scope = is_wasm_mod;
        visit::visit_item_mod(self, item_mod);
        self.in_test_scope = prev_test;
        self.in_wasm32_scope = prev_wasm;
    }
}

fn is_dedicated_wasm_test_path(filepath: &Path) -> bool {
    let stem = filepath.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    stem.ends_with("_wasm_tests") || stem == "wasm_tests"
}

fn summarize_ast(
    ast: &syn::File,
    initial_test_scope: bool,
    initial_wasm32_scope: bool,
) -> FileWasmAstSummary {
    let mut visitor = WasmAstVisitor {
        in_test_scope: initial_test_scope,
        in_wasm32_scope: initial_wasm32_scope || attrs_require_wasm32(&ast.attrs),
        summary: FileWasmAstSummary::default(),
    };
    visitor.visit_file(ast);
    visitor.summary
}

fn summarize_source(
    filepath: &Path,
    content: &str,
    initial_test_scope: bool,
) -> FileWasmAstSummary {
    let Ok(ast) = syn::parse_file(content) else {
        return FileWasmAstSummary::default();
    };
    summarize_ast(
        &ast,
        initial_test_scope,
        is_dedicated_wasm_test_path(filepath),
    )
}

fn resolve_outline_mod_file(parent_file: &Path, outline: &OutlineTestMod) -> Option<PathBuf> {
    let parent_dir = parent_file.parent().unwrap_or_else(|| Path::new("."));
    let stem = parent_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let is_root_like = matches!(stem, "lib" | "main" | "mod");
    let stem_dir = parent_dir.join(stem);

    let mut candidates = Vec::new();
    if let Some(rel) = &outline.explicit_path {
        candidates.push(parent_dir.join(rel));
        if !is_root_like {
            candidates.push(stem_dir.join(rel));
        }
    } else {
        if !is_root_like {
            candidates.push(stem_dir.join(format!("{}.rs", outline.ident)));
            candidates.push(stem_dir.join(&outline.ident).join("mod.rs"));
        }
        candidates.push(parent_dir.join(format!("{}.rs", outline.ident)));
        candidates.push(parent_dir.join(&outline.ident).join("mod.rs"));
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn outline_tree_has_wasm_tests(
    file: &Path,
    summary: &FileWasmAstSummary,
    visited: &mut HashSet<PathBuf>,
    depth_remaining: usize,
) -> bool {
    if summary.executable_wasm_tests > 0 {
        return true;
    }
    if depth_remaining == 0 {
        return false;
    }
    for outline in &summary.outline_test_mods {
        let Some(child_path) = resolve_outline_mod_file(file, outline) else {
            continue;
        };
        let canonical = child_path
            .canonicalize()
            .unwrap_or_else(|_| child_path.clone());
        if !visited.insert(canonical) {
            continue;
        }
        let Ok(child_src) = fs::read_to_string(&child_path) else {
            continue;
        };
        let child_summary = summarize_source(&child_path, &child_src, true);
        if outline_tree_has_wasm_tests(&child_path, &child_summary, visited, depth_remaining - 1) {
            return true;
        }
    }
    false
}

fn is_dedicated_test_source_path(filepath: &Path) -> bool {
    let stem = filepath.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem == "tests" || stem.ends_with("_tests") || stem.starts_with("tests_") {
        return true;
    }
    filepath
        .components()
        .any(|c| c.as_os_str().to_str() == Some("tests"))
}

pub fn find_orphaned_wasm_unit_tests_ast(filepath: &Path, ast: &syn::File) -> Vec<String> {
    let summary = summarize_ast(
        ast,
        is_dedicated_test_source_path(filepath),
        is_dedicated_wasm_test_path(filepath),
    );
    summary.orphaned_wasm_unit_tests
}

pub fn find_orphaned_wasm_unit_tests(filepath: &Path, content: &str) -> Vec<String> {
    let Ok(ast) = syn::parse_file(content) else {
        return Vec::new();
    };
    find_orphaned_wasm_unit_tests_ast(filepath, &ast)
}

pub fn count_file_wasm_coverage(filepath: &Path, content: &str) -> (usize, usize) {
    let in_test_file = is_dedicated_test_source_path(filepath);
    let summary = summarize_source(filepath, content, in_test_file);
    if summary.wasm_paths == 0 {
        return (0, 0);
    }
    let mut visited = HashSet::new();
    if let Ok(canon) = filepath.canonicalize() {
        visited.insert(canon);
    }
    let covered = if outline_tree_has_wasm_tests(filepath, &summary, &mut visited, 2) {
        summary.wasm_paths
    } else {
        0
    };
    (summary.wasm_paths, covered)
}

#[cfg(test)]
#[path = "wasm_coverage_tests.rs"]
mod tests;
