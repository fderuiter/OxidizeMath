use super::ApiItem;
use super::visitor::ApiVisitor;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_MODULES_PER_CRATE: usize = 10_000;
const MAX_SYNTAX_DEPTH: usize = 4_096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ExportFilter {
    AllPublic,
    Selected(BTreeSet<String>),
}

impl ExportFilter {
    pub(crate) fn allows_ident(&self, ident: &str) -> bool {
        match self {
            Self::AllPublic => true,
            Self::Selected(names) => names.contains(ident),
        }
    }

    pub(crate) fn merge(&mut self, other: Self) -> bool {
        match (self, other) {
            (Self::AllPublic, _) => false,
            (this, Self::AllPublic) => {
                *this = Self::AllPublic;
                true
            }
            (Self::Selected(existing), Self::Selected(incoming)) => {
                let prev_len = existing.len();
                existing.extend(incoming);
                existing.len() > prev_len
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum UseTarget {
    Name(String),
    Rename { ident: String, alias: String },
    Glob,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct UseLeaf {
    pub(crate) prefix: Vec<String>,
    pub(crate) target: UseTarget,
}

impl UseLeaf {
    pub(crate) fn exported_ident(&self) -> Option<&str> {
        match &self.target {
            UseTarget::Name(name) => Some(name.as_str()),
            UseTarget::Rename { alias, .. } => Some(alias.as_str()),
            UseTarget::Glob => None,
        }
    }

    pub(crate) fn format_signature(&self) -> String {
        let prefix_tokens: Vec<proc_macro2::Ident> = self
            .prefix
            .iter()
            .map(|seg| quote::format_ident!("{}", seg))
            .collect();
        match &self.target {
            UseTarget::Name(name) => {
                let id = quote::format_ident!("{}", name);
                if prefix_tokens.is_empty() {
                    quote::quote!(pub use #id).to_string()
                } else {
                    quote::quote!(pub use #(#prefix_tokens ::)* #id).to_string()
                }
            }
            UseTarget::Rename { ident, alias } => {
                let id = quote::format_ident!("{}", ident);
                let ren = quote::format_ident!("{}", alias);
                if prefix_tokens.is_empty() {
                    quote::quote!(pub use #id as #ren).to_string()
                } else {
                    quote::quote!(pub use #(#prefix_tokens ::)* #id as #ren).to_string()
                }
            }
            UseTarget::Glob => {
                if prefix_tokens.is_empty() {
                    quote::quote!(pub use *).to_string()
                } else {
                    quote::quote!(pub use #(#prefix_tokens ::)* *).to_string()
                }
            }
        }
    }
}

pub(crate) fn flatten_use_tree(root: &syn::UseTree) -> Vec<UseLeaf> {
    let mut leaves = Vec::new();
    let mut stack: Vec<(Vec<String>, &syn::UseTree)> = vec![(Vec::new(), root)];
    let mut steps = 0;

    while let Some((mut prefix, node)) = stack.pop() {
        if steps >= MAX_SYNTAX_DEPTH {
            break;
        }
        steps += 1;

        match node {
            syn::UseTree::Path(p) => {
                prefix.push(p.ident.to_string());
                stack.push((prefix, &p.tree));
            }
            syn::UseTree::Name(n) => {
                leaves.push(UseLeaf {
                    prefix,
                    target: UseTarget::Name(n.ident.to_string()),
                });
            }
            syn::UseTree::Rename(r) => {
                leaves.push(UseLeaf {
                    prefix,
                    target: UseTarget::Rename {
                        ident: r.ident.to_string(),
                        alias: r.rename.to_string(),
                    },
                });
            }
            syn::UseTree::Glob(_) => {
                leaves.push(UseLeaf {
                    prefix,
                    target: UseTarget::Glob,
                });
            }
            syn::UseTree::Group(g) => {
                for item in g.items.iter().rev() {
                    stack.push((prefix.clone(), item));
                }
            }
        }
    }
    leaves
}

fn meta_requires_test(meta: &syn::Meta) -> bool {
    let mut stack = vec![meta.clone()];
    let mut steps = 0;

    while let Some(curr) = stack.pop() {
        if steps >= MAX_SYNTAX_DEPTH {
            break;
        }
        steps += 1;

        match curr {
            syn::Meta::Path(p) if p.is_ident("test") => return true,
            syn::Meta::List(list) if list.path.is_ident("all") => {
                if let Ok(nested) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) {
                    stack.extend(nested);
                }
            }
            syn::Meta::List(list) if list.path.is_ident("any") => {
                if let Ok(nested) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) && !nested.is_empty()
                    && nested.iter().all(meta_is_direct_test)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn meta_is_direct_test(meta: &syn::Meta) -> bool {
    matches!(meta, syn::Meta::Path(p) if p.is_ident("test"))
}

pub(crate) fn has_cfg_test(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if !attr.path().is_ident("cfg") {
            continue;
        }
        let Ok(nested) = attr.parse_args_with(
            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
        ) else {
            continue;
        };
        if nested.iter().any(meta_requires_test) {
            return true;
        }
    }
    false
}

pub(crate) fn collect_facade_exports(
    items: &[syn::Item],
    parent_filter: &ExportFilter,
) -> BTreeMap<String, ExportFilter> {
    let mut facades: BTreeMap<String, ExportFilter> = BTreeMap::new();

    for item in items {
        let syn::Item::Use(item_use) = item else {
            continue;
        };
        if !matches!(item_use.vis, syn::Visibility::Public(_)) || has_cfg_test(&item_use.attrs) {
            continue;
        }
        for leaf in flatten_use_tree(&item_use.tree) {
            if !leaf_allowed_by_filter(&leaf, parent_filter) {
                continue;
            }
            register_leaf_facade(&leaf, &mut facades);
        }
    }
    facades
}

pub(crate) fn leaf_allowed_by_filter(leaf: &UseLeaf, filter: &ExportFilter) -> bool {
    match filter {
        ExportFilter::AllPublic => true,
        ExportFilter::Selected(allowed) => leaf
            .exported_ident()
            .is_some_and(|ident| allowed.contains(ident)),
    }
}

fn register_leaf_facade(leaf: &UseLeaf, facades: &mut BTreeMap<String, ExportFilter>) {
    let stripped: &[String] = if leaf.prefix.first().map(String::as_str) == Some("self") {
        &leaf.prefix[1..]
    } else {
        &leaf.prefix[..]
    };

    if stripped.is_empty() {
        let mod_ident = match &leaf.target {
            UseTarget::Name(name) => Some(name.as_str()),
            UseTarget::Rename { ident, .. } => Some(ident.as_str()),
            UseTarget::Glob => None,
        };
        if let Some(name) = mod_ident {
            upsert_facade(facades, name.to_string(), ExportFilter::AllPublic);
        }
        return;
    }

    let first_mod = &stripped[0];
    if first_mod == "crate" || first_mod == "super" {
        return;
    }

    let child_filter = if stripped.len() > 1 {
        let mut set = BTreeSet::new();
        set.insert(stripped[1].clone());
        ExportFilter::Selected(set)
    } else {
        match &leaf.target {
            UseTarget::Glob => ExportFilter::AllPublic,
            UseTarget::Name(name) => {
                let mut set = BTreeSet::new();
                set.insert(name.clone());
                ExportFilter::Selected(set)
            }
            UseTarget::Rename { ident, .. } => {
                let mut set = BTreeSet::new();
                set.insert(ident.clone());
                ExportFilter::Selected(set)
            }
        }
    };
    upsert_facade(facades, first_mod.clone(), child_filter);
}

fn upsert_facade(
    facades: &mut BTreeMap<String, ExportFilter>,
    mod_name: String,
    filter: ExportFilter,
) {
    if let Some(existing) = facades.get_mut(&mod_name) {
        existing.merge(filter);
    } else {
        facades.insert(mod_name, filter);
    }
}

fn extract_path_attr(attrs: &[syn::Attribute]) -> Option<PathBuf> {
    for attr in attrs {
        if !attr.path().is_ident("path") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) = &nv.value
        {
            return Some(PathBuf::from(s.value()));
        }
    }
    None
}

pub(crate) fn child_module_dir_for_file(file_path: &Path) -> PathBuf {
    let parent = file_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = file_path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if file_name == "lib.rs" || file_name == "mod.rs" {
        parent.to_path_buf()
    } else {
        let stem = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        parent.join(stem)
    }
}

pub(crate) fn resolve_child_mod_file(
    module_dir: &Path,
    ident: &str,
    attrs: &[syn::Attribute],
) -> Option<PathBuf> {
    if let Some(custom_rel) = extract_path_attr(attrs) {
        let candidate = module_dir.join(custom_rel);
        return candidate.is_file().then_some(candidate);
    }
    let direct = module_dir.join(format!("{ident}.rs"));
    if direct.is_file() {
        return Some(direct);
    }
    let nested = module_dir.join(ident).join("mod.rs");
    nested.is_file().then_some(nested)
}

pub(crate) fn traverse_crate_root(
    crate_lib: &Path,
    workspace_root: &Path,
    apis: &mut BTreeMap<String, Vec<ApiItem>>,
) {
    if !crate_lib.is_file() {
        return;
    }

    let mut visited: BTreeMap<PathBuf, ExportFilter> = BTreeMap::new();
    let mut worklist: Vec<PathBuf> = vec![crate_lib.to_path_buf()];
    visited.insert(crate_lib.to_path_buf(), ExportFilter::AllPublic);

    let mut iterations = 0;
    while let Some(file_path) = worklist.pop() {
        if iterations >= MAX_MODULES_PER_CRATE {
            break;
        }
        iterations += 1;

        let Some(active_filter) = visited.get(&file_path).cloned() else {
            continue;
        };
        let Ok(content) = fs::read_to_string(&file_path) else {
            continue;
        };
        let Ok(ast) = syn::parse_file(&content) else {
            continue;
        };

        let mod_dir = child_module_dir_for_file(&file_path);
        let mut visitor = ApiVisitor::default();
        let child_files = visitor.visit_module_ast(&ast, &active_filter, &mod_dir);

        if !visitor.items.is_empty() {
            let mut items = visitor.items;
            items.sort();
            items.dedup();
            let rel = file_path.strip_prefix(workspace_root).unwrap_or(&file_path);
            let norm_path = rel.to_string_lossy().replace('\\', "/");
            apis.insert(norm_path, items);
        }

        for (child_path, child_filter) in child_files {
            let should_enqueue = if let Some(prev) = visited.get_mut(&child_path) {
                prev.merge(child_filter)
            } else {
                visited.insert(child_path.clone(), child_filter);
                true
            };
            if should_enqueue {
                worklist.push(child_path);
            }
        }
    }
}
