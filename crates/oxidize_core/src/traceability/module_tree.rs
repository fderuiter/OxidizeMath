//! Standard Rust module tree and `#[path = "..."]` submodule resolution helpers.

pub(super) struct ExternalSubmodule {
    pub ident: String,
    pub custom_path: Option<String>,
}

#[derive(Default)]
struct SubmoduleCollector {
    submodules: Vec<ExternalSubmodule>,
}

fn extract_path_attr(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("path")
            && let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) = &nv.value
        {
            return Some(s.value());
        }
    }
    None
}

impl<'ast> syn::visit::Visit<'ast> for SubmoduleCollector {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if node.content.is_none() {
            self.submodules.push(ExternalSubmodule {
                ident: node.ident.to_string(),
                custom_path: extract_path_attr(&node.attrs),
            });
        }
        syn::visit::visit_item_mod(self, node);
    }
}

pub(super) fn collect_external_submodules(ast: Option<&syn::File>) -> Vec<ExternalSubmodule> {
    let Some(ast) = ast else {
        return Vec::new();
    };
    let mut collector = SubmoduleCollector::default();
    syn::visit::Visit::visit_file(&mut collector, ast);
    collector.submodules
}

pub(super) fn module_parent_dir(norm_file_path: &str) -> String {
    norm_file_path
        .rsplit_once('/')
        .map_or_else(String::new, |(parent, _)| parent.to_string())
}

pub(super) fn module_child_dir(norm_file_path: &str) -> String {
    let mut dir_parts: Vec<&str> = norm_file_path.split('/').collect();
    let is_mod_rs = norm_file_path.ends_with("mod.rs")
        || norm_file_path.ends_with("lib.rs")
        || norm_file_path.ends_with("main.rs");
    if is_mod_rs {
        dir_parts.pop();
    } else if let Some(stem) = dir_parts.pop().map(|s| s.strip_suffix(".rs").unwrap_or(s)) {
        dir_parts.push(stem);
    }
    dir_parts.join("/")
}
