use std::collections::HashMap;
use syn::spanned::Spanned;

pub(super) fn flatten_use_tree(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    mappings: &mut HashMap<String, String>,
    violations: &mut Vec<(String, usize)>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            flatten_use_tree(&p.tree, prefix, mappings, violations);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let mut full_path = prefix.clone();
            full_path.push(n.ident.to_string());
            let joined = full_path.join("::");
            if is_std_rng_path(&joined) {
                violations.push(("StdRng".to_string(), n.ident.span().start().line));
            }
            mappings.insert(n.ident.to_string(), joined);
        }
        syn::UseTree::Rename(r) => {
            let mut full_path = prefix.clone();
            full_path.push(r.ident.to_string());
            let joined = full_path.join("::");
            if is_std_rng_path(&joined) {
                violations.push(("StdRng".to_string(), r.ident.span().start().line));
            }
            mappings.insert(r.rename.to_string(), joined);
        }
        syn::UseTree::Group(g) => {
            for item in &g.items {
                flatten_use_tree(item, prefix, mappings, violations);
            }
        }
        syn::UseTree::Glob(_) => {}
    }
}

pub(crate) fn resolve_path(p: &syn::Path, mappings: &HashMap<String, String>) -> String {
    let mut parts = Vec::new();
    for (i, seg) in p.segments.iter().enumerate() {
        if i == 0 && p.leading_colon.is_none() {
            let first = seg.ident.to_string();
            if let Some(mapped) = mappings.get(&first) {
                parts.push(mapped.clone());
            } else {
                parts.push(first);
            }
        } else {
            parts.push(seg.ident.to_string());
        }
    }
    parts.join("::")
}

pub(super) fn is_std_rng_path(resolved: &str) -> bool {
    resolved == "StdRng"
        || resolved.ends_with("::StdRng")
        || resolved.starts_with("StdRng::")
        || resolved.contains("::StdRng::")
}

pub(super) fn is_oxidize_rng_default_path(resolved: &str) -> bool {
    resolved == "OxidizeRng::default" || resolved.ends_with("::OxidizeRng::default")
}

pub(super) fn is_qself_oxidize_rng_default(
    p: &syn::ExprPath,
    mappings: &HashMap<String, String>,
) -> bool {
    let Some(qself) = &p.qself else {
        return false;
    };
    let Some(last) = p.path.segments.last() else {
        return false;
    };
    if last.ident != "default" {
        return false;
    }
    let syn::Type::Path(tp) = &*qself.ty else {
        return false;
    };
    let ty_resolved = resolve_path(&tp.path, mappings);
    ty_resolved == "OxidizeRng" || ty_resolved.ends_with("::OxidizeRng")
}

pub(super) fn check_expr_path(
    p: &syn::ExprPath,
    mappings: &HashMap<String, String>,
    allow_default_rng: bool,
    violations: &mut Vec<(String, usize)>,
) {
    let line = p.span().start().line;
    if !allow_default_rng && is_qself_oxidize_rng_default(p, mappings) {
        violations.push(("OxidizeRng::default()".to_string(), line));
        return;
    }
    let resolved = resolve_path(&p.path, mappings);
    if resolved != "StdRng" && resolved != "OxidizeRng" {
        if is_std_rng_path(&resolved) {
            violations.push(("StdRng".to_string(), line));
        } else if !allow_default_rng && is_oxidize_rng_default_path(&resolved) {
            violations.push(("OxidizeRng::default()".to_string(), line));
        }
    }
}

pub(super) fn check_type_path(
    tp: &syn::TypePath,
    mappings: &HashMap<String, String>,
    violations: &mut Vec<(String, usize)>,
) {
    if tp.qself.is_some() {
        return;
    }
    let resolved = resolve_path(&tp.path, mappings);
    if tp.path.segments.len() >= 2 && is_std_rng_path(&resolved) {
        violations.push(("StdRng".to_string(), tp.path.span().start().line));
    }
}
