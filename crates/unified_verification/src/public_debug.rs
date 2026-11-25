use crate::api_drift::traversal::{
    ExportFilter, collect_facade_exports, has_cfg_test, resolve_child_mod_file,
};
use crate::ast_helpers::impl_base_ident;
use oxidize_core::traceability::AstCache;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[cfg(test)]
#[path = "public_debug_tests.rs"]
mod tests;

const MAX_MODULES_PER_CRATE: usize = 10_000;
const MAX_INLINE_MODULES: usize = 2_048;

fn meta_derives_debug(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::List(list) if list.path.is_ident("derive") => list
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            )
            .is_ok_and(|paths| {
                paths
                    .iter()
                    .any(|p| p.segments.last().is_some_and(|seg| seg.ident == "Debug"))
            }),
        syn::Meta::List(list) if list.path.is_ident("cfg_attr") => list
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            )
            .is_ok_and(|nested| nested.iter().skip(1).any(meta_derives_debug)),
        _ => false,
    }
}

pub(crate) fn attrs_derive_debug(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| meta_derives_debug(&attr.meta))
}

fn collect_manual_debug_impls_in_items(items: &[syn::Item], out: &mut BTreeSet<String>) {
    let mut stack: Vec<&[syn::Item]> = vec![items];
    let mut steps = 0;
    while let Some(curr) = stack.pop() {
        if steps >= MAX_INLINE_MODULES {
            break;
        }
        steps += 1;
        for item in curr {
            match item {
                syn::Item::Impl(item_impl) if !has_cfg_test(&item_impl.attrs) => {
                    if let Some((_, trait_path, _)) = &item_impl.trait_
                        && trait_path
                            .segments
                            .last()
                            .is_some_and(|seg| seg.ident == "Debug")
                        && let Some(base) = impl_base_ident(&item_impl.self_ty)
                    {
                        out.insert(base);
                    }
                }
                syn::Item::Mod(item_mod) if !has_cfg_test(&item_mod.attrs) => {
                    if let Some((_, inline_items)) = &item_mod.content {
                        stack.push(inline_items);
                    }
                }
                _ => {}
            }
        }
    }
}

struct MissingCandidate {
    rel_path: String,
    kind: &'static str,
    ident: String,
}

struct ModuleScanCtx<'a> {
    rel_path: &'a str,
    filter: &'a ExportFilter,
    facades: &'a BTreeMap<String, ExportFilter>,
    mod_dir: &'a Path,
}

fn inspect_module_item<'a>(
    item: &'a syn::Item,
    ctx: &ModuleScanCtx<'_>,
    missing: &mut Vec<MissingCandidate>,
    stack: &mut Vec<(&'a [syn::Item], ExportFilter, PathBuf)>,
    child_files: &mut Vec<(PathBuf, ExportFilter)>,
) {
    match item {
        syn::Item::Struct(s)
            if matches!(s.vis, syn::Visibility::Public(_))
                && !has_cfg_test(&s.attrs)
                && ctx.filter.allows_ident(&s.ident.to_string())
                && !attrs_derive_debug(&s.attrs) =>
        {
            missing.push(MissingCandidate {
                rel_path: ctx.rel_path.to_string(),
                kind: "struct",
                ident: s.ident.to_string(),
            });
        }
        syn::Item::Enum(e)
            if matches!(e.vis, syn::Visibility::Public(_))
                && !has_cfg_test(&e.attrs)
                && ctx.filter.allows_ident(&e.ident.to_string())
                && !attrs_derive_debug(&e.attrs) =>
        {
            missing.push(MissingCandidate {
                rel_path: ctx.rel_path.to_string(),
                kind: "enum",
                ident: e.ident.to_string(),
            });
        }
        syn::Item::Mod(item_mod) if !has_cfg_test(&item_mod.attrs) => {
            let mod_name = item_mod.ident.to_string();
            let is_pub = matches!(item_mod.vis, syn::Visibility::Public(_));
            let child_filter = if is_pub && ctx.filter.allows_ident(&mod_name) {
                Some(ExportFilter::AllPublic)
            } else {
                ctx.facades.get(&mod_name).cloned()
            };
            if let Some(cf) = child_filter {
                if let Some((_, inline_items)) = &item_mod.content {
                    stack.push((inline_items, cf, ctx.mod_dir.join(mod_name)));
                } else if let Some(path) =
                    resolve_child_mod_file(ctx.mod_dir, &mod_name, &item_mod.attrs)
                {
                    child_files.push((path, cf));
                }
            }
        }
        _ => {}
    }
}

fn scan_module_ast(
    file: &syn::File,
    rel_path: &str,
    root_filter: &ExportFilter,
    root_mod_dir: &Path,
    manual_debug: &mut BTreeSet<String>,
    missing: &mut Vec<MissingCandidate>,
) -> Vec<(PathBuf, ExportFilter)> {
    if has_cfg_test(&file.attrs) {
        return Vec::new();
    }
    collect_manual_debug_impls_in_items(&file.items, manual_debug);

    let mut child_files = Vec::new();
    let mut stack: Vec<(&[syn::Item], ExportFilter, PathBuf)> =
        vec![(&file.items, root_filter.clone(), root_mod_dir.to_path_buf())];
    let mut steps = 0;

    while let Some((mod_items, filter, mod_dir)) = stack.pop() {
        if steps >= MAX_INLINE_MODULES {
            break;
        }
        steps += 1;
        let facades = collect_facade_exports(mod_items, &filter);
        let ctx = ModuleScanCtx {
            rel_path,
            filter: &filter,
            facades: &facades,
            mod_dir: &mod_dir,
        };
        for item in mod_items {
            inspect_module_item(item, &ctx, missing, &mut stack, &mut child_files);
        }
    }
    child_files
}

fn discover_reachable_modules(
    crate_lib: &Path,
    cache: &mut AstCache,
) -> BTreeMap<PathBuf, ExportFilter> {
    let mut visited: BTreeMap<PathBuf, ExportFilter> = BTreeMap::new();
    visited.insert(crate_lib.to_path_buf(), ExportFilter::AllPublic);
    let mut queue = vec![crate_lib.to_path_buf()];
    let mut steps = 0;

    while let Some(file_path) = queue.pop() {
        if steps >= MAX_MODULES_PER_CRATE {
            break;
        }
        steps += 1;
        let Some(filter) = visited.get(&file_path).cloned() else {
            continue;
        };
        let mod_dir = module_search_dir(&file_path, crate_lib);
        let Some(ast) = cache
            .get_or_load_fs(&file_path)
            .and_then(|e| e.ast.as_ref())
        else {
            continue;
        };
        let children = scan_module_ast(
            ast,
            "",
            &filter,
            &mod_dir,
            &mut BTreeSet::new(),
            &mut Vec::new(),
        );
        for (child_path, child_filter) in children {
            let should_enqueue = match visited.get_mut(&child_path) {
                Some(existing) => existing.merge(child_filter),
                None => {
                    visited.insert(child_path.clone(), child_filter);
                    true
                }
            };
            if should_enqueue {
                queue.push(child_path);
            }
        }
    }
    visited
}

pub fn collect_crate_public_debug_violations(
    crate_lib: &Path,
    workspace_root: &Path,
    cache: &mut AstCache,
) -> Vec<String> {
    if !crate_lib.is_file() {
        return Vec::new();
    }
    let visited = discover_reachable_modules(crate_lib, cache);

    let mut manual_debug = BTreeSet::new();
    let mut candidates = Vec::new();
    for (file_path, filter) in &visited {
        let rel = file_path
            .strip_prefix(workspace_root)
            .unwrap_or(file_path)
            .to_string_lossy()
            .replace('\\', "/");
        let mod_dir = module_search_dir(file_path, crate_lib);
        if let Some(ast) = cache.get_or_load_fs(file_path).and_then(|e| e.ast.as_ref()) {
            let _ = scan_module_ast(
                ast,
                &rel,
                filter,
                &mod_dir,
                &mut manual_debug,
                &mut candidates,
            );
        }
    }

    let mut violations: Vec<String> = candidates
        .into_iter()
        .filter(|c| !manual_debug.contains(&c.ident))
        .map(|c| {
            format!(
                "{}: `pub {} {}` is missing `Debug` (derive `Debug` or implement `core::fmt::Debug`)",
                c.rel_path, c.kind, c.ident
            )
        })
        .collect();
    violations.sort();
    violations.dedup();
    violations
}

fn report_public_debug_violations(violations: &[String]) -> bool {
    if violations.is_empty() {
        return true;
    }
    println!("[!] Public `Debug` coverage check failed:");
    for v in violations {
        println!("  - {v}");
    }
    false
}

pub fn check_public_debug_in_dir_with_cache(workspace_root: &Path, cache: &mut AstCache) -> bool {
    let mut violations = Vec::new();
    for crate_lib in crate::api_drift::collect_crate_roots(workspace_root) {
        violations.extend(collect_crate_public_debug_violations(
            &crate_lib,
            workspace_root,
            cache,
        ));
    }
    report_public_debug_violations(&violations)
}

pub fn audit_members_public_debug(members: &[&str], cache: &mut AstCache) -> bool {
    let workspace_root = crate::manifest_cache::workspace_root_dir();
    let mut violations = Vec::new();
    for member in members {
        if super::policy_audit::lint_policy::is_vendored_egui_plot(member) {
            continue;
        }
        let member_dir = super::policy_audit::lint_policy::resolve_member_dir(member);
        let lib_rs = member_dir.join("src/lib.rs");
        if lib_rs.is_file() {
            violations.extend(collect_crate_public_debug_violations(
                &lib_rs,
                &workspace_root,
                cache,
            ));
        }
    }
    report_public_debug_violations(&violations)
}

pub fn run_check_public_debug_cli(args: &[String]) -> bool {
    let mut cache = AstCache::new();
    let passed = if args.is_empty() {
        let root = crate::manifest_cache::workspace_root_dir();
        check_public_debug_in_dir_with_cache(&root, &mut cache)
    } else {
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        audit_members_public_debug(&refs, &mut cache)
    };
    if passed {
        println!("Public Debug coverage check passed.");
    }
    passed
}

fn module_search_dir(file_path: &Path, crate_lib: &Path) -> PathBuf {
    let parent = file_path.parent().unwrap_or_else(|| Path::new("."));
    if file_path == crate_lib || file_path.file_name().and_then(|n| n.to_str()) == Some("mod.rs") {
        parent.to_path_buf()
    } else {
        match file_path.file_stem() {
            Some(stem) => parent.join(stem),
            None => parent.to_path_buf(),
        }
    }
}
