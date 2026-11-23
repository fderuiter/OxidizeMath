use super::traversal::{
    ExportFilter, child_module_dir_for_file, collect_facade_exports, has_cfg_test,
    resolve_child_mod_file,
};
use crate::ast_helpers::impl_base_ident;
use oxidize_core::traceability::AstCache;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const MAX_MODULES_PER_CRATE: usize = 10_000;
const MAX_INLINE_MODULES: usize = 2_048;

pub(crate) const MATHEMATICAL_CRATES: &[(&str, &str)] = &[
    ("crates/math_commons", "math_commons"),
    ("crates/pure_math", "pure_math"),
    ("crates/domain_ai", "domain_ai"),
    ("crates/domain_applied", "domain_applied"),
    ("crates/domain_biology", "domain_biology"),
    ("crates/domain_climate", "domain_climate"),
    ("crates/domain_epidemiology", "domain_epidemiology"),
    ("crates/domain_physics", "domain_physics"),
];

pub(crate) fn is_mathematical_crate_member(member: &str, package_name: Option<&str>) -> bool {
    let norm = member.replace('\\', "/");
    let trimmed = norm.trim_end_matches('/');
    MATHEMATICAL_CRATES.iter().any(|(rel, pkg)| {
        package_name == Some(*pkg) || trimmed == *rel || trimmed.ends_with(&format!("/{rel}"))
    })
}

fn has_derive_debug(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if !attr.path().is_ident("derive") {
            continue;
        }
        let Ok(paths) = attr.parse_args_with(
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        ) else {
            continue;
        };
        if paths
            .iter()
            .any(|p| p.segments.last().is_some_and(|seg| seg.ident == "Debug"))
        {
            return true;
        }
    }
    false
}

fn collect_manual_debug_impls_in_ast(items: &[syn::Item], out: &mut BTreeSet<String>) {
    let mut stack: Vec<&[syn::Item]> = vec![items];
    let mut steps = 0;
    while let Some(mod_items) = stack.pop() {
        if steps >= MAX_INLINE_MODULES {
            break;
        }
        steps += 1;
        for item in mod_items {
            match item {
                syn::Item::Mod(m) if !has_cfg_test(&m.attrs) => {
                    if let Some((_, inline_items)) = &m.content {
                        stack.push(inline_items);
                    }
                }
                syn::Item::Impl(imp) if !has_cfg_test(&imp.attrs) => {
                    if let Some((None, trait_path, _)) = &imp.trait_
                        && trait_path
                            .segments
                            .last()
                            .is_some_and(|seg| seg.ident == "Debug")
                        && let Some(ident) = impl_base_ident(&imp.self_ty)
                    {
                        out.insert(ident);
                    }
                }
                _ => {}
            }
        }
    }
}

fn collect_crate_manual_debug_impls(crate_dir: &Path, cache: &mut AstCache) -> BTreeSet<String> {
    let mut manual = BTreeSet::new();
    let src_dir = crate_dir.join("src");
    if !src_dir.is_dir() {
        return manual;
    }
    for path in crate::utils::collect_rs_files_filtered(&[src_dir], &["target"]) {
        if path.file_name().and_then(|s| s.to_str()) == Some("tests.rs") {
            continue;
        }
        let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) else {
            continue;
        };
        if has_cfg_test(&ast.attrs) {
            continue;
        }
        collect_manual_debug_impls_in_ast(&ast.items, &mut manual);
    }
    manual
}

struct ModuleDebugScan<'a> {
    display_path: &'a str,
    manual_debug: &'a BTreeSet<String>,
    checked_count: &'a mut usize,
    violations: &'a mut Vec<String>,
}

impl ModuleDebugScan<'_> {
    fn check_type(&mut self, kind: &str, ident: &syn::Ident, attrs: &[syn::Attribute]) {
        *self.checked_count += 1;
        let name = ident.to_string();
        if !has_derive_debug(attrs) && !self.manual_debug.contains(&name) {
            self.violations.push(format!(
                "{}: public {kind} `{name}` does not implement `Debug`",
                self.display_path
            ));
        }
    }
}

fn scan_reachable_module_ast(
    file: &syn::File,
    root_filter: &ExportFilter,
    root_mod_dir: &Path,
    scan: &mut ModuleDebugScan<'_>,
) -> Vec<(PathBuf, ExportFilter)> {
    if has_cfg_test(&file.attrs) {
        return Vec::new();
    }
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
        for item in mod_items {
            match item {
                syn::Item::Struct(s)
                    if matches!(s.vis, syn::Visibility::Public(_))
                        && !has_cfg_test(&s.attrs)
                        && filter.allows_ident(&s.ident.to_string()) =>
                {
                    scan.check_type("struct", &s.ident, &s.attrs);
                }
                syn::Item::Enum(e)
                    if matches!(e.vis, syn::Visibility::Public(_))
                        && !has_cfg_test(&e.attrs)
                        && filter.allows_ident(&e.ident.to_string()) =>
                {
                    scan.check_type("enum", &e.ident, &e.attrs);
                }
                syn::Item::Mod(m) if !has_cfg_test(&m.attrs) => {
                    let mod_name = m.ident.to_string();
                    let is_pub = matches!(m.vis, syn::Visibility::Public(_));
                    let next_filter = if is_pub && filter.allows_ident(&mod_name) {
                        Some(ExportFilter::AllPublic)
                    } else {
                        facades.get(&mod_name).cloned()
                    };
                    if let Some(cf) = next_filter {
                        if let Some((_, inline_items)) = &m.content {
                            stack.push((inline_items, cf, mod_dir.join(mod_name)));
                        } else if let Some(p) =
                            resolve_child_mod_file(&mod_dir, &mod_name, &m.attrs)
                        {
                            child_files.push((p, cf));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    child_files
}

pub(crate) fn check_crate_public_debug_with_cache(
    crate_dir: &Path,
    display_root: &Path,
    cache: &mut AstCache,
) -> (usize, Vec<String>) {
    let crate_lib = crate_dir.join("src/lib.rs");
    if !crate_lib.is_file() {
        return (0, Vec::new());
    }
    let manual_debug = collect_crate_manual_debug_impls(crate_dir, cache);
    let mut visited: BTreeMap<PathBuf, ExportFilter> = BTreeMap::new();
    let mut worklist = vec![crate_lib.clone()];
    visited.insert(crate_lib, ExportFilter::AllPublic);

    let mut checked_count = 0usize;
    let mut violations = Vec::new();
    let mut iterations = 0usize;

    while let Some(file_path) = worklist.pop() {
        if iterations >= MAX_MODULES_PER_CRATE {
            break;
        }
        iterations += 1;
        let Some(active_filter) = visited.get(&file_path).cloned() else {
            continue;
        };
        let mod_dir = child_module_dir_for_file(&file_path);
        let rel = file_path.strip_prefix(display_root).unwrap_or(&file_path);
        let norm_path = rel.to_string_lossy().replace('\\', "/");

        let child_files = {
            let Some(ast) = cache
                .get_or_load_fs(&file_path)
                .and_then(|e| e.ast.as_ref())
            else {
                continue;
            };
            let mut scan = ModuleDebugScan {
                display_path: &norm_path,
                manual_debug: &manual_debug,
                checked_count: &mut checked_count,
                violations: &mut violations,
            };
            scan_reachable_module_ast(ast, &active_filter, &mod_dir, &mut scan)
        };

        for (child_path, child_filter) in child_files {
            let enqueue = if let Some(prev) = visited.get_mut(&child_path) {
                prev.merge(child_filter)
            } else {
                visited.insert(child_path.clone(), child_filter);
                true
            };
            if enqueue {
                worklist.push(child_path);
            }
        }
    }

    violations.sort();
    violations.dedup();
    (checked_count, violations)
}

pub(crate) fn check_member_public_debug_with_cache(
    member_dir: &Path,
    cache: &mut AstCache,
) -> bool {
    let (_, violations) = check_crate_public_debug_with_cache(member_dir, member_dir, cache);
    if violations.is_empty() {
        return true;
    }
    for v in &violations {
        println!(
            "[!] Missing Debug implementation in {}: {v}",
            member_dir.display()
        );
    }
    false
}

pub(crate) fn check_workspace_public_debug_with_cache(
    workspace_root: &Path,
    cache: &mut AstCache,
) -> (usize, Vec<String>) {
    let mut total_checked = 0usize;
    let mut all_violations = Vec::new();
    for (rel, _) in MATHEMATICAL_CRATES {
        let crate_dir = crate::manifest_cache::resolve_workspace_path_from(workspace_root, rel);
        let (checked, violations) =
            check_crate_public_debug_with_cache(&crate_dir, workspace_root, cache);
        total_checked += checked;
        all_violations.extend(violations);
    }
    all_violations.sort();
    all_violations.dedup();
    (total_checked, all_violations)
}

pub fn run_public_debug_check() -> bool {
    let mut cache = AstCache::new();
    let root = crate::manifest_cache::workspace_root_dir();
    let (checked, violations) = check_workspace_public_debug_with_cache(root, &mut cache);
    if violations.is_empty() {
        println!("Public Debug coverage check passed ({checked} public structs/enums verified).");
        return true;
    }
    println!("Public Debug coverage check failed:");
    for v in &violations {
        println!("  - {v}");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_utils::TempTestDir;
    use std::fs;

    #[test]
    fn test_public_debug_accepts_derive_and_manual_impl_and_ignores_private_mods() {
        let dir = TempTestDir::new("public_debug_ok");
        let crate_dir = dir.join("crates/pure_math");
        fs::create_dir_all(crate_dir.join("src")).unwrap();
        fs::write(
            crate_dir.join("src/lib.rs"),
            r#"
mod hidden;
pub mod manual_mod;

#[derive(Clone, Debug, PartialEq)]
pub struct PlainConfig {
    pub x: f64,
}

#[derive(core::fmt::Debug)]
pub enum Mode {
    Fast,
}
"#,
        )
        .unwrap();
        fs::write(
            crate_dir.join("src/hidden.rs"),
            "pub struct UnreachableWithoutDebug { pub x: i32 }\n",
        )
        .unwrap();
        fs::write(
            crate_dir.join("src/manual_mod.rs"),
            r#"
pub struct ClosureHolder<F> {
    pub f: F,
}

impl<F> std::fmt::Debug for ClosureHolder<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureHolder").finish_non_exhaustive()
    }
}
"#,
        )
        .unwrap();

        let mut cache = AstCache::new();
        let (checked, violations) =
            check_crate_public_debug_with_cache(&crate_dir, &dir, &mut cache);
        assert_eq!(checked, 3);
        assert!(
            violations.is_empty(),
            "Unexpected violations: {violations:?}"
        );
    }

    #[test]
    fn test_public_debug_flags_missing_debug_on_reachable_struct_and_reexported_enum() {
        let dir = TempTestDir::new("public_debug_fail");
        let crate_dir = dir.join("crates/domain_physics");
        fs::create_dir_all(crate_dir.join("src")).unwrap();
        fs::write(
            crate_dir.join("src/lib.rs"),
            r#"
mod inner;
pub use inner::ReexportedEnum;

#[derive(Clone)]
pub struct MissingDebugStruct;
"#,
        )
        .unwrap();
        fs::write(
            crate_dir.join("src/inner.rs"),
            "pub enum ReexportedEnum { A, B }\npub struct NotReexported;\n",
        )
        .unwrap();

        let mut cache = AstCache::new();
        let (checked, violations) =
            check_crate_public_debug_with_cache(&crate_dir, &dir, &mut cache);
        assert_eq!(checked, 2);
        assert_eq!(violations.len(), 2);
        assert!(violations.iter().any(|v| v.contains("MissingDebugStruct")));
        assert!(violations.iter().any(|v| v.contains("ReexportedEnum")));
    }

    #[test]
    fn test_workspace_mathematical_crates_have_100_percent_public_debug() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let root = crate::manifest_cache::workspace_root_dir();
        let mut cache = AstCache::new();
        let (checked, violations) = check_workspace_public_debug_with_cache(root, &mut cache);
        assert!(
            checked >= 430,
            "Expected at least 430 reachable public structs/enums across 8 math crates, got {checked}"
        );
        assert!(
            violations.is_empty(),
            "Missing Debug on reachable public types: {violations:#?}"
        );
    }
}
