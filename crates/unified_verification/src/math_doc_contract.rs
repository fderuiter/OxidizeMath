pub(crate) mod visitor;

#[cfg(test)]
mod tests;

use crate::whitelist::{
    PathMatchOptions, VerificationWhitelist, lossy_slash_path, path_matches_lossy,
};
use oxidize_core::bibtex::BibtexDatabase;
use oxidize_core::traceability::AstCache;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;
use visitor::{ContractViolation, MathDocContractVisitor};

pub const MATH_CONTRACT_CRATES: &[&str] = &[
    "crates/math_commons",
    "crates/pure_math",
    "crates/domain_ai",
    "crates/domain_applied",
    "crates/domain_biology",
    "crates/domain_climate",
    "crates/domain_epidemiology",
    "crates/domain_physics",
];

const PATH_MATCH_OPTS: PathMatchOptions = PathMatchOptions {
    trim_dot_slash: true,
    require_non_empty: true,
    match_file_name: false,
};

pub(crate) fn is_math_contract_member(member: &str) -> bool {
    let member_path = Path::new(member.trim_end_matches('/'));
    MATH_CONTRACT_CRATES
        .iter()
        .any(|target| path_matches_lossy(member_path, target, PATH_MATCH_OPTS))
}

pub(crate) fn is_test_source_file(rel_or_abs: &str) -> bool {
    let norm = rel_or_abs.replace('\\', "/");
    norm.ends_with("/tests.rs") || norm.ends_with("_tests.rs") || norm.contains("/tests/")
}

fn resolve_workspace_root_for_whitelist(whitelist_path: &Path) -> PathBuf {
    if let Some(parent) = whitelist_path.parent() {
        let base = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if base.join("papers/references.bib").is_file() || base.join("Cargo.toml").is_file() {
            return base.to_path_buf();
        }
    }
    crate::manifest_cache::workspace_root_dir()
}

fn load_workspace_bib(workspace_root: &Path) -> Result<BibtexDatabase, String> {
    let bib_path = workspace_root.join("papers/references.bib");
    let raw = fs::read_to_string(&bib_path).map_err(|err| {
        format!(
            "Failed to read BibTeX database at {}: {}",
            bib_path.display(),
            err
        )
    })?;
    BibtexDatabase::parse(&raw)
        .map_err(|err| format!("Invalid BibTeX syntax in {}: {err}", bib_path.display()))
}

fn lookup_file_cap<'a>(
    file_path: &str,
    whitelist: &'a VerificationWhitelist,
) -> Option<(&'a str, usize)> {
    let cfg = whitelist.math_doc_contract.as_ref()?;
    let actual = Path::new(file_path);
    cfg.files
        .iter()
        .find(|(key, _)| path_matches_lossy(actual, key, PATH_MATCH_OPTS))
        .map(|(key, &cap)| (key.as_str(), cap))
}

pub(crate) fn collect_file_violations(
    ast: &syn::File,
    workspace_root: &Path,
    bib_db: &BibtexDatabase,
) -> Vec<ContractViolation> {
    let mut visitor = MathDocContractVisitor::new(workspace_root, bib_db);
    visitor.visit_file(ast);
    visitor.violations
}

fn evaluate_file_against_cap(
    file_path: &str,
    violations: &[ContractViolation],
    whitelist: &VerificationWhitelist,
    matched_keys: &mut HashSet<String>,
) -> bool {
    let matched = lookup_file_cap(file_path, whitelist);
    if let Some((key, _)) = matched {
        matched_keys.insert(key.to_string());
    }
    let cap = matched.map_or(0, |(_, c)| c);
    let actual = violations.len();

    if actual > cap {
        for v in violations {
            let loc = crate::ast_helpers::format_file_line(file_path, v.line);
            println!(
                "[!] Math doc contract violation at {loc} ({}): missing {}",
                v.item_label,
                v.missing_parts.join("; ")
            );
        }
        println!(
            "[!] Math doc contract failed in {file_path}: found {actual} violation(s), whitelisted cap is {cap}"
        );
        return false;
    }
    if actual < cap {
        println!(
            "[!] Unratcheted math_doc_contract cap in {file_path}: found {actual} violation(s), but verification_whitelist.toml allows {cap}; lower or remove the cap"
        );
        return false;
    }
    true
}

fn crate_prefix_of_whitelist_key(key: &str) -> Option<String> {
    let norm = key.trim().replace('\\', "/");
    let trimmed = norm.trim_start_matches("./");
    let segs: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
    if segs.len() >= 3 && (segs[0] == "crates" || segs[0] == "apps") {
        Some(format!("{}/{}", segs[0], segs[1]))
    } else if segs.len() >= 2 && segs[0] != "crates" && segs[0] != "apps" {
        Some(segs[0].to_string())
    } else {
        None
    }
}

fn check_stale_or_non_math_caps(
    audited_member_dirs: &[PathBuf],
    whitelist: &VerificationWhitelist,
    matched_keys: &HashSet<String>,
) -> bool {
    let Some(cfg) = &whitelist.math_doc_contract else {
        return true;
    };
    let mut ok = true;
    for (key, &cap) in &cfg.files {
        let Some(crate_prefix) = crate_prefix_of_whitelist_key(key) else {
            println!(
                "[!] Invalid math_doc_contract cap key `{key}` in verification_whitelist.toml"
            );
            ok = false;
            continue;
        };
        if !is_math_contract_member(&crate_prefix) {
            println!(
                "[!] Invalid math_doc_contract cap `{key}` in verification_whitelist.toml: `{crate_prefix}` is not one of the 8 math crates"
            );
            ok = false;
            continue;
        }
        let was_audited = audited_member_dirs
            .iter()
            .any(|dir| path_matches_lossy(dir, &crate_prefix, PATH_MATCH_OPTS));
        if was_audited && !matched_keys.contains(key) {
            println!(
                "[!] Stale math_doc_contract cap in verification_whitelist.toml for missing or test-only file {key} (cap = {cap}); remove the entry"
            );
            ok = false;
        }
    }
    ok
}

pub(crate) fn audit_members_with_whitelist_and_cache(
    members: &[&str],
    whitelist_path: &Path,
    whitelist: &VerificationWhitelist,
    cache: &mut AstCache,
) -> bool {
    let workspace_root = resolve_workspace_root_for_whitelist(whitelist_path);
    audit_members_in_root(&workspace_root, members, whitelist, cache)
}

pub(crate) fn audit_members_in_root(
    workspace_root: &Path,
    members: &[&str],
    whitelist: &VerificationWhitelist,
    cache: &mut AstCache,
) -> bool {
    let math_members: Vec<&str> = members
        .iter()
        .copied()
        .filter(|m| is_math_contract_member(m))
        .collect();
    if math_members.is_empty() {
        return check_stale_or_non_math_caps(&[], whitelist, &HashSet::new());
    }

    let bib_db = match load_workspace_bib(workspace_root) {
        Ok(db) => db,
        Err(err) => {
            eprintln!("[!] {err}");
            return false;
        }
    };

    let mut success = true;
    let mut matched_keys = HashSet::new();
    let mut audited_dirs = Vec::with_capacity(math_members.len());

    for member in math_members {
        let member_dir = crate::manifest_cache::resolve_workspace_path_from(workspace_root, member);
        let src_dir = member_dir.join("src");
        for path in crate::utils::collect_rs_files_filtered(&[&src_dir], &["target"]) {
            let file_str = lossy_slash_path(&path);
            if is_test_source_file(&file_str) {
                continue;
            }
            if let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) {
                let violations = collect_file_violations(ast, workspace_root, &bib_db);
                if !evaluate_file_against_cap(&file_str, &violations, whitelist, &mut matched_keys)
                {
                    success = false;
                }
            } else {
                eprintln!("[!] Failed to parse Rust AST for {file_str}");
                success = false;
            }
        }
        audited_dirs.push(member_dir);
    }

    if !check_stale_or_non_math_caps(&audited_dirs, whitelist, &matched_keys) {
        success = false;
    }
    success
}

pub fn run_math_doc_contract_in_workspace(
    workspace_root: &Path,
    members: &[String],
    cache: &mut AstCache,
) -> bool {
    let candidate = crate::manifest_cache::resolve_workspace_path_from(
        workspace_root,
        crate::whitelist::WHITELIST_FILE_NAME,
    );
    let whitelist = match crate::whitelist::load_whitelist_strict(None, &[candidate.as_path()]) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("[!] {err}");
            return false;
        }
    };
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    audit_members_in_root(workspace_root, &member_refs, &whitelist, cache)
}
