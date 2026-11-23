mod adr_validation;
mod comment_scanner;
mod git_range;

#[cfg(test)]
mod tests;

use std::env;
use std::path::Path;

/// Source of Git changes inspected during ADR verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DiffSource {
    /// Staged index only (`git diff --cached`).
    Staged,
    /// Staged index and branch commits relative to a resolved merge-base (`git diff --cached <base>`).
    CachedBase(String),
    /// Explicit commit range (`git diff <base>...<head>`).
    Range { base: String, head: String },
    /// Single `HEAD` commit (`git diff-tree` / `git show HEAD`).
    Head,
}

/// Returns `true` if `path` belongs to a core mathematical or runtime crate requiring an ADR.
pub(crate) fn is_core_path(path: &str) -> bool {
    path.starts_with("math_explorer/")
        || (path.starts_with("crates/") && !path.starts_with("crates/unified_verification/"))
}

/// CLI entrypoint for `unified_verification verify-records`.
pub fn verify_records(args: &[String]) -> bool {
    let force_staged = args.iter().any(|a| a == "--staged");
    let raw_base = env::var("BASE_SHA").ok();
    let raw_head = env::var("HEAD_SHA").ok();
    let root = crate::manifest_cache::workspace_root_dir();
    verify_records_in_dir_opts(
        &root,
        raw_base.as_deref(),
        raw_head.as_deref(),
        force_staged,
    )
}

#[cfg(test)]
pub fn verify_records_in_dir(
    repo_dir: &Path,
    raw_base: Option<&str>,
    raw_head: Option<&str>,
) -> bool {
    verify_records_in_dir_opts(repo_dir, raw_base, raw_head, false)
}

/// Verifies ADR structure, unique prefixes, and core-crate ADR update rules inside `repo_dir`.
pub fn verify_records_in_dir_opts(
    repo_dir: &Path,
    raw_base: Option<&str>,
    raw_head: Option<&str>,
    force_staged: bool,
) -> bool {
    let base_sha = git_range::sanitize_sha(raw_base);
    let head_sha = git_range::sanitize_sha(raw_head);

    let Some((changed_files, source)) = git_range::select_changed_files(
        repo_dir,
        base_sha.as_deref(),
        head_sha.as_deref(),
        force_staged,
    ) else {
        return false;
    };

    if !adr_validation::validate_repository_adrs(repo_dir, &source) {
        return false;
    }

    let changed_files = changed_files.replace('\\', "/");
    println!("Changed files:\n{}", changed_files);

    let changed_lines: Vec<&str> = changed_files
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let core_modified = changed_lines.iter().any(|l| is_core_path(l));

    if !core_modified {
        println!("No core logic areas modified. Skipping ADR verification.");
        return true;
    }

    if comment_scanner::is_pure_deletion_for_source(repo_dir, &source) {
        println!("Pure deletion detected in core logic. Skipping ADR verification.");
        return true;
    }

    println!("Core logic areas (math_explorer/ or crates/) were modified.");
    let active_files = git_range::active_changed_files(repo_dir, &source);
    let adr_modified = active_files
        .iter()
        .any(|p| adr_validation::is_valid_adr_rel_path(p));
    if !adr_modified {
        println!(
            "Verification failed! An active Markdown ADR in docs/adr/ must be created or updated when modifying core logic."
        );
        return false;
    }
    println!("ADR verification passed.");
    true
}
