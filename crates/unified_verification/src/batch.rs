use oxidize_core::traceability::AstCache;
use std::path::Path;
use std::process::Command;

fn print_debt_items(debt: Vec<String>) -> bool {
    let passed = debt.is_empty();
    for item in debt {
        println!("{}", item);
    }
    passed
}

pub fn run_traceability_with_cache(auto_fix: bool, cache: &mut AstCache) -> bool {
    match oxidize_core::traceability::run_workspace_traceability_with_cache(auto_fix, cache) {
        Ok(passed) => passed,
        Err(err) => {
            eprintln!("Error scanning repository: {}", err);
            false
        }
    }
}

pub fn run_traceability_in_process(args: &[String]) -> bool {
    let _root_guard = crate::manifest_cache::WorkspaceRootGuard::enter();
    let auto_fix = args.iter().any(|arg| arg == "--auto-fix");
    let mut cache = AstCache::new();
    run_traceability_with_cache(auto_fix, &mut cache)
}

pub fn run_batch_ast_with_cache(members: &[String], cache: &mut AstCache) -> bool {
    let mut manifest_cache = crate::manifest_cache::ManifestCache::new();
    run_batch_ast_with_caches(members, cache, &mut manifest_cache)
}

pub(crate) fn run_batch_ast_with_caches(
    members: &[String],
    cache: &mut AstCache,
    manifest_cache: &mut crate::manifest_cache::ManifestCache,
) -> bool {
    println!("=== Running Single-Pass AST Verification Batch ===");
    let mut passed = true;

    if !crate::ast_visitor::run_clone_detector_with_cache(cache) {
        passed = false;
    }
    if !crate::api_drift::check_api_drift_with_cache(cache) {
        passed = false;
    }
    if !run_traceability_with_cache(false, cache) {
        passed = false;
    }
    if !print_debt_items(crate::entropy_guard::check_entropy_with_cache(&[], cache)) {
        passed = false;
    }
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    if !crate::policy_audit::run_policy_audit_with_caches(&member_refs, cache, manifest_cache) {
        passed = false;
    }

    println!(
        "Single-pass AST batch finished ({} unique Rust files parsed).",
        cache.parse_calls()
    );
    passed
}

pub fn run_batch_ast() -> bool {
    let _root_guard = crate::manifest_cache::WorkspaceRootGuard::enter();
    let (_, members) = crate::manifest_cache::discover_workspace();
    let mut cache = AstCache::new();
    run_batch_ast_with_cache(&members, &mut cache)
}

pub fn run_batch_static(args: &[String]) -> bool {
    let mut cache = AstCache::new();
    let mut manifest_cache = crate::manifest_cache::ManifestCache::new();
    run_batch_static_with_caches(args, &mut cache, &mut manifest_cache)
}

pub(crate) fn run_batch_static_with_caches(
    args: &[String],
    cache: &mut AstCache,
    manifest_cache: &mut crate::manifest_cache::ManifestCache,
) -> bool {
    println!("=== Running Batched Static Verification Suite ===");
    let _root_guard = crate::manifest_cache::WorkspaceRootGuard::enter();
    let (_, members) = crate::manifest_cache::discover_workspace();
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut passed = true;

    println!("\n--- Static Gate 1: Cargo Profile Verification ---");
    if !crate::profile::check_profiles_with_cache(&member_refs, false, manifest_cache) {
        passed = false;
    }

    println!("\n--- Static Gate 2: Architectural Decision Records ---");
    if !crate::records::verify_records(args) {
        passed = false;
    }

    println!("\n--- Static Gate 3: File Length Constraints ---");
    if !print_debt_items(crate::utils::check_file_lengths(&members)) {
        passed = false;
    }

    println!("\n--- Static Gate 4: CI Workflow Linting ---");
    if !crate::workflow_linter::lint_workflows() {
        passed = false;
    }

    println!("\n--- Static Gate 5: Crate Documentation Audit ---");
    if !crate::doc_audit::run_doc_audit_for_members(&members, &[]) {
        passed = false;
    }

    println!("\n--- Static Gate 6: Single-Pass AST Checks ---");
    if !run_batch_ast_with_caches(&members, cache, manifest_cache) {
        passed = false;
    }

    passed
}

fn git_cached_paths(extra_args: &[&str]) -> Vec<String> {
    let mut args = vec![
        "-c",
        "core.quotepath=false",
        "diff",
        "--cached",
        "--no-renames",
        "--name-only",
    ];
    args.extend_from_slice(extra_args);
    let Ok(output) = Command::new("git").args(&args).output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().replace('\\', "/"))
        .filter(|l| !l.is_empty())
        .collect()
}

pub(crate) fn is_root_escalation_path(path: &str) -> bool {
    matches!(
        path,
        "Cargo.toml"
            | "Cargo.lock"
            | "rust-toolchain.toml"
            | "rust-toolchain"
            | ".clippy.toml"
            | "clippy.toml"
            | ".rustfmt.toml"
            | "rustfmt.toml"
            | crate::whitelist::WHITELIST_FILE_NAME
    ) || path.starts_with(".cargo/")
}

pub(crate) fn select_touched_policy_members(
    all_staged: &[String],
    workspace_members: &[String],
) -> Vec<String> {
    if all_staged.iter().any(|p| is_root_escalation_path(p)) {
        return workspace_members.to_vec();
    }

    let any_md_staged = all_staged.iter().any(|p| p.ends_with(".md"));
    let mut touched = Vec::new();
    for member in workspace_members {
        let prefix = format!("{member}/");
        let manifest = format!("{member}/Cargo.toml");
        let is_touched = (any_md_staged && member == "crates/markdown_tests")
            || all_staged.iter().any(|p| {
                p == &manifest
                    || (p.starts_with(&prefix)
                        && (p.ends_with(".rs") || p.ends_with(".md") || p.ends_with("/Cargo.toml")))
            });
        if is_touched {
            touched.push(member.clone());
        }
    }
    touched
}

fn run_staged_ast_and_policy_gates(
    all_staged: &[String],
    existing_staged: &[String],
    workspace_members: &[String],
    escalate: bool,
    cache: &mut AstCache,
    manifest_cache: &mut crate::manifest_cache::ManifestCache,
) -> bool {
    let mut passed = true;

    if !workspace_members.is_empty() {
        if !crate::ast_visitor::run_clone_detector_with_cache(cache) {
            passed = false;
        }
        if !run_traceability_with_cache(false, cache) {
            passed = false;
        }
    }

    let entropy_targets: &[String] = if escalate { &[] } else { existing_staged };
    if ((escalate && !workspace_members.is_empty()) || !existing_staged.is_empty())
        && !print_debt_items(crate::entropy_guard::check_entropy_with_cache(
            entropy_targets,
            cache,
        ))
    {
        passed = false;
    }

    let touched_members = select_touched_policy_members(all_staged, workspace_members);
    if !touched_members.is_empty() {
        let member_refs: Vec<&str> = touched_members.iter().map(String::as_str).collect();
        if !crate::policy_audit::run_policy_audit_with_caches(&member_refs, cache, manifest_cache) {
            passed = false;
        }
    }

    passed
}

pub(crate) fn run_batch_staged_for_paths_in_dir(
    repo_dir: &Path,
    all_staged: &[String],
    existing_staged: &[String],
    workspace_members: &[String],
    cache: &mut AstCache,
) -> bool {
    if all_staged.is_empty() {
        return true;
    }

    let mut passed = true;
    if !crate::records::verify_records_in_dir_opts(repo_dir, None, None, true) {
        passed = false;
    }

    let escalate = all_staged.iter().any(|p| is_root_escalation_path(p));
    let mut manifest_cache = crate::manifest_cache::ManifestCache::new();

    if !workspace_members.is_empty() {
        let all_member_refs: Vec<&str> = workspace_members.iter().map(String::as_str).collect();
        if !crate::profile::check_profiles_with_cache(&all_member_refs, false, &mut manifest_cache)
        {
            passed = false;
        }
    }

    let length_targets: Vec<String> = if escalate {
        workspace_members.to_vec()
    } else {
        existing_staged
            .iter()
            .filter(|p| p.ends_with(".rs"))
            .cloned()
            .collect()
    };
    if !length_targets.is_empty()
        && !print_debt_items(crate::utils::check_file_lengths(&length_targets))
    {
        passed = false;
    }

    if !workspace_members.is_empty()
        && !crate::doc_audit::run_doc_audit_for_members(workspace_members, &[])
    {
        passed = false;
    }

    if !run_staged_ast_and_policy_gates(
        all_staged,
        existing_staged,
        workspace_members,
        escalate,
        cache,
        &mut manifest_cache,
    ) {
        passed = false;
    }

    if repo_dir == Path::new(".") {
        crate::utils::check_staged_duplicates();
    }

    passed
}

pub(crate) fn run_batch_staged_for_paths(
    all_staged: &[String],
    existing_staged: &[String],
    workspace_members: &[String],
    cache: &mut AstCache,
) -> bool {
    run_batch_staged_for_paths_in_dir(
        Path::new("."),
        all_staged,
        existing_staged,
        workspace_members,
        cache,
    )
}

pub fn run_batch_staged() -> bool {
    let _root_guard = crate::manifest_cache::WorkspaceRootGuard::enter();
    let (_, members) = crate::manifest_cache::discover_workspace();
    let all_staged = git_cached_paths(&[]);
    let existing_staged: Vec<String> = git_cached_paths(&["--diff-filter=ACMRT"])
        .into_iter()
        .filter(|p| Path::new(p).is_file())
        .collect();
    let mut cache = AstCache::new();
    run_batch_staged_for_paths(&all_staged, &existing_staged, &members, &mut cache)
}

#[cfg(test)]
mod tests;
