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
    let auto_fix = args.iter().any(|arg| arg == "--auto-fix");
    let mut cache = AstCache::new();
    run_traceability_with_cache(auto_fix, &mut cache)
}

pub fn run_batch_ast_with_cache(members: &[String], cache: &mut AstCache) -> bool {
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
    if !crate::policy_audit::run_policy_audit_with_cache(&member_refs, cache) {
        passed = false;
    }

    println!(
        "Single-pass AST batch finished ({} unique Rust files parsed).",
        cache.parse_calls()
    );
    passed
}

pub fn run_batch_ast() -> bool {
    let (_, members) = oxidize_core::workspace::get_workspace_members();
    let mut cache = AstCache::new();
    run_batch_ast_with_cache(&members, &mut cache)
}

pub fn run_batch_static(args: &[String]) -> bool {
    println!("=== Running Batched Static Verification Suite ===");
    let (_, members) = oxidize_core::workspace::get_workspace_members();
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut passed = true;

    println!("\n--- Static Gate 1: Cargo Profile Verification ---");
    if !crate::profile::check_profiles(&member_refs, false) {
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
    if !crate::doc_audit::run_doc_audit(&[]) {
        passed = false;
    }

    println!("\n--- Static Gate 6: Single-Pass AST Checks ---");
    let mut cache = AstCache::new();
    if !run_batch_ast_with_cache(&members, &mut cache) {
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

pub(crate) fn select_touched_policy_members(
    all_staged: &[String],
    workspace_members: &[String],
) -> Vec<String> {
    if all_staged
        .iter()
        .any(|p| p == "Cargo.toml" || p == crate::whitelist::WHITELIST_FILE_NAME)
    {
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

pub(crate) fn run_batch_staged_for_paths(
    all_staged: &[String],
    existing_staged: &[String],
    workspace_members: &[String],
    cache: &mut AstCache,
) -> bool {
    if all_staged.is_empty() {
        return true;
    }

    let mut passed = true;
    let staged_rs: Vec<String> = existing_staged
        .iter()
        .filter(|p| p.ends_with(".rs"))
        .cloned()
        .collect();

    if !staged_rs.is_empty() && !print_debt_items(crate::utils::check_file_lengths(&staged_rs)) {
        passed = false;
    }

    if !existing_staged.is_empty()
        && !print_debt_items(crate::entropy_guard::check_entropy_with_cache(
            existing_staged,
            cache,
        ))
    {
        passed = false;
    }

    let touched_members = select_touched_policy_members(all_staged, workspace_members);
    if !touched_members.is_empty() {
        let member_refs: Vec<&str> = touched_members.iter().map(String::as_str).collect();
        if !crate::policy_audit::run_policy_audit_with_cache(&member_refs, cache) {
            passed = false;
        }
    }

    crate::utils::check_staged_duplicates();

    if all_staged.iter().any(|p| p.starts_with(".github/"))
        && !crate::workflow_linter::lint_workflows()
    {
        passed = false;
    }

    passed
}

pub fn run_batch_staged() -> bool {
    let all_staged = git_cached_paths(&[]);
    let existing_staged: Vec<String> = git_cached_paths(&["--diff-filter=ACMR"])
        .into_iter()
        .filter(|p| Path::new(p).is_file())
        .collect();
    let (_, members) = oxidize_core::workspace::get_workspace_members();
    let mut cache = AstCache::new();
    run_batch_staged_for_paths(&all_staged, &existing_staged, &members, &mut cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_utils::CurrentDirGuard;
    use std::collections::HashSet;
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_select_touched_policy_members_and_global_triggers() {
        let members = vec![
            "crates/pure_math".to_string(),
            "crates/domain_physics".to_string(),
            "math_explorer_gui".to_string(),
        ];

        assert!(select_touched_policy_members(&["README.md".to_string()], &members).is_empty());

        assert_eq!(
            select_touched_policy_members(
                &[
                    "crates/domain_physics/src/quantum.rs".to_string(),
                    "docs/README.md".to_string()
                ],
                &members
            ),
            vec!["crates/domain_physics".to_string()]
        );

        assert_eq!(
            select_touched_policy_members(&["math_explorer_gui/Cargo.toml".to_string()], &members),
            vec!["math_explorer_gui".to_string()]
        );

        assert_eq!(
            select_touched_policy_members(&["verification_whitelist.toml".to_string()], &members),
            members
        );

        let full_members = vec![
            "crates/diagnostics".to_string(),
            "crates/domain_physics".to_string(),
            "crates/markdown_tests".to_string(),
            "math_explorer_gui".to_string(),
        ];
        assert_eq!(
            select_touched_policy_members(
                &["crates/diagnostics/README.md".to_string()],
                &full_members
            ),
            vec![
                "crates/diagnostics".to_string(),
                "crates/markdown_tests".to_string()
            ]
        );
        assert_eq!(
            select_touched_policy_members(&["docs/README.md".to_string()], &full_members),
            vec!["crates/markdown_tests".to_string()]
        );
    }

    #[test]
    fn test_batch_ast_parses_each_unique_workspace_file_at_most_once() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let orig = env::current_dir().unwrap();
        let _guard = CurrentDirGuard(orig);
        if !Path::new("crates").is_dir() && Path::new("../../crates").is_dir() {
            env::set_current_dir("../..").unwrap();
        }

        let (_, members) = oxidize_core::workspace::get_workspace_members();
        let mut cache = AstCache::new();

        assert!(crate::ast_visitor::run_clone_detector_with_cache(
            &mut cache
        ));
        let apis = crate::api_drift::extract_apis_in_dir_with_cache(Path::new("."), &mut cache);
        assert!(!apis.is_empty());
        assert!(run_traceability_with_cache(false, &mut cache));
        assert!(crate::entropy_guard::check_entropy_with_cache(&[], &mut cache).is_empty());
        let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
        assert!(crate::policy_audit::run_policy_audit_with_cache(
            &member_refs,
            &mut cache
        ));

        assert!(cache.parse_calls() > 500);
        assert_eq!(cache.parse_calls(), cache.len());

        let canonical_files: HashSet<PathBuf> = cache
            .keys()
            .map(|k| fs::canonicalize(k).unwrap_or_else(|_| PathBuf::from(k)))
            .collect();
        assert_eq!(
            canonical_files.len(),
            cache.parse_calls(),
            "Every physical Rust file in the workspace must be parsed by syn::parse_file at most once"
        );
    }
}
