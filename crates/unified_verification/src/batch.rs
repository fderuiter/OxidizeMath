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

    let mut touched = Vec::new();
    for member in workspace_members {
        let prefix = format!("{member}/");
        let manifest = format!("{member}/Cargo.toml");
        let is_touched = all_staged.iter().any(|p| {
            p == &manifest
                || (p.starts_with(&prefix) && (p.ends_with(".rs") || p.ends_with("/Cargo.toml")))
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
) -> bool {
    let mut passed = true;

    if !crate::ast_visitor::run_clone_detector_with_cache(cache) {
        passed = false;
    }
    if !run_traceability_with_cache(false, cache) {
        passed = false;
    }

    let entropy_targets: &[String] = if escalate { &[] } else { existing_staged };
    if (escalate || !existing_staged.is_empty())
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
        if !crate::policy_audit::run_policy_audit_with_cache(&member_refs, cache) {
            passed = false;
        }
    }

    passed
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
    let escalate = all_staged.iter().any(|p| is_root_escalation_path(p));
    let all_member_refs: Vec<&str> = workspace_members.iter().map(String::as_str).collect();

    if !crate::profile::check_profiles(&all_member_refs, false) {
        passed = false;
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

    if !crate::doc_audit::run_doc_audit_for_members(workspace_members, &[]) {
        passed = false;
    }

    if !run_staged_ast_and_policy_gates(
        all_staged,
        existing_staged,
        workspace_members,
        escalate,
        cache,
    ) {
        passed = false;
    }

    crate::utils::check_staged_duplicates();
    passed
}

pub fn run_batch_staged() -> bool {
    let (is_root, members) = oxidize_core::workspace::get_workspace_members();
    if !is_root && let Some(root) = oxidize_core::workspace::find_workspace_root() {
        let _ = std::env::set_current_dir(root);
    }
    let all_staged = git_cached_paths(&[]);
    let existing_staged: Vec<String> = git_cached_paths(&["--diff-filter=ACMRT"])
        .into_iter()
        .filter(|p| Path::new(p).is_file())
        .collect();
    let mut cache = AstCache::new();
    run_batch_staged_for_paths(&all_staged, &existing_staged, &members, &mut cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_utils::{CurrentDirGuard, TempTestDir};
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
        for root_cfg in [
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".clippy.toml",
            ".rustfmt.toml",
            ".cargo/config.toml",
        ] {
            assert_eq!(
                select_touched_policy_members(&[root_cfg.to_string()], &members),
                members,
                "Expected full member escalation for {root_cfg}"
            );
        }
    }

    #[test]
    fn test_run_batch_staged_for_paths_catches_profile_and_doc_failures() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let tmp = TempTestDir::new("batch_staged_profile_doc");
        let orig = env::current_dir().unwrap();
        let _guard = CurrentDirGuard(orig);

        env::set_current_dir(&*tmp).unwrap();
        fs::create_dir_all("crates/demo/src").unwrap();
        fs::write(
            "Cargo.toml",
            "[workspace]\nmembers = [\"crates/demo\"]\n\n[profile.release]\noverflow-checks = false\n",
        )
        .unwrap();
        fs::write(
            "crates/demo/Cargo.toml",
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lints]\nworkspace = true\n",
        )
        .unwrap();
        fs::write(
            "crates/demo/src/lib.rs",
            "//! Demo crate.\npub fn ok() {}\n",
        )
        .unwrap();

        let staged = vec!["Cargo.toml".to_string()];
        let members = vec!["crates/demo".to_string()];
        let mut cache = AstCache::new();
        assert!(
            !run_batch_staged_for_paths(&staged, &staged, &members, &mut cache),
            "Disabled overflow-checks or missing crate README must fail batch-staged"
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
