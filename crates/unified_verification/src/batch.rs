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
    if !crate::doc_audit::run_doc_audit(&[]) {
        passed = false;
    }

    println!("\n--- Static Gate 6: Single-Pass AST Checks ---");
    if !run_batch_ast_with_caches(&members, cache, manifest_cache) {
        passed = false;
    }

    println!("\n--- Static Gate 7: Verification Certificate AST Parity ---");
    if !crate::report::certificate::check_ast_certificate_parity_with_cache(&members, cache) {
        passed = false;
    }

    passed
}

const NON_TRACEABILITY_MEMBERS: &[&str] = &[
    "crates/unified_verification",
    "crates/diagnostics",
    "crates/federated_registry",
    "crates/oxidize_core",
    "crates/verified_engine",
    "crates/verified_engine_macros",
    "apps/xtask",
    "crates/math_commons",
    "egui_plot",
];

pub(crate) fn should_run_staged_traceability(
    all_staged: &[String],
    workspace_members: &[String],
) -> bool {
    all_staged.iter().any(|raw| {
        let norm = raw.replace('\\', "/");
        let p = norm.trim_start_matches("./");
        if p == "traceability.toml"
            || p == "Cargo.toml"
            || p.starts_with("papers/")
            || p.starts_with("math_explorer_gui/src/tabs/")
        {
            return true;
        }
        workspace_members.iter().any(|member| {
            if NON_TRACEABILITY_MEMBERS.contains(&member.as_str()) || member == "math_explorer_gui"
            {
                return false;
            }
            let manifest = format!("{member}/Cargo.toml");
            let src_prefix = format!("{member}/src/");
            p == manifest || p.starts_with(&src_prefix)
        })
    })
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

    if should_run_staged_traceability(all_staged, workspace_members)
        && !run_traceability_with_cache(false, cache)
    {
        passed = false;
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
    }

    #[test]
    fn test_should_run_staged_traceability_scopes_domain_and_registry_edits() {
        let members = vec![
            "crates/pure_math".to_string(),
            "crates/domain_physics".to_string(),
            "crates/scientific_metadata".to_string(),
            "crates/unified_verification".to_string(),
            "apps/xtask".to_string(),
            "math_explorer_gui".to_string(),
        ];

        assert!(!should_run_staged_traceability(
            &[
                "README.md".to_string(),
                "docs/adr/0036-test.md".to_string(),
                "crates/unified_verification/src/main.rs".to_string(),
                "apps/xtask/src/main.rs".to_string(),
                "math_explorer_gui/src/app/theme.rs".to_string(),
            ],
            &members
        ));
        assert!(should_run_staged_traceability(
            &["traceability.toml".to_string()],
            &members
        ));
        assert!(should_run_staged_traceability(
            &["papers/grpo.tex".to_string()],
            &members
        ));
        assert!(should_run_staged_traceability(
            &["crates/pure_math/src/lib.rs".to_string()],
            &members
        ));
        assert!(should_run_staged_traceability(
            &["math_explorer_gui/src/tabs/neuroscience/hodgkin_huxley.rs".to_string()],
            &members
        ));
    }

    #[test]
    fn test_batch_ast_and_report_parse_each_unique_workspace_file_at_most_once() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let orig = env::current_dir().unwrap();
        let _guard = CurrentDirGuard(orig);
        let root = crate::manifest_cache::workspace_root_dir();
        if root != Path::new(".") {
            env::set_current_dir(root).unwrap();
        }

        let (_, members) = crate::manifest_cache::discover_workspace();
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

        let rs_files = crate::report::collect_rs_files();
        let metrics = crate::report::analyze_files_with_cache(&rs_files, &mut cache);
        assert!(metrics.total_funcs > 0.0);
        let unverified = crate::report::find_unverified_crates_with_cache(&members, &mut cache);
        assert!(!unverified.contains(&"crates/pure_math".to_string()));

        assert!(cache.parse_calls() > 500);
        assert_eq!(cache.parse_calls(), cache.len());

        let canonical_files: HashSet<PathBuf> = cache
            .keys()
            .map(|k| fs::canonicalize(k).unwrap_or_else(|_| PathBuf::from(k)))
            .collect();
        assert_eq!(
            canonical_files.len(),
            cache.parse_calls(),
            "Every physical Rust file in the workspace must be parsed by syn::parse_file at most once across batch_ast + report"
        );
    }

    #[test]
    fn test_batch_static_parses_each_member_manifest_and_checks_lints_once() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let orig = env::current_dir().unwrap();
        let _guard = CurrentDirGuard(orig);
        let root = crate::manifest_cache::workspace_root_dir();
        if root != Path::new(".") {
            env::set_current_dir(root).unwrap();
        }

        let (_, members) = crate::manifest_cache::discover_workspace();
        let mut ast_cache = AstCache::new();
        let mut manifest_cache = crate::manifest_cache::ManifestCache::new();

        assert!(run_batch_static_with_caches(
            &[],
            &mut ast_cache,
            &mut manifest_cache,
        ));

        assert_eq!(manifest_cache.lint_check_calls(), members.len());
        assert_eq!(manifest_cache.parse_calls(), members.len() + 1);
        for member in &members {
            let manifest = Path::new(member).join("Cargo.toml");
            assert_eq!(manifest_cache.parse_count_for(&manifest), 1);
        }
    }
}
