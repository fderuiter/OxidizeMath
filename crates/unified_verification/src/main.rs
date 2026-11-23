#![doc = include_str!("../README.md")]
use std::env;
mod api_drift;
mod ast_helpers;
mod ast_visitor;
mod batch;
mod doc_audit;
mod entropy_guard;
mod manifest_cache;
mod maturity;
mod policy_audit;
mod profile;
mod records;
mod release_tag;
mod report;
mod utils;
mod vulnerabilities;
mod wasm_coverage;
mod whitelist;
mod workflow_linter;
use utils::{check_file_lengths, check_staged_duplicates};

#[cfg(test)]
/// Test mutex to synchronize tests that modify process working directory.
pub static DIR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run_check_profiles_cli() {
    let (_, members) = oxidize_core::workspace::get_workspace_members();
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    if !profile::check_profiles(&member_refs, false) {
        std::process::exit(1);
    }
}

fn run_policy_audit_cli(args: &[String]) {
    let targets = if args.is_empty() {
        oxidize_core::workspace::get_workspace_members().1
    } else {
        args.to_vec()
    };
    let member_refs: Vec<&str> = targets.iter().map(String::as_str).collect();
    if !policy_audit::run_policy_audit(&member_refs) {
        std::process::exit(1);
    }
}

fn run_check_entropy_cli(args: &[String]) {
    let debt = entropy_guard::check_entropy(args);
    let has_debt = !debt.is_empty();
    for d in debt {
        println!("{}", d);
    }
    if has_debt {
        std::process::exit(1);
    }
}

fn run_check_file_lengths_cli(args: &[String]) {
    let targets = if args.is_empty() {
        oxidize_core::workspace::get_workspace_members().1
    } else {
        args.to_vec()
    };
    let debt = check_file_lengths(&targets);
    let has_debt = !debt.is_empty();
    for d in debt {
        println!("{}", d);
    }
    if has_debt {
        std::process::exit(1);
    }
}

fn exit_if_failed(passed: bool) {
    if !passed {
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: unified_verification <command>");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "verify-records" => exit_if_failed(records::verify_records(&args[2..])),
        "traceability" => exit_if_failed(batch::run_traceability_in_process(&args[2..])),
        "batch-ast" | "check-ast-batch" => exit_if_failed(batch::run_batch_ast()),
        "batch-static" => exit_if_failed(batch::run_batch_static(&args[2..])),
        "batch-staged" | "lint-staged" => exit_if_failed(batch::run_batch_staged()),
        "verify-suite" => verify_suite(&args[2..]),
        "regenerate-baseline" => api_drift::regenerate_baseline_with_args(&args[2..]),
        "check-api-drift" => exit_if_failed(api_drift::check_api_drift()),
        "check-maturity" => exit_if_failed(maturity::check_maturity(&args[2..])),
        "check-duplicates" => exit_if_failed(ast_visitor::run_clone_detector()),
        "check-profiles" => run_check_profiles_cli(),
        "check-release-tag" => exit_if_failed(release_tag::run_check_release_tag_cli(&args[2..])),
        "check-staged-duplicates" => check_staged_duplicates(),
        "test-policy-audit" => run_policy_audit_cli(&args[2..]),
        "check-entropy" => run_check_entropy_cli(&args[2..]),
        "check-vulnerabilities" => exit_if_failed(vulnerabilities::check_osv_vulnerabilities()),
        "check-file-lengths" => run_check_file_lengths_cli(&args[2..]),
        "lint-workflows" => exit_if_failed(workflow_linter::lint_workflows()),
        "check-docs" => exit_if_failed(doc_audit::run_doc_audit(&args[2..])),
        "list-examples" => exit_if_failed(doc_audit::list_registered_examples_cli()),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}

pub(crate) fn run_security_checks_with_hooks(
    member_refs: &[&str],
    auto_fix: bool,
    cache: &mut oxidize_core::traceability::AstCache,
    manifest_cache: &mut manifest_cache::ManifestCache,
    check_osv: fn() -> bool,
    check_deps_ast: fn() -> bool,
) -> (bool, Vec<String>) {
    println!("Running security checks...");
    let mut passed_security = true;

    if !profile::check_profiles_with_cache(member_refs, auto_fix, manifest_cache) {
        passed_security = false;
    }
    if !check_osv() {
        passed_security = false;
    }
    if !check_deps_ast() {
        passed_security = false;
    }
    if !ast_visitor::run_clone_detector_with_cache(cache) {
        passed_security = false;
    }
    if !policy_audit::run_policy_audit_with_caches(member_refs, cache, manifest_cache) {
        passed_security = false;
    }
    let entropy_debt = entropy_guard::check_entropy_with_cache(&[], cache);
    if !entropy_debt.is_empty() {
        passed_security = false;
    }
    (passed_security, entropy_debt)
}

fn run_security_checks(
    member_refs: &[&str],
    auto_fix: bool,
    cache: &mut oxidize_core::traceability::AstCache,
    manifest_cache: &mut manifest_cache::ManifestCache,
) -> Vec<String> {
    let (passed_security, entropy_debt) = run_security_checks_with_hooks(
        member_refs,
        auto_fix,
        cache,
        manifest_cache,
        vulnerabilities::check_osv_vulnerabilities,
        ast_visitor::run_ast_visitor,
    );
    if !passed_security {
        eprintln!("Security verification failed!");
        std::process::exit(1);
    }
    entropy_debt
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VerifySuiteOptions {
    pub auto_fix: bool,
    pub coverage_only: bool,
}

pub(crate) fn parse_verify_suite_args(args: &[String]) -> Result<VerifySuiteOptions, String> {
    let mut opts = VerifySuiteOptions::default();
    for arg in args {
        match arg.as_str() {
            "--auto-fix" => opts.auto_fix = true,
            "--coverage-only" => opts.coverage_only = true,
            other => return Err(format!("Unknown verify-suite argument: {other}")),
        }
    }
    Ok(opts)
}

pub(crate) fn evaluate_coverage_and_report_with<F>(
    members: &[String],
    entropy_debt: Vec<String>,
    coverage_only: bool,
    collect_cov: F,
) -> Result<bool, String>
where
    F: FnOnce() -> Result<serde_json::Value, String>,
{
    println!("Gathering native execution coverage...");

    let cov_json = collect_cov()?;
    let rs_files = report::collect_rs_files();
    let (native_lines_total, native_lines_covered) = report::parse_coverage(&cov_json);
    let m = report::analyze_files(&rs_files);
    let mut all_debt = Vec::new();
    all_debt.extend(report::check_unverified_modules(members));
    if !coverage_only {
        all_debt.extend(check_file_lengths(members));
    }
    all_debt.extend(entropy_debt);

    let native_cov_pct = if native_lines_total > 0.0 {
        native_lines_covered / native_lines_total * 100.0
    } else {
        0.0
    };
    let wasm_cov_pct = if m.wasm_paths > 0 {
        (m.wasm_covered as f64 / m.wasm_paths as f64) * 100.0
    } else {
        100.0
    };

    Ok(report::print_report(
        native_cov_pct,
        wasm_cov_pct,
        &m,
        &all_debt,
    ))
}

fn run_default_verify_suite_static_gates(members: &[String], auto_fix: bool) -> Vec<String> {
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut cache = oxidize_core::traceability::AstCache::new();
    let mut manifests = manifest_cache::ManifestCache::new();
    let entropy_debt = run_security_checks(&member_refs, auto_fix, &mut cache, &mut manifests);

    println!("Running ADR and API drift checks...");
    if !records::verify_records(&[]) {
        std::process::exit(1);
    }
    if !api_drift::check_api_drift_with_cache(&mut cache) {
        std::process::exit(1);
    }
    entropy_debt
}

pub(crate) fn run_verify_suite_with<S, C>(
    opts: VerifySuiteOptions,
    members: &[String],
    run_static_gates: S,
    collect_cov: C,
) -> Result<bool, String>
where
    S: FnOnce(&[String], bool) -> Vec<String>,
    C: FnOnce() -> Result<serde_json::Value, String>,
{
    let entropy_debt = if opts.coverage_only {
        Vec::new()
    } else {
        run_static_gates(members, opts.auto_fix)
    };
    evaluate_coverage_and_report_with(members, entropy_debt, opts.coverage_only, collect_cov)
}

fn verify_suite(args: &[String]) {
    println!("=== High-Integrity Verified Suite ===");
    let opts = match parse_verify_suite_args(args) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let _root_guard = manifest_cache::WorkspaceRootGuard::enter();
    let (_, members) = manifest_cache::discover_workspace();
    let passed = match run_verify_suite_with(
        opts,
        &members,
        run_default_verify_suite_static_gates,
        report::collect_llvm_cov_json,
    ) {
        Ok(passed) => passed,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };
    if !passed {
        std::process::exit(1);
    }
    println!("\nAll integrity checks passed!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test_utils::CurrentDirGuard;

    fn sample_cov_json(count: u64, covered: u64) -> serde_json::Value {
        serde_json::json!({
            "data": [{
                "files": [{
                    "summary": {
                        "lines": {
                            "count": count,
                            "covered": covered
                        }
                    }
                }]
            }]
        })
    }

    #[test]
    fn test_parse_verify_suite_args_flags_and_unknown_rejection() {
        assert_eq!(
            parse_verify_suite_args(&[]),
            Ok(VerifySuiteOptions {
                auto_fix: false,
                coverage_only: false,
            })
        );
        assert_eq!(
            parse_verify_suite_args(&["--coverage-only".to_string()]),
            Ok(VerifySuiteOptions {
                auto_fix: false,
                coverage_only: true,
            })
        );
        assert_eq!(
            parse_verify_suite_args(&["--auto-fix".to_string(), "--coverage-only".to_string()]),
            Ok(VerifySuiteOptions {
                auto_fix: true,
                coverage_only: true,
            })
        );
        assert!(parse_verify_suite_args(&["--invalid-flag".to_string()]).is_err());
    }

    #[test]
    fn test_verify_suite_coverage_only_skips_static_gates_and_enforces_50_pct_threshold() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let _guard = CurrentDirGuard(env::current_dir().unwrap());
        let root = manifest_cache::workspace_root_dir();
        if root != std::path::Path::new(".") {
            env::set_current_dir(root).unwrap();
        }

        let (_, members) = manifest_cache::discover_workspace();
        let cov_only = parse_verify_suite_args(&["--coverage-only".to_string()]).unwrap();
        let mut static_calls = 0usize;

        let low = run_verify_suite_with(
            cov_only,
            &members,
            |_, _| {
                static_calls += 1;
                Vec::new()
            },
            || Ok(sample_cov_json(1000, 499)),
        )
        .expect("coverage JSON should evaluate cleanly");
        assert!(!low, "49.9% coverage must fail 50.0% gate");
        assert_eq!(
            static_calls, 0,
            "--coverage-only must skip static/OSV gates"
        );

        let ok_cov = run_verify_suite_with(
            cov_only,
            &members,
            |_, _| {
                static_calls += 1;
                Vec::new()
            },
            || Ok(sample_cov_json(1000, 500)),
        )
        .expect("coverage JSON should evaluate cleanly");
        assert!(ok_cov && static_calls == 0, "50.0% coverage must pass");

        let default_opts = parse_verify_suite_args(&[]).unwrap();
        let full = run_verify_suite_with(
            default_opts,
            &members,
            |_, _| {
                static_calls += 1;
                Vec::new()
            },
            || Ok(sample_cov_json(1000, 750)),
        )
        .expect("default verify-suite should evaluate cleanly");
        assert!(full && static_calls == 1);
        assert!(
            run_verify_suite_with(
                cov_only,
                &members,
                |_, _| Vec::new(),
                || Err("llvm-cov failed".to_string()),
            )
            .is_err()
        );
    }

    #[test]
    fn test_verify_suite_security_checks_parse_each_member_manifest_once() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let _guard = CurrentDirGuard(env::current_dir().unwrap());
        let root = manifest_cache::workspace_root_dir();
        if root != std::path::Path::new(".") {
            env::set_current_dir(root).unwrap();
        }

        let (_, members) = manifest_cache::discover_workspace();
        let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
        let mut ast_cache = oxidize_core::traceability::AstCache::new();
        let mut manifests = manifest_cache::ManifestCache::new();

        let (passed, entropy_debt) = run_security_checks_with_hooks(
            &member_refs,
            false,
            &mut ast_cache,
            &mut manifests,
            || true,
            || true,
        );
        assert!(passed);
        assert!(entropy_debt.is_empty());
        assert_eq!(manifests.lint_check_calls(), members.len());
        assert_eq!(manifests.parse_calls(), members.len() + 1);
        for member in &members {
            let manifest = std::path::Path::new(member).join("Cargo.toml");
            assert_eq!(manifests.parse_count_for(&manifest), 1);
        }
    }
}
