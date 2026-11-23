#![doc = include_str!("../README.md")]
use std::env;
mod api_drift;
mod ast_helpers;
mod ast_visitor;
mod batch;
mod doc_audit;
mod entropy_guard;
mod manifest_cache;
mod policy_audit;
mod profile;
mod records;
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

fn run_check_certificate_cli() -> bool {
    let _root_guard = manifest_cache::WorkspaceRootGuard::enter();
    let (_, members) = manifest_cache::discover_workspace();
    let mut cache = oxidize_core::traceability::AstCache::new();
    report::certificate::check_ast_certificate_parity_with_cache(&members, &mut cache)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VerifySuiteOptions {
    mode: report::ReportMode,
    auto_fix: bool,
}

fn parse_verify_suite_args(args: &[String]) -> Result<VerifySuiteOptions, String> {
    let mut saw_check = false;
    let mut saw_write = false;
    let mut auto_fix = false;

    for arg in args {
        match arg.as_str() {
            "--check" => saw_check = true,
            "--write" => saw_write = true,
            "--auto-fix" => auto_fix = true,
            other => {
                return Err(format!(
                    "Unknown verify-suite flag `{other}` (expected `--check`, `--write`, or `--write --auto-fix`)"
                ));
            }
        }
    }
    if saw_check && saw_write {
        return Err("Cannot pass both `--check` and `--write` to `verify-suite`".to_string());
    }
    let mode = if saw_write {
        report::ReportMode::Write
    } else {
        report::ReportMode::Check
    };
    if auto_fix && mode != report::ReportMode::Write {
        return Err(
            "`--auto-fix` requires `--write` so read-only `--check` mode never mutates files"
                .to_string(),
        );
    }
    Ok(VerifySuiteOptions { mode, auto_fix })
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
        "check-certificate" => exit_if_failed(run_check_certificate_cli()),
        "regenerate-baseline" => api_drift::regenerate_baseline(),
        "check-api-drift" => exit_if_failed(api_drift::check_api_drift()),
        "check-duplicates" => exit_if_failed(ast_visitor::run_clone_detector()),
        "check-profiles" => run_check_profiles_cli(),
        "check-staged-duplicates" => check_staged_duplicates(),
        "test-policy-audit" => run_policy_audit_cli(&args[2..]),
        "check-entropy" => run_check_entropy_cli(&args[2..]),
        "check-vulnerabilities" => exit_if_failed(vulnerabilities::check_osv_vulnerabilities()),
        "check-file-lengths" => run_check_file_lengths_cli(&args[2..]),
        "lint-workflows" => exit_if_failed(workflow_linter::lint_workflows()),
        "check-docs" => exit_if_failed(doc_audit::run_doc_audit(&args[2..])),
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
    println!("Running security and traceability checks...");
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
    if !batch::run_traceability_with_cache(auto_fix, cache) {
        passed_security = false;
    }
    if !policy_audit::run_policy_audit_with_caches(member_refs, cache, manifest_cache) {
        passed_security = false;
    }
    let entropy_debt = entropy_guard::check_entropy_with_cache(&[], cache);
    if !entropy_debt.is_empty() {
        for item in &entropy_debt {
            println!("{item}");
        }
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

fn evaluate_coverage_and_report(
    members: &[String],
    entropy_debt: &[String],
    mode: report::ReportMode,
    cache: &mut oxidize_core::traceability::AstCache,
) -> bool {
    println!("Gathering native execution coverage...");

    let output = report::get_llvm_cov_output(mode);
    if !output.status.success() {
        eprintln!(
            "Error running cargo llvm-cov:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::process::exit(1);
    }
    let cov_json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("Failed to parse coverage JSON");

    let rs_files = report::collect_rs_files();
    let (native_lines_total, native_lines_covered) = report::parse_coverage(&cov_json);
    let m = report::analyze_files_with_cache(&rs_files, cache);
    let unverified_crates = report::find_unverified_crates_with_cache(members, cache);
    let file_len_debt = check_file_lengths(members);

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

    let debt = report::DebtSources {
        unverified_crates: &unverified_crates,
        file_length_debt: &file_len_debt,
        entropy_debt,
        extra_debt: &[],
    };
    report::run_report_with_mode(native_cov_pct, wasm_cov_pct, &m, &debt, mode)
}

fn verify_suite(args: &[String]) {
    let opts = match parse_verify_suite_args(args) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("Error: {err}");
            std::process::exit(1);
        }
    };
    println!("=== High-Integrity Verified Suite ({:?}) ===", opts.mode);

    let _root_guard = manifest_cache::WorkspaceRootGuard::enter();
    let (_, members) = manifest_cache::discover_workspace();
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut cache = oxidize_core::traceability::AstCache::new();
    let mut manifests = manifest_cache::ManifestCache::new();
    let entropy_debt = run_security_checks(&member_refs, opts.auto_fix, &mut cache, &mut manifests);

    println!("Running ADR and API drift checks...");
    if !records::verify_records(&[]) {
        std::process::exit(1);
    }
    if !api_drift::check_api_drift_with_cache(&mut cache) {
        std::process::exit(1);
    }

    if !evaluate_coverage_and_report(&members, &entropy_debt, opts.mode, &mut cache) {
        std::process::exit(1);
    }
    println!(
        "\nAll integrity checks passed! ({} unique Rust files parsed)",
        cache.parse_calls()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::ReportMode;
    use crate::utils::test_utils::CurrentDirGuard;

    fn parse(args: &[&str]) -> Result<VerifySuiteOptions, String> {
        let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
        parse_verify_suite_args(&owned)
    }

    #[test]
    fn verify_suite_defaults_to_read_only_check_and_validates_flags() {
        assert_eq!(
            parse(&[]).unwrap(),
            VerifySuiteOptions {
                mode: ReportMode::Check,
                auto_fix: false
            }
        );
        assert_eq!(
            parse(&["--check"]).unwrap(),
            VerifySuiteOptions {
                mode: ReportMode::Check,
                auto_fix: false
            }
        );
        assert_eq!(
            parse(&["--write"]).unwrap(),
            VerifySuiteOptions {
                mode: ReportMode::Write,
                auto_fix: false
            }
        );
        assert_eq!(
            parse(&["--write", "--auto-fix"]).unwrap(),
            VerifySuiteOptions {
                mode: ReportMode::Write,
                auto_fix: true
            }
        );
        assert!(parse(&["--check", "--write"]).is_err());
        assert!(parse(&["--auto-fix"]).is_err());
        assert!(parse(&["--check", "--auto-fix"]).is_err());
        assert!(parse(&["--unknown"]).is_err());
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
