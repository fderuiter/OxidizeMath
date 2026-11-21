#![doc = include_str!("../README.md")]
use std::env;
mod api_drift;
mod ast_visitor;
mod batch;
mod doc_audit;
mod entropy_guard;
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

fn run_security_checks(
    member_refs: &[&str],
    auto_fix: bool,
    cache: &mut oxidize_core::traceability::AstCache,
) -> Vec<String> {
    println!("Running security checks...");
    let mut passed_security = true;

    if !profile::check_profiles(member_refs, auto_fix) {
        passed_security = false;
    }
    if !vulnerabilities::check_osv_vulnerabilities() {
        passed_security = false;
    }
    if !ast_visitor::run_ast_visitor() {
        passed_security = false;
    }
    if !ast_visitor::run_clone_detector_with_cache(cache) {
        passed_security = false;
    }
    if !policy_audit::run_policy_audit_with_cache(member_refs, cache) {
        passed_security = false;
    }
    let entropy_debt = entropy_guard::check_entropy_with_cache(&[], cache);
    if !entropy_debt.is_empty() {
        passed_security = false;
    }
    if !passed_security {
        eprintln!("Security verification failed!");
        std::process::exit(1);
    }
    entropy_debt
}

fn evaluate_coverage_and_report(members: &[String], entropy_debt: Vec<String>) -> bool {
    println!("Gathering native execution coverage...");

    let output = report::get_llvm_cov_output();
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
    let m = report::analyze_files(&rs_files);
    let mut all_debt = Vec::new();
    all_debt.extend(report::check_unverified_modules(members));
    all_debt.extend(check_file_lengths(members));
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

    report::print_report(native_cov_pct, wasm_cov_pct, &m, &all_debt)
}

fn verify_suite(args: &[String]) {
    println!("=== High-Integrity Verified Suite ===");
    let auto_fix = args.contains(&"--auto-fix".to_string());

    let (_, members) = oxidize_core::workspace::get_workspace_members();
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut cache = oxidize_core::traceability::AstCache::new();
    let entropy_debt = run_security_checks(&member_refs, auto_fix, &mut cache);

    println!("Running ADR and API drift checks...");
    if !records::verify_records(&[]) {
        std::process::exit(1);
    }
    if !api_drift::check_api_drift_with_cache(&mut cache) {
        std::process::exit(1);
    }

    if !evaluate_coverage_and_report(&members, entropy_debt) {
        std::process::exit(1);
    }
    println!("\nAll integrity checks passed!");
}
