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
        "regenerate-baseline" => api_drift::regenerate_baseline(),
        "check-api-drift" => exit_if_failed(api_drift::check_api_drift()),
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct VerifySuiteOptions {
    pub auto_fix: bool,
    pub coverage_only: bool,
    pub output_cert: Option<std::path::PathBuf>,
    pub write_tracked: bool,
}

fn parse_non_empty_cert_path(raw: &str) -> Result<std::path::PathBuf, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("--output-cert requires a non-empty file path".to_string());
    }
    Ok(std::path::PathBuf::from(trimmed))
}

pub(crate) fn parse_verify_suite_args(args: &[String]) -> Result<VerifySuiteOptions, String> {
    let mut opts = VerifySuiteOptions::default();
    let mut idx = 0;
    while idx < args.len() {
        let arg = args[idx].as_str();
        match arg {
            "--auto-fix" => opts.auto_fix = true,
            "--coverage-only" => opts.coverage_only = true,
            "--write-tracked" => opts.write_tracked = true,
            "--output-cert" => {
                idx += 1;
                let Some(next) = args.get(idx) else {
                    return Err("Missing value for --output-cert <path>".to_string());
                };
                opts.output_cert = Some(parse_non_empty_cert_path(next)?);
            }
            _ if arg.starts_with("--output-cert=") => {
                let raw = &arg["--output-cert=".len()..];
                opts.output_cert = Some(parse_non_empty_cert_path(raw)?);
            }
            other => return Err(format!("Unknown verify-suite argument: {other}")),
        }
        idx += 1;
    }
    Ok(opts)
}

pub(crate) fn evaluate_coverage_and_report_with<F>(
    members: &[String],
    entropy_debt: Vec<String>,
    opts: &VerifySuiteOptions,
    cache: &mut oxidize_core::traceability::AstCache,
    collect_cov: F,
) -> Result<bool, String>
where
    F: FnOnce() -> Result<serde_json::Value, String>,
{
    println!("Gathering native execution coverage...");

    let whitelist_path = whitelist::find_whitelist_path();
    let loaded_whitelist = whitelist::load_whitelist_strict(
        whitelist_path.as_deref(),
        whitelist::WORKSPACE_WHITELIST_CANDIDATES,
    )?;
    if whitelist_path.is_some() && loaded_whitelist.verification_debt.is_none() {
        return Err(
            "Missing required [verification_debt] table in verification_whitelist.toml".to_string(),
        );
    }

    let cov_json = collect_cov()?;
    let rs_files = report::collect_rs_files();
    let (native_lines_total, native_lines_covered) = report::parse_coverage(&cov_json);
    let m = report::analyze_workspace_with_cache(&rs_files, cache)?;
    let default_exempt = ["egui_plot".to_string()];
    let exempt_crates = loaded_whitelist
        .verification_debt
        .as_ref()
        .map(|d| d.exempt_crates.as_slice())
        .unwrap_or(&default_exempt);
    let mut all_debt = Vec::new();
    all_debt.extend(report::check_unverified_modules_with_cache(
        members,
        exempt_crates,
        cache,
    ));
    if !opts.coverage_only {
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

    let write_opts = report::ReportWriteOptions {
        output_cert: opts.output_cert.clone(),
        write_tracked: opts.write_tracked,
    };
    report::print_report_with_options(
        native_cov_pct,
        wasm_cov_pct,
        &m,
        &all_debt,
        loaded_whitelist.verification_debt.as_ref(),
        &write_opts,
    )
}

fn run_default_verify_suite_static_gates(
    members: &[String],
    auto_fix: bool,
    cache: &mut oxidize_core::traceability::AstCache,
) -> Vec<String> {
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut manifests = manifest_cache::ManifestCache::new();
    let entropy_debt = run_security_checks(&member_refs, auto_fix, cache, &mut manifests);

    println!("Running ADR and API drift checks...");
    if !records::verify_records(&[]) {
        std::process::exit(1);
    }
    if !api_drift::check_api_drift_with_cache(cache) {
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
    S: FnOnce(&[String], bool, &mut oxidize_core::traceability::AstCache) -> Vec<String>,
    C: FnOnce() -> Result<serde_json::Value, String>,
{
    let mut cache = oxidize_core::traceability::AstCache::new();
    let entropy_debt = if opts.coverage_only {
        Vec::new()
    } else {
        run_static_gates(members, opts.auto_fix, &mut cache)
    };
    evaluate_coverage_and_report_with(members, entropy_debt, &opts, &mut cache, collect_cov)
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
mod tests;
