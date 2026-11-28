#![doc = include_str!("../README.md")]
use std::env;
mod api_drift;
mod ast_helpers;
mod ast_visitor;
mod batch;
mod dag_gate;
mod doc_audit;
mod entropy_guard;
mod error_contracts;
mod manifest_cache;
mod maturity;
mod policy_audit;
mod profile;
mod pyramid_cli;
mod records;
mod release_tag;
mod report;
mod staged_traceability;
mod utils;
mod vulnerabilities;
mod wasm_coverage;
mod whitelist;
mod workflow_linter;
use utils::{check_file_lengths, check_staged_duplicates};

#[cfg(test)]
/// Test mutex to synchronize tests that modify process working directory.
pub static DIR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
/// Executes `f` against a shared read-only workspace `AstCache` owned by a dedicated worker
/// thread so `!Send` `syn::File` spans never cross threads while avoiding repeated re-parses.
pub(crate) fn with_workspace_ast_cache<R: Send + 'static>(
    f: impl FnOnce(&mut oxidize_core::traceability::AstCache) -> R + Send + 'static,
) -> R {
    type Job = Box<dyn FnOnce(&mut oxidize_core::traceability::AstCache) + Send>;
    static WORKER: std::sync::OnceLock<std::sync::Mutex<std::sync::mpsc::SyncSender<Job>>> =
        std::sync::OnceLock::new();
    let tx = WORKER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Job>(0);
        std::thread::spawn(move || {
            let mut cache = oxidize_core::traceability::AstCache::new();
            while let Ok(job) = rx.recv() {
                job(&mut cache);
            }
        });
        std::sync::Mutex::new(tx)
    });
    let (res_tx, res_rx) = std::sync::mpsc::sync_channel(1);
    tx.lock()
        .unwrap_or_else(|e| e.into_inner())
        .send(Box::new(move |cache| {
            let _ = res_tx.send(f(cache));
        }))
        .expect("workspace AST cache worker exited");
    res_rx.recv().expect("workspace AST cache job panicked")
}

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

fn parse_check_certificate_mode(args: &[String]) -> Result<report::ReportMode, String> {
    let mut saw_check = false;
    let mut saw_strict = false;
    let mut saw_write = false;
    for arg in args {
        match arg.as_str() {
            "--check" => saw_check = true,
            "--strict-parity" | "--strict-certificate" => saw_strict = true,
            "--write" | "--write-tracked" => saw_write = true,
            _ => {
                return Err(
                    "Usage: unified_verification check-certificate [--check | --strict-parity | --write]"
                        .to_string(),
                );
            }
        }
    }
    if saw_write && (saw_check || saw_strict) {
        return Err(
            "Cannot combine `--write` with `--check` or `--strict-parity` in `check-certificate`"
                .to_string(),
        );
    }
    if saw_write {
        Ok(report::ReportMode::Write)
    } else if saw_strict {
        Ok(report::ReportMode::StrictParity)
    } else {
        Ok(report::ReportMode::Check)
    }
}

fn run_check_certificate_cli(args: &[String]) -> bool {
    let mode = match parse_check_certificate_mode(args) {
        Ok(mode) => mode,
        Err(err) => {
            eprintln!("Error: {err}");
            return false;
        }
    };
    let _root_guard = manifest_cache::WorkspaceRootGuard::enter();
    let (_, members) = manifest_cache::discover_workspace();
    let mut cache = oxidize_core::traceability::AstCache::new();
    report::certificate::sync_ast_certificate_with_cache(&members, mode, &mut cache)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifySuiteOptions {
    pub mode: report::ReportMode,
    pub auto_fix: bool,
    pub coverage_only: bool,
    pub output_cert: Option<std::path::PathBuf>,
}

fn parse_non_empty_cert_path(raw: &str) -> Result<std::path::PathBuf, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("--output-cert requires a non-empty file path".to_string());
    }
    Ok(std::path::PathBuf::from(trimmed))
}

pub(crate) fn parse_verify_suite_args(args: &[String]) -> Result<VerifySuiteOptions, String> {
    let mut saw_check = false;
    let mut saw_strict = false;
    let mut saw_write = false;
    let mut auto_fix = false;
    let mut coverage_only = false;
    let mut output_cert = None;

    let mut idx = 0usize;
    while idx < args.len() {
        let arg = args[idx].as_str();
        match arg {
            "--check" => saw_check = true,
            "--strict-parity" | "--strict-certificate" => saw_strict = true,
            "--write" | "--write-tracked" => saw_write = true,
            "--auto-fix" => auto_fix = true,
            "--coverage-only" => coverage_only = true,
            "--output-cert" => {
                idx += 1;
                let Some(next) = args.get(idx) else {
                    return Err("Missing value for --output-cert <path>".to_string());
                };
                output_cert = Some(parse_non_empty_cert_path(next)?);
            }
            _ if arg.starts_with("--output-cert=") => {
                let raw = &arg["--output-cert=".len()..];
                output_cert = Some(parse_non_empty_cert_path(raw)?);
            }
            other => {
                return Err(format!(
                    "Unknown verify-suite flag `{other}` (expected `--check`, `--strict-parity`, `--write`, `--write-tracked`, `--coverage-only`, `--output-cert <path>`, or `--write --auto-fix`)"
                ));
            }
        }
        idx += 1;
    }
    if saw_write && (saw_check || saw_strict) {
        return Err(
            "Cannot pass `--write` together with `--check` or `--strict-parity` to `verify-suite`"
                .to_string(),
        );
    }
    let mode = resolve_suite_mode(saw_write, saw_strict, auto_fix)?;
    Ok(VerifySuiteOptions {
        mode,
        auto_fix,
        coverage_only,
        output_cert,
    })
}

fn resolve_suite_mode(
    saw_write: bool,
    saw_strict: bool,
    auto_fix: bool,
) -> Result<report::ReportMode, String> {
    let mode = if saw_write {
        report::ReportMode::Write
    } else if saw_strict {
        report::ReportMode::StrictParity
    } else {
        report::ReportMode::Check
    };
    if auto_fix && mode != report::ReportMode::Write {
        return Err(
            "`--auto-fix` requires `--write` so read-only `--check` mode never mutates files"
                .to_string(),
        );
    }
    Ok(mode)
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
        "check-dag" => exit_if_failed(dag_gate::run_check_dag_cli()),
        "verify-suite" => verify_suite(&args[2..]),
        "check-certificate" => exit_if_failed(run_check_certificate_cli(&args[2..])),
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
        cmd if pyramid_cli::is_pyramid_command(cmd) => {
            let mut ok = pyramid_cli::dispatch_pyramid_cmd(cmd);
            if ok && cmd == "verify-pyramid" && args[2..].iter().any(|a| a == "--write-certificate")
            {
                ok = run_check_certificate_cli(&["--write".to_string()]);
            }
            exit_if_failed(ok);
        }
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

pub(crate) fn evaluate_coverage_and_report_with<F>(
    members: &[String],
    entropy_debt: &[String],
    opts: &VerifySuiteOptions,
    cache: &mut oxidize_core::traceability::AstCache,
    collect_cov: F,
) -> Result<bool, String>
where
    F: FnOnce(report::ReportMode) -> Result<serde_json::Value, String>,
{
    println!("Gathering native execution coverage...");

    let cov_json = collect_cov(opts.mode)?;
    let rs_files = report::collect_rs_files();
    let (native_lines_total, native_lines_covered) = report::parse_coverage(&cov_json);
    let m = report::analyze_files_with_cache(&rs_files, cache);
    let unverified_crates = report::find_unverified_crates_with_cache(members, cache);
    let file_len_debt = if opts.coverage_only {
        Vec::new()
    } else {
        check_file_lengths(members)
    };

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
    Ok(report::run_report_with_options(
        native_cov_pct,
        wasm_cov_pct,
        &m,
        &debt,
        opts.mode,
        opts.output_cert.as_deref(),
    ))
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
    if !pyramid_cli::dispatch_pyramid_cmd("verify-pyramid") {
        std::process::exit(1);
    }
    entropy_debt
}

pub(crate) fn run_verify_suite_with<S, C>(
    opts: VerifySuiteOptions,
    members: &[String],
    cache: &mut oxidize_core::traceability::AstCache,
    run_static_gates: S,
    collect_cov: C,
) -> Result<bool, String>
where
    S: FnOnce(&[String], bool, &mut oxidize_core::traceability::AstCache) -> Vec<String>,
    C: FnOnce(report::ReportMode) -> Result<serde_json::Value, String>,
{
    let entropy_debt = if opts.coverage_only {
        Vec::new()
    } else {
        run_static_gates(members, opts.auto_fix, cache)
    };
    evaluate_coverage_and_report_with(members, &entropy_debt, &opts, cache, collect_cov)
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
    let mut cache = oxidize_core::traceability::AstCache::new();
    if opts.mode == report::ReportMode::Write
        && !opts.coverage_only
        && !report::certificate::sync_ast_certificate_with_cache(
            &members,
            report::ReportMode::Write,
            &mut cache,
        )
    {
        std::process::exit(1);
    }
    let passed = match run_verify_suite_with(
        opts,
        &members,
        &mut cache,
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
    println!(
        "\nAll integrity checks passed! ({} unique Rust files parsed)",
        cache.parse_calls()
    );
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
