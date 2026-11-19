use crate::markdown_links::check_markdown_links;
use std::path::PathBuf;
use std::process::{Command, exit};
use std::sync::atomic::{AtomicBool, Ordering};

static VERIFICATION_BUILT: AtomicBool = AtomicBool::new(false);

pub const HOST_CLIPPY_ARGS: &[&str] = &[
    "clippy",
    "--workspace",
    "--all-targets",
    "--all-features",
    "--",
    "-D",
    "warnings",
];

pub const WASM_CLIPPY_ARGS: &[&str] = &[
    "clippy",
    "--workspace",
    "--lib",
    "--bins",
    "--tests",
    "--target",
    "wasm32-unknown-unknown",
    "--all-features",
    "--",
    "-D",
    "warnings",
];

pub fn run_cmd(cmd: &str, args: &[&str]) {
    run_cmd_with_env(cmd, args, &[]);
}

pub fn run_cmd_with_env(cmd: &str, args: &[&str], envs: &[(&str, &str)]) {
    println!("Running: {} {}", cmd, args.join(" "));
    let mut command = Command::new(cmd);
    command.args(args);
    for (k, v) in envs {
        command.env(k, v);
    }
    let status = command.status();
    match status {
        Ok(s) if s.success() => (),
        _ => {
            eprintln!("Command failed!");
            exit(1);
        }
    }
}

fn unified_verification_bin() -> PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    let mut bin = target_dir.join("release").join("unified_verification");
    if !std::env::consts::EXE_EXTENSION.is_empty() {
        bin.set_extension(std::env::consts::EXE_EXTENSION);
    }
    bin
}

fn ensure_verification_built() -> PathBuf {
    if !VERIFICATION_BUILT.swap(true, Ordering::SeqCst) {
        run_cmd(
            "cargo",
            &["build", "-p", "unified_verification", "--release", "-q"],
        );
    }
    unified_verification_bin()
}

pub fn run_verification_with_args(subcmd: &str, extra_args: &[&str]) {
    let bin = ensure_verification_built();
    let bin_str = bin.to_string_lossy();
    let mut args = Vec::with_capacity(1 + extra_args.len());
    args.push(subcmd);
    args.extend_from_slice(extra_args);
    run_cmd(&bin_str, &args);
}

pub fn run_verification(subcmd: &str) {
    run_verification_with_args(subcmd, &[]);
}

fn run_clippy_lints() {
    println!("\n--- Step 7a: Clippy Lints (Host) ---");
    run_cmd("cargo", HOST_CLIPPY_ARGS);

    println!("\n--- Step 7b: Clippy Lints (WebAssembly) ---");
    run_cmd("cargo", WASM_CLIPPY_ARGS);
}

fn run_fast_preflight_gates() {
    println!("\n--- Step 1: Format Check ---");
    run_cmd("cargo", &["fmt", "--all", "--", "--check"]);

    println!("\n--- Step 2: Cargo Profile Verification ---");
    check_profiles();

    println!("\n--- Step 3: AST Code Clone Detection ---");
    check_duplicates();

    println!("\n--- Step 4: Public API Drift Check ---");
    check_api_drift();

    println!("\n--- Step 5: Staged Architectural Decision Records ---");
    verify_records_staged();

    println!("\n--- Step 6: File Length Checks ---");
    check_file_lengths();
}

fn run_compile_and_doc_gates() {
    println!("\n--- Step 7: Clippy Lints ---");
    run_clippy_lints();

    println!("\n--- Step 8: Cargo Doc Verification ---");
    check_cargo_docs();

    println!("\n--- Step 9: Doc Tests ---");
    run_cmd("cargo", &["test", "--doc", "--workspace", "--all-features"]);

    println!("\n--- Step 10: Markdown Documentation Links ---");
    check_markdown_links();
}

fn run_audit_gates() {
    println!("\n--- Step 11: CI Workflow Linting ---");
    lint_workflows();

    println!("\n--- Step 12: Crate Documentation Audit ---");
    check_docs();

    println!("\n--- Step 13: Traceability Check ---");
    traceability();

    println!("\n--- Step 14: Entropy and Secret Scanning ---");
    check_entropy();

    println!("\n--- Step 15: Dependency Vulnerability Check ---");
    check_vulnerabilities();

    println!("\n--- Step 16: Policy Audit ---");
    test_policy_audit();
}

pub fn lint_all() {
    println!("=== Running Workspace Lint Suite ===");
    run_fast_preflight_gates();
    run_compile_and_doc_gates();
    run_audit_gates();
    println!("\n=== All Lint Checks Passed Successfully ===");
}

pub fn verify_suite(args: &[String]) {
    println!("=== High-Integrity Verified Suite ===");
    let extra: Vec<&str> = args.iter().map(String::as_str).collect();
    run_verification_with_args("verify-suite", &extra);
}

pub fn verify_records(args: &[String]) {
    println!("=== Verify Records ===");
    let extra: Vec<&str> = args.iter().map(String::as_str).collect();
    run_verification_with_args("verify-records", &extra);
}

pub fn verify_records_staged() {
    println!("=== Verify Staged Records ===");
    run_verification_with_args("verify-records", &["--staged"]);
}

pub fn traceability() {
    println!("=== Traceability Report ===");
    run_verification("traceability");
}

pub fn lint_workflows() {
    println!("=== Lint CI Workflows ===");
    run_verification("lint-workflows");
}

pub fn check_docs() {
    println!("=== Check Crate Documentation ===");
    run_verification("check-docs");
}

pub fn check_cargo_docs() {
    println!("=== Cargo Doc Verification ===");
    run_cmd_with_env(
        "cargo",
        &["doc", "--workspace", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    );
}

pub fn check_file_lengths() {
    println!("=== Check File Lengths ===");
    run_verification("check-file-lengths");
}

pub fn check_duplicates() {
    println!("=== Check AST Code Duplicates ===");
    run_verification("check-duplicates");
}

pub fn check_api_drift() {
    println!("=== Check Public API Drift ===");
    run_verification("check-api-drift");
}

pub fn check_profiles() {
    println!("=== Check Cargo Profiles ===");
    run_verification("check-profiles");
}

pub fn check_entropy() {
    println!("=== Check Entropy and Secrets ===");
    run_verification("check-entropy");
}

pub fn check_vulnerabilities() {
    println!("=== Check Dependency Vulnerabilities ===");
    run_verification("check-vulnerabilities");
}

pub fn test_policy_audit() {
    println!("=== Run Policy Audit ===");
    run_verification("test-policy-audit");
}

pub fn check_staged_duplicates() {
    run_verification("check-staged-duplicates");
}

pub fn regenerate_baseline() {
    println!("=== Regenerate Public API Baseline ===");
    run_verification("regenerate-baseline");
}
