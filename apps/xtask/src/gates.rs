use crate::markdown_links::check_markdown_links;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "../../../crates/oxidize_core/src/workspace_wasm.rs"]
pub mod workspace_wasm;

static VERIFICATION_BUILT: AtomicBool = AtomicBool::new(false);

pub(crate) fn resolve_workspace_root() -> PathBuf {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| workspace_wasm::find_workspace_root(&cwd))
        .or_else(|| workspace_wasm::find_workspace_root(Path::new(env!("CARGO_MANIFEST_DIR"))))
        .unwrap_or_else(|| {
            eprintln!("Error: Unable to locate workspace root Cargo.toml");
            exit(1);
        })
}

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

pub(crate) fn unified_verification_bin() -> PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    let mut bin = target_dir.join("debug").join("unified_verification");
    if !std::env::consts::EXE_EXTENSION.is_empty() {
        bin.set_extension(std::env::consts::EXE_EXTENSION);
    }
    bin
}

fn ensure_verification_built() -> PathBuf {
    if !VERIFICATION_BUILT.swap(true, Ordering::SeqCst) {
        run_cmd("cargo", &["build", "-p", "unified_verification", "-q"]);
    }
    unified_verification_bin()
}

pub fn run_verification_with_env(subcmd: &str, extra_args: &[&str], envs: &[(&str, &str)]) {
    let bin = ensure_verification_built();
    let bin_str = bin.to_string_lossy();
    let mut args = Vec::with_capacity(1 + extra_args.len());
    args.push(subcmd);
    args.extend_from_slice(extra_args);
    run_cmd_with_env(&bin_str, &args, envs);
}

pub fn run_verification_with_args(subcmd: &str, extra_args: &[&str]) {
    run_verification_with_env(subcmd, extra_args, &[]);
}

pub fn run_verification(subcmd: &str) {
    run_verification_with_args(subcmd, &[]);
}

fn git_trim_stdout(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn branch_records_base_sha() -> Option<String> {
    if std::env::var_os("BASE_SHA").is_some() {
        return None;
    }
    if git_trim_stdout(&["diff", "--cached", "--name-only"]).is_some() {
        return None;
    }
    let head = git_trim_stdout(&["rev-parse", "HEAD"])?;
    let merge_base = git_trim_stdout(&["merge-base", "HEAD", "main"])?;
    (merge_base != head).then_some(merge_base)
}

fn run_clippy_lints() {
    println!("\n--- Step 3a: Clippy Lints (Host) ---");
    run_cmd("cargo", HOST_CLIPPY_ARGS);

    println!("\n--- Step 3b: Clippy Lints (WebAssembly) ---");
    run_cmd("cargo", WASM_CLIPPY_ARGS);
}

fn run_fast_preflight_gates() {
    println!("\n--- Step 1: Format Check ---");
    run_cmd("cargo", &["fmt", "--all", "--", "--check"]);

    println!("\n--- Step 2: Batched Static & Single-Pass AST Verification ---");
    if let Some(base_sha) = branch_records_base_sha() {
        run_verification_with_env(
            "batch-static",
            &[],
            &[("BASE_SHA", base_sha.as_str()), ("HEAD_SHA", "HEAD")],
        );
    } else {
        run_verification("batch-static");
    }
}

fn run_compile_and_doc_gates() {
    println!("\n--- Step 3: Clippy Lints ---");
    run_clippy_lints();

    println!("\n--- Step 4: Cargo Doc Verification ---");
    check_cargo_docs();

    println!("\n--- Step 5: Doc Tests ---");
    run_cmd("cargo", &["test", "--doc", "--workspace", "--all-features"]);

    println!("\n--- Step 6: Markdown Documentation Links ---");
    check_markdown_links();
}

fn run_audit_gates() {
    println!("\n--- Step 7: Dependency Vulnerability Check ---");
    check_vulnerabilities();
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

pub(crate) fn run_cmd_in_path(cmd: &str, args: &[&str], dir: &Path) {
    println!("Running (in {}): {cmd} {}", dir.display(), args.join(" "));
    let status = Command::new(cmd).args(args).current_dir(dir).status();
    match status {
        Ok(s) if s.success() => (),
        _ => {
            eprintln!("Command failed!");
            exit(1);
        }
    }
}

fn parse_test_wasm_args(args: &[String], caller_cwd: &Path) -> (bool, Option<PathBuf>) {
    let mut skip_build = false;
    let mut screenshot_dir: Option<PathBuf> = None;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--skip-build" => skip_build = true,
            "--screenshot-dir" if idx + 1 < args.len() => {
                let raw = Path::new(&args[idx + 1]);
                let resolved = if raw.is_absolute() {
                    raw.to_path_buf()
                } else {
                    caller_cwd.join(raw)
                };
                screenshot_dir = Some(resolved);
                idx += 1;
            }
            other => {
                eprintln!("Unknown test-wasm argument: {other}");
                exit(1);
            }
        }
        idx += 1;
    }
    (skip_build, screenshot_dir)
}

pub fn test_wasm(args: &[String]) {
    println!("=== Running WebAssembly Unit Tests & Headless Chromium Smoke Suite ===");
    let caller_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let (skip_build, screenshot_dir) = parse_test_wasm_args(args, &caller_cwd);
    let ws_root = resolve_workspace_root();

    let wasm_crates = workspace_wasm::discover_wasm_crates(&ws_root).unwrap_or_else(|err| {
        eprintln!("Failed to discover wasm32 workspace crates: {err}");
        exit(1);
    });
    let invocations = workspace_wasm::build_wasm_cargo_test_invocations(&wasm_crates);
    for (step_idx, cargo_args) in invocations.iter().enumerate() {
        println!(
            "\n--- Step 1{}: WASM Workspace Crate Tests ({} crates discovered) ---",
            (b'a' + step_idx as u8) as char,
            wasm_crates.len()
        );
        let arg_refs: Vec<&str> = cargo_args.iter().map(String::as_str).collect();
        run_cmd_in_path("cargo", &arg_refs, &ws_root);
    }

    if !skip_build {
        println!("\n--- Step 2: Build Release Web Bundle & Verify Budget ---");
        run_cmd_in_path(
            "trunk",
            &["build", "--release"],
            &ws_root.join("math_explorer_gui"),
        );
        run_cmd_in_path(
            "bash",
            &["scripts/verify_wasm_budget.sh", "math_explorer_gui/dist"],
            &ws_root,
        );
    }

    println!("\n--- Step 3: Headless Chromium CSP & Fault-Injection Smoke Checks ---");
    let screenshot_str = screenshot_dir.map(|p| p.to_string_lossy().into_owned());
    let mut smoke_args = vec![
        "scripts/playwright_smoke.py",
        "--dist",
        "math_explorer_gui/dist",
        "--verify-regression-detection",
    ];
    if let Some(dir) = screenshot_str.as_deref() {
        smoke_args.push("--screenshot-dir");
        smoke_args.push(dir);
    }
    run_cmd_in_path("python3", &smoke_args, &ws_root);
}
