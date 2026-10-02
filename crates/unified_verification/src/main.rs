#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_lines, clippy::cognitive_complexity)]
#![allow(clippy::all)]
use std::env;
use std::process::Command;
mod api_drift;
mod ast_visitor;
mod entropy_guard;
mod policy_audit;
mod profile;
mod report;
mod utils;
mod vulnerabilities;
mod workflow_linter;
use utils::{check_file_lengths, check_staged_duplicates};
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: unified_verification <command>");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "verify-records" => {
            if !verify_records() {
                std::process::exit(1);
            }
        }
        "traceability" => traceability(),
        "verify-suite" => verify_suite(&args[2..]),
        "regenerate-baseline" => api_drift::regenerate_baseline(),
        "check-api-drift" => {
            if !api_drift::check_api_drift() {
                std::process::exit(1);
            }
        }
        "check-duplicates" => {
            if !ast_visitor::run_clone_detector() {
                std::process::exit(1);
            }
        }
        "check-staged-duplicates" => check_staged_duplicates(),
        "test-policy-audit" => {
            let (_, members) = oxidize_core::workspace::get_workspace_members();
            let member_refs: Vec<&str> = members.iter().map(|s| s.as_str()).collect();
            if !policy_audit::run_policy_audit(&member_refs) {
                std::process::exit(1);
            }
        }
        "check-entropy" => {
            let (_, members) = oxidize_core::workspace::get_workspace_members();
            let debt = entropy_guard::check_entropy(&members);
            for d in debt {
                println!("{}", d);
            }
        }
        "check-file-lengths" => {
            let (_, members) = oxidize_core::workspace::get_workspace_members();
            let debt = check_file_lengths(&members);
            let has_debt = !debt.is_empty();
            for d in debt {
                println!("{}", d);
            }
            if has_debt {
                std::process::exit(1);
            } else {
                std::process::exit(0);
            }
        }
        "lint-workflows" => {
            if !workflow_linter::lint_workflows() {
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}
fn verify_records() -> bool {
    let base_sha = env::var("BASE_SHA")
        .ok()
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty());
    let head_sha = env::var("HEAD_SHA")
        .ok()
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty());

    let (_commit_msgs, changed_files) =
        if let (Some(base), Some(head)) = (base_sha.as_ref(), head_sha.as_ref()) {
            let msg_out = Command::new("git")
                .args(["log", "--format=%B", &format!("{}..{}", base, head)])
                .output()
                .unwrap();
            let msgs = String::from_utf8_lossy(&msg_out.stdout).to_string();

            let diff_out = Command::new("git")
                .args(["diff", "--name-only", &format!("{}...{}", base, head)])
                .output()
                .unwrap();
            let mut diffs = String::from_utf8_lossy(&diff_out.stdout).to_string();
            if diffs.trim().is_empty() {
                let diff_out = Command::new("git")
                    .args(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"])
                    .output()
                    .unwrap();
                diffs = String::from_utf8_lossy(&diff_out.stdout).to_string();
            }
            (msgs, diffs)
        } else {
            let msg_out = Command::new("git")
                .args(["log", "-1", "--pretty=%B"])
                .output()
                .unwrap();
            let msgs = String::from_utf8_lossy(&msg_out.stdout).to_string();

            let diff_out = Command::new("git")
                .args(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"])
                .output()
                .unwrap();
            let diffs = String::from_utf8_lossy(&diff_out.stdout).to_string();
            (msgs, diffs)
        };

    let changed_files = changed_files.replace("\\", "/");
    println!("Changed files:\n{}", changed_files);

    let changed_lines: Vec<&str> = changed_files.lines().collect();
    let core_modified = changed_lines.iter().any(|l| {
        l.starts_with("math_explorer/")
            || (l.starts_with("crates/") && !l.starts_with("crates/unified_verification/"))
    });

    if core_modified {
        if is_pure_deletion(base_sha.as_deref(), head_sha.as_deref()) {
            println!("Pure deletion detected in core logic. Skipping ADR verification.");
        } else {
            println!("Core logic areas (math_explorer/ or crates/) were modified.");

            let adr_modified = changed_lines
                .iter()
                .any(|l| l.starts_with("docs/adr/") && l.ends_with(".md"));

            if !adr_modified {
                println!(
                    "Verification failed! A Markdown ADR in docs/adr/ must be created or updated when modifying core logic."
                );
                return false;
            }

            println!("ADR verification passed.");
        }
    } else {
        println!("No core logic areas modified. Skipping ADR verification.");
    }

    true
}

fn is_pure_deletion(base_sha: Option<&str>, head_sha: Option<&str>) -> bool {
    let mut output_str = String::new();

    if let (Some(base), Some(head)) = (base_sha, head_sha) {
        if let Some(out) = Command::new("git")
            .args(["diff", &format!("{}...{}", base, head)])
            .output()
            .ok()
        {
            output_str = String::from_utf8_lossy(&out.stdout).to_string();
        }
    }

    if output_str.trim().is_empty() {
        if let Some(out) = Command::new("git")
            .args(["diff", "HEAD~1", "HEAD"])
            .output()
            .ok()
        {
            output_str = String::from_utf8_lossy(&out.stdout).to_string();
        }
    }

    let diff = output_str;
    let mut in_core_file = false;
    let mut added_code_lines = 0;

    for line in diff.lines() {
        if line.starts_with("diff --git") {
            let is_core = line.contains(" a/math_explorer/")
                || (line.contains(" a/crates/")
                    && !line.contains(" a/crates/unified_verification/"));
            in_core_file = is_core;
        } else if in_core_file {
            if line.starts_with('+') && !line.starts_with("+++") {
                let content = line[1..].trim();
                if !content.is_empty()
                    && !content.starts_with("//")
                    && !content.starts_with("/*")
                    && !content.starts_with("*")
                {
                    added_code_lines += 1;
                }
            }
        }
    }
    added_code_lines == 0
}

fn traceability() {
    println!("=== Traceability Report ===");
    println!("Delegating to unified Rust Traceability Engine...");

    let status = Command::new("cargo")
        .args(["run", "--bin", "traceability_cli"])
        .status()
        .expect("Failed to run traceability_cli");

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn verify_suite(args: &[String]) {
    println!("=== High-Integrity Verified Suite ===");

    println!("Running security checks...");
    let mut passed_security = true;
    let auto_fix = args.contains(&"--auto-fix".to_string());

    let (_, members) = oxidize_core::workspace::get_workspace_members();
    let member_refs: Vec<&str> = members.iter().map(|s| s.as_str()).collect();

    if !profile::check_profiles(&member_refs, auto_fix) {
        passed_security = false;
    }

    if !vulnerabilities::check_osv_vulnerabilities() {
        passed_security = false;
    }

    if !ast_visitor::run_ast_visitor() {
        passed_security = false;
    }

    if !ast_visitor::run_clone_detector() {
        passed_security = false;
    }

    if !policy_audit::run_policy_audit(&member_refs) {
        passed_security = false;
    }

    let entropy_debt = entropy_guard::check_entropy(&members);

    if !passed_security {
        eprintln!("Security verification failed!");
        std::process::exit(1);
    }

    println!("Running ADR and API drift checks...");
    if !verify_records() {
        std::process::exit(1);
    }
    if !api_drift::check_api_drift() {
        std::process::exit(1);
    }

    println!("Gathering native execution coverage...");

    let output = report::get_llvm_cov_output();
    let cov_json: serde_json::Value = if output.status.success() {
        serde_json::from_slice(&output.stdout).expect("Failed to parse coverage JSON")
    } else {
        println!("Warning: llvm-cov failed, using empty coverage.");
        serde_json::json!({})
    };

    let rs_files = report::collect_rs_files();
    let (native_lines_total, native_lines_covered) = report::parse_coverage(&cov_json);
    let m = report::analyze_files(&rs_files);
    let mut all_debt = Vec::new();
    all_debt.extend(report::check_unverified_modules(&members));
    all_debt.extend(check_file_lengths(&members));
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

    let passed = report::print_report(native_cov_pct, wasm_cov_pct, &m, &all_debt);

    if !passed {
        std::process::exit(1);
    }
    println!("\nAll integrity checks passed!");
}
