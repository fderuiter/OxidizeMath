#![doc = include_str!("../README.md")]
mod markdown_links;
mod papers;

use markdown_links::check_markdown_links;
use std::env;
#[cfg(unix)]
use std::fs;
use std::path::Path;
use std::process::{Command, exit};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "-h" || args[1] == "--help" {
        print_help();
        if args.len() < 2 {
            exit(1);
        } else {
            exit(0);
        }
    }
    match args[1].as_str() {
        "setup" => setup(),
        "lint-all" | "lint" => lint_all(),
        "check-cargo-docs" | "check-workspace-docs" => check_cargo_docs(),
        "check-docs" => check_docs(),
        "check-entropy" => check_entropy(),
        "test-features" => test_features(&args[2..]),
        "verify-suite" => verify_suite(&args[2..]),
        "verify-records" => verify_records(),
        "compile-papers" => compile_papers(),
        "traceability" => traceability(),
        "check-file-lengths" => check_file_lengths(),
        "check-staged-duplicates" => check_staged_duplicates(),
        "check-markdown-links" => check_markdown_links(),
        "check-vulnerabilities" => check_vulnerabilities(),
        "test-policy-audit" => test_policy_audit(),
        "lint-workflows" => lint_workflows(),
        "regenerate-baseline" => regenerate_baseline(),
        _ => {
            println!("Unknown command");
            exit(1);
        }
    }
}

fn print_help() {
    println!("Usage: xtask <command> [args...]\n");
    println!("Active Quality Gates:");
    println!("  check-cargo-docs         - Verify workspace documentation builds without warnings");
    println!("  check-docs               - Run crate documentation audit");
    println!("  check-entropy            - Scan codebase for high entropy strings and secrets");
    println!(
        "  check-file-lengths       - Run centralized verification suite for file-length constraints"
    );
    println!(
        "  check-markdown-links     - Check all internal relative markdown documentation links"
    );
    println!("  check-staged-duplicates  - Run staged duplicates check");
    println!("  check-vulnerabilities    - Scan workspace dependencies for known vulnerabilities");
    println!(
        "  lint-all                 - Run format, clippy, file-length, duplicate, doc, doctest, workflow, traceability, entropy, vulnerability, and policy audit checks"
    );
    println!("  lint-workflows           - Lint CI workflow definitions");
    println!("  test-features            - Run tests across feature combinations");
    println!(
        "  test-policy-audit        - Detect unauthorized dead-code bypasses and unused dependencies"
    );
    println!("  traceability             - Generate traceability report");
    println!("  verify-records           - Verify verification records");
    println!("  verify-suite             - Run high-integrity verified suite\n");
    println!("Legacy Utility Tasks:");
    println!("  compile-papers           - Compile academic papers with Tectonic");
    println!("  regenerate-baseline      - Regenerate public API baseline");
    println!("  setup                    - Run setup script and configure git hooks");
}

fn run_cmd(cmd: &str, args: &[&str]) {
    run_cmd_with_env(cmd, args, &[]);
}

fn run_cmd_with_env(cmd: &str, args: &[&str], envs: &[(&str, &str)]) {
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

fn run_clippy() {
    run_cmd(
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
    );
}

fn lint_all() {
    println!("=== Running Workspace Lint Suite ===");
    println!("\n--- Step 1: Format Check ---");
    run_cmd("cargo", &["fmt", "--all", "--", "--check"]);

    println!("\n--- Step 2: Clippy Lints ---");
    run_clippy();

    println!("\n--- Step 3: Staged Duplicates Check ---");
    check_staged_duplicates();

    println!("\n--- Step 4: File Length Checks ---");
    check_file_lengths();

    println!("\n--- Step 5: Cargo Doc Verification ---");
    check_cargo_docs();

    println!("\n--- Step 6: Doc Tests ---");
    run_cmd("cargo", &["test", "--doc", "--workspace", "--all-features"]);

    println!("\n--- Step 7: Markdown Documentation Links ---");
    check_markdown_links();

    println!("\n--- Step 8: CI Workflow Linting ---");
    lint_workflows();

    println!("\n--- Step 9: Crate Documentation Audit ---");
    check_docs();

    println!("\n--- Step 10: Traceability Check ---");
    traceability();

    println!("\n--- Step 11: Entropy and Secret Scanning ---");
    check_entropy();

    println!("\n--- Step 12: Dependency Vulnerability Check ---");
    check_vulnerabilities();

    println!("\n--- Step 13: Policy Audit ---");
    test_policy_audit();

    println!("\n=== All Lint Checks Passed Successfully ===");
}

fn configure_git_hooks() {
    println!("\n--- Configuring Git Hooks ---");
    run_cmd("git", &["config", "core.hooksPath", ".githooks"]);

    let hook_path = Path::new(".githooks/pre-commit");
    if hook_path.exists() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = fs::metadata(hook_path) {
                let mut perms = metadata.permissions();
                if perms.mode() & 0o111 == 0 {
                    perms.set_mode(0o755);
                    if let Err(e) = fs::set_permissions(hook_path, perms) {
                        eprintln!(
                            "Warning: Failed to set executable permissions on {}: {}",
                            hook_path.display(),
                            e
                        );
                    } else {
                        println!("Pre-commit hook permissions set to executable (0755).");
                    }
                } else {
                    println!("Pre-commit hook executable permissions verified.");
                }
            }
        }
        println!("Git core.hooksPath configured to .githooks");
    } else {
        eprintln!(
            "Warning: Expected hook script not found at {}.",
            hook_path.display()
        );
    }
}

fn setup() {
    println!("=== Math Explorer Setup Script ===");
    run_cmd("cargo", &["build"]);
    run_cmd("cargo", &["test"]);

    configure_git_hooks();
    println!("=== Setup Complete ===");
}

fn test_features(args: &[String]) {
    if args.is_empty() {
        let features = [
            "pure_math",
            "applied",
            "ai",
            "biology",
            "climate",
            "epidemiology",
            "physics",
        ];
        println!("=== Running core-only (no features) ===");
        run_cmd(
            "cargo",
            &["test", "-p", "math_explorer", "--no-default-features"],
        );
        for f in features.iter() {
            println!("=== Running tests for feature: {} ===", f);
            run_cmd(
                "cargo",
                &[
                    "test",
                    "-p",
                    "math_explorer",
                    "--no-default-features",
                    "--features",
                    f,
                ],
            );
        }
        println!("=== Running all features ===");
        run_cmd("cargo", &["test", "-p", "math_explorer", "--all-features"]);
        println!("All feature combinations passed successfully!");
    } else {
        let feature = &args[0];
        if feature == "core-only" || feature.is_empty() {
            run_cmd(
                "cargo",
                &["test", "-p", "math_explorer", "--no-default-features"],
            );
        } else {
            run_cmd(
                "cargo",
                &[
                    "test",
                    "-p",
                    "math_explorer",
                    "--no-default-features",
                    "--features",
                    feature,
                ],
            );
        }
    }
}

fn verify_suite(args: &[String]) {
    println!("=== High-Integrity Verified Suite ===");
    let mut cmd_args = vec![
        "run",
        "-p",
        "unified_verification",
        "--release",
        "--",
        "verify-suite",
    ];
    let args_str: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    cmd_args.extend(args_str);
    run_cmd("cargo", &cmd_args);
}

fn run_verification(subcmd: &str) {
    run_cmd(
        "cargo",
        &[
            "run",
            "-p",
            "unified_verification",
            "--release",
            "--",
            subcmd,
        ],
    );
}

fn verify_records() {
    println!("=== Verify Records ===");
    run_verification("verify-records");
}

fn compile_papers() {
    papers::compile_papers();
}

fn traceability() {
    println!("=== Traceability Report ===");
    run_verification("traceability");
}

fn lint_workflows() {
    println!("=== Lint CI Workflows ===");
    run_verification("lint-workflows");
}

fn check_docs() {
    println!("=== Check Crate Documentation ===");
    run_verification("check-docs");
}

fn check_cargo_docs() {
    println!("=== Cargo Doc Verification ===");
    run_cmd_with_env(
        "cargo",
        &["doc", "--workspace", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    );
}

fn check_file_lengths() {
    println!("=== Check File Lengths ===");
    run_verification("check-file-lengths");
}

fn check_entropy() {
    println!("=== Check Entropy and Secrets ===");
    run_verification("check-entropy");
}

fn check_vulnerabilities() {
    println!("=== Check Dependency Vulnerabilities ===");
    run_verification("check-vulnerabilities");
}

fn test_policy_audit() {
    println!("=== Run Policy Audit ===");
    run_verification("test-policy-audit");
}

fn check_staged_duplicates() {
    run_verification("check-staged-duplicates");
}

fn regenerate_baseline() {
    println!("=== Regenerate Public API Baseline ===");
    run_verification("regenerate-baseline");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markdown_links::{collect_md_files, extract_md_links};

    #[test]
    fn test_git_hooks_path_configuration() {
        configure_git_hooks();
        let output = Command::new("git")
            .args(["config", "core.hooksPath"])
            .output()
            .expect("failed to execute git");
        assert!(output.status.success());
        let hooks_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        assert_eq!(hooks_path, ".githooks");
    }

    #[test]
    #[cfg(unix)]
    fn test_pre_commit_hook_executable() {
        use std::os::unix::fs::PermissionsExt;
        let hook_path = Path::new(".githooks/pre-commit");
        if hook_path.exists() {
            let metadata = fs::metadata(hook_path).expect("failed to read metadata");
            assert_ne!(
                metadata.permissions().mode() & 0o111,
                0,
                "Hook should be executable"
            );
        }
    }

    #[test]
    fn test_xtask_helpers() {
        print_help();
        assert_eq!(extract_md_links("[d](README.md)"), vec!["README.md"]);
        let mut md = Vec::new();
        collect_md_files(Path::new("."), &mut md);
        assert!(!md.is_empty());
        check_markdown_links();
    }

    #[test]
    fn test_verification_gates() {
        let output = Command::new("cargo")
            .args(["run", "-p", "xtask", "--", "--help"])
            .output()
            .expect("failed to execute xtask --help");
        assert!(output.status.success());
        let help = String::from_utf8_lossy(&output.stdout);
        assert!(help.contains("lint-workflows"));
        assert!(help.contains("check-docs"));
        assert!(help.contains("check-cargo-docs"));
        assert!(help.contains("traceability"));
        assert!(help.contains("lint-all"));
        assert!(help.contains("check-entropy"));
        assert!(help.contains("check-vulnerabilities"));
        assert!(help.contains("test-policy-audit"));
    }
}
