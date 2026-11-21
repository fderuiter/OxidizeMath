#![doc = include_str!("../README.md")]
mod gates;
mod markdown_links;
mod papers;

use gates::run_cmd;
use markdown_links::check_markdown_links;
use std::env;
#[cfg(any(unix, test))]
use std::fs;
use std::path::Path;
use std::process::exit;

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
    dispatch_command(&args[1], &args[2..]);
}

fn dispatch_command(cmd: &str, rest: &[String]) {
    match cmd {
        "setup" => setup(),
        "lint-all" | "lint" => gates::lint_all(),
        "check-cargo-docs" | "check-workspace-docs" => gates::check_cargo_docs(),
        "check-docs" => gates::check_docs(),
        "check-entropy" => gates::check_entropy(),
        "test-features" => test_features(rest),
        "verify-suite" => gates::verify_suite(rest),
        "verify-records" => gates::verify_records(rest),
        "compile-papers" => papers::compile_papers(),
        "traceability" => gates::traceability(),
        "check-file-lengths" => gates::check_file_lengths(),
        "check-duplicates" => gates::check_duplicates(),
        "check-api-drift" => gates::check_api_drift(),
        "check-profiles" => gates::check_profiles(),
        "check-staged-duplicates" => gates::check_staged_duplicates(),
        "check-markdown-links" => check_markdown_links(),
        "check-vulnerabilities" => gates::check_vulnerabilities(),
        "test-policy-audit" => gates::test_policy_audit(),
        "test-wasm" => gates::test_wasm(rest),
        "lint-workflows" => gates::lint_workflows(),
        "regenerate-baseline" => gates::regenerate_baseline(),
        _ => {
            println!("Unknown command");
            exit(1);
        }
    }
}

fn print_help() {
    println!("Usage: xtask <command> [args...]\n");
    println!("Active Quality Gates:");
    println!("  check-api-drift          - Validate public API surface against baseline");
    println!("  check-cargo-docs         - Verify workspace documentation builds without warnings");
    println!("  check-docs               - Run crate documentation audit");
    println!("  check-duplicates         - Run AST-level code clone detector across crates");
    println!("  check-entropy            - Scan codebase for high entropy strings and secrets");
    println!(
        "  check-file-lengths       - Run centralized verification suite for file-length constraints"
    );
    println!(
        "  check-markdown-links     - Check all internal relative markdown documentation links"
    );
    println!("  check-profiles           - Verify Cargo release profile invariants");
    println!("  check-staged-duplicates  - Run staged duplicates check");
    println!("  check-vulnerabilities    - Scan workspace dependencies for known vulnerabilities");
    println!(
        "  lint-all                 - Run format, profile, AST duplicate, API drift, staged ADR, file-length, clippy, doc, doctest, markdown-link, workflow, traceability, entropy, vulnerability, and policy audit checks"
    );
    println!("  lint-workflows           - Lint CI workflow definitions");
    println!("  test-features            - Run tests across feature combinations");
    println!(
        "  test-policy-audit        - Detect unauthorized dead-code bypasses and unused dependencies"
    );
    println!(
        "  test-wasm                - Run wasm32 unit tests and headless Chromium smoke checks"
    );
    println!("  traceability             - Generate traceability report");
    println!(
        "  verify-records           - Verify architectural decision records (--staged supported)"
    );
    println!("  verify-suite             - Run high-integrity verified suite\n");
    println!("Legacy Utility Tasks:");
    println!("  compile-papers           - Compile academic papers with Tectonic");
    println!("  regenerate-baseline      - Regenerate public API baseline");
    println!("  setup                    - Run setup script and configure git hooks");
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
        for f in &features {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{
        HOST_CLIPPY_ARGS, WASM_CLIPPY_ARGS, WASM_GUI_TEST_ARGS, WASM_OXIDIZE_CORE_TEST_ARGS,
    };
    use crate::markdown_links::{collect_md_files, extract_md_links};
    use std::process::Command;

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
        for cmd in [
            "check-api-drift",
            "check-duplicates",
            "check-profiles",
            "lint-workflows",
            "check-docs",
            "check-cargo-docs",
            "traceability",
            "lint-all",
            "check-entropy",
            "check-vulnerabilities",
            "test-policy-audit",
            "test-wasm",
            "verify-records",
            "verify-suite",
            "regenerate-baseline",
        ] {
            assert!(help.contains(cmd), "Missing help entry for {cmd}");
        }
    }

    #[test]
    fn test_wasm_clippy_args_and_ci_parity() {
        assert_eq!(
            HOST_CLIPPY_ARGS,
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ]
        );
        assert_eq!(
            WASM_CLIPPY_ARGS,
            &[
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
            ]
        );
        assert!(!WASM_CLIPPY_ARGS.contains(&"--all-targets"));
        assert!(!WASM_CLIPPY_ARGS.contains(&"--benches"));

        let ci_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml");
        let ci_yaml = fs::read_to_string(&ci_path)
            .expect("failed to read ci.yml")
            .replace("\r\n", "\n");
        let expected_host_cmd = format!("cargo {}", HOST_CLIPPY_ARGS.join(" "));
        assert!(ci_yaml.contains("name: Run Clippy (Native)"));
        assert!(ci_yaml.contains(&expected_host_cmd));
        let expected_wasm_cmd = format!("cargo {}", WASM_CLIPPY_ARGS.join(" "));
        assert!(ci_yaml.contains("name: Run Clippy (WASM)"));
        assert!(ci_yaml.contains(&expected_wasm_cmd));
        assert!(ci_yaml.contains(
            "- name: Run Workspace Tests (Native)\n        if: runner.os != 'Linux' && matrix.target != 'wasm32-unknown-unknown'\n        run: cargo test --workspace --lib --tests --all-features"
        ));

        for preflight_cmd in [
            "cargo run -p unified_verification --release -- check-profiles",
            "cargo run -p unified_verification --release -- check-duplicates",
            "cargo run -p unified_verification --release -- check-api-drift",
        ] {
            assert!(
                ci_yaml.contains(preflight_cmd),
                "ci.yml missing pre-flight command: {preflight_cmd}"
            );
        }
    }

    #[test]
    fn test_wasm_test_and_smoke_ci_parity() {
        assert_eq!(
            WASM_OXIDIZE_CORE_TEST_ARGS,
            &[
                "test",
                "--target",
                "wasm32-unknown-unknown",
                "-p",
                "oxidize_core",
                "--lib"
            ]
        );
        assert_eq!(
            WASM_GUI_TEST_ARGS,
            &[
                "test",
                "--target",
                "wasm32-unknown-unknown",
                "-p",
                "math_explorer_gui",
                "--bin",
                "math_explorer_gui"
            ]
        );
        let ci_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/ci.yml");
        let ci_yaml = fs::read_to_string(&ci_path)
            .expect("failed to read ci.yml")
            .replace("\r\n", "\n");
        let build_pos = ci_yaml
            .find("trunk build --release")
            .expect("ci.yml must build web app with trunk");
        let smoke_pos = ci_yaml
            .find("cargo run -p xtask -- test-wasm --skip-build")
            .expect("ci.yml must run xtask test-wasm gate");
        let upload_pos = ci_yaml
            .find("name: Upload Web Dist Artifact")
            .expect("ci.yml must upload wasm_web_dist artifact");
        assert!(smoke_pos > build_pos && smoke_pos < upload_pos);
    }
}
