#![doc = include_str!("../README.md")]
mod features;
mod gates;
mod markdown_links;
mod papers;
mod pyramid;
mod setup;
mod smoke;
mod staged;

use gates::resolve_workspace_root;
use markdown_links::check_markdown_links;
#[cfg(test)]
pub(crate) use setup::configure_git_hooks_in_dir;
use std::env;
#[cfg(test)]
use std::fs;
#[cfg(test)]
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
        "setup" => setup(rest),
        "lint-all" | "lint" => gates::lint_all(),
        "lint-push" => staged::lint_push(rest),
        "lint-staged" => staged::lint_staged(),
        "check-benches" => smoke::check_benches(),
        "check-cargo-docs" | "check-workspace-docs" => gates::check_cargo_docs(),
        "check-docs" => gates::check_docs(),
        "check-entropy" => gates::check_entropy(),
        "smoke-examples" => smoke::smoke_examples(),
        "test-features" => features::test_features(rest),
        "verify-suite" => gates::verify_suite(rest),
        "check-certificate" => gates::check_certificate(rest),
        "verify-records" => gates::verify_records(rest),
        "compile-papers" => papers::compile_papers(),
        "traceability" => gates::traceability(),
        "check-file-lengths" => gates::check_file_lengths(),
        "check-duplicates" => gates::check_duplicates(),
        "check-api-drift" => gates::check_api_drift(),
        "check-maturity" => gates::check_maturity(rest),
        "check-dag" => gates::check_dag(),
        "check-profiles" => gates::check_profiles(),
        "check-release-tag" => gates::check_release_tag(rest),
        "check-staged-duplicates" => gates::check_staged_duplicates(),
        "check-markdown-links" => check_markdown_links(),
        "check-vulnerabilities" => gates::check_vulnerabilities(),
        "test-policy-audit" => gates::test_policy_audit(),
        "test-wasm" => gates::test_wasm(rest),
        "lint-workflows" => gates::lint_workflows(),
        "regenerate-baseline" => gates::regenerate_baseline(rest),
        other => {
            if let Some((canonical, banner)) = pyramid::resolve_pyramid_subcommand(other) {
                pyramid::run_pyramid_gate(canonical, banner);
            } else {
                println!("Unknown command");
                exit(1);
            }
        }
    }
}

fn print_help() {
    println!("Usage: xtask <command> [args...]\n");
    println!("Active Quality Gates:");
    println!("  check-api-drift          - Validate public API surface against baseline");
    println!("  check-benches            - Build and run workspace benchmark targets in test mode");
    println!(
        "  check-cargo-docs         - Verify workspace documentation builds with --all-features and zero warnings"
    );
    println!(
        "  check-certificate        - Verify Verification_Certificate.json schema and [verify_suite] floors"
    );
    println!(
        "  check-dag                - Verify strict 5-tier DAG and isolated single-domain feature closures"
    );
    println!("  check-docs               - Run crate documentation audit");
    println!("  check-duplicates         - Run AST-level code clone detector across crates");
    println!("  check-entropy            - Scan codebase for high entropy strings and secrets");
    println!(
        "  check-file-lengths       - Run centralized verification suite for file-length constraints"
    );
    println!(
        "  check-markdown-links     - Check all internal relative markdown documentation links"
    );
    println!(
        "  check-maturity           - Verify Experimental, Validated, and Certified maturity qualification gates"
    );
    println!("  check-profiles           - Verify Cargo release profile invariants");
    println!(
        "  check-release-tag        - Verify git release tag parity and WiX pre-release mapping"
    );
    println!("  check-staged-duplicates  - Run staged duplicates check");
    println!("  check-vulnerabilities    - Scan workspace dependencies for known vulnerabilities");
    println!(
        "  compile-papers           - Compile academic papers with TeXLive (pdflatex and BibTeX)"
    );
    println!(
        "  lint-all                 - Run format, profile, AST duplicate, API drift, ADR, file-length, clippy, doc, doctest, markdown-link, workflow, traceability, entropy, vulnerability, and policy audit checks"
    );
    println!(
        "  lint-push                - Run offline pre-push reverse-dependent Clippy and full workspace traceability"
    );
    println!("  lint-staged              - Run fast scoped pre-commit checks on staged files only");
    println!("  lint-workflows           - Lint CI workflow definitions");
    println!(
        "  smoke-examples           - Build and run all registered math_explorer examples with MATH_EXPLORER_SMOKE=1"
    );
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
    println!("  verify-suite             - Run high-integrity verified suite");
    pyramid::print_pyramid_help();
    print_legacy_help();
}

fn print_legacy_help() {
    println!("Legacy Utility Tasks:");
    println!("  regenerate-baseline      - Regenerate public API baseline");
    println!(
        "  setup                    - Configure .githooks and check workspace (--hooks-only, --check-only, --web)"
    );
}

fn setup(args: &[String]) {
    let ws_root = resolve_workspace_root();
    setup::run_setup(args, &ws_root);
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod ci_filter_tests;
