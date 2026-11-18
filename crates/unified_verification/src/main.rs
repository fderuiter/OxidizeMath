#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_lines, clippy::cognitive_complexity)]
#![allow(clippy::all)]
use std::env;
use std::path::Path;
use std::process::Command;
mod api_drift;
mod ast_visitor;
mod doc_audit;
mod entropy_guard;
mod policy_audit;
mod profile;
mod report;
mod utils;
mod vulnerabilities;
mod workflow_linter;
use utils::{check_file_lengths, check_staged_duplicates};

#[cfg(test)]
/// Test mutex to synchronize tests that modify process working directory.
pub static DIR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
            let custom_paths: Vec<String> = if args.len() > 2 {
                args[2..].to_vec()
            } else {
                Vec::new()
            };
            let debt = entropy_guard::check_entropy(&custom_paths);
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
        "check-vulnerabilities" => {
            if !vulnerabilities::check_osv_vulnerabilities() {
                std::process::exit(1);
            } else {
                std::process::exit(0);
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
        "check-docs" => {
            if !doc_audit::run_doc_audit(&args[2..]) {
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}
fn sanitize_sha(raw: Option<&str>) -> Option<String> {
    raw.map(|s| s.trim().trim_matches('"').trim())
        .filter(|s| !s.is_empty() && !s.chars().all(|c| c == '0'))
        .map(ToString::to_string)
}

fn git_stdout(repo_dir: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .current_dir(repo_dir)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
}

fn inspect_head_commit(repo_dir: &Path) -> (String, String) {
    let msgs = git_stdout(repo_dir, &["log", "-1", "--pretty=%B"]).unwrap_or_default();
    let diffs = git_stdout(
        repo_dir,
        &["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
    )
    .unwrap_or_default();
    (msgs, diffs)
}

fn inspect_commit_range(repo_dir: &Path, base: &str, head: &str) -> Option<(String, String)> {
    let Some(mut diffs) = git_stdout(
        repo_dir,
        &["diff", "--name-only", &format!("{base}...{head}")],
    ) else {
        eprintln!(
            "Warning: git diff failed for range {base}...{head}; falling back to HEAD inspection."
        );
        return None;
    };

    let msgs = git_stdout(
        repo_dir,
        &["log", "--format=%B", &format!("{base}..{head}")],
    )
    .unwrap_or_default();
    if diffs.trim().is_empty() {
        diffs = inspect_head_commit(repo_dir).1;
    }
    Some((msgs, diffs))
}

fn verify_records() -> bool {
    let raw_base = env::var("BASE_SHA").ok();
    let raw_head = env::var("HEAD_SHA").ok();
    verify_records_in_dir(Path::new("."), raw_base.as_deref(), raw_head.as_deref())
}

fn verify_records_in_dir(repo_dir: &Path, raw_base: Option<&str>, raw_head: Option<&str>) -> bool {
    let base_sha = sanitize_sha(raw_base);
    let head_sha = sanitize_sha(raw_head);

    let mut range_valid = false;
    let (_commit_msgs, changed_files) =
        if let (Some(base), Some(head)) = (base_sha.as_deref(), head_sha.as_deref()) {
            if let Some(res) = inspect_commit_range(repo_dir, base, head) {
                range_valid = true;
                res
            } else {
                inspect_head_commit(repo_dir)
            }
        } else {
            inspect_head_commit(repo_dir)
        };

    let changed_files = changed_files.replace("\\", "/");
    println!("Changed files:\n{}", changed_files);

    let changed_lines: Vec<&str> = changed_files.lines().collect();
    let core_modified = changed_lines.iter().any(|l| {
        l.starts_with("math_explorer/")
            || (l.starts_with("crates/") && !l.starts_with("crates/unified_verification/"))
    });

    if core_modified {
        let (eff_base, eff_head) = if range_valid {
            (base_sha.as_deref(), head_sha.as_deref())
        } else {
            (None, None)
        };
        if is_pure_deletion_in_dir(repo_dir, eff_base, eff_head) {
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

fn is_pure_deletion_in_dir(
    repo_dir: &Path,
    base_sha: Option<&str>,
    head_sha: Option<&str>,
) -> bool {
    let mut output_str = match (base_sha, head_sha) {
        (Some(base), Some(head)) => {
            git_stdout(repo_dir, &["diff", &format!("{base}...{head}")]).unwrap_or_default()
        }
        _ => String::new(),
    };
    if output_str.trim().is_empty() {
        output_str = git_stdout(repo_dir, &["diff", "HEAD~1", "HEAD"]).unwrap_or_default();
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

    let entropy_debt = entropy_guard::check_entropy(&[]);
    if !entropy_debt.is_empty() {
        passed_security = false;
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static REPO_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .current_dir(dir)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {:?} failed", args);
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn init_temp_repo() -> PathBuf {
        let id = REPO_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            env::temp_dir().join(format!("oxidize_verify_test_{}_{}", std::process::id(), id));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        git(&dir, &["init"]);
        git(&dir, &["config", "user.email", "test@example.com"]);
        git(&dir, &["config", "user.name", "Test"]);
        fs::write(dir.join("README.md"), "init\n").unwrap();
        git(&dir, &["add", "."]);
        git(&dir, &["commit", "-m", "initial commit"]);
        dir
    }

    fn commit_file(dir: &Path, rel_path: &str, content: &str, msg: &str) -> String {
        let full = dir.join(rel_path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(full, content).unwrap();
        git(dir, &["add", rel_path]);
        git(dir, &["commit", "-m", msg]);
        git(dir, &["rev-parse", "HEAD"])
    }

    #[test]
    fn test_sanitize_sha() {
        assert_eq!(sanitize_sha(None), None);
        assert_eq!(sanitize_sha(Some("")), None);
        assert_eq!(sanitize_sha(Some("   ")), None);
        assert_eq!(sanitize_sha(Some("\"\"")), None);
        let zero40 = "0".repeat(40);
        let zero64 = "0".repeat(64);
        assert_eq!(sanitize_sha(Some(&zero40)), None);
        assert_eq!(sanitize_sha(Some(&zero64)), None);
        assert_eq!(sanitize_sha(Some(&format!("\"{}\"", zero40))), None);
        assert_eq!(
            sanitize_sha(Some("  \"2fc42a4b\"  ")),
            Some("2fc42a4b".to_string())
        );
    }

    #[test]
    fn test_multi_commit_range_and_fallbacks() {
        let dir = init_temp_repo();
        let base = git(&dir, &["rev-parse", "HEAD"]);

        commit_file(&dir, "crates/pure_math/src/lib.rs", "pub fn x() {}\n", "c1");
        let head = commit_file(&dir, "README.md", "updated\n", "c2");

        // Single-commit HEAD only touches README.md -> passes
        assert!(verify_records_in_dir(&dir, None, None));
        // Multi-commit range base..head catches c1 core edit without ADR -> fails
        assert!(!verify_records_in_dir(&dir, Some(&base), Some(&head)));

        // All-zero BASE_SHA (branch creation / tag push) falls back to HEAD -> passes
        let zero_sha = "0".repeat(40);
        assert!(verify_records_in_dir(&dir, Some(&zero_sha), Some(&head)));

        // Missing remote base SHA falls back gracefully to HEAD -> passes
        let missing_sha = "deadbeef".repeat(5);
        assert!(verify_records_in_dir(&dir, Some(&missing_sha), Some(&head)));

        // Adding ADR in a follow-up commit satisfies multi-commit range check -> passes
        let head_with_adr = commit_file(&dir, "docs/adr/0001-test.md", "# ADR\n", "c3");
        assert!(verify_records_in_dir(
            &dir,
            Some(&base),
            Some(&head_with_adr)
        ));

        // Core edit at HEAD without ADR fails even when falling back from zero/missing BASE_SHA
        let head_bad = commit_file(&dir, "math_explorer/src/lib.rs", "pub fn y() {}\n", "c4");
        assert!(!verify_records_in_dir(
            &dir,
            Some(&zero_sha),
            Some(&head_bad)
        ));
        assert!(!verify_records_in_dir(
            &dir,
            Some(&missing_sha),
            Some(&head_bad)
        ));

        let _ = fs::remove_dir_all(&dir);
    }
}
