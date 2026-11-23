use super::graph::{expand_reverse_dependents, is_root_config_escalation, load_workspace_crates};
use super::{
    WorkspaceCrate, build_host_clippy_args, build_wasm_clippy_args, select_direct_touched_crates,
};
use crate::gates::{
    HOST_CLIPPY_ARGS, WASM_CLIPPY_ARGS, run_cmd_in_path, run_verification,
    run_verification_with_args, workspace_root,
};
use std::collections::BTreeSet;
use std::io::{IsTerminal, Read};
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PushRefUpdate {
    pub local_ref: String,
    pub local_oid: String,
    pub remote_ref: String,
    pub remote_oid: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PushLintPlan {
    pub full_workspace: bool,
    pub host_clippy_packages: Vec<String>,
    pub host_clippy_args: Option<Vec<String>>,
    pub wasm_clippy_args: Option<Vec<String>>,
    pub run_markdown_doctest_policy: bool,
    pub run_full_traceability: bool,
}

fn is_zero_oid(oid: &str) -> bool {
    !oid.is_empty() && oid.chars().all(|c| c == '0')
}

pub(crate) fn parse_pre_push_stdin(input: &str) -> Vec<PushRefUpdate> {
    let mut updates = Vec::new();
    for line in input.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            updates.push(PushRefUpdate {
                local_ref: parts[0].to_string(),
                local_oid: parts[1].to_string(),
                remote_ref: parts[2].to_string(),
                remote_oid: parts[3].to_string(),
            });
        }
    }
    updates
}

fn git_lines_in(repo_root: &Path, args: &[&str]) -> Vec<String> {
    let Ok(output) = Command::new("git")
        .current_dir(repo_root)
        .args(args)
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().replace('\\', "/"))
        .filter(|l| !l.is_empty())
        .collect()
}

fn git_commit_exists(repo_root: &Path, rev: &str) -> bool {
    let spec = format!("{rev}^{{commit}}");
    Command::new("git")
        .current_dir(repo_root)
        .args([
            "--no-optional-locks",
            "rev-parse",
            "--verify",
            "--quiet",
            &spec,
        ])
        .status()
        .is_ok_and(|s| s.success())
}

fn find_branch_base_for_commit(repo_root: &Path, commit_oid: &str) -> Option<String> {
    for candidate in ["@{upstream}", "origin/main", "main"] {
        if !git_commit_exists(repo_root, candidate) {
            continue;
        }
        let lines = git_lines_in(
            repo_root,
            &["--no-optional-locks", "merge-base", commit_oid, candidate],
        );
        if let Some(base) = lines.into_iter().next() {
            return Some(base);
        }
    }
    None
}

fn collect_paths_for_hook_updates(repo_root: &Path, updates: &[PushRefUpdate]) -> Vec<String> {
    let mut paths = BTreeSet::new();
    for upd in updates {
        if is_zero_oid(&upd.local_oid) {
            continue;
        }
        let base_oid =
            if !is_zero_oid(&upd.remote_oid) && git_commit_exists(repo_root, &upd.remote_oid) {
                Some(upd.remote_oid.clone())
            } else {
                find_branch_base_for_commit(repo_root, &upd.local_oid)
            };
        let diff_args: Vec<String> = match base_oid {
            Some(base) => vec![
                "--no-optional-locks".into(),
                "-c".into(),
                "core.quotepath=false".into(),
                "diff".into(),
                "--name-only".into(),
                format!("{base}...{}", upd.local_oid),
            ],
            None => vec![
                "--no-optional-locks".into(),
                "-c".into(),
                "core.quotepath=false".into(),
                "diff-tree".into(),
                "--no-commit-id".into(),
                "--name-only".into(),
                "-r".into(),
                "--root".into(),
                upd.local_oid.clone(),
            ],
        };
        let arg_refs: Vec<&str> = diff_args.iter().map(String::as_str).collect();
        paths.extend(git_lines_in(repo_root, &arg_refs));
    }
    paths.into_iter().collect()
}

fn collect_paths_for_cli_push(repo_root: &Path) -> Vec<String> {
    let mut paths = BTreeSet::new();
    if let Some(base) = find_branch_base_for_commit(repo_root, "HEAD") {
        let range = format!("{base}...HEAD");
        paths.extend(git_lines_in(
            repo_root,
            &[
                "--no-optional-locks",
                "-c",
                "core.quotepath=false",
                "diff",
                "--name-only",
                &range,
            ],
        ));
    }
    paths.extend(git_lines_in(
        repo_root,
        &[
            "--no-optional-locks",
            "-c",
            "core.quotepath=false",
            "diff",
            "--cached",
            "--name-only",
        ],
    ));
    paths.extend(git_lines_in(
        repo_root,
        &[
            "--no-optional-locks",
            "-c",
            "core.quotepath=false",
            "diff",
            "--name-only",
        ],
    ));
    paths.into_iter().collect()
}

pub(crate) fn build_push_plan(changed_paths: &[String], crates: &[WorkspaceCrate]) -> PushLintPlan {
    let full_workspace = changed_paths.iter().any(|p| is_root_config_escalation(p));
    if full_workspace {
        return PushLintPlan {
            full_workspace: true,
            host_clippy_packages: crates.iter().map(|c| c.package_name.clone()).collect(),
            host_clippy_args: Some(HOST_CLIPPY_ARGS.iter().map(|s| (*s).to_string()).collect()),
            wasm_clippy_args: Some(WASM_CLIPPY_ARGS.iter().map(|s| (*s).to_string()).collect()),
            run_markdown_doctest_policy: true,
            run_full_traceability: true,
        };
    }

    let direct_crates = select_direct_touched_crates(changed_paths, crates);
    let affected_crates = expand_reverse_dependents(&direct_crates, crates);
    let host_clippy_packages: Vec<String> = affected_crates
        .iter()
        .map(|c| c.package_name.clone())
        .collect();
    let host_clippy_args = build_host_clippy_args(&host_clippy_packages);
    let touched_web: Vec<&WorkspaceCrate> =
        affected_crates.into_iter().filter(|c| c.is_web).collect();
    let run_markdown_doctest_policy = changed_paths.iter().any(|p| p.ends_with(".md"));

    PushLintPlan {
        full_workspace: false,
        host_clippy_packages,
        host_clippy_args,
        wasm_clippy_args: build_wasm_clippy_args(&touched_web),
        run_markdown_doctest_policy,
        run_full_traceability: true,
    }
}

fn read_hook_updates() -> Vec<PushRefUpdate> {
    if let Ok(raw) = std::env::var("PRE_PUSH_REFS") {
        return parse_pre_push_stdin(&raw);
    }
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return Vec::new();
    }
    let mut buf = String::new();
    let _ = stdin.lock().read_to_string(&mut buf);
    parse_pre_push_stdin(&buf)
}

pub fn lint_push(args: &[String]) {
    println!("=== Running Offline Pre-Push Verification Suite ===");
    let repo_root = workspace_root();
    let hook_mode = args.iter().any(|a| a == "--hook");

    let changed_paths = if hook_mode {
        let updates = read_hook_updates();
        let non_deleted: Vec<_> = updates
            .iter()
            .filter(|u| !is_zero_oid(&u.local_oid))
            .collect();
        if non_deleted.is_empty() {
            println!("No pushed branch updates detected; skipping lint-push.");
            return;
        }
        collect_paths_for_hook_updates(&repo_root, &updates)
    } else {
        collect_paths_for_cli_push(&repo_root)
    };

    let crates = load_workspace_crates(&repo_root);
    let plan = build_push_plan(&changed_paths, &crates);

    if let Some(host_args) = &plan.host_clippy_args {
        println!("\n--- Pre-Push Reverse-Dependent Clippy Lints (Host) ---");
        let refs: Vec<&str> = host_args.iter().map(String::as_str).collect();
        run_cmd_in_path("cargo", &refs, &repo_root);
    }

    if let Some(wasm_args) = &plan.wasm_clippy_args {
        println!("\n--- Pre-Push Reverse-Dependent Clippy Lints (WebAssembly) ---");
        let refs: Vec<&str> = wasm_args.iter().map(String::as_str).collect();
        run_cmd_in_path("cargo", &refs, &repo_root);
    }

    if plan.run_markdown_doctest_policy {
        println!("\n--- Pre-Push Workspace Markdown Doctest Policy Audit ---");
        run_verification_with_args("test-policy-audit", &["crates/markdown_tests"]);
    }

    if plan.run_full_traceability {
        println!("\n--- Pre-Push Full Workspace Traceability Verification ---");
        run_verification("traceability");
    }

    println!("\n=== Pre-Push Verification Passed Successfully ===");
}
