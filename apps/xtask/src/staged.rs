mod graph;
mod snapshot;

use crate::gates::{
    HOST_CLIPPY_ARGS, WASM_CLIPPY_ARGS, resolve_target_dir, run_cmd_in_dir_with_env,
    run_verification_in_dir_with_env, workspace_root,
};
use crate::markdown_links::{check_markdown_links_for_files_in_dir, check_markdown_links_in_dir};
pub(crate) use graph::load_workspace_crates;
use graph::{expand_reverse_dependents, is_root_config_escalation};
use snapshot::{GitSnapshotEnv, sync_index_snapshot};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceCrate {
    pub dir: String,
    pub package_name: String,
    pub edition: String,
    pub has_lib: bool,
    pub has_bins: bool,
    pub is_web: bool,
    pub normal_deps: Vec<String>,
    pub dev_deps: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StagedLintPlan {
    pub full_workspace: bool,
    pub rustfmt_by_edition: BTreeMap<String, Vec<String>>,
    pub run_workflows: bool,
    pub scan_all_markdown: bool,
    pub markdown_files: Vec<PathBuf>,
    pub host_clippy_packages: Vec<String>,
    pub host_clippy_args: Option<Vec<String>>,
    pub wasm_clippy_args: Option<Vec<String>>,
}

fn git_staged_paths_in(repo_root: &Path, extra_args: &[&str]) -> Vec<String> {
    let mut args = vec![
        "--no-optional-locks",
        "-c",
        "core.quotepath=false",
        "diff",
        "--cached",
        "--name-only",
    ];
    args.extend_from_slice(extra_args);
    let Ok(output) = Command::new("git")
        .current_dir(repo_root)
        .args(&args)
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

fn find_owning_crate<'a>(path: &str, crates: &'a [WorkspaceCrate]) -> Option<&'a WorkspaceCrate> {
    crates
        .iter()
        .filter(|c| {
            path == format!("{}/Cargo.toml", c.dir) || path.starts_with(&format!("{}/", c.dir))
        })
        .max_by_key(|c| c.dir.len())
}

fn build_wasm_clippy_args(touched_web: &[&WorkspaceCrate]) -> Option<Vec<String>> {
    if touched_web.is_empty() {
        return None;
    }
    let mut args = vec!["clippy".to_string()];
    for krate in touched_web {
        args.push("-p".to_string());
        args.push(krate.package_name.clone());
    }
    if touched_web.iter().any(|c| c.has_lib) {
        args.push("--lib".to_string());
    }
    if touched_web.iter().any(|c| c.has_bins) {
        args.push("--bins".to_string());
    }
    args.extend(
        [
            "--tests",
            "--target",
            "wasm32-unknown-unknown",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ]
        .into_iter()
        .map(String::from),
    );
    Some(args)
}

fn build_host_clippy_args(packages: &[String]) -> Option<Vec<String>> {
    if packages.is_empty() {
        return None;
    }
    let mut args = vec!["clippy".to_string()];
    for pkg in packages {
        args.push("-p".to_string());
        args.push(pkg.clone());
    }
    args.extend(
        ["--all-targets", "--all-features", "--", "-D", "warnings"]
            .into_iter()
            .map(String::from),
    );
    Some(args)
}

fn collect_rustfmt_by_edition(
    existing_staged: &[String],
    crates: &[WorkspaceCrate],
) -> BTreeMap<String, Vec<String>> {
    let mut rustfmt_by_edition: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in existing_staged.iter().filter(|p| p.ends_with(".rs")) {
        let edition = find_owning_crate(path, crates)
            .map(|c| c.edition.clone())
            .unwrap_or_else(|| "2024".to_string());
        rustfmt_by_edition
            .entry(edition)
            .or_default()
            .push(path.clone());
    }
    rustfmt_by_edition
}

fn select_direct_touched_crates<'a>(
    all_staged: &[String],
    crates: &'a [WorkspaceCrate],
) -> Vec<&'a WorkspaceCrate> {
    let mut touched = Vec::new();
    for krate in crates {
        let manifest = format!("{}/Cargo.toml", krate.dir);
        let prefix = format!("{}/", krate.dir);
        let has_code_or_manifest_edit = all_staged.iter().any(|p| {
            p == &manifest
                || (p.starts_with(&prefix) && (p.ends_with(".rs") || p.ends_with("/Cargo.toml")))
        });
        if has_code_or_manifest_edit {
            touched.push(krate);
        }
    }
    touched
}

pub(crate) fn build_staged_plan(
    all_staged: &[String],
    existing_staged: &[String],
    crates: &[WorkspaceCrate],
) -> StagedLintPlan {
    let full_workspace = all_staged.iter().any(|p| is_root_config_escalation(p));
    let rustfmt_by_edition = collect_rustfmt_by_edition(existing_staged, crates);
    let markdown_files: Vec<PathBuf> = existing_staged
        .iter()
        .filter(|p| p.ends_with(".md"))
        .map(PathBuf::from)
        .collect();

    let existing_set: HashSet<&str> = existing_staged.iter().map(String::as_str).collect();
    let has_deleted_or_renamed = all_staged
        .iter()
        .any(|p| !existing_set.contains(p.as_str()));

    if full_workspace {
        return StagedLintPlan {
            full_workspace: true,
            rustfmt_by_edition,
            run_workflows: true,
            scan_all_markdown: true,
            markdown_files,
            host_clippy_packages: crates.iter().map(|c| c.package_name.clone()).collect(),
            host_clippy_args: Some(HOST_CLIPPY_ARGS.iter().map(|s| (*s).to_string()).collect()),
            wasm_clippy_args: Some(WASM_CLIPPY_ARGS.iter().map(|s| (*s).to_string()).collect()),
        };
    }

    let direct_crates = select_direct_touched_crates(all_staged, crates);
    let affected_crates = expand_reverse_dependents(&direct_crates, crates);
    let host_clippy_packages: Vec<String> = affected_crates
        .iter()
        .map(|c| c.package_name.clone())
        .collect();
    let host_clippy_args = build_host_clippy_args(&host_clippy_packages);
    let touched_web: Vec<&WorkspaceCrate> =
        affected_crates.into_iter().filter(|c| c.is_web).collect();

    StagedLintPlan {
        full_workspace: false,
        rustfmt_by_edition,
        run_workflows: all_staged.iter().any(|p| p.starts_with(".github/")),
        scan_all_markdown: has_deleted_or_renamed,
        markdown_files,
        host_clippy_packages,
        host_clippy_args,
        wasm_clippy_args: build_wasm_clippy_args(&touched_web),
    }
}

fn run_staged_format_gates(
    plan: &StagedLintPlan,
    snapshot_dir: &Path,
    cargo_envs: &[(&str, &str)],
) {
    if plan.full_workspace {
        println!("\n--- Staged Workspace Format Check ---");
        run_cmd_in_dir_with_env(
            "cargo",
            &["fmt", "--all", "--", "--check"],
            snapshot_dir,
            cargo_envs,
        );
        return;
    }
    for (edition, files) in &plan.rustfmt_by_edition {
        println!("\n--- Staged Format Check (edition {edition}) ---");
        let mut args = vec!["--edition", edition.as_str(), "--check"];
        let file_refs: Vec<&str> = files.iter().map(String::as_str).collect();
        args.extend(file_refs);
        run_cmd_in_dir_with_env("rustfmt", &args, snapshot_dir, &[]);
    }
}

fn run_staged_markdown_and_clippy_gates(
    plan: &StagedLintPlan,
    snapshot_dir: &Path,
    cargo_envs: &[(&str, &str)],
) {
    if plan.scan_all_markdown {
        println!("\n--- Staged Full-Repository Markdown Link Check ---");
        check_markdown_links_in_dir(snapshot_dir);
    } else if !plan.markdown_files.is_empty() {
        println!("\n--- Staged Markdown Link Check ---");
        check_markdown_links_for_files_in_dir(snapshot_dir, &plan.markdown_files);
    }

    if let Some(host_args) = &plan.host_clippy_args {
        println!("\n--- Targeted Clippy Lints (Host) ---");
        let host_refs: Vec<&str> = host_args.iter().map(String::as_str).collect();
        run_cmd_in_dir_with_env("cargo", &host_refs, snapshot_dir, cargo_envs);
    }

    if let Some(wasm_args) = &plan.wasm_clippy_args {
        println!("\n--- Targeted Clippy Lints (WebAssembly) ---");
        let wasm_refs: Vec<&str> = wasm_args.iter().map(String::as_str).collect();
        run_cmd_in_dir_with_env("cargo", &wasm_refs, snapshot_dir, cargo_envs);
    }
}

fn prepare_snapshot_and_plan(
    repo_root: &Path,
    all_staged: &[String],
) -> (GitSnapshotEnv, PathBuf, PathBuf, StagedLintPlan) {
    let git_env = GitSnapshotEnv::resolve(repo_root).unwrap_or_else(|err| {
        eprintln!("Failed to resolve Git repository metadata: {err}");
        exit(1);
    });
    let target_dir = resolve_target_dir(repo_root);
    let snapshot_dir = target_dir.join("lint-staged-snapshot");
    let manifest_path = target_dir.join("lint-staged-snapshot.manifest");

    println!("\n--- Synchronizing Staged Git Index Snapshot ---");
    if let Err(err) = sync_index_snapshot(repo_root, &snapshot_dir, &manifest_path) {
        eprintln!("Failed to synchronize staged Git index snapshot: {err}");
        exit(1);
    }

    let existing_staged: Vec<String> =
        git_staged_paths_in(repo_root, &["--no-renames", "--diff-filter=ACMRT"])
            .into_iter()
            .filter(|p| snapshot_dir.join(p).is_file())
            .collect();
    let crates = load_workspace_crates(&snapshot_dir);
    let mut plan = build_staged_plan(all_staged, &existing_staged, &crates);
    if !git_staged_paths_in(repo_root, &["-M", "--diff-filter=DR"]).is_empty() {
        plan.scan_all_markdown = true;
    }
    (git_env, target_dir, snapshot_dir, plan)
}

pub fn lint_staged() {
    println!("=== Running Scoped Staged Lint Suite ===");
    let repo_root = workspace_root();
    let all_staged = git_staged_paths_in(&repo_root, &["--no-renames"]);
    if all_staged.is_empty() {
        println!("No staged changes detected; skipping lint-staged.");
        return;
    }

    let (git_env, target_dir, snapshot_dir, plan) =
        prepare_snapshot_and_plan(&repo_root, &all_staged);
    let target_str = target_dir.to_string_lossy().into_owned();
    let git_dir_str = git_env.git_dir.to_string_lossy().into_owned();
    let git_idx_str = git_env
        .git_index_file
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());

    let cargo_envs = [("CARGO_TARGET_DIR", target_str.as_str())];
    let mut verif_envs = vec![
        ("CARGO_TARGET_DIR", target_str.as_str()),
        ("GIT_DIR", git_dir_str.as_str()),
    ];
    if let Some(idx_s) = &git_idx_str {
        verif_envs.push(("GIT_INDEX_FILE", idx_s.as_str()));
    }

    run_staged_format_gates(&plan, &snapshot_dir, &cargo_envs);

    println!("\n--- Staged Static & Single-Pass AST Checks ---");
    run_verification_in_dir_with_env("batch-staged", &[], &snapshot_dir, &target_dir, &verif_envs);

    if plan.run_workflows {
        println!("\n--- Staged CI Workflow Linting ---");
        run_verification_in_dir_with_env(
            "lint-workflows",
            &[],
            &snapshot_dir,
            &target_dir,
            &verif_envs,
        );
    }

    run_staged_markdown_and_clippy_gates(&plan, &snapshot_dir, &cargo_envs);

    println!("\n=== Staged Lint Checks Passed Successfully ===");
}

#[cfg(test)]
mod tests;
