use crate::gates::{run_cmd, run_verification};
use crate::markdown_links::check_markdown_links_for_files;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const WEB_PACKAGE_NAMES: &[&str] = &["oxidize_core", "egui_plot", "math_explorer_gui"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceCrate {
    pub dir: String,
    pub package_name: String,
    pub edition: String,
    pub has_lib: bool,
    pub has_bins: bool,
    pub is_web: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StagedLintPlan {
    pub rustfmt_by_edition: BTreeMap<String, Vec<String>>,
    pub run_workflows: bool,
    pub markdown_files: Vec<PathBuf>,
    pub host_clippy_packages: Vec<String>,
    pub wasm_clippy_args: Option<Vec<String>>,
}

fn git_staged_paths(extra_args: &[&str]) -> Vec<String> {
    let mut args = vec![
        "-c",
        "core.quotepath=false",
        "diff",
        "--cached",
        "--name-only",
    ];
    args.extend_from_slice(extra_args);
    let Ok(output) = Command::new("git").args(&args).output() else {
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

pub(crate) fn parse_workspace_members(root_manifest: &str) -> Vec<String> {
    let mut in_workspace = false;
    let mut in_members = false;
    let mut members = Vec::new();

    for raw_line in root_manifest.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && !in_members {
            in_workspace = line == "[workspace]";
            continue;
        }
        if in_workspace && !in_members && line.starts_with("members") {
            in_members = true;
            collect_quoted_strings(line, &mut members);
            if line.contains(']') {
                break;
            }
            continue;
        }
        if in_members {
            collect_quoted_strings(line, &mut members);
            if line.contains(']') {
                break;
            }
        }
    }
    members
}

fn collect_quoted_strings(line: &str, out: &mut Vec<String>) {
    let mut rest = line;
    while let Some(start) = rest.find('"') {
        rest = &rest[start + 1..];
        if let Some(end) = rest.find('"') {
            let val = rest[..end].trim();
            if !val.is_empty() {
                out.push(val.to_string());
            }
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }
}

fn parse_package_fields(manifest: &str, fallback_name: &str) -> (String, String) {
    let mut in_package = false;
    let mut name = fallback_name.to_string();
    let mut edition = "2024".to_string();

    for raw_line in manifest.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(rest) = line.strip_prefix("name") {
            let mut vals = Vec::new();
            collect_quoted_strings(rest, &mut vals);
            if let Some(first) = vals.into_iter().next() {
                name = first;
            }
        } else if let Some(rest) = line.strip_prefix("edition") {
            let mut vals = Vec::new();
            collect_quoted_strings(rest, &mut vals);
            if let Some(first) = vals.into_iter().next() {
                edition = first;
            }
        }
    }
    (name, edition)
}

pub(crate) fn load_workspace_crates(root: &Path) -> Vec<WorkspaceCrate> {
    let Ok(root_toml) = fs::read_to_string(root.join("Cargo.toml")) else {
        return Vec::new();
    };
    let mut crates = Vec::new();
    for dir in parse_workspace_members(&root_toml) {
        let member_root = root.join(&dir);
        let fallback = dir.rsplit('/').next().unwrap_or(&dir);
        let manifest = fs::read_to_string(member_root.join("Cargo.toml")).unwrap_or_default();
        let (package_name, edition) = parse_package_fields(&manifest, fallback);
        let has_lib = member_root.join("src/lib.rs").is_file()
            || manifest.lines().any(|l| l.trim() == "[lib]");
        let has_bins = member_root.join("src/main.rs").is_file()
            || member_root.join("src/bin").is_dir()
            || manifest.lines().any(|l| l.trim() == "[[bin]]");
        let is_web = WEB_PACKAGE_NAMES.contains(&package_name.as_str());
        crates.push(WorkspaceCrate {
            dir,
            package_name,
            edition,
            has_lib,
            has_bins,
            is_web,
        });
    }
    crates
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

pub(crate) fn build_staged_plan(
    all_staged: &[String],
    existing_staged: &[String],
    crates: &[WorkspaceCrate],
) -> StagedLintPlan {
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

    let markdown_files: Vec<PathBuf> = existing_staged
        .iter()
        .filter(|p| p.ends_with(".md"))
        .map(PathBuf::from)
        .collect();

    let mut touched_crates: Vec<&WorkspaceCrate> = Vec::new();
    for krate in crates {
        let manifest = format!("{}/Cargo.toml", krate.dir);
        let prefix = format!("{}/", krate.dir);
        let has_code_or_manifest_edit = all_staged.iter().any(|p| {
            p == &manifest
                || (p.starts_with(&prefix) && (p.ends_with(".rs") || p.ends_with("/Cargo.toml")))
        });
        if has_code_or_manifest_edit {
            touched_crates.push(krate);
        }
    }

    let host_clippy_packages = touched_crates
        .iter()
        .map(|c| c.package_name.clone())
        .collect();
    let touched_web: Vec<&WorkspaceCrate> =
        touched_crates.into_iter().filter(|c| c.is_web).collect();

    StagedLintPlan {
        rustfmt_by_edition,
        run_workflows: all_staged.iter().any(|p| p.starts_with(".github/")),
        markdown_files,
        host_clippy_packages,
        wasm_clippy_args: build_wasm_clippy_args(&touched_web),
    }
}

fn build_host_clippy_args(packages: &[String]) -> Vec<String> {
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
    args
}

pub fn lint_staged() {
    println!("=== Running Scoped Staged Lint Suite ===");
    let all_staged = git_staged_paths(&[]);
    if all_staged.is_empty() {
        println!("No staged changes detected; skipping lint-staged.");
        return;
    }

    let existing_staged: Vec<String> = git_staged_paths(&["--diff-filter=ACMR"])
        .into_iter()
        .filter(|p| Path::new(p).is_file())
        .collect();
    let crates = load_workspace_crates(Path::new("."));
    let plan = build_staged_plan(&all_staged, &existing_staged, &crates);

    for (edition, files) in &plan.rustfmt_by_edition {
        println!("\n--- Staged Format Check (edition {edition}) ---");
        let mut args = vec!["--edition", edition.as_str(), "--check"];
        let file_refs: Vec<&str> = files.iter().map(String::as_str).collect();
        args.extend(file_refs);
        run_cmd("rustfmt", &args);
    }

    println!("\n--- Staged Static & Single-Pass AST Checks ---");
    run_verification("batch-staged");

    if !plan.markdown_files.is_empty() {
        println!("\n--- Staged Markdown Link Check ---");
        check_markdown_links_for_files(&plan.markdown_files);
    }

    if !plan.host_clippy_packages.is_empty() {
        println!("\n--- Targeted Clippy Lints (Host) ---");
        let host_args = build_host_clippy_args(&plan.host_clippy_packages);
        let host_refs: Vec<&str> = host_args.iter().map(String::as_str).collect();
        run_cmd("cargo", &host_refs);
    }

    if let Some(wasm_args) = &plan.wasm_clippy_args {
        println!("\n--- Targeted Clippy Lints (WebAssembly) ---");
        let wasm_refs: Vec<&str> = wasm_args.iter().map(String::as_str).collect();
        run_cmd("cargo", &wasm_refs);
    }

    println!("\n=== Staged Lint Checks Passed Successfully ===");
}

#[cfg(test)]
mod tests;
