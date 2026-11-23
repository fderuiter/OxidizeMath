//! Static validation of Cargo CLI commands in Markdown guides and example doc comments.

use super::cargo_cli_parser::{
    ParsedCargoCmd, is_placeholder, parse_cargo_invocation, split_shell_words,
};
use super::manifest_model::WorkspaceManifestModel;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const ACTIVE_DOC_DIRS: &[&str] = &[
    "apps",
    "crates",
    "docs",
    "math_explorer",
    "math_explorer_gui",
];

fn collect_md_recursive(dir: &Path, adr_dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(
            name,
            "target" | ".git" | "dist" | "artifacts" | "node_modules"
        ) || path.starts_with(adr_dir)
        {
            continue;
        }
        if path.is_dir() {
            collect_md_recursive(&path, adr_dir, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("md") {
            out.push(path);
        }
    }
}

/// Collects all active first-party Markdown documentation files (excluding `docs/adr/` and vendored `egui_plot/`).
pub fn collect_active_markdown_files(workspace_root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(workspace_root) {
        let mut root_md: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("md"))
            .collect();
        root_md.sort();
        files.extend(root_md);
    }
    let adr_dir = workspace_root.join("docs").join("adr");
    for sub in ACTIVE_DOC_DIRS {
        collect_md_recursive(&workspace_root.join(sub), &adr_dir, &mut files);
    }
    files
}

fn is_shell_fence(tag: &str) -> bool {
    matches!(
        tag.split_whitespace().next().unwrap_or(""),
        "bash" | "sh" | "shell" | "console"
    )
}

struct TargetCheckCtx<'a> {
    path: &'a Path,
    line_num: usize,
    pkg_name: &'a str,
    cmd: &'a ParsedCargoCmd,
    enabled: &'a BTreeSet<String>,
}

fn validate_parsed_cargo_cmd(
    path: &Path,
    line_num: usize,
    cmd: &ParsedCargoCmd,
    cwd_pkg: Option<&str>,
    model: &WorkspaceManifestModel,
) -> Vec<String> {
    let mut violations = Vec::new();
    if cmd.subcmd == "doc" && cmd.workspace && cmd.no_deps && !cmd.all_features {
        violations.push(format!(
            "{}:{}: `cargo doc --workspace --no-deps` omits `--all-features`",
            path.display(),
            line_num
        ));
    }

    let mut active_pkgs = cmd.packages.clone();
    if active_pkgs.is_empty()
        && let Some(cwd) = cwd_pkg
    {
        active_pkgs.push(cwd.to_string());
    }

    let needs_concrete_pkg = cmd.lib
        || cmd.bins
        || cmd.bin_target.is_some()
        || cmd.example.is_some()
        || cmd.bench.is_some();
    if active_pkgs.is_empty() && !cmd.workspace && needs_concrete_pkg {
        violations.push(format!(
            "{}:{}: `cargo {}` with target selector (`--lib`/`--bin`/`--example`) omits `-p <package>` on virtual workspace root",
            path.display(),
            line_num,
            cmd.subcmd
        ));
        return violations;
    }

    for pkg_name in &active_pkgs {
        if !is_placeholder(pkg_name) {
            validate_pkg_target_and_features(path, line_num, pkg_name, cmd, model, &mut violations);
        }
    }
    violations
}

fn validate_pkg_target_and_features(
    path: &Path,
    line_num: usize,
    pkg_name: &str,
    cmd: &ParsedCargoCmd,
    model: &WorkspaceManifestModel,
    violations: &mut Vec<String>,
) {
    let Some(pkg) = model.packages_by_name.get(pkg_name) else {
        violations.push(format!(
            "{}:{}: Unknown workspace package `{pkg_name}` in Cargo command",
            path.display(),
            line_num
        ));
        return;
    };
    if cmd.lib && !cmd.workspace && !pkg.has_lib {
        violations.push(format!(
            "{}:{}: Package `{pkg_name}` has no library target for `--lib`",
            path.display(),
            line_num
        ));
    }

    let enabled = match model.resolve_enabled_features(
        pkg_name,
        &cmd.features,
        cmd.all_features,
        cmd.no_default_features,
    ) {
        Ok(set) => set,
        Err(err) => {
            violations.push(format!("{}:{}: {err}", path.display(), line_num));
            return;
        }
    };

    let ctx = TargetCheckCtx {
        path,
        line_num,
        pkg_name,
        cmd,
        enabled: &enabled,
    };
    if let Some(ex_name) = &cmd.example {
        validate_named_target(&ctx, "example", ex_name, &pkg.examples, violations);
    }
    if let Some(bench_name) = &cmd.bench {
        validate_named_target(&ctx, "bench", bench_name, &pkg.benches, violations);
    }
}

fn validate_named_target(
    ctx: &TargetCheckCtx<'_>,
    kind: &str,
    target_name: &str,
    targets: &BTreeMap<String, Vec<String>>,
    violations: &mut Vec<String>,
) {
    if is_placeholder(target_name) {
        let all_gated = !targets.is_empty() && targets.values().all(|req| !req.is_empty());
        if all_gated
            && ctx.cmd.features.is_empty()
            && !ctx.cmd.has_feature_placeholder
            && !ctx.cmd.all_features
        {
            violations.push(format!(
                "{}:{}: Template `--{kind} {target_name}` for `{}` omits required `--features` flag",
                ctx.path.display(),
                ctx.line_num,
                ctx.pkg_name
            ));
        }
        return;
    }

    let Some(required) = targets.get(target_name) else {
        violations.push(format!(
            "{}:{}: Package `{}` has no registered `[[{kind}]]` named `{target_name}`",
            ctx.path.display(),
            ctx.line_num,
            ctx.pkg_name
        ));
        return;
    };

    if ctx.cmd.has_feature_placeholder {
        return;
    }
    let missing: Vec<&str> = required
        .iter()
        .filter(|req| !ctx.enabled.contains(*req))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        violations.push(format!(
            "{}:{}: `--{kind} {target_name}` in `{}` requires feature(s) {:?}, missing {:?}",
            ctx.path.display(),
            ctx.line_num,
            ctx.pkg_name,
            required,
            missing
        ));
    }
}

fn infer_cwd_package(
    md_path: &Path,
    workspace_root: &Path,
    cd_arg: &str,
    model: &WorkspaceManifestModel,
) -> Option<String> {
    let base = md_path.parent().unwrap_or(workspace_root);
    let candidate = base.join(cd_arg);
    let rel = candidate.strip_prefix(workspace_root).ok()?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    model
        .packages_by_dir
        .get(rel_str.trim_matches('/'))
        .cloned()
}

fn default_fence_cwd_pkg(
    md_path: &Path,
    workspace_root: &Path,
    model: &WorkspaceManifestModel,
) -> Option<String> {
    let rel = md_path.strip_prefix(workspace_root).ok()?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    model
        .packages_by_dir
        .iter()
        .filter(|(dir, _)| rel_str == format!("{dir}/README.md"))
        .map(|(_, pkg)| pkg.clone())
        .next()
}

struct ShellFenceState {
    cwd_pkg: Option<String>,
    continued: String,
    continued_line: usize,
}

/// Validates Cargo CLI invocations inside shell fences and inline backticks of a Markdown guide.
pub fn check_markdown_cargo_commands(
    md_path: &Path,
    workspace_root: &Path,
    content: &str,
    model: &WorkspaceManifestModel,
) -> Vec<String> {
    let mut violations = Vec::new();
    let mut in_fence = false;
    let mut shell_fence = false;
    let mut state = ShellFenceState {
        cwd_pkg: None,
        continued: String::new(),
        continued_line: 1,
    };

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(after) = trimmed.strip_prefix("```") {
            in_fence = !in_fence;
            shell_fence = in_fence && is_shell_fence(after);
            state.cwd_pkg = default_fence_cwd_pkg(md_path, workspace_root, model);
            state.continued.clear();
            continue;
        }
        if in_fence {
            if shell_fence {
                process_shell_line(
                    md_path,
                    workspace_root,
                    idx + 1,
                    trimmed,
                    &mut state,
                    model,
                    &mut violations,
                );
            }
        } else {
            check_inline_backtick_commands(md_path, idx + 1, trimmed, model, &mut violations);
        }
    }
    violations
}

fn process_shell_line(
    md_path: &Path,
    workspace_root: &Path,
    line_num: usize,
    trimmed: &str,
    state: &mut ShellFenceState,
    model: &WorkspaceManifestModel,
    violations: &mut Vec<String>,
) {
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return;
    }
    if state.continued.is_empty() {
        state.continued_line = line_num;
    }
    if let Some(prefix) = trimmed.strip_suffix('\\') {
        state.continued.push_str(prefix);
        state.continued.push(' ');
        return;
    }
    state.continued.push_str(trimmed);
    let full_cmd = std::mem::take(&mut state.continued);

    for segment in full_cmd.split("&&").flat_map(|s| s.split(';')) {
        let words = split_shell_words(segment.trim());
        if words.len() == 2 && words[0] == "cd" {
            state.cwd_pkg = infer_cwd_package(md_path, workspace_root, &words[1], model);
        } else if let Some(parsed) = parse_cargo_invocation(&words) {
            violations.extend(validate_parsed_cargo_cmd(
                md_path,
                state.continued_line,
                &parsed,
                state.cwd_pkg.as_deref(),
                model,
            ));
        }
    }
}

fn check_inline_backtick_commands(
    md_path: &Path,
    line_num: usize,
    line: &str,
    model: &WorkspaceManifestModel,
    violations: &mut Vec<String>,
) {
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        rest = &rest[start + 1..];
        let Some(end) = rest.find('`') else {
            break;
        };
        let snippet = rest[..end].trim();
        rest = &rest[end + 1..];
        if snippet.contains("cargo ")
            && let Some(parsed) = parse_cargo_invocation(&split_shell_words(snippet))
        {
            violations.extend(validate_parsed_cargo_cmd(
                md_path, line_num, &parsed, None, model,
            ));
        }
    }
}

/// Checks `//!` header doc comments in example `.rs` files for Cargo command feature parity.
pub fn check_rustdoc_cargo_commands(
    rs_path: &Path,
    content: &str,
    model: &WorkspaceManifestModel,
) -> Vec<String> {
    let mut violations = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let Some(doc) = line.trim_start().strip_prefix("//!") else {
            continue;
        };
        let trimmed = doc.trim();
        if trimmed.starts_with("cargo ")
            && let Some(parsed) = parse_cargo_invocation(&split_shell_words(trimmed))
        {
            violations.extend(validate_parsed_cargo_cmd(
                rs_path,
                idx + 1,
                &parsed,
                None,
                model,
            ));
        }
    }
    violations
}
