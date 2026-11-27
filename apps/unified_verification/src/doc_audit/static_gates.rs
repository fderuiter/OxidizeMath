//! Static documentation gates blocking unapproved `rust,ignore` fences, invalid binary-GUI
//! imports (`math_explorer_gui::`), and stale workspace crate counts.

use super::manifest_model::WorkspaceManifestModel;
use crate::whitelist::{expand_package_pattern_in_ws, path_matches_lossy};
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_APPROVED_IGNORE_FENCES: &[(&str, usize)] = &[
    ("docs/architecture/interactive_tool_walkthrough.md", 6),
    ("math_explorer_gui:README.md", 3),
];

const GUI_CONTEXT_TOKENS: &[&str] = &["crate::", "egui", "eframe", "ui.", "ui,"];
const IGNORE_MATCH_OPTS: crate::whitelist::PathMatchOptions = crate::whitelist::PathMatchOptions {
    trim_dot_slash: true,
    require_non_empty: true,
    match_file_name: false,
};

/// Loads approved `rust,ignore` per-file caps from `verification_whitelist.toml` (falling back to defaults).
pub fn load_approved_ignore_fences(workspace_root: &Path) -> BTreeMap<String, usize> {
    let candidate = workspace_root.join(crate::whitelist::WHITELIST_FILE_NAME);
    let cfg = crate::whitelist::load_whitelist_or_default(None, &[candidate.as_path()]);
    if let Some(map) = cfg.doc_verification.and_then(|d| d.approved_ignore_fences) {
        return map;
    }
    DEFAULT_APPROVED_IGNORE_FENCES
        .iter()
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect()
}

fn is_ignore_fence_tag(info: &str) -> bool {
    info.split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .any(|tok| tok == "ignore" || tok.starts_with("ignore-"))
}

fn is_rust_fence_tag(info: &str) -> bool {
    let trimmed = info.trim();
    trimmed.is_empty()
        || matches!(
            trimmed,
            "rust" | "no_run" | "should_panic" | "compile_fail" | "ignore"
        )
        || trimmed.starts_with("rust,")
        || trimmed.starts_with("rust ")
}

fn parse_fence_ast(raw_block: &str) -> Result<syn::File, syn::Error> {
    let mut cleaned = String::new();
    for line in raw_block.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            cleaned.push_str(rest);
        } else if trimmed != "#" {
            cleaned.push_str(line);
        }
        cleaned.push('\n');
    }
    syn::parse_file(&cleaned)
        .or_else(|_| syn::parse_file(&format!("fn __doctest_wrap() {{\n{cleaned}\n}}")))
}

fn rel_slash_path(path: &Path, workspace_root: &Path) -> String {
    path.strip_prefix(workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}

/// Rejects unapproved `rust,ignore` / `ignore` code fences in an active Markdown file.
pub fn check_markdown_ignore_fences(
    md_path: &Path,
    workspace_root: &Path,
    content: &str,
    approved_caps: &BTreeMap<String, usize>,
) -> Vec<String> {
    let mut violations = Vec::new();
    let rel = rel_slash_path(md_path, workspace_root);
    let cap = approved_caps
        .get(&rel)
        .copied()
        .or_else(|| {
            approved_caps
                .iter()
                .find(|(k, _)| {
                    expand_package_pattern_in_ws(workspace_root, k)
                        .is_some_and(|e| e.trim_end_matches('/') == rel)
                        || path_matches_lossy(Path::new(&rel), k, IGNORE_MATCH_OPTS)
                })
                .map(|(_, &c)| c)
        })
        .unwrap_or(0);

    let mut in_fence = false;
    let mut active_ignore = false;
    let mut fence_line = 1usize;
    let mut fence_tag = String::new();
    let mut block = String::new();
    let mut ignore_count = 0usize;

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(after) = trimmed.strip_prefix("```") {
            if !in_fence {
                in_fence = true;
                fence_line = idx + 1;
                fence_tag = after.trim().to_string();
                active_ignore = is_ignore_fence_tag(&fence_tag);
                block.clear();
            } else {
                if active_ignore {
                    ignore_count += 1;
                    validate_ignore_block(
                        md_path,
                        fence_line,
                        &fence_tag,
                        &block,
                        cap,
                        ignore_count,
                        &mut violations,
                    );
                }
                in_fence = false;
                active_ignore = false;
                block.clear();
            }
            continue;
        }
        if in_fence && active_ignore {
            block.push_str(line);
            block.push('\n');
        }
    }
    violations
}

fn is_out_of_line_mod_declaration(ast: &syn::File) -> bool {
    !ast.items.is_empty()
        && ast
            .items
            .iter()
            .all(|item| matches!(item, syn::Item::Mod(m) if m.content.is_none()))
}

fn validate_ignore_block(
    md_path: &Path,
    fence_line: usize,
    fence_tag: &str,
    block: &str,
    cap: usize,
    ignore_count: usize,
    violations: &mut Vec<String>,
) {
    if cap == 0 {
        violations.push(format!(
            "{}:{}: Unapproved `{fence_tag}` code fence in active guide (use a compiled `rust` doctest or `text`/`bash` fence)",
            md_path.display(),
            fence_line
        ));
        return;
    }
    if ignore_count > cap {
        violations.push(format!(
            "{}:{}: File exceeds approved `rust,ignore` fence cap ({ignore_count} > {cap})",
            md_path.display(),
            fence_line
        ));
    }
    match parse_fence_ast(block) {
        Ok(ast) => {
            let needs_gui_ctx = GUI_CONTEXT_TOKENS.iter().any(|tok| block.contains(tok))
                || is_out_of_line_mod_declaration(&ast);
            if block.contains("TheoryDescribable for ") || !needs_gui_ctx {
                violations.push(format!(
                    "{}:{}: `rust,ignore` snippet must be a compiled `rust` doctest in `markdown_tests` (does not require binary-only GUI context)",
                    md_path.display(),
                    fence_line
                ));
            }
        }
        Err(err) => violations.push(format!(
            "{}:{}: Invalid Rust syntax inside `{fence_tag}` fence: {err}",
            md_path.display(),
            fence_line
        )),
    }
}

fn build_binary_import_regex(binary_pkgs: &BTreeSet<String>) -> Option<Regex> {
    if binary_pkgs.is_empty() {
        return None;
    }
    let joined: Vec<String> = binary_pkgs.iter().map(|s| regex::escape(s)).collect();
    Regex::new(&format!(r"(?:^|[^:\w])({})::", joined.join("|"))).ok()
}

/// Collects binary-only package names (such as `math_explorer_gui`) that have no library target.
pub fn collect_binary_only_packages(model: &WorkspaceManifestModel) -> BTreeSet<String> {
    let mut set: BTreeSet<String> = model
        .packages_by_name
        .values()
        .filter(|pkg| !pkg.has_lib)
        .map(|pkg| pkg.name.clone())
        .collect();
    if !model
        .packages_by_name
        .get("math_explorer_gui")
        .is_some_and(|p| p.has_lib)
    {
        set.insert("math_explorer_gui".to_string());
    }
    set
}

/// Checks Markdown Rust fences (including `rust,ignore` and `rust,no_run`) for invalid imports from binary-only crates.
pub fn check_markdown_binary_gui_imports(
    md_path: &Path,
    content: &str,
    binary_pkgs: &BTreeSet<String>,
) -> Vec<String> {
    let mut violations = Vec::new();
    let Some(re) = build_binary_import_regex(binary_pkgs) else {
        return violations;
    };
    let mut in_fence = false;
    let mut is_rust = false;

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(after) = trimmed.strip_prefix("```") {
            is_rust = !in_fence && is_rust_fence_tag(after);
            in_fence = !in_fence;
            continue;
        }
        if !in_fence || !is_rust || trimmed.starts_with("//") {
            continue;
        }
        let code = trimmed.split("//").next().unwrap_or(trimmed);
        if let Some(caps) = re.captures(code) {
            let pkg = caps.get(1).unwrap().as_str();
            violations.push(format!(
                "{}:{}: Invalid `{pkg}::` import from binary-only package (use `crate::` paths inside `{pkg}`)",
                md_path.display(),
                idx + 1
            ));
        }
    }
    violations
}

/// Checks Rust source code and `///`/`//!` doc-comment fences for invalid `math_explorer_gui::` binary-crate imports.
pub fn check_rs_binary_gui_imports(
    rs_path: &Path,
    content: &str,
    binary_pkgs: &BTreeSet<String>,
) -> Vec<String> {
    let mut violations = Vec::new();
    let Some(re) = build_binary_import_regex(binary_pkgs) else {
        return violations;
    };
    let doc_re = Regex::new(r"^\s*(?://!|///)\s?(.*)$").expect("Valid regex");
    let mut in_doc_fence = false;
    let mut doc_is_rust = false;

    for (idx, line) in content.lines().enumerate() {
        if let Some(caps) = doc_re.captures(line) {
            let doc_line = caps.get(1).map_or("", |m| m.as_str()).trim();
            if let Some(after) = doc_line.strip_prefix("```") {
                doc_is_rust = !in_doc_fence && is_rust_fence_tag(after);
                in_doc_fence = !in_doc_fence;
                continue;
            }
            if in_doc_fence && doc_is_rust && !doc_line.starts_with("//") {
                let code = doc_line.split("//").next().unwrap_or(doc_line);
                if let Some(m) = re.captures(code) {
                    let pkg = m.get(1).unwrap().as_str();
                    violations.push(format!(
                        "{}:{}: Rustdoc snippet imports binary-only package `{pkg}::` (use `crate::` paths)",
                        rs_path.display(),
                        idx + 1
                    ));
                }
            }
            continue;
        }
        in_doc_fence = false;
        doc_is_rust = false;
        let trimmed = line.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        let code = trimmed.split("//").next().unwrap_or(trimmed);
        if let Some(m) = re.captures(code) {
            let pkg = m.get(1).unwrap().as_str();
            violations.push(format!(
                "{}:{}: Source file imports binary-only package `{pkg}::` (use `crate::` paths)",
                rs_path.display(),
                idx + 1
            ));
        }
    }
    violations
}

/// Checks active Markdown prose (outside code fences) for drifted workspace crate counts.
pub fn check_markdown_crate_counts(
    md_path: &Path,
    content: &str,
    expected_count: usize,
) -> Vec<String> {
    let mut violations = Vec::new();
    if expected_count == 0 {
        return violations;
    }
    let count_re = Regex::new(
        r"(?i)\b(?:all\s+(\d+)\s+(?:workspace\s+|member\s+)*crates|(\d+)\s+(?:workspace|member)\s+(?:member\s+)?crates|(?:catalog\s+of|across|following|contains|declares)\s+(?:all\s+|the\s+)?(\d+)\s+(?:workspace\s+|member\s+)*crates)\b",
    )
    .expect("Valid regex");

    let mut in_fence = false;
    for (idx, line) in content.lines().enumerate() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        for caps in count_re.captures_iter(line) {
            let num_str = caps
                .get(1)
                .or_else(|| caps.get(2))
                .or_else(|| caps.get(3))
                .map(|m| m.as_str())
                .unwrap_or("");
            if let Ok(found) = num_str.parse::<usize>()
                && found != expected_count
            {
                violations.push(format!(
                    "{}:{}: Stale workspace crate count `{found}` (root Cargo.toml declares `{expected_count}` workspace crates)",
                    md_path.display(),
                    idx + 1
                ));
            }
        }
    }
    violations
}

/// Validates that `docs/architecture/system_overview.md` catalogs all workspace crates in its Mermaid topology and table.
pub fn audit_system_overview_crate_catalog(
    workspace_root: &Path,
    model: &WorkspaceManifestModel,
) -> Vec<String> {
    let mut violations = Vec::new();
    let overview_path = workspace_root.join("docs/architecture/system_overview.md");
    let Ok(content) = fs::read_to_string(&overview_path) else {
        return violations;
    };
    if model.packages_by_dir.is_empty() {
        return violations;
    }

    let row_re = Regex::new(r"^\|\s*`([^`]+)`\s*\|").expect("Valid regex");
    let mut table_dirs = BTreeSet::new();
    for line in content.lines() {
        if let Some(caps) = row_re.captures(line.trim()) {
            table_dirs.insert(caps.get(1).unwrap().as_str().to_string());
        }
    }
    for expected_dir in model.packages_by_dir.keys() {
        if !table_dirs.contains(expected_dir) {
            violations.push(format!(
                "{}: Workspace Crate Catalog table omits member `{expected_dir}` declared in Cargo.toml",
                overview_path.display()
            ));
        }
    }
    for listed_dir in &table_dirs {
        if !model.packages_by_dir.contains_key(listed_dir) {
            violations.push(format!(
                "{}: Workspace Crate Catalog table lists non-existent member `{listed_dir}`",
                overview_path.display()
            ));
        }
    }

    if let Some(mermaid) = extract_second_mermaid_block(&content) {
        for pkg_name in model.packages_by_name.keys() {
            if !mermaid.contains(pkg_name) {
                violations.push(format!(
                    "{}: Crate Dependency Topology Mermaid diagram omits workspace package `{pkg_name}`",
                    overview_path.display()
                ));
            }
        }
    }
    violations
}

fn extract_second_mermaid_block(content: &str) -> Option<String> {
    let mut blocks = Vec::new();
    let mut in_mermaid = false;
    let mut buf = String::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if !in_mermaid && trimmed == "```mermaid" {
            in_mermaid = true;
            buf.clear();
        } else if in_mermaid && trimmed == "```" {
            blocks.push(std::mem::take(&mut buf));
            in_mermaid = false;
        } else if in_mermaid {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    blocks.into_iter().nth(1)
}

fn collect_rs_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for p in paths {
        if p.is_dir() {
            collect_rs_recursive(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

/// Runs all static documentation gates across active Markdown files and GUI Rust sources.
pub fn audit_static_doc_gates(
    workspace_root: &Path,
    active_md: &[PathBuf],
    model: &WorkspaceManifestModel,
) -> Vec<String> {
    let mut violations = Vec::new();
    let approved_caps = load_approved_ignore_fences(workspace_root);
    let binary_pkgs = collect_binary_only_packages(model);
    let expected_crates = model.packages_by_dir.len();

    for md_path in active_md {
        let Ok(content) = fs::read_to_string(md_path) else {
            continue;
        };
        violations.extend(check_markdown_ignore_fences(
            md_path,
            workspace_root,
            &content,
            &approved_caps,
        ));
        violations.extend(check_markdown_binary_gui_imports(
            md_path,
            &content,
            &binary_pkgs,
        ));
        violations.extend(check_markdown_crate_counts(
            md_path,
            &content,
            expected_crates,
        ));
    }

    let mut gui_rs = Vec::new();
    for pkg in [
        "math_explorer_gui",
        "oxidizemath",
        "math_explorer",
        "oxidize-cli",
    ] {
        let dir = model
            .packages_by_name
            .get(pkg)
            .map_or(pkg, |p| p.member_dir.as_str());
        collect_rs_recursive(&workspace_root.join(dir).join("src"), &mut gui_rs);
        collect_rs_recursive(&workspace_root.join(dir).join("examples"), &mut gui_rs);
    }
    for rs_path in gui_rs {
        if let Ok(src) = fs::read_to_string(&rs_path) {
            violations.extend(check_rs_binary_gui_imports(&rs_path, &src, &binary_pkgs));
        }
    }

    violations.extend(audit_system_overview_crate_catalog(workspace_root, model));
    violations
}
