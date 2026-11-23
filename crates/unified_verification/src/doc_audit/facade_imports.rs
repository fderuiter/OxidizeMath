use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

const DOMAIN_CRATES: &[&str] = &[
    "pure_math",
    "domain_ai",
    "domain_applied",
    "domain_biology",
    "domain_climate",
    "domain_epidemiology",
    "domain_physics",
];

fn is_rust_code_fence(tag: &str) -> bool {
    let trimmed = tag.trim();
    trimmed.is_empty()
        || matches!(
            trimmed,
            "rust" | "no_run" | "should_panic" | "compile_fail" | "ignore"
        )
        || trimmed.starts_with("rust,")
        || trimmed.starts_with("rust ")
}

fn collect_files_with_ext(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name == "target" || name == ".git" {
            continue;
        }
        if path.is_dir() {
            collect_files_with_ext(&path, ext, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}

fn collect_active_guide_paths(workspace_root: &Path) -> Vec<PathBuf> {
    let mut guides = Vec::new();
    let root_readme = workspace_root.join("README.md");
    if root_readme.is_file() {
        guides.push(root_readme);
    }
    collect_files_with_ext(&workspace_root.join("math_explorer"), "md", &mut guides);
    for crate_name in DOMAIN_CRATES {
        collect_files_with_ext(
            &workspace_root.join("crates").join(crate_name),
            "md",
            &mut guides,
        );
    }

    let mut docs_md = Vec::new();
    collect_files_with_ext(&workspace_root.join("docs"), "md", &mut docs_md);
    let adr_dir = workspace_root.join("docs").join("adr");
    for md_path in docs_md {
        if !md_path.starts_with(&adr_dir) {
            guides.push(md_path);
        }
    }
    guides
}

struct FacadeImportPatterns {
    doc_re: Regex,
    forbidden_re: Regex,
    bypass_re: Regex,
    std_rng_re: Regex,
    default_rng_re: Regex,
}

impl FacadeImportPatterns {
    fn new() -> Self {
        Self {
            doc_re: Regex::new(r"^\s*(?://!|///)\s?(.*)$").expect("Valid regex"),
            forbidden_re: Regex::new(r"(?:^|[^:\w])(domain_[a-z0-9_]+|pure_math)::")
                .expect("Valid regex"),
            bypass_re: Regex::new(
                r"(?:^|[^:\w])(domain_[a-z0-9_]+|pure_math|math_commons|oxidize_core|nalgebra|num_complex|petgraph|rand)::",
            )
            .expect("Valid regex"),
            std_rng_re: Regex::new(r"\bStdRng\b").expect("Valid regex"),
            default_rng_re: Regex::new(
                r"(?:\bOxidizeRng\s*::\s*default\b|<\s*[^>\n]*\bOxidizeRng\s+as\s+[^>\n]*\bDefault\s*>\s*::\s*default\b)",
            )
            .expect("Valid regex"),
        }
    }
}

fn check_rng_snippet_line(
    path: &Path,
    line_no: usize,
    context_label: &str,
    raw_line: &str,
    patterns: &FacadeImportPatterns,
    allow_default_rng: bool,
    violations: &mut Vec<String>,
) {
    let unhidden = raw_line.strip_prefix("# ").unwrap_or(raw_line).trim();
    if unhidden.is_empty() || unhidden == "#" || unhidden.starts_with("//") {
        return;
    }
    let code_part = unhidden.split("//").next().unwrap_or(unhidden);
    if patterns.std_rng_re.is_match(code_part) {
        violations.push(format!(
            "{}:{line_no}: {context_label} references forbidden `StdRng`; use `math_explorer::rng::OxidizeRng`",
            path.display()
        ));
    }
    if !allow_default_rng && patterns.default_rng_re.is_match(code_part) {
        violations.push(format!(
            "{}:{line_no}: {context_label} calls unallowlisted `OxidizeRng::default`; pass an explicit seed via `OxidizeRng::new` or `OxidizeRng::from_seed`",
            path.display()
        ));
    }
}

fn check_markdown_facade_imports_with_patterns(
    path: &Path,
    content: &str,
    patterns: &FacadeImportPatterns,
) -> Vec<String> {
    let mut violations = Vec::new();
    let mut in_fence = false;
    let mut is_rust = false;

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(after_ticks) = trimmed.strip_prefix("```") {
            if !in_fence {
                in_fence = true;
                is_rust = is_rust_code_fence(after_ticks);
            } else {
                in_fence = false;
                is_rust = false;
            }
            continue;
        }
        if !in_fence || !is_rust || trimmed.starts_with("//") {
            continue;
        }
        check_rng_snippet_line(
            path,
            idx + 1,
            "Consumer guide snippet",
            trimmed,
            patterns,
            false,
            &mut violations,
        );
        if trimmed.starts_with("# ") || trimmed == "#" {
            continue;
        }
        let code_part = trimmed.split("//").next().unwrap_or(trimmed);
        if let Some(caps) = patterns.forbidden_re.captures(code_part) {
            violations.push(format!(
                "{}:{}: Consumer guide snippet imports internal crate `{}` directly instead of `math_explorer::`",
                path.display(),
                idx + 1,
                caps.get(1).unwrap().as_str()
            ));
        }
    }
    violations
}

/// Checks Markdown content for visible internal `domain_*::` or `pure_math::` imports/paths
/// and forbidden `StdRng` / `OxidizeRng::default` calls inside Rust code fences.
#[cfg(test)]
pub fn check_markdown_facade_imports(path: &Path, content: &str) -> Vec<String> {
    let patterns = FacadeImportPatterns::new();
    check_markdown_facade_imports_with_patterns(path, content, &patterns)
}

fn check_rustdoc_facade_imports_with_patterns(
    path: &Path,
    content: &str,
    patterns: &FacadeImportPatterns,
    check_facade: bool,
    allow_default_rng: bool,
) -> Vec<String> {
    let mut violations = Vec::new();
    let mut in_fence = false;
    let mut is_rust = false;

    for (idx, line) in content.lines().enumerate() {
        let Some(caps) = patterns.doc_re.captures(line) else {
            in_fence = false;
            is_rust = false;
            continue;
        };
        let doc_line = caps.get(1).map_or("", |m| m.as_str()).trim();
        if let Some(after_ticks) = doc_line.strip_prefix("```") {
            if !in_fence {
                in_fence = true;
                is_rust = is_rust_code_fence(after_ticks);
            } else {
                in_fence = false;
                is_rust = false;
            }
            continue;
        }
        if !in_fence || !is_rust || doc_line.starts_with("//") {
            continue;
        }
        check_rng_snippet_line(
            path,
            idx + 1,
            "Rustdoc snippet",
            doc_line,
            patterns,
            allow_default_rng,
            &mut violations,
        );
        if !check_facade || doc_line.starts_with("# ") || doc_line == "#" {
            continue;
        }
        let code_part = doc_line.split("//").next().unwrap_or(doc_line);
        if let Some(caps) = patterns.forbidden_re.captures(code_part) {
            violations.push(format!(
                "{}:{}: Visible rustdoc snippet imports internal crate `{}` directly instead of `math_explorer::`",
                path.display(),
                idx + 1,
                caps.get(1).unwrap().as_str()
            ));
        }
    }
    violations
}

/// Checks Rust source doc-comment fences (`///` / `//!`) for visible internal
/// `domain_*::` or `pure_math::` imports/paths and forbidden RNG calls.
#[cfg(test)]
pub fn check_rustdoc_facade_imports(path: &Path, content: &str) -> Vec<String> {
    let patterns = FacadeImportPatterns::new();
    check_rustdoc_facade_imports_with_patterns(path, content, &patterns, true, false)
}

fn check_example_facade_imports_with_patterns(
    path: &Path,
    content: &str,
    patterns: &FacadeImportPatterns,
) -> Vec<String> {
    let mut violations = Vec::new();

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        check_rng_snippet_line(
            path,
            idx + 1,
            "Example/benchmark",
            trimmed,
            patterns,
            false,
            &mut violations,
        );
        let code_part = trimmed.split("//").next().unwrap_or(trimmed);
        for caps in patterns.bypass_re.captures_iter(code_part) {
            let ident_match = caps.get(1).unwrap();
            let prefix_before = &code_part[..ident_match.start()];
            if prefix_before.ends_with("math_explorer::") {
                continue;
            }
            violations.push(format!(
                "{}:{}: Example/benchmark bypasses `math_explorer` facade via `{}`",
                path.display(),
                idx + 1,
                ident_match.as_str()
            ));
        }
    }
    violations
}

/// Checks `math_explorer/examples/*.rs` and `math_explorer/benches/*.rs` for facade-bypassing imports.
#[cfg(test)]
pub fn check_example_facade_imports(path: &Path, content: &str) -> Vec<String> {
    let patterns = FacadeImportPatterns::new();
    check_example_facade_imports_with_patterns(path, content, &patterns)
}

fn is_default_rng_allowlisted(workspace_root: &Path, path: &Path, allowlist: &[String]) -> bool {
    let rel = path.strip_prefix(workspace_root).unwrap_or(path);
    let norm = rel.to_string_lossy().replace('\\', "/");
    let norm = norm.strip_prefix("./").unwrap_or(&norm);
    allowlist.iter().any(|entry| entry == norm)
}

/// Audits active Markdown guides, domain/core rustdoc blocks, and `math_explorer` examples/benchmarks.
fn audit_non_domain_and_example_hygiene(
    workspace_root: &Path,
    patterns: &FacadeImportPatterns,
    default_allow: &[String],
    violations: &mut Vec<String>,
) {
    for extra_src in [
        workspace_root.join("crates/oxidize_core/src"),
        workspace_root.join("crates/math_commons/src"),
        workspace_root.join("math_explorer/src"),
        workspace_root.join("math_explorer_gui/src"),
    ] {
        let mut rs_files = Vec::new();
        collect_files_with_ext(&extra_src, "rs", &mut rs_files);
        for rs_path in rs_files {
            if let Ok(content) = fs::read_to_string(&rs_path) {
                let allow_def = is_default_rng_allowlisted(workspace_root, &rs_path, default_allow);
                violations.extend(check_rustdoc_facade_imports_with_patterns(
                    &rs_path, &content, patterns, false, allow_def,
                ));
            }
        }
    }

    for sub in ["examples", "benches"] {
        let mut targets = Vec::new();
        collect_files_with_ext(
            &workspace_root.join("math_explorer").join(sub),
            "rs",
            &mut targets,
        );
        for target_path in targets {
            if let Ok(content) = fs::read_to_string(&target_path) {
                violations.extend(check_example_facade_imports_with_patterns(
                    &target_path,
                    &content,
                    patterns,
                ));
            }
        }
    }
}

pub fn audit_facade_import_hygiene(workspace_root: &Path) -> Vec<String> {
    let patterns = FacadeImportPatterns::new();
    let wl_candidate = workspace_root.join(crate::whitelist::WHITELIST_FILE_NAME);
    let default_allow = crate::whitelist::load_whitelist_or_default(
        wl_candidate.is_file().then_some(wl_candidate.as_path()),
        crate::whitelist::WORKSPACE_WHITELIST_CANDIDATES,
    )
    .entropy_guard
    .and_then(|cfg| cfg.default_rng_allowlist)
    .unwrap_or_default();
    let mut violations = Vec::new();

    for md_path in collect_active_guide_paths(workspace_root) {
        if let Ok(content) = fs::read_to_string(&md_path) {
            violations.extend(check_markdown_facade_imports_with_patterns(
                &md_path, &content, &patterns,
            ));
        }
    }

    for crate_name in DOMAIN_CRATES {
        let mut rs_files = Vec::new();
        collect_files_with_ext(
            &workspace_root.join("crates").join(crate_name).join("src"),
            "rs",
            &mut rs_files,
        );
        for rs_path in rs_files {
            if let Ok(content) = fs::read_to_string(&rs_path) {
                let allow_def =
                    is_default_rng_allowlisted(workspace_root, &rs_path, &default_allow);
                violations.extend(check_rustdoc_facade_imports_with_patterns(
                    &rs_path, &content, &patterns, true, allow_def,
                ));
            }
        }
    }

    audit_non_domain_and_example_hygiene(
        workspace_root,
        &patterns,
        &default_allow,
        &mut violations,
    );

    violations
}

fn audit_active_markdown_contracts(
    workspace_root: &Path,
    active_md: &[PathBuf],
    model: &super::manifest_model::WorkspaceManifestModel,
    violations: &mut Vec<String>,
) {
    let trait_registry = super::trait_contracts::TraitContractRegistry::load(workspace_root);
    for md_path in active_md {
        let Ok(content) = fs::read_to_string(md_path) else {
            continue;
        };
        violations.extend(super::cargo_cli_checks::check_markdown_cargo_commands(
            md_path,
            workspace_root,
            &content,
            model,
        ));
        violations.extend(super::toml_dep_checks::check_markdown_toml_dependencies(
            md_path,
            workspace_root,
            &content,
            model,
        ));
        violations.extend(
            super::trait_contracts::check_markdown_trait_and_gui_contracts(
                md_path,
                &content,
                &trait_registry,
            ),
        );
    }
}

/// Audits Cargo CLI commands, TOML `[dependencies]` blocks, examples catalog parity,
/// paper/rustdoc gate instructions, trait/GUI onboarding contracts, and static doc gates.
pub fn audit_workspace_guides_and_manifests(workspace_root: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    let model = match super::manifest_model::WorkspaceManifestModel::load(workspace_root) {
        Ok(m) => m,
        Err(err) => {
            violations.push(err);
            return violations;
        }
    };

    let active_md = super::cargo_cli_checks::collect_active_markdown_files(workspace_root);
    audit_active_markdown_contracts(workspace_root, &active_md, &model, &mut violations);

    let mut ex_files = Vec::new();
    collect_files_with_ext(
        &workspace_root.join("math_explorer/examples"),
        "rs",
        &mut ex_files,
    );
    for ex_path in ex_files {
        if let Ok(content) = fs::read_to_string(&ex_path) {
            violations.extend(super::cargo_cli_checks::check_rustdoc_cargo_commands(
                &ex_path, &content, &model,
            ));
        }
    }

    violations.extend(super::catalog_and_guide_checks::audit_examples_catalog(
        workspace_root,
        &model,
    ));
    violations.extend(
        super::catalog_and_guide_checks::audit_paper_and_gate_instructions(
            workspace_root,
            &active_md,
        ),
    );
    violations.extend(super::static_gates::audit_static_doc_gates(
        workspace_root,
        &active_md,
        &model,
    ));
    violations
}

#[cfg(test)]
#[path = "facade_imports_tests.rs"]
mod tests;
