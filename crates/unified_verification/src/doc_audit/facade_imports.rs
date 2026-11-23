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

/// Checks Markdown content for visible internal `domain_*::` or `pure_math::` imports/paths
/// inside Rust code fences (ignoring hidden `# ` lines and comments).
pub fn check_markdown_facade_imports(path: &Path, content: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let forbidden_re =
        Regex::new(r"(?:^|[^:\w])(domain_[a-z0-9_]+|pure_math)::").expect("Valid regex");

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
        if !in_fence || !is_rust {
            continue;
        }
        if trimmed.starts_with("# ") || trimmed == "#" || trimmed.starts_with("//") {
            continue;
        }
        let code_part = trimmed.split("//").next().unwrap_or(trimmed);
        if let Some(caps) = forbidden_re.captures(code_part) {
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

/// Checks Rust source doc-comment fences (`///` / `//!`) for visible internal
/// `domain_*::` or `pure_math::` imports/paths.
pub fn check_rustdoc_facade_imports(path: &Path, content: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let doc_re = Regex::new(r"^\s*(?://!|///)\s?(.*)$").expect("Valid regex");
    let forbidden_re =
        Regex::new(r"(?:^|[^:\w])(domain_[a-z0-9_]+|pure_math)::").expect("Valid regex");

    let mut in_fence = false;
    let mut is_rust = false;

    for (idx, line) in content.lines().enumerate() {
        let Some(caps) = doc_re.captures(line) else {
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
        if !in_fence || !is_rust {
            continue;
        }
        if doc_line.starts_with("# ") || doc_line == "#" || doc_line.starts_with("//") {
            continue;
        }
        let code_part = doc_line.split("//").next().unwrap_or(doc_line);
        if let Some(caps) = forbidden_re.captures(code_part) {
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

/// Checks `math_explorer/examples/*.rs` and `math_explorer/benches/*.rs` for facade-bypassing imports.
pub fn check_example_facade_imports(path: &Path, content: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let bypass_re = Regex::new(
        r"(?:^|[^:\w])(domain_[a-z0-9_]+|pure_math|math_commons|oxidize_core|nalgebra|num_complex|petgraph|rand)::",
    )
    .expect("Valid regex");

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        let code_part = trimmed.split("//").next().unwrap_or(trimmed);
        for caps in bypass_re.captures_iter(code_part) {
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

/// Audits active Markdown guides, domain rustdoc blocks, and `math_explorer` examples/benchmarks.
pub fn audit_facade_import_hygiene(workspace_root: &Path) -> Vec<String> {
    let mut violations = Vec::new();

    for md_path in collect_active_guide_paths(workspace_root) {
        if let Ok(content) = fs::read_to_string(&md_path) {
            violations.extend(check_markdown_facade_imports(&md_path, &content));
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
                violations.extend(check_rustdoc_facade_imports(&rs_path, &content));
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
                violations.extend(check_example_facade_imports(&target_path, &content));
            }
        }
    }

    violations
}

/// Audits Cargo CLI commands, TOML `[dependencies]` blocks, examples catalog parity,
/// paper/rustdoc gate instructions, and trait/GUI onboarding contracts across `workspace_root`.
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
    let trait_registry = super::trait_contracts::TraitContractRegistry::load(workspace_root);

    for md_path in &active_md {
        let Ok(content) = fs::read_to_string(md_path) else {
            continue;
        };
        violations.extend(super::cargo_cli_checks::check_markdown_cargo_commands(
            md_path,
            workspace_root,
            &content,
            &model,
        ));
        violations.extend(super::toml_dep_checks::check_markdown_toml_dependencies(
            md_path,
            workspace_root,
            &content,
            &model,
        ));
        violations.extend(
            super::trait_contracts::check_markdown_trait_and_gui_contracts(
                md_path,
                &content,
                &trait_registry,
            ),
        );
    }

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
    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_facade_import_check_allows_hidden_shims_and_rejects_visible_internal_imports() {
        let clean = "```toml\ndomain_applied = \"0.4\"\n```\n\n```rust\n# use domain_applied::__doctest_facade as math_explorer;\nuse math_explorer::applied::battery_degradation::PowerLawModel;\n```\n";
        assert!(check_markdown_facade_imports(Path::new("guide.md"), clean).is_empty());

        let dirty =
            "```rust\nuse domain_applied::applied::battery_degradation::PowerLawModel;\n```\n";
        let violations = check_markdown_facade_imports(Path::new("guide.md"), dirty);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("domain_applied"));
    }

    #[test]
    fn rustdoc_and_adr_exclusion_work_end_to_end() {
        let dir = crate::utils::test_utils::create_temp_test_dir("facade_import_hygiene");
        let adr_dir = dir.join("docs/adr");
        let arch_dir = dir.join("docs/architecture");
        let dom_src = dir.join("crates/domain_ai/src");
        let ex_dir = dir.join("math_explorer/examples");
        fs::create_dir_all(&adr_dir).unwrap();
        fs::create_dir_all(&arch_dir).unwrap();
        fs::create_dir_all(&dom_src).unwrap();
        fs::create_dir_all(&ex_dir).unwrap();

        // Historical ADR with legacy domain_ai import must be ignored
        fs::write(
            adr_dir.join("0001-legacy.md"),
            "```rust\nuse domain_ai::ai::transformer::Encoder;\n```\n",
        )
        .unwrap();
        // Active guide with direct domain_climate import must be flagged
        fs::write(
            arch_dir.join("guide.md"),
            "```rust\nuse domain_climate::climate::cera::Cera;\n```\n",
        )
        .unwrap();
        // Rustdoc with visible pure_math import must be flagged
        fs::write(
            dom_src.join("lib.rs"),
            "//! ```rust\n//! use pure_math::pure_math::number_theory;\n//! ```\n",
        )
        .unwrap();
        // Example bypassing math_explorer::nalgebra must be flagged
        fs::write(
            ex_dir.join("demo.rs"),
            "use nalgebra::DMatrix;\nfn main() {}\n",
        )
        .unwrap();

        let violations = audit_facade_import_hygiene(&dir);
        assert_eq!(violations.len(), 3, "Violations: {violations:?}");
        assert!(!violations.iter().any(|v| v.contains("0001-legacy.md")));
        let _ = fs::remove_dir_all(&dir);
    }
}
