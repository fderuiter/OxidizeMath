//! Scoped documentation, facade import, and guide verification for `batch-staged`.

use super::facade_imports::{
    DOMAIN_CRATES, FacadeImportPatterns, check_example_facade_imports_with_regex,
    check_markdown_facade_imports_with_regex, check_rustdoc_facade_imports_with_regexes,
};
use super::manifest_model::WorkspaceManifestModel;
use super::trait_contracts::TraitContractRegistry;
use std::fs;
use std::path::{Path, PathBuf};

const TRACKED_TRAIT_SOURCES: &[&str] = &[
    "crates/scientific_metadata/src/theory.rs",
    "math_explorer_gui/src/framework/mod.rs",
    "math_explorer_gui/src/tabs/mod.rs",
];

fn normalize_rel(path: &str) -> &str {
    path.trim_start_matches("./")
}

fn has_excluded_segment(rel: &str) -> bool {
    rel.split('/').any(|seg| {
        matches!(
            seg,
            "target" | ".git" | "dist" | "artifacts" | "node_modules" | "egui_plot"
        )
    })
}

fn is_active_facade_guide_rel(rel: &str) -> bool {
    if !rel.ends_with(".md") || has_excluded_segment(rel) || rel.starts_with("docs/adr/") {
        return false;
    }
    rel == "README.md"
        || rel.starts_with("math_explorer/")
        || rel.starts_with("docs/")
        || DOMAIN_CRATES
            .iter()
            .any(|c| rel.starts_with(&format!("crates/{c}/")))
}

fn is_domain_src_rs_rel(rel: &str) -> bool {
    rel.ends_with(".rs")
        && !has_excluded_segment(rel)
        && DOMAIN_CRATES
            .iter()
            .any(|c| rel.starts_with(&format!("crates/{c}/src/")))
}

fn is_example_or_bench_rs_rel(rel: &str) -> bool {
    rel.ends_with(".rs")
        && !has_excluded_segment(rel)
        && (rel.starts_with("math_explorer/examples/") || rel.starts_with("math_explorer/benches/"))
}

pub(super) fn is_active_markdown_rel(rel: &str) -> bool {
    if !rel.ends_with(".md") || has_excluded_segment(rel) || rel.starts_with("docs/adr/") {
        return false;
    }
    !rel.contains('/')
        || rel.starts_with("apps/")
        || rel.starts_with("crates/")
        || rel.starts_with("docs/")
        || rel.starts_with("math_explorer/")
        || rel.starts_with("math_explorer_gui/")
}

/// Audits facade import hygiene strictly on staged active guides, domain `.rs`, and example/bench `.rs` files.
pub fn audit_staged_facade_import_hygiene(
    workspace_root: &Path,
    existing_staged: &[String],
) -> Vec<String> {
    if existing_staged.is_empty() {
        return Vec::new();
    }
    let patterns = FacadeImportPatterns::new();
    let mut violations = Vec::new();
    for raw in existing_staged {
        let norm = raw.replace('\\', "/");
        let rel = normalize_rel(&norm);
        let full_path = workspace_root.join(rel);
        if is_active_facade_guide_rel(rel) {
            if let Ok(content) = fs::read_to_string(&full_path) {
                violations.extend(check_markdown_facade_imports_with_regex(
                    &full_path,
                    &content,
                    &patterns.forbidden_re,
                ));
            }
        } else if is_domain_src_rs_rel(rel) {
            if let Ok(content) = fs::read_to_string(&full_path) {
                violations.extend(check_rustdoc_facade_imports_with_regexes(
                    &full_path,
                    &content,
                    &patterns.doc_re,
                    &patterns.forbidden_re,
                ));
            }
        } else if is_example_or_bench_rs_rel(rel)
            && let Ok(content) = fs::read_to_string(&full_path)
        {
            violations.extend(check_example_facade_imports_with_regex(
                &full_path,
                &content,
                &patterns.bypass_re,
            ));
        }
    }
    violations
}

fn collect_staged_active_md(workspace_root: &Path, existing_staged: &[String]) -> Vec<PathBuf> {
    existing_staged
        .iter()
        .map(|p| p.replace('\\', "/"))
        .filter(|p| is_active_markdown_rel(normalize_rel(p)))
        .map(|p| workspace_root.join(normalize_rel(&p)))
        .filter(|p| p.is_file())
        .collect()
}

fn collect_staged_gui_rs(workspace_root: &Path, existing_staged: &[String]) -> Vec<PathBuf> {
    existing_staged
        .iter()
        .map(|p| p.replace('\\', "/"))
        .filter(|p| {
            let rel = normalize_rel(p);
            rel.ends_with(".rs")
                && !has_excluded_segment(rel)
                && (rel.starts_with("math_explorer_gui/src/")
                    || rel.starts_with("math_explorer/src/")
                    || rel.starts_with("math_explorer/examples/"))
        })
        .map(|p| workspace_root.join(normalize_rel(&p)))
        .filter(|p| p.is_file())
        .collect()
}

fn collect_staged_example_rs(workspace_root: &Path, existing_staged: &[String]) -> Vec<PathBuf> {
    existing_staged
        .iter()
        .map(|p| p.replace('\\', "/"))
        .filter(|p| {
            let rel = normalize_rel(p);
            rel.starts_with("math_explorer/examples/")
                && rel.ends_with(".rs")
                && !has_excluded_segment(rel)
        })
        .map(|p| workspace_root.join(normalize_rel(&p)))
        .filter(|p| p.is_file())
        .collect()
}

fn audit_staged_md_contracts(
    workspace_root: &Path,
    staged_md: &[PathBuf],
    trait_src_touched: bool,
    model: Option<&WorkspaceManifestModel>,
    violations: &mut Vec<String>,
) {
    let trait_targets = if trait_src_touched {
        super::cargo_cli_checks::collect_active_markdown_files(workspace_root)
    } else {
        staged_md.to_vec()
    };
    let trait_registry = if trait_src_touched {
        Some(TraitContractRegistry::load(workspace_root))
    } else {
        None
    };

    for md_path in staged_md {
        let Ok(content) = fs::read_to_string(md_path) else {
            continue;
        };
        if let Some(m) = model {
            violations.extend(super::cargo_cli_checks::check_markdown_cargo_commands(
                md_path,
                workspace_root,
                &content,
                m,
            ));
            violations.extend(super::toml_dep_checks::check_markdown_toml_dependencies(
                md_path,
                workspace_root,
                &content,
                m,
            ));
        }
    }

    let mut lazy_registry: Option<TraitContractRegistry> = trait_registry;
    for md_path in &trait_targets {
        let Ok(content) = fs::read_to_string(md_path) else {
            continue;
        };
        let needs_traits = ["TheoryDescribable", "InteractiveTool", "ExplorerTab"]
            .iter()
            .any(|t| content.contains(&format!("{t} for ")));
        let empty_reg = TraitContractRegistry::default();
        let reg = if needs_traits {
            lazy_registry.get_or_insert_with(|| TraitContractRegistry::load(workspace_root))
        } else {
            &empty_reg
        };
        violations.extend(
            super::trait_contracts::check_markdown_trait_and_gui_contracts(md_path, &content, reg),
        );
    }
}

fn load_staged_manifest_model_if_needed(
    workspace_root: &Path,
    catalog_touched: bool,
    overview_touched: bool,
    staged_ex: &[PathBuf],
    staged_md: &[PathBuf],
    staged_gui_rs: &[PathBuf],
) -> Result<Option<WorkspaceManifestModel>, String> {
    let has_root_manifest = workspace_root.join("Cargo.toml").is_file();
    let needs_model = catalog_touched
        || overview_touched
        || !staged_ex.is_empty()
        || (has_root_manifest && (!staged_md.is_empty() || !staged_gui_rs.is_empty()))
        || staged_md.iter().any(|p| {
            fs::read_to_string(p)
                .map(|c| c.contains("cargo ") || c.contains("```toml"))
                .unwrap_or(false)
        });
    if needs_model {
        WorkspaceManifestModel::load(workspace_root).map(Some)
    } else {
        Ok(None)
    }
}

fn audit_staged_static_doc_gates(
    workspace_root: &Path,
    staged_md: &[PathBuf],
    staged_gui_rs: &[PathBuf],
    overview_touched: bool,
    model: Option<&WorkspaceManifestModel>,
    violations: &mut Vec<String>,
) {
    if staged_md.is_empty() && staged_gui_rs.is_empty() && !overview_touched {
        return;
    }
    let approved_caps = super::static_gates::load_approved_ignore_fences(workspace_root);
    let default_model = WorkspaceManifestModel::default();
    let binary_pkgs =
        super::static_gates::collect_binary_only_packages(model.unwrap_or(&default_model));
    let expected_crates = model.map_or(0, |m| m.packages_by_dir.len());

    for md_path in staged_md {
        let Ok(content) = fs::read_to_string(md_path) else {
            continue;
        };
        violations.extend(super::static_gates::check_markdown_ignore_fences(
            md_path,
            workspace_root,
            &content,
            &approved_caps,
        ));
        violations.extend(super::static_gates::check_markdown_binary_gui_imports(
            md_path,
            &content,
            &binary_pkgs,
        ));
        violations.extend(super::static_gates::check_markdown_crate_counts(
            md_path,
            &content,
            expected_crates,
        ));
    }
    for rs_path in staged_gui_rs {
        if let Ok(src) = fs::read_to_string(rs_path) {
            violations.extend(super::static_gates::check_rs_binary_gui_imports(
                rs_path,
                &src,
                &binary_pkgs,
            ));
        }
    }
    if overview_touched && let Some(m) = model {
        violations.extend(super::static_gates::audit_system_overview_crate_catalog(
            workspace_root,
            m,
        ));
    }
}

fn audit_staged_examples(
    workspace_root: &Path,
    staged_ex: &[PathBuf],
    catalog_touched: bool,
    model: Option<&WorkspaceManifestModel>,
    violations: &mut Vec<String>,
) {
    let Some(m) = model else {
        return;
    };
    for ex_path in staged_ex {
        if let Ok(content) = fs::read_to_string(ex_path) {
            violations.extend(super::cargo_cli_checks::check_rustdoc_cargo_commands(
                ex_path, &content, m,
            ));
        }
    }
    if catalog_touched {
        violations.extend(super::catalog_and_guide_checks::audit_examples_catalog(
            workspace_root,
            m,
        ));
    }
}

fn audit_staged_paper_guides(
    workspace_root: &Path,
    all_staged: &[String],
    staged_md: Vec<PathBuf>,
    gate_infra_touched: bool,
    violations: &mut Vec<String>,
) {
    if staged_md.is_empty() && !gate_infra_touched {
        return;
    }
    let paper_targets = if all_staged
        .iter()
        .any(|p| normalize_rel(p) == "apps/xtask/src/papers.rs")
    {
        super::cargo_cli_checks::collect_active_markdown_files(workspace_root)
    } else {
        staged_md
    };
    violations.extend(
        super::catalog_and_guide_checks::audit_paper_and_gate_instructions(
            workspace_root,
            &paper_targets,
        ),
    );
}

/// Audits Cargo CLI commands, TOML blocks, trait contracts, examples catalog, and paper/gate
/// instructions strictly on staged guides and touched catalog/gate paths.
pub fn audit_staged_guides_and_manifests(
    workspace_root: &Path,
    all_staged: &[String],
    existing_staged: &[String],
) -> Vec<String> {
    let mut violations = Vec::new();
    let staged_md = collect_staged_active_md(workspace_root, existing_staged);
    let staged_ex = collect_staged_example_rs(workspace_root, existing_staged);
    let staged_gui_rs = collect_staged_gui_rs(workspace_root, existing_staged);
    let has_staged = |pred: fn(&str) -> bool| all_staged.iter().any(|p| pred(normalize_rel(p)));
    let catalog_touched =
        has_staged(|r| r == "math_explorer/Cargo.toml" || r.starts_with("math_explorer/examples/"));
    let overview_touched = has_staged(|r| r == "docs/architecture/system_overview.md");
    let trait_src_touched = has_staged(|r| TRACKED_TRAIT_SOURCES.contains(&r));
    let gate_infra_touched = has_staged(|r| {
        matches!(
            r,
            "apps/xtask/src/papers.rs"
                | "apps/xtask/src/main.rs"
                | "apps/xtask/src/gates.rs"
                | ".github/workflows/ci.yml"
        )
    });

    let model = match load_staged_manifest_model_if_needed(
        workspace_root,
        catalog_touched,
        overview_touched,
        &staged_ex,
        &staged_md,
        &staged_gui_rs,
    ) {
        Ok(m) => m,
        Err(err) => return vec![err],
    };

    let m_ref = model.as_ref();
    audit_staged_static_doc_gates(
        workspace_root,
        &staged_md,
        &staged_gui_rs,
        overview_touched,
        m_ref,
        &mut violations,
    );
    audit_staged_md_contracts(
        workspace_root,
        &staged_md,
        trait_src_touched,
        m_ref,
        &mut violations,
    );
    audit_staged_examples(
        workspace_root,
        &staged_ex,
        catalog_touched,
        m_ref,
        &mut violations,
    );
    audit_staged_paper_guides(
        workspace_root,
        all_staged,
        staged_md,
        gate_infra_touched,
        &mut violations,
    );
    violations
}
