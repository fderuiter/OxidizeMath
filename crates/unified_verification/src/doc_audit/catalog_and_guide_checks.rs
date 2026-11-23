use super::manifest_model::WorkspaceManifestModel;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

fn heading_to_domain_feature(heading_line: &str) -> Option<&'static str> {
    let lower = heading_line.to_ascii_lowercase();
    for (needle, feat) in [
        ("pure math", "pure_math"),
        ("pure_math", "pure_math"),
        ("epidemiology", "epidemiology"),
        ("climate", "climate"),
        ("biology", "biology"),
        ("applied", "applied"),
        ("physics", "physics"),
    ] {
        if lower.contains(needle) {
            return Some(feat);
        }
    }
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .collect();
    words.contains(&"ai").then_some("ai")
}

/// Validates 1-to-1 parity across `math_explorer/examples/*.rs`, `math_explorer/Cargo.toml`
/// `[[example]]` entries, and `math_explorer/examples/README.md` domain sections.
pub fn audit_examples_catalog(
    workspace_root: &Path,
    model: &WorkspaceManifestModel,
) -> Vec<String> {
    let mut violations = Vec::new();
    let Some(pkg) = model.packages_by_name.get("math_explorer") else {
        return violations;
    };
    let examples_dir = workspace_root.join("math_explorer").join("examples");
    let readme_path = examples_dir.join("README.md");
    if !examples_dir.is_dir() {
        return violations;
    }

    check_disk_vs_manifest_examples(&examples_dir, &pkg.examples, &mut violations);

    let Ok(readme_content) = fs::read_to_string(&readme_path) else {
        violations.push(format!(
            "{}: Missing examples catalog README.md",
            readme_path.display()
        ));
        return violations;
    };

    check_readme_vs_manifest_examples(
        &readme_path,
        &readme_content,
        model,
        &pkg.examples,
        &mut violations,
    );
    violations
}

fn check_disk_vs_manifest_examples(
    examples_dir: &Path,
    manifest_examples: &BTreeMap<String, Vec<String>>,
    violations: &mut Vec<String>,
) {
    let mut disk_stems = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(examples_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("rs")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                disk_stems.insert(stem.to_string());
                if !manifest_examples.contains_key(stem) {
                    violations.push(format!(
                        "{}: Unregistered example `.rs` file missing `[[example]]` entry in `math_explorer/Cargo.toml`",
                        path.display()
                    ));
                }
            }
        }
    }

    for (name, req_features) in manifest_examples {
        if !disk_stems.contains(name) {
            violations.push(format!(
                "math_explorer/Cargo.toml: Registered `[[example]]` `{name}` has no `{name}.rs` in `math_explorer/examples/`"
            ));
        }
        if req_features.is_empty() {
            violations.push(format!(
                "math_explorer/Cargo.toml: `[[example]]` `{name}` omits `required-features` while `default = []`"
            ));
        }
    }
}

fn check_readme_vs_manifest_examples(
    readme_path: &Path,
    content: &str,
    model: &WorkspaceManifestModel,
    manifest_examples: &BTreeMap<String, Vec<String>>,
    violations: &mut Vec<String>,
) {
    let item_re = Regex::new(r"^-\s+`([a-zA-Z0-9_]+)`\s*:").expect("Valid regex");
    let mut current_domain: Option<(&'static str, usize)> = None;
    let mut cataloged: BTreeMap<String, (&'static str, usize)> = BTreeMap::new();

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("### ") {
            match heading_to_domain_feature(trimmed) {
                Some(dom) => current_domain = Some((dom, idx + 1)),
                None => violations.push(format!(
                    "{}:{}: Unrecognized domain heading `{trimmed}` in examples catalog",
                    readme_path.display(),
                    idx + 1
                )),
            }
            continue;
        }
        let Some(caps) = item_re.captures(trimmed) else {
            continue;
        };
        let ex_name = caps.get(1).unwrap().as_str();
        let Some((section_dom, _)) = current_domain else {
            violations.push(format!(
                "{}:{}: Example `{ex_name}` appears outside any `### <Domain>` heading",
                readme_path.display(),
                idx + 1
            ));
            continue;
        };
        if cataloged
            .insert(ex_name.to_string(), (section_dom, idx + 1))
            .is_some()
        {
            violations.push(format!(
                "{}:{}: Duplicate example entry `{ex_name}` in catalog",
                readme_path.display(),
                idx + 1
            ));
        }
    }

    verify_catalog_completeness_and_domains(
        readme_path,
        model,
        manifest_examples,
        &cataloged,
        violations,
    );
}

fn verify_catalog_completeness_and_domains(
    readme_path: &Path,
    model: &WorkspaceManifestModel,
    manifest_examples: &BTreeMap<String, Vec<String>>,
    cataloged: &BTreeMap<String, (&'static str, usize)>,
    violations: &mut Vec<String>,
) {
    for (ex_name, (section_dom, line_num)) in cataloged {
        let Some(req_features) = manifest_examples.get(ex_name) else {
            violations.push(format!(
                "{}:{}: Catalog lists unregistered example `{ex_name}`",
                readme_path.display(),
                line_num
            ));
            continue;
        };
        if let Some(expected_dom) = model.primary_domain_feature("math_explorer", req_features)
            && expected_dom != *section_dom
        {
            violations.push(format!(
                "{}:{}: Example `{ex_name}` requires `{expected_dom}` in Cargo.toml but is listed under `{section_dom}`",
                readme_path.display(),
                line_num
            ));
        }
    }

    for manifest_name in manifest_examples.keys() {
        if !cataloged.contains_key(manifest_name) {
            violations.push(format!(
                "{}: Missing registered example `{manifest_name}` from domain catalog",
                readme_path.display()
            ));
        }
    }
}

/// Rejects stale `Tectonic` paper instructions when `apps/xtask/src/papers.rs` uses TeXLive `pdflatex`/`bibtex`,
/// and checks `gates.rs` / `ci.yml` full-feature rustdoc gates.
pub fn audit_paper_and_gate_instructions(
    workspace_root: &Path,
    active_md_files: &[std::path::PathBuf],
) -> Vec<String> {
    let mut violations = Vec::new();
    let papers_rs = workspace_root.join("apps/xtask/src/papers.rs");
    let uses_texlive = fs::read_to_string(&papers_rs)
        .map(|src| src.contains("\"pdflatex\"") && !src.contains("\"tectonic\""))
        .unwrap_or(true);

    if uses_texlive {
        let tectonic_re = Regex::new(r"(?i)\btectonic\b").expect("Valid regex");
        for md_path in active_md_files {
            if let Ok(content) = fs::read_to_string(md_path) {
                scan_forbidden_pattern(
                    md_path,
                    &content,
                    &tectonic_re,
                    "Stale `Tectonic` reference (paper compiler uses TeXLive `pdflatex` and `bibtex`)",
                    &mut violations,
                );
            }
        }
        let xtask_main = workspace_root.join("apps/xtask/src/main.rs");
        if let Ok(content) = fs::read_to_string(&xtask_main) {
            let prod_content = content.split("#[cfg(test)]").next().unwrap_or(&content);
            scan_forbidden_pattern(
                &xtask_main,
                prod_content,
                &tectonic_re,
                "Stale `Tectonic` help text (paper compiler uses TeXLive `pdflatex` and `bibtex`)",
                &mut violations,
            );
        }
    }

    check_full_feature_rustdoc_gates(workspace_root, &mut violations);
    violations
}

fn scan_forbidden_pattern(
    path: &Path,
    content: &str,
    re: &Regex,
    reason: &str,
    violations: &mut Vec<String>,
) {
    for (idx, line) in content.lines().enumerate() {
        if re.is_match(line) {
            violations.push(format!("{}:{}: {reason}", path.display(), idx + 1));
        }
    }
}

fn check_full_feature_rustdoc_gates(workspace_root: &Path, violations: &mut Vec<String>) {
    let gates_rs = workspace_root.join("apps/xtask/src/gates.rs");
    if let Ok(src) = fs::read_to_string(&gates_rs)
        && src.contains("\"doc\"")
        && src.contains("\"--no-deps\"")
        && !src.contains("\"--all-features\"")
    {
        violations.push(format!(
            "{}: `check_cargo_docs` omits `\"--all-features\"`",
            gates_rs.display()
        ));
    }

    let ci_yml = workspace_root.join(".github/workflows/ci.yml");
    if let Ok(ci_src) = fs::read_to_string(&ci_yml) {
        for (idx, line) in ci_src.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.contains("cargo doc --workspace --no-deps")
                && !trimmed.contains("--all-features")
            {
                violations.push(format!(
                    "{}:{}: CI `cargo doc --workspace --no-deps` omits `--all-features`",
                    ci_yml.display(),
                    idx + 1
                ));
            }
        }
    }
}
