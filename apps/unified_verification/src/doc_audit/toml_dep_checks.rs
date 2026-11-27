//! Static validation of consumer `[dependencies]` TOML blocks referencing `oxidizemath` or `math_explorer`.

use super::manifest_model::WorkspaceManifestModel;
use oxidize_core::workspace::FacadeRole;
use std::path::Path;

fn expected_domain_for_guide(
    md_path: &Path,
    workspace_root: &Path,
    model: &WorkspaceManifestModel,
) -> Option<String> {
    let rel = md_path.strip_prefix(workspace_root).ok()?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    for (dir, pkg_name) in &model.packages_by_dir {
        if (rel_str == *dir || rel_str.starts_with(&format!("{dir}/")))
            && let Some(pkg) = model.packages_by_name.get(pkg_name)
            && pkg.metadata.facade_role == FacadeRole::Domain
        {
            return Some(
                pkg.metadata
                    .facade_module
                    .clone()
                    .unwrap_or_else(|| pkg.name.trim_start_matches("domain_").to_string()),
            );
        }
    }
    for seg in rel_str.split('/') {
        if seg == "pure_math" {
            return Some("pure_math".to_string());
        }
        if let Some(dom) = seg.strip_prefix("domain_")
            && !dom.is_empty()
        {
            return Some(dom.to_string());
        }
    }
    None
}

/// Validates consumer `[dependencies]` TOML snippets referencing `oxidizemath` or `math_explorer`.
pub fn check_markdown_toml_dependencies(
    md_path: &Path,
    workspace_root: &Path,
    content: &str,
    model: &WorkspaceManifestModel,
) -> Vec<String> {
    let mut violations = Vec::new();
    let mut in_fence = false;
    let mut toml_fence = false;
    let mut fence_start = 1usize;
    let mut buf = String::new();

    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(after) = trimmed.strip_prefix("```") {
            if !in_fence {
                in_fence = true;
                toml_fence = after.trim() == "toml";
                fence_start = idx + 1;
                buf.clear();
            } else {
                if toml_fence {
                    validate_toml_block(
                        md_path,
                        workspace_root,
                        fence_start,
                        &buf,
                        model,
                        &mut violations,
                    );
                }
                in_fence = false;
                toml_fence = false;
                buf.clear();
            }
            continue;
        }
        if in_fence && toml_fence {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    violations
}

fn validate_toml_block(
    md_path: &Path,
    workspace_root: &Path,
    fence_start: usize,
    raw_toml: &str,
    model: &WorkspaceManifestModel,
    violations: &mut Vec<String>,
) {
    if !raw_toml.contains("oxidizemath") && !raw_toml.contains("math_explorer") {
        return;
    }
    let parsed = toml::from_str::<toml::Value>(raw_toml)
        .or_else(|_| toml::from_str::<toml::Value>(&format!("[dependencies]\n{raw_toml}")));
    let doc = match parsed {
        Ok(v) => v,
        Err(err) => {
            violations.push(format!(
                "{}:{}: Invalid TOML dependency snippet: {err}",
                md_path.display(),
                fence_start
            ));
            return;
        }
    };

    for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
        for facade_name in ["oxidizemath", "math_explorer"] {
            if let Some(dep_val) = doc.get(section).and_then(|s| s.get(facade_name)) {
                validate_facade_dep_entry(
                    md_path,
                    workspace_root,
                    fence_start,
                    facade_name,
                    dep_val,
                    model,
                    violations,
                );
            }
        }
    }
}

fn validate_facade_dep_entry(
    md_path: &Path,
    workspace_root: &Path,
    fence_start: usize,
    facade_name: &str,
    dep_val: &toml::Value,
    model: &WorkspaceManifestModel,
    violations: &mut Vec<String>,
) {
    let features: Vec<String> = dep_val
        .as_table()
        .and_then(|t| t.get("features"))
        .and_then(toml::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(toml::Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default();

    if features.is_empty() {
        violations.push(format!(
            "{}:{}: `{facade_name}` dependency snippet omits explicit `features = [...]` (`default = []` enables zero domains)",
            md_path.display(),
            fence_start
        ));
        return;
    }

    let resolve_pkg = if model.packages_by_name.contains_key(facade_name) {
        facade_name
    } else if let Some(primary) = model.primary_facade_pkg() {
        primary.name.as_str()
    } else {
        facade_name
    };
    if let Err(err) = model.resolve_enabled_features(resolve_pkg, &features, false, false) {
        violations.push(format!("{}:{}: {err}", md_path.display(), fence_start));
        return;
    }

    if let Some(expected_dom) = expected_domain_for_guide(md_path, workspace_root, model)
        && !features.contains(&expected_dom)
    {
        violations.push(format!(
            "{}:{}: `{facade_name}` dependency snippet in `{expected_dom}` guide must enable feature `\"{expected_dom}\"` (found {:?})",
            md_path.display(),
            fence_start,
            features
        ));
    }
}
