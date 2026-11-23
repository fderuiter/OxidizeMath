//! Static validation of consumer `[dependencies]` TOML blocks referencing `math_explorer`.

use super::manifest_model::WorkspaceManifestModel;
use std::path::Path;

fn expected_domain_for_guide(md_path: &Path, workspace_root: &Path) -> Option<String> {
    let rel = md_path.strip_prefix(workspace_root).ok()?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    if rel_str.starts_with("crates/pure_math/") {
        return Some("pure_math".to_string());
    }
    let rest = rel_str.strip_prefix("crates/domain_")?;
    let domain = rest.split('/').next()?;
    (!domain.is_empty()).then(|| domain.to_string())
}

/// Validates consumer `[dependencies]` TOML snippets referencing `math_explorer`.
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
    if !raw_toml.contains("math_explorer") {
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
        if let Some(dep_val) = doc.get(section).and_then(|s| s.get("math_explorer")) {
            validate_math_explorer_dep_entry(
                md_path,
                workspace_root,
                fence_start,
                dep_val,
                model,
                violations,
            );
        }
    }
}

fn validate_math_explorer_dep_entry(
    md_path: &Path,
    workspace_root: &Path,
    fence_start: usize,
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
            "{}:{}: `math_explorer` dependency snippet omits explicit `features = [...]` (`default = []` enables zero domains)",
            md_path.display(),
            fence_start
        ));
        return;
    }

    let _enabled = match model.resolve_enabled_features("math_explorer", &features, false, false) {
        Ok(set) => set,
        Err(err) => {
            violations.push(format!("{}:{}: {err}", md_path.display(), fence_start));
            return;
        }
    };

    if let Some(expected_dom) = expected_domain_for_guide(md_path, workspace_root)
        && !features.contains(&expected_dom)
    {
        violations.push(format!(
            "{}:{}: `math_explorer` dependency snippet in `{expected_dom}` guide must enable feature `\"{expected_dom}\"` (found {:?})",
            md_path.display(),
            fence_start,
            features
        ));
    }
}
