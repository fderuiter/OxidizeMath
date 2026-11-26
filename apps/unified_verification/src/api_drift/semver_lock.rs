//! Tier-aware API drift policies and SemVer signature locks for `Certified` modules.

use super::ApiItem;
use oxidize_core::traceability::{MaturityRegistry, MaturityTier};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Parses a `MAJOR.MINOR.PATCH` SemVer prefix (ignoring pre-release/build suffixes) for monotonic comparison.
pub fn parse_semver_triple(raw: &str) -> Option<(u64, u64, u64)> {
    let core = raw.trim().split(['-', '+']).next()?;
    let mut parts = core.split('.');
    let major: u64 = parts.next()?.parse().ok()?;
    let minor: u64 = parts.next()?.parse().ok()?;
    let patch: u64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// Reads `[workspace.package].version` from `<workspace_root>/Cargo.toml`.
pub fn read_workspace_package_version(workspace_root: &Path) -> Option<String> {
    let content = fs::read_to_string(workspace_root.join("Cargo.toml")).ok()?;
    let parsed: toml::Table = content.parse().ok()?;
    parsed
        .get("workspace")?
        .as_table()?
        .get("package")?
        .as_table()?
        .get("version")?
        .as_str()
        .map(str::to_string)
}

/// Returns `true` when `current_version` is strictly newer in SemVer than `locked_version`.
pub fn is_semver_bumped(current_version: Option<&str>, locked_version: Option<&str>) -> bool {
    let (Some(curr), Some(locked)) = (
        current_version.and_then(parse_semver_triple),
        locked_version.and_then(parse_semver_triple),
    ) else {
        return false;
    };
    curr > locked
}

/// Collects breaking signature or file removals on `Certified` modules (excluding expired deprecations).
pub fn collect_certified_breaking_removals(
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    registry: &MaturityRegistry,
    today: &str,
) -> Vec<String> {
    let mut breaking = Vec::new();
    for (path, base_items) in baseline_apis {
        if registry.tier_for_repo_path(path) != MaturityTier::Certified {
            continue;
        }
        let Some(curr_items) = current_apis.get(path) else {
            let has_non_expired = base_items.iter().any(|item| {
                !item
                    .deprecation_expires
                    .as_deref()
                    .is_some_and(|d| d <= today)
            });
            if has_non_expired {
                breaking.push(format!(
                    "SemVer Lock Violation [Certified]: Public file removed without SemVer bump: {path}"
                ));
            }
            continue;
        };
        for item in base_items {
            if curr_items.iter().any(|c| c.signature == item.signature) {
                continue;
            }
            if item
                .deprecation_expires
                .as_deref()
                .is_some_and(|d| d <= today)
            {
                continue;
            }
            breaking.push(format!(
                "SemVer Lock Violation [Certified]: Breaking signature removal in {path}: `- {}`",
                item.signature
            ));
        }
    }
    breaking
}

/// Updates `certified_baseline_version = "<new_version>"` in `<workspace_root>/traceability.toml` when present.
pub fn sync_certified_baseline_version(workspace_root: &Path, new_version: &str) {
    let path = workspace_root.join("traceability.toml");
    let Ok(content) = fs::read_to_string(&path) else {
        return;
    };
    let mut updated = Vec::new();
    let mut in_lock = false;
    let mut changed = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_lock = trimmed == "[semver_lock]";
            updated.push(line.to_string());
            continue;
        }
        if in_lock && trimmed.starts_with("certified_baseline_version") {
            let replacement = format!("certified_baseline_version = \"{new_version}\"");
            if line != replacement {
                changed = true;
            }
            updated.push(replacement);
        } else {
            updated.push(line.to_string());
        }
    }
    if changed {
        let _ = fs::write(&path, format!("{}\n", updated.join("\n")));
    }
}

/// Verifies whether `regenerate-baseline` is permitted to write `current_apis` over `baseline_apis`.
pub fn check_regenerate_baseline_semver_lock(
    workspace_root: &Path,
    current_apis: &BTreeMap<String, Vec<ApiItem>>,
    baseline_apis: &BTreeMap<String, Vec<ApiItem>>,
    today: &str,
    allow_breaking_override: bool,
) -> Result<(), Vec<String>> {
    if !workspace_root.join("traceability.toml").is_file() {
        return Ok(());
    }
    let registry = MaturityRegistry::load_or_embedded(Some(workspace_root));
    let breaking =
        collect_certified_breaking_removals(current_apis, baseline_apis, &registry, today);
    if breaking.is_empty() {
        return Ok(());
    }
    let ws_ver = read_workspace_package_version(workspace_root);
    if allow_breaking_override
        || is_semver_bumped(
            ws_ver.as_deref(),
            registry.certified_baseline_version.as_deref(),
        )
    {
        if let Some(ver) = ws_ver {
            sync_certified_baseline_version(workspace_root, &ver);
        }
        return Ok(());
    }
    let locked = registry
        .certified_baseline_version
        .unwrap_or_else(|| "unknown".to_string());
    let current_v = ws_ver.unwrap_or_else(|| "unknown".to_string());
    let mut errors = breaking;
    errors.push(format!(
        "Refusing to regenerate baseline: breaking API removal detected in `Certified` module(s) while `[workspace.package].version` is still `{current_v}` (locked at `{locked}`)."
    ));
    errors.push(
        "  Hint: Bump `[workspace.package].version` in root `Cargo.toml` (preserving lockstep workspace inheritance) + record an ADR, or pass `--allow-breaking-certified` (`OXIDIZE_ALLOW_BREAKING_CERTIFIED=1`) for an explicit override."
            .to_string(),
    );
    Err(errors)
}
