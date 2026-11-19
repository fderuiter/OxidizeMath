use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Deserialize, Debug, Clone, Default)]
struct WhitelistFile {
    #[serde(default)]
    workflow_linter: Option<WorkflowLinterConfig>,
}

/// Configuration for `workflow_linter` loaded from `verification_whitelist.toml`.
#[derive(Deserialize, Debug, Clone, Default)]
pub struct WorkflowLinterConfig {
    /// Explicit job-level write permission exceptions for deployment workflows.
    #[serde(default, alias = "allowed_job_permissions")]
    pub exceptions: Vec<JobPermissionException>,
}

/// Scoped exception allowing specific write permissions on a named job in a specific workflow.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct JobPermissionException {
    /// Relative path to the workflow file (e.g., `.github/workflows/pages.yml`).
    pub workflow: String,
    /// Job identifier within the `jobs` mapping (e.g., `deploy`).
    pub job: String,
    /// Permission scopes explicitly permitted to use `write` on this job (e.g., `["pages", "id-token"]`).
    #[serde(default, alias = "permissions", alias = "write_scopes")]
    pub allow_write: Vec<String>,
    /// Optional human-readable justification for the exception.
    #[serde(default)]
    pub reason: Option<String>,
}

/// Locates `verification_whitelist.toml` relative to the current working directory.
pub fn find_whitelist_path() -> Option<PathBuf> {
    let candidates = [
        Path::new("verification_whitelist.toml"),
        Path::new("../../verification_whitelist.toml"),
    ];
    for candidate in &candidates {
        if candidate.exists() {
            return Some(candidate.to_path_buf());
        }
    }
    None
}

/// Loads configured job-level permission exceptions from `verification_whitelist.toml`.
pub fn load_workflow_exceptions(path: Option<&Path>) -> Vec<JobPermissionException> {
    let resolved = match path {
        Some(p) => Some(p.to_path_buf()),
        None => find_whitelist_path(),
    };
    if let Some(toml_path) = resolved {
        if let Ok(content) = fs::read_to_string(&toml_path) {
            if let Ok(parsed) = toml::from_str::<WhitelistFile>(&content) {
                return parsed.workflow_linter.unwrap_or_default().exceptions;
            }
        }
    }
    Vec::new()
}

fn workflow_path_matches(workflow_path: &Path, configured_workflow: &str) -> bool {
    let norm_actual = workflow_path.to_string_lossy().replace('\\', "/");
    let norm_actual = norm_actual.trim_start_matches("./");
    let norm_expected = configured_workflow.replace('\\', "/");
    let norm_expected = norm_expected.trim_start_matches("./");
    !norm_expected.is_empty()
        && (norm_actual == norm_expected || norm_actual.ends_with(&format!("/{}", norm_expected)))
}

fn is_job_write_scope_exempted(
    workflow_path: Option<&Path>,
    job_name: &str,
    scope: &str,
    exceptions: &[JobPermissionException],
) -> bool {
    let Some(path) = workflow_path else {
        return false;
    };
    exceptions.iter().any(|exc| {
        let valid_reason = exc.reason.as_deref().is_none_or(|r| !r.trim().is_empty());
        valid_reason
            && exc.job == job_name
            && workflow_path_matches(path, &exc.workflow)
            && exc.allow_write.iter().any(|allowed| allowed == scope)
    })
}

fn validate_root_permissions(yaml: &Value) -> Result<(), String> {
    let permissions = yaml
        .get("permissions")
        .ok_or_else(|| "Workflow is missing a global 'permissions' block.".to_string())?;

    let mut is_read_only = false;
    if let Value::String(s) = permissions {
        if s == "read-all" || s == "none" {
            is_read_only = true;
        }
    } else if let Value::Object(map) = permissions {
        is_read_only = map
            .values()
            .all(|v| matches!(v, Value::String(s) if s == "read" || s == "none"));
    }

    if !is_read_only {
        return Err(
            "Workflow 'permissions' block must explicitly be read-only globally. (e.g., 'permissions: read-all' or specific 'read' permissions)".to_string(),
        );
    }

    Ok(())
}

fn validate_single_job_permissions(
    job_name: &str,
    permissions: &Value,
    workflow_path: Option<&Path>,
    exceptions: &[JobPermissionException],
) -> Result<(), String> {
    match permissions {
        Value::String(s) => {
            if s == "read-all" || s == "none" {
                Ok(())
            } else {
                Err(format!(
                    "Job '{}' has invalid permissions '{}': job-level permissions must be read-only ('read-all' or 'none').",
                    job_name, s
                ))
            }
        }
        Value::Object(map) => {
            for (scope, level_val) in map {
                let Some(level) = level_val.as_str() else {
                    return Err(format!(
                        "Job '{}' has invalid permission value for '{}': job-level permissions must be 'read' or 'none'.",
                        job_name, scope
                    ));
                };
                if level == "read" || level == "none" {
                    continue;
                }
                if level == "write"
                    && is_job_write_scope_exempted(workflow_path, job_name, scope, exceptions)
                {
                    continue;
                }
                return Err(format!(
                    "Job '{}' has invalid permission '{}: {}': job-level permissions must be 'read' or 'none'.",
                    job_name, scope, level
                ));
            }
            Ok(())
        }
        _ => Err(format!(
            "Job '{}' has invalid 'permissions' block: job-level permissions must be 'read-all', 'none', or a mapping of 'read'/'none' scopes.",
            job_name
        )),
    }
}

/// Validates job-level `permissions` blocks across all jobs in a workflow YAML document.
pub fn validate_job_permissions(
    yaml: &Value,
    workflow_path: Option<&Path>,
    exceptions: &[JobPermissionException],
) -> Result<(), String> {
    let jobs = match yaml.get("jobs") {
        Some(Value::Object(map)) => map,
        Some(_) => return Err("Workflow 'jobs' block must be a mapping.".to_string()),
        None => return Ok(()),
    };

    for (job_name, job_val) in jobs {
        let job_map = match job_val {
            Value::Object(map) => map,
            _ => return Err(format!("Job '{}' must be a mapping.", job_name)),
        };

        if let Some(job_permissions) = job_map.get("permissions") {
            validate_single_job_permissions(job_name, job_permissions, workflow_path, exceptions)?;
        }
    }

    Ok(())
}

/// Validates both root-level and job-level `permissions` blocks with explicit workflow exceptions.
pub fn validate_permissions_with_exceptions(
    yaml: &Value,
    workflow_path: Option<&Path>,
    exceptions: &[JobPermissionException],
) -> Result<(), String> {
    validate_root_permissions(yaml)?;
    validate_job_permissions(yaml, workflow_path, exceptions)
}

/// Validates both root-level and job-level `permissions` blocks as strictly read-only.
pub fn validate_permissions_block(yaml: &Value) -> Result<(), String> {
    validate_permissions_with_exceptions(yaml, None, &[])
}

pub fn validate_concurrency_block(yaml: &Value) -> Result<(), &'static str> {
    let concurrency = yaml
        .get("concurrency")
        .ok_or("Workflow is missing a global 'concurrency' block.")?;

    match concurrency {
        Value::String(s) => {
            if s.trim().is_empty() {
                return Err("Workflow 'concurrency' block cannot be empty.");
            }
        }
        Value::Object(map) => {
            let group = map
                .get("group")
                .ok_or("Workflow 'concurrency' block must specify a 'group'.")?;
            if let Value::String(s) = group {
                if s.trim().is_empty() {
                    return Err("Workflow 'concurrency' group cannot be empty.");
                }
            } else {
                return Err("Workflow 'concurrency' group must be a string.");
            }
        }
        _ => {
            return Err("Workflow 'concurrency' block must be a valid string or mapping.");
        }
    }

    Ok(())
}

pub fn validate_job_timeouts(yaml: &Value) -> Result<(), String> {
    let jobs = match yaml.get("jobs") {
        Some(Value::Object(map)) => map,
        Some(_) => return Err("Workflow 'jobs' block must be a mapping.".to_string()),
        None => return Err("Workflow is missing a 'jobs' block.".to_string()),
    };

    for (job_name, job_val) in jobs {
        let job_map = match job_val {
            Value::Object(map) => map,
            _ => return Err(format!("Job '{}' must be a mapping.", job_name)),
        };

        match job_map.get("timeout-minutes") {
            Some(Value::Number(num)) if num.as_u64().unwrap_or(0) > 0 => {}
            Some(_) => {
                return Err(format!(
                    "Job '{}' timeout-minutes must be a positive integer.",
                    job_name
                ));
            }
            None => {
                return Err(format!(
                    "Job '{}' is missing a 'timeout-minutes' specification.",
                    job_name
                ));
            }
        }
    }

    Ok(())
}

fn is_pinned_commit_hash(action_ref: &str) -> bool {
    let trimmed = action_ref.trim();
    if trimmed.starts_with("./") {
        return true;
    }
    match trimmed.rsplit_once('@') {
        Some((_action, ref_part)) => {
            let ref_trimmed = ref_part.trim();
            ref_trimmed.len() == 40 && ref_trimmed.chars().all(|c| c.is_ascii_hexdigit())
        }
        None => false,
    }
}

pub fn validate_action_pinning(yaml: &Value) -> Result<(), String> {
    let jobs = match yaml.get("jobs") {
        Some(Value::Object(map)) => map,
        Some(_) => return Err("Workflow 'jobs' block must be a mapping.".to_string()),
        None => return Ok(()),
    };

    for (job_name, job_val) in jobs {
        let Some(steps) = job_val.get("steps").and_then(|s| s.as_array()) else {
            continue;
        };

        for step in steps {
            if let Some(uses_val) = step.get("uses") {
                let Some(uses_str) = uses_val.as_str() else {
                    return Err(format!(
                        "Job '{}' contains a step with an invalid 'uses' attribute.",
                        job_name
                    ));
                };
                if !is_pinned_commit_hash(uses_str) {
                    return Err(format!(
                        "Action '{}' in job '{}' must be pinned to a 40-character commit hash.",
                        uses_str, job_name
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Lints all `.yml`/`.yaml` workflows inside `workflows_dir` using exceptions from `whitelist_path`.
pub fn lint_workflows_in_dir(workflows_dir: &Path, whitelist_path: Option<&Path>) -> bool {
    if !workflows_dir.exists() {
        println!("No .github/workflows directory found. Skipping workflow linting.");
        return true;
    }

    let exceptions = load_workflow_exceptions(whitelist_path);
    let mut all_passed = true;

    for entry in WalkDir::new(workflows_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.path().is_file() {
            let path_str = entry.path().to_string_lossy();
            if path_str.ends_with(".yml") || path_str.ends_with(".yaml") {
                let content = match fs::read_to_string(entry.path()) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Error reading {}: {}", path_str, e);
                        all_passed = false;
                        continue;
                    }
                };

                let yaml: Value = match serde_saphyr::from_str(&content) {
                    Ok(y) => y,
                    Err(e) => {
                        eprintln!("Invalid YAML syntax in {}: {}", path_str, e);
                        all_passed = false;
                        continue;
                    }
                };

                let perm_res = if exceptions.is_empty() {
                    validate_permissions_block(&yaml)
                } else {
                    validate_permissions_with_exceptions(&yaml, Some(entry.path()), &exceptions)
                };
                if let Err(msg) = perm_res {
                    eprintln!("Error in {}: {}", path_str, msg);
                    all_passed = false;
                }

                if let Err(msg) = validate_concurrency_block(&yaml) {
                    eprintln!("Error in {}: {}", path_str, msg);
                    all_passed = false;
                }

                if let Err(msg) = validate_job_timeouts(&yaml) {
                    eprintln!("Error in {}: {}", path_str, msg);
                    all_passed = false;
                }

                if let Err(msg) = validate_action_pinning(&yaml) {
                    eprintln!("Error in {}: {}", path_str, msg);
                    all_passed = false;
                }
            }
        }
    }

    if all_passed {
        println!(
            "All workflows passed linting (read-only permissions, concurrency controls, job timeouts, and action pinning enforced)."
        );
    }

    all_passed
}

/// Lints all GitHub Actions workflows in `.github/workflows`.
pub fn lint_workflows() -> bool {
    lint_workflows_in_dir(Path::new(".github/workflows"), None)
}

#[cfg(test)]
mod tests;
