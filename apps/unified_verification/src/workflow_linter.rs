use crate::whitelist::PathMatchOptions;
pub use crate::whitelist::{JobPermissionException, WorkflowLinterConfig, find_whitelist_path};
use serde_json::Value;
use std::fs;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use walkdir::WalkDir;

/// Loads configured job-level permission exceptions from `verification_whitelist.toml`.
pub fn load_workflow_exceptions(path: Option<&Path>) -> Vec<JobPermissionException> {
    let resolved = path.map(Path::to_path_buf).or_else(find_whitelist_path);
    let cfg: WorkflowLinterConfig =
        crate::whitelist::load_whitelist_or_default(resolved.as_deref(), &[] as &[&Path])
            .workflow_linter
            .unwrap_or_default();
    cfg.exceptions
}

fn workflow_path_matches(workflow_path: &Path, configured_workflow: &str) -> bool {
    crate::whitelist::path_matches_lossy(
        workflow_path,
        configured_workflow,
        PathMatchOptions {
            trim_dot_slash: true,
            require_non_empty: true,
            match_file_name: false,
        },
    )
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

/// Validates root and optional job-level `concurrency` blocks, enforcing run-unique groups and
/// non-canceling rules across `on.push.branches` triggers.
pub fn validate_concurrency_block(yaml: &Value) -> Result<(), String> {
    concurrency::validate_concurrency_block(yaml)
}

/// Validates job-level `timeout-minutes` settings, exempting `jobs.<id>.uses` reusable workflow
/// callers and rejecting schema-invalid `timeout-minutes` on `uses` jobs.
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

        if job_map.contains_key("uses") {
            if job_map.contains_key("timeout-minutes") {
                return Err(format!(
                    "Job '{}' calls a reusable workflow via 'uses' and must not specify 'timeout-minutes'.",
                    job_name
                ));
            }
            continue;
        }

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

/// Validates job-level reusable workflow `uses` and step-level action `uses` relative to
/// `repo_root`, forwarding `exceptions` when validating local reusable workflows.
pub fn validate_action_pinning_with_exceptions(
    yaml: &Value,
    repo_root: &Path,
    exceptions: &[JobPermissionException],
) -> Result<(), String> {
    let mut state = pinning::PinningState::with_exceptions(exceptions);
    pinning::validate_workflow_jobs(yaml, repo_root, &mut state)
}

/// Validates job-level reusable workflow `uses` and step-level action `uses` relative to `repo_root`.
pub fn validate_action_pinning_in_dir(yaml: &Value, repo_root: &Path) -> Result<(), String> {
    validate_action_pinning_with_exceptions(yaml, repo_root, &[])
}

/// Validates job-level reusable workflow `uses` and step-level action `uses` relative to `.`.
pub fn validate_action_pinning(yaml: &Value) -> Result<(), String> {
    validate_action_pinning_in_dir(yaml, Path::new("."))
}

/// Validates that workflow `run` scripts do not interpolate unsafe `${{ github.* }}`,
/// `${{ inputs.* }}`, or non-static/unmapped `${{ env.* }}` expressions.
///
/// Untrusted or dynamic context values must be bound through `env:` blocks and read as shell
/// variables rather than interpolated directly into `run` scripts; only static scalar `env:`
/// literals without `${{` may be referenced via `${{ env.<KEY> }}` or `${{ env['<KEY>'] }}`.
pub fn validate_run_script_expressions(yaml: &Value) -> Result<(), String> {
    let jobs = match yaml.get("jobs") {
        Some(Value::Object(map)) => map,
        Some(_) => return Err("Workflow 'jobs' block must be a mapping.".to_string()),
        None => return Ok(()),
    };
    let root_env = expressions::SafeEnvScope::from_opt_value(yaml.get("env"), "Workflow")?;

    for (job_name, job_val) in jobs {
        let job_map = match job_val {
            Value::Object(map) => map,
            _ => return Err(format!("Job '{}' must be a mapping.", job_name)),
        };
        let owner = format!("Job '{}'", job_name);
        let job_env = root_env.merged_with(job_map.get("env"), &owner)?;
        let Some(steps_val) = job_map.get("steps") else {
            continue;
        };
        let Some(steps) = steps_val.as_array() else {
            return Err(format!(
                "Job '{}' 'steps' block must be a sequence.",
                job_name
            ));
        };
        expressions::validate_steps_run_expressions(steps, &job_env, &owner)?;
    }

    Ok(())
}

fn lint_workflow_file(
    path: &Path,
    repo_root: &Path,
    exceptions: &[JobPermissionException],
) -> bool {
    let path_str = path.to_string_lossy();
    if let Err(msg) = docker::canonicalize_within(path, repo_root, &path_str, "repository root") {
        eprintln!("Error in {}: {}", path_str, msg);
        return false;
    }
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {}: {}", path_str, e);
            return false;
        }
    };

    let yaml: Value = match serde_saphyr::from_str(&content) {
        Ok(y) => y,
        Err(e) => {
            eprintln!("Invalid YAML syntax in {}: {}", path_str, e);
            return false;
        }
    };

    let mut passed = true;
    let perm_res = if exceptions.is_empty() {
        validate_permissions_block(&yaml)
    } else {
        validate_permissions_with_exceptions(&yaml, Some(path), exceptions)
    };
    let pin_res = match (exceptions.is_empty(), repo_root == Path::new(".")) {
        (true, true) => validate_action_pinning(&yaml),
        (true, false) => validate_action_pinning_in_dir(&yaml, repo_root),
        (false, _) => validate_action_pinning_with_exceptions(&yaml, repo_root, exceptions),
    };
    for check in [
        perm_res,
        validate_concurrency_block(&yaml),
        validate_job_timeouts(&yaml),
        pin_res,
        validate_run_script_expressions(&yaml),
    ] {
        if let Err(msg) = check {
            eprintln!("Error in {}: {}", path_str, msg);
            passed = false;
        }
    }

    passed
}

/// Lints all `.yml`/`.yaml` workflows inside `workflows_dir` and `.github/actions` manifests using `whitelist_path`.
pub fn lint_workflows_in_dir(workflows_dir: &Path, whitelist_path: Option<&Path>) -> bool {
    let (repo_root, actions_dir) = docker::infer_repo_and_actions_dirs(workflows_dir);
    if !workflows_dir.exists() && !actions_dir.exists() {
        println!("No .github/workflows directory found. Skipping workflow linting.");
        return true;
    }

    let exceptions = load_workflow_exceptions(whitelist_path);
    let mut all_passed = true;

    if workflows_dir.exists() {
        for entry in WalkDir::new(workflows_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            let path_str = path.to_string_lossy();
            if entry.file_type().is_symlink()
                && let Err(msg) =
                    docker::canonicalize_within(path, &repo_root, &path_str, "repository root")
            {
                eprintln!("Error in {}: {}", path_str, msg);
                all_passed = false;
                continue;
            }
            if path.is_file()
                && (path_str.ends_with(".yml") || path_str.ends_with(".yaml"))
                && !lint_workflow_file(path, &repo_root, &exceptions)
            {
                all_passed = false;
            }
        }
    }

    if !docker::lint_actions_dir(&actions_dir, &repo_root) {
        all_passed = false;
    }

    if all_passed {
        println!(
            "All workflows passed linting (read-only permissions, concurrency controls, job timeouts, action pinning, and safe run expressions enforced)."
        );
    }

    all_passed
}

/// Lints all GitHub Actions workflows in `.github/workflows` and composite actions in `.github/actions`.
pub fn lint_workflows() -> bool {
    let workflows_dir = crate::manifest_cache::resolve_workspace_path(".github/workflows");
    lint_workflows_in_dir(&workflows_dir, None)
}

#[cfg(test)]
#[path = "workflow_linter/cargo_cache_tests.rs"]
mod cargo_cache_tests;
mod concurrency;
mod concurrency_eval;
#[cfg(test)]
#[path = "workflow_linter/concurrency_tests.rs"]
mod concurrency_tests;
mod docker;
mod expressions;
mod pinning;
#[cfg(test)]
#[path = "workflow_linter/policy_tests.rs"]
mod policy_tests;
#[cfg(test)]
#[path = "workflow_linter/reusable_caller_tests.rs"]
mod reusable_caller_tests;
#[cfg(test)]
mod tests;
