use super::docker::{
    canonicalize_within, extract_safe_local_rel_path, is_pinned_docker_uri,
    validate_action_runs_image,
};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub(super) struct PinningState<'a> {
    active_stack: HashSet<PathBuf>,
    validated: HashSet<PathBuf>,
    exceptions: &'a [super::JobPermissionException],
}

impl<'a> PinningState<'a> {
    pub(super) fn with_exceptions(exceptions: &'a [super::JobPermissionException]) -> Self {
        Self {
            exceptions,
            ..Self::default()
        }
    }
}

fn is_forty_char_hex(s: &str) -> bool {
    s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn has_valid_remote_segments(path: &str, min_segments: usize) -> bool {
    if path.is_empty()
        || path.starts_with('/')
        || path.ends_with('/')
        || path.contains('\\')
        || path.chars().any(char::is_whitespace)
    {
        return false;
    }
    let segments: Vec<&str> = path.split('/').collect();
    segments.len() >= min_segments
        && segments
            .iter()
            .all(|seg| !seg.is_empty() && *seg != "." && *seg != "..")
}

fn is_yaml_filename(path: &str) -> bool {
    path.ends_with(".yml") || path.ends_with(".yaml")
}

fn is_pinned_external_action(action_ref: &str) -> bool {
    let Some((action_path, ref_part)) = action_ref.trim().rsplit_once('@') else {
        return false;
    };
    has_valid_remote_segments(action_path.trim(), 2) && is_forty_char_hex(ref_part.trim())
}

fn is_pinned_external_reusable_workflow(workflow_ref: &str) -> bool {
    let Some((wf_path, ref_part)) = workflow_ref.trim().rsplit_once('@') else {
        return false;
    };
    let wf_trimmed = wf_path.trim();
    has_valid_remote_segments(wf_trimmed, 3)
        && is_yaml_filename(wf_trimmed)
        && is_forty_char_hex(ref_part.trim())
}

fn read_and_validate_reusable_workflow(
    canonical_wf: &Path,
    uses_str: &str,
    repo_root: &Path,
    state: &mut PinningState<'_>,
) -> Result<(), String> {
    let content = fs::read_to_string(canonical_wf).map_err(|e| {
        format!(
            "Failed to read local reusable workflow '{}': {}",
            uses_str, e
        )
    })?;
    let yaml: Value = serde_saphyr::from_str(&content).map_err(|e| {
        format!(
            "Invalid YAML in local reusable workflow '{}': {}",
            uses_str, e
        )
    })?;
    let wrap = |e: String| format!("Local reusable workflow '{}': {}", uses_str, e);
    super::validate_permissions_with_exceptions(&yaml, Some(canonical_wf), state.exceptions)
        .map_err(wrap)?;
    super::validate_concurrency_block(&yaml).map_err(|e| wrap(e.to_string()))?;
    super::validate_job_timeouts(&yaml).map_err(wrap)?;
    super::validate_run_script_expressions(&yaml).map_err(wrap)?;
    validate_workflow_jobs(&yaml, repo_root, state).map_err(wrap)
}

fn validate_local_reusable_workflow(
    uses_str: &str,
    job_name: &str,
    repo_root: &Path,
    state: &mut PinningState<'_>,
) -> Result<(), String> {
    let rel_path = extract_safe_local_rel_path(uses_str, false)
        .map_err(|e| format!("Job '{}' invalid local reusable workflow: {}", job_name, e))?;
    let parts: Vec<&str> = rel_path.iter().filter_map(|os| os.to_str()).collect();
    if parts.len() < 3
        || parts[0] != ".github"
        || parts[1] != "workflows"
        || !is_yaml_filename(parts[parts.len() - 1])
    {
        return Err(format!(
            "Reusable workflow '{}' in job '{}' must resolve to a .yml/.yaml file inside '.github/workflows'.",
            uses_str, job_name
        ));
    }
    let full_path = repo_root.join(&rel_path);
    if !full_path.is_file() {
        return Err(format!(
            "Local reusable workflow '{}' in job '{}' does not exist on disk.",
            uses_str, job_name
        ));
    }
    canonicalize_within(&full_path, repo_root, uses_str, "repository root")
        .map_err(|e| format!("Job '{}': {}", job_name, e))?;
    let workflows_root = repo_root.join(".github").join("workflows");
    let canonical_wf =
        canonicalize_within(&full_path, &workflows_root, uses_str, ".github/workflows")
            .map_err(|e| format!("Job '{}': {}", job_name, e))?;

    if state.validated.contains(&canonical_wf) {
        return Ok(());
    }
    if !state.active_stack.insert(canonical_wf.clone()) {
        return Err(format!(
            "Cyclic local reusable workflow reference detected at '{}' in job '{}'.",
            uses_str, job_name
        ));
    }
    let res = read_and_validate_reusable_workflow(&canonical_wf, uses_str, repo_root, state);
    state.active_stack.remove(&canonical_wf);
    if res.is_ok() {
        state.validated.insert(canonical_wf);
    }
    res
}

fn validate_job_uses(
    uses_val: &Value,
    job_name: &str,
    repo_root: &Path,
    state: &mut PinningState<'_>,
) -> Result<(), String> {
    let Some(uses_str) = uses_val.as_str() else {
        return Err(format!(
            "Job '{}' contains an invalid 'uses' attribute.",
            job_name
        ));
    };
    let trimmed = uses_str.trim();
    if trimmed.starts_with("./") {
        return validate_local_reusable_workflow(trimmed, job_name, repo_root, state);
    }
    if trimmed.starts_with("..") || trimmed.contains("/../") || trimmed.ends_with("/..") {
        return Err(format!(
            "Reusable workflow '{}' in job '{}' must not contain '..' path traversal segments.",
            uses_str, job_name
        ));
    }
    if !is_pinned_external_reusable_workflow(trimmed) {
        return Err(format!(
            "Reusable workflow '{}' in job '{}' must specify a non-empty workflow path and be pinned to a 40-character commit hash.",
            uses_str, job_name
        ));
    }
    Ok(())
}

fn resolve_local_action_manifest(
    uses_str: &str,
    owner_desc: &str,
    repo_root: &Path,
) -> Result<PathBuf, String> {
    let rel_path = extract_safe_local_rel_path(uses_str, true)
        .map_err(|e| format!("Invalid local action in {}: {}", owner_desc, e))?;
    let action_dir = repo_root.join(&rel_path);
    if !action_dir.is_dir() {
        return Err(format!(
            "Local action '{}' in {} does not exist on disk (missing action.yml or action.yaml).",
            uses_str, owner_desc
        ));
    }
    canonicalize_within(&action_dir, repo_root, uses_str, "repository root")
        .map_err(|e| format!("Invalid local action in {}: {}", owner_desc, e))?;

    let yml_candidate = action_dir.join("action.yml");
    let yaml_candidate = action_dir.join("action.yaml");
    let manifest_path = if yml_candidate.is_file() {
        yml_candidate
    } else if yaml_candidate.is_file() {
        yaml_candidate
    } else {
        return Err(format!(
            "Local action '{}' in {} is missing an action.yml or action.yaml manifest.",
            uses_str, owner_desc
        ));
    };

    canonicalize_within(&manifest_path, repo_root, uses_str, "repository root")
        .map_err(|e| format!("Invalid local action in {}: {}", owner_desc, e))
}

fn validate_step_uses(
    step: &Value,
    owner_desc: &str,
    invalid_attr_msg: &str,
    repo_root: &Path,
    state: &mut PinningState<'_>,
) -> Result<(), String> {
    let Some(uses_val) = step.get("uses") else {
        return Ok(());
    };
    let Some(uses_str) = uses_val.as_str() else {
        return Err(invalid_attr_msg.to_string());
    };
    let trimmed = uses_str.trim();
    if trimmed.starts_with("./") {
        let canonical_manifest = resolve_local_action_manifest(trimmed, owner_desc, repo_root)?;
        return validate_action_manifest_with_state(&canonical_manifest, repo_root, state);
    }
    if trimmed.starts_with("docker://") {
        if !is_pinned_docker_uri(trimmed) {
            return Err(format!(
                "Docker action '{}' in {} must be pinned to a '@sha256:<64-hex>' digest.",
                uses_str, owner_desc
            ));
        }
        return Ok(());
    }
    if trimmed.starts_with("..") || trimmed.contains("/../") || trimmed.ends_with("/..") {
        return Err(format!(
            "Action '{}' in {} must not contain '..' path traversal segments.",
            uses_str, owner_desc
        ));
    }
    if !is_pinned_external_action(trimmed) {
        return Err(format!(
            "Action '{}' in {} must be pinned to a 40-character commit hash.",
            uses_str, owner_desc
        ));
    }
    Ok(())
}

fn validate_action_manifest_body(
    yaml: &Value,
    canonical_manifest: &Path,
    manifest_label: &str,
    repo_root: &Path,
    state: &mut PinningState<'_>,
) -> Result<(), String> {
    let Some(runs_map) = yaml.get("runs").and_then(Value::as_object) else {
        return Err(format!(
            "Action manifest '{}' is missing a valid 'runs' mapping.",
            manifest_label
        ));
    };
    let action_dir = canonical_manifest.parent().unwrap_or(repo_root);
    validate_action_runs_image(runs_map, action_dir, repo_root, manifest_label)?;

    let is_composite = runs_map.get("using").and_then(Value::as_str) == Some("composite");
    if is_composite && !runs_map.get("steps").is_some_and(Value::is_array) {
        return Err(format!(
            "Composite action manifest '{}' must define a 'runs.steps' sequence.",
            manifest_label
        ));
    }
    if let Some(steps_val) = runs_map.get("steps") {
        let Some(steps) = steps_val.as_array() else {
            return Err(format!(
                "Action manifest '{}' 'runs.steps' must be a sequence.",
                manifest_label
            ));
        };
        let owner_desc = format!("action '{}'", manifest_label);
        let invalid_attr_msg = format!(
            "Action '{}' contains a step with an invalid 'uses' attribute.",
            manifest_label
        );
        for step in steps {
            validate_step_uses(step, &owner_desc, &invalid_attr_msg, repo_root, state)?;
        }
        super::expressions::validate_steps_run_expressions(
            steps,
            &super::expressions::SafeEnvScope::default(),
            &format!("Action '{}'", manifest_label),
        )?;
    }
    Ok(())
}

pub(super) fn validate_action_manifest_with_state(
    manifest_path: &Path,
    repo_root: &Path,
    state: &mut PinningState<'_>,
) -> Result<(), String> {
    let label = manifest_path.to_string_lossy().to_string();
    let canonical_manifest =
        canonicalize_within(manifest_path, repo_root, &label, "repository root")?;

    if state.validated.contains(&canonical_manifest) {
        return Ok(());
    }
    if !state.active_stack.insert(canonical_manifest.clone()) {
        return Err(format!(
            "Cyclic local action reference detected at '{}'.",
            label
        ));
    }

    let res = (|| -> Result<(), String> {
        let content = fs::read_to_string(&canonical_manifest)
            .map_err(|e| format!("Failed to read action manifest '{}': {}", label, e))?;
        let yaml: Value = serde_saphyr::from_str(&content)
            .map_err(|e| format!("Invalid YAML syntax in '{}': {}", label, e))?;
        validate_action_manifest_body(&yaml, &canonical_manifest, &label, repo_root, state)
    })();

    state.active_stack.remove(&canonical_manifest);
    if res.is_ok() {
        state.validated.insert(canonical_manifest);
    }
    res
}

pub(super) fn validate_workflow_jobs(
    yaml: &Value,
    repo_root: &Path,
    state: &mut PinningState<'_>,
) -> Result<(), String> {
    let jobs = match yaml.get("jobs") {
        Some(Value::Object(map)) => map,
        Some(_) => return Err("Workflow 'jobs' block must be a mapping.".to_string()),
        None => return Ok(()),
    };

    for (job_name, job_val) in jobs {
        let Some(job_map) = job_val.as_object() else {
            return Err(format!("Job '{}' must be a mapping.", job_name));
        };
        if let Some(job_uses) = job_map.get("uses") {
            validate_job_uses(job_uses, job_name, repo_root, state)?;
        }
        let Some(steps_val) = job_map.get("steps") else {
            continue;
        };
        let Some(steps) = steps_val.as_array() else {
            return Err(format!(
                "Job '{}' 'steps' block must be a sequence.",
                job_name
            ));
        };
        let owner_desc = format!("job '{}'", job_name);
        let invalid_attr_msg = format!(
            "Job '{}' contains a step with an invalid 'uses' attribute.",
            job_name
        );
        for step in steps {
            validate_step_uses(step, &owner_desc, &invalid_attr_msg, repo_root, state)?;
        }
    }

    Ok(())
}
