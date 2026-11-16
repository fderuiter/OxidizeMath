use serde_json::Value;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

pub fn validate_permissions_block(yaml: &Value) -> Result<(), &'static str> {
    let permissions = yaml
        .get("permissions")
        .ok_or("Workflow is missing a global 'permissions' block.")?;

    let mut is_read_only = false;
    if let Value::String(s) = permissions {
        if s == "read-all" || s == "none" {
            is_read_only = true;
        }
    } else if let Value::Object(map) = permissions {
        let mut only_read = true;
        for (_k, v) in map {
            if let Value::String(v_str) = v {
                if v_str != "read" && v_str != "none" {
                    only_read = false;
                    break;
                }
            } else {
                only_read = false;
                break;
            }
        }
        if only_read {
            is_read_only = true;
        }
    }

    if !is_read_only {
        return Err(
            "Workflow 'permissions' block must explicitly be read-only globally. (e.g., 'permissions: read-all' or specific 'read' permissions)",
        );
    }

    Ok(())
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

pub fn lint_workflows() -> bool {
    let workflows_dir = Path::new(".github/workflows");
    if !workflows_dir.exists() {
        println!("No .github/workflows directory found. Skipping workflow linting.");
        return true;
    }

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

                if let Err(msg) = validate_permissions_block(&yaml) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_str(content: &str) -> Result<(), String> {
        let yaml: Value =
            serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
        validate_permissions_block(&yaml).map_err(|e| e.to_string())
    }

    #[test]
    fn test_valid_read_all() {
        let yaml = r#"
name: CI
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
"#;
        assert!(validate_str(yaml).is_ok());
    }

    #[test]
    fn test_valid_none() {
        let yaml = r#"
name: CI
permissions: none
"#;
        assert!(validate_str(yaml).is_ok());
    }

    #[test]
    fn test_valid_mapping_permissions() {
        let yaml = r#"
name: CI
permissions:
  contents: read
  issues: read
  pull-requests: none
"#;
        assert!(validate_str(yaml).is_ok());
    }

    #[test]
    fn test_missing_permissions() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
"#;
        assert_eq!(
            validate_str(yaml),
            Err("Workflow is missing a global 'permissions' block.".to_string())
        );
    }

    #[test]
    fn test_invalid_write_permission() {
        let yaml = r#"
name: CI
permissions:
  contents: write
"#;
        assert_eq!(
            validate_str(yaml),
            Err("Workflow 'permissions' block must explicitly be read-only globally. (e.g., 'permissions: read-all' or specific 'read' permissions)".to_string())
        );
    }

    #[test]
    fn test_invalid_yaml_syntax() {
        let yaml = r#"
name: CI
permissions: [unclosed list
"#;
        assert!(validate_str(yaml).is_err());
    }

    fn validate_concurrency_str(content: &str) -> Result<(), String> {
        let yaml: Value =
            serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
        validate_concurrency_block(&yaml).map_err(|e| e.to_string())
    }

    #[test]
    fn test_valid_concurrency_mapping() {
        let yaml = r#"
name: CI
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
"#;
        assert!(validate_concurrency_str(yaml).is_ok());
    }

    #[test]
    fn test_valid_concurrency_string() {
        let yaml = r#"
name: CI
concurrency: ci-run
"#;
        assert!(validate_concurrency_str(yaml).is_ok());
    }

    #[test]
    fn test_missing_concurrency() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
"#;
        assert_eq!(
            validate_concurrency_str(yaml),
            Err("Workflow is missing a global 'concurrency' block.".to_string())
        );
    }

    #[test]
    fn test_invalid_concurrency_missing_group() {
        let yaml = r#"
name: CI
concurrency:
  cancel-in-progress: true
"#;
        assert_eq!(
            validate_concurrency_str(yaml),
            Err("Workflow 'concurrency' block must specify a 'group'.".to_string())
        );
    }

    #[test]
    fn test_invalid_concurrency_empty_group() {
        let yaml = r#"
name: CI
concurrency:
  group: "   "
"#;
        assert_eq!(
            validate_concurrency_str(yaml),
            Err("Workflow 'concurrency' group cannot be empty.".to_string())
        );
    }

    #[test]
    fn test_invalid_concurrency_empty_string() {
        let yaml = r#"
name: CI
concurrency: "   "
"#;
        assert_eq!(
            validate_concurrency_str(yaml),
            Err("Workflow 'concurrency' block cannot be empty.".to_string())
        );
    }

    fn validate_timeouts_str(content: &str) -> Result<(), String> {
        let yaml: Value =
            serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
        validate_job_timeouts(&yaml)
    }

    fn validate_action_str(content: &str) -> Result<(), String> {
        let yaml: Value =
            serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
        validate_action_pinning(&yaml)
    }

    #[test]
    fn test_valid_job_timeouts() {
        let yaml = "jobs:\n  build:\n    timeout-minutes: 15\n  test:\n    timeout-minutes: 30\n";
        assert!(validate_timeouts_str(yaml).is_ok());
    }

    #[test]
    fn test_missing_job_timeout() {
        let yaml = "jobs:\n  deploy:\n    runs-on: ubuntu-latest\n";
        assert_eq!(
            validate_timeouts_str(yaml),
            Err("Job 'deploy' is missing a 'timeout-minutes' specification.".to_string())
        );
    }

    #[test]
    fn test_invalid_job_timeout_zero() {
        let yaml = "jobs:\n  test:\n    timeout-minutes: 0\n";
        assert_eq!(
            validate_timeouts_str(yaml),
            Err("Job 'test' timeout-minutes must be a positive integer.".to_string())
        );
    }

    #[test]
    fn test_invalid_job_timeout_string() {
        let yaml = "jobs:\n  test:\n    timeout-minutes: \"15\"\n";
        assert_eq!(
            validate_timeouts_str(yaml),
            Err("Job 'test' timeout-minutes must be a positive integer.".to_string())
        );
    }

    #[test]
    fn test_invalid_job_timeout_negative() {
        let yaml = "jobs:\n  test:\n    timeout-minutes: -10\n";
        assert_eq!(
            validate_timeouts_str(yaml),
            Err("Job 'test' timeout-minutes must be a positive integer.".to_string())
        );
    }

    #[test]
    fn test_valid_action_pinning_commit_hash() {
        let yaml = "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683\n";
        assert!(validate_action_str(yaml).is_ok());
    }

    #[test]
    fn test_valid_action_pinning_local_action() {
        let yaml = "jobs:\n  test:\n    steps:\n      - uses: ./.github/actions/local-test\n";
        assert!(validate_action_str(yaml).is_ok());
    }

    #[test]
    fn test_invalid_action_pinning_tag() {
        let yaml = "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@v4\n";
        assert_eq!(
            validate_action_str(yaml),
            Err("Action 'actions/checkout@v4' in job 'test' must be pinned to a 40-character commit hash.".to_string())
        );
    }

    #[test]
    fn test_invalid_action_pinning_branch() {
        let yaml = "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@main\n";
        assert_eq!(
            validate_action_str(yaml),
            Err("Action 'actions/checkout@main' in job 'test' must be pinned to a 40-character commit hash.".to_string())
        );
    }

    #[test]
    fn test_invalid_action_pinning_missing_ref() {
        let yaml = "jobs:\n  test:\n    steps:\n      - uses: actions/checkout\n";
        assert_eq!(
            validate_action_str(yaml),
            Err("Action 'actions/checkout' in job 'test' must be pinned to a 40-character commit hash.".to_string())
        );
    }

    #[test]
    fn test_invalid_action_pinning_non_hex_sha() {
        let yaml = "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n";
        assert_eq!(
            validate_action_str(yaml),
            Err("Action 'actions/checkout@zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz' in job 'test' must be pinned to a 40-character commit hash.".to_string())
        );
    }
}
