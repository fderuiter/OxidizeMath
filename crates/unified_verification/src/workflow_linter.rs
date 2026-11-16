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
    let jobs = yaml
        .get("jobs")
        .ok_or_else(|| "Workflow is missing a 'jobs' block.".to_string())?;

    let jobs_map = jobs
        .as_object()
        .ok_or_else(|| "Workflow 'jobs' block must be a mapping.".to_string())?;

    if jobs_map.is_empty() {
        return Err("Workflow 'jobs' block cannot be empty.".to_string());
    }

    for (job_name, job_val) in jobs_map {
        let job_obj = job_val
            .as_object()
            .ok_or_else(|| format!("Job '{}' must be a mapping.", job_name))?;

        let timeout = job_obj
            .get("timeout-minutes")
            .ok_or_else(|| format!("Job '{}' is missing a 'timeout-minutes' setting.", job_name))?;

        let is_positive_integer = match timeout {
            Value::Number(n) => {
                if let Some(val) = n.as_i64() {
                    val > 0
                } else if let Some(val) = n.as_u64() {
                    val > 0
                } else {
                    false
                }
            }
            _ => false,
        };

        if !is_positive_integer {
            return Err(format!(
                "Job '{}' has an invalid 'timeout-minutes' value (must be a positive integer).",
                job_name
            ));
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
            }
        }
    }

    if all_passed {
        println!(
            "All workflows passed linting (read-only permissions and concurrency controls enforced)."
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

    fn validate_timeout_str(content: &str) -> Result<(), String> {
        let yaml: Value =
            serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
        validate_job_timeouts(&yaml)
    }

    #[test]
    fn test_valid_job_timeouts() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
  deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 20
"#;
        assert!(validate_timeout_str(yaml).is_ok());
    }

    #[test]
    fn test_missing_job_timeouts() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
"#;
        assert_eq!(
            validate_timeout_str(yaml),
            Err("Job 'test' is missing a 'timeout-minutes' setting.".to_string())
        );
    }

    #[test]
    fn test_non_numeric_timeout_string() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: "15"
"#;
        assert_eq!(
            validate_timeout_str(yaml),
            Err(
                "Job 'test' has an invalid 'timeout-minutes' value (must be a positive integer)."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_non_numeric_timeout_bool() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: true
"#;
        assert_eq!(
            validate_timeout_str(yaml),
            Err(
                "Job 'test' has an invalid 'timeout-minutes' value (must be a positive integer)."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_negative_timeout() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: -10
"#;
        assert_eq!(
            validate_timeout_str(yaml),
            Err(
                "Job 'test' has an invalid 'timeout-minutes' value (must be a positive integer)."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_zero_timeout() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 0
"#;
        assert_eq!(
            validate_timeout_str(yaml),
            Err(
                "Job 'test' has an invalid 'timeout-minutes' value (must be a positive integer)."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_float_timeout() {
        let yaml = r#"
name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15.5
"#;
        assert_eq!(
            validate_timeout_str(yaml),
            Err(
                "Job 'test' has an invalid 'timeout-minutes' value (must be a positive integer)."
                    .to_string()
            )
        );
    }

    #[test]
    fn test_missing_jobs_block_for_timeouts() {
        let yaml = r#"
name: CI
concurrency: ci-run
"#;
        assert_eq!(
            validate_timeout_str(yaml),
            Err("Workflow is missing a 'jobs' block.".to_string())
        );
    }
}
