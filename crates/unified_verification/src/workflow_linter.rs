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
}
