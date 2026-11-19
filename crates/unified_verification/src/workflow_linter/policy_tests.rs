use super::*;

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
        Err(
            "Action 'actions/checkout' in job 'test' must be pinned to a 40-character commit hash."
                .to_string()
        )
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
