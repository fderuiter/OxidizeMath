use super::*;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

static WORKFLOW_TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn validate_str(content: &str) -> Result<(), String> {
    let yaml: Value =
        serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
    validate_permissions_block(&yaml).map_err(|e| e.to_string())
}

fn validate_str_with_exceptions(
    content: &str,
    workflow_path: Option<&Path>,
    exceptions: &[JobPermissionException],
) -> Result<(), String> {
    let yaml: Value =
        serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
    validate_permissions_with_exceptions(&yaml, workflow_path, exceptions)
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

#[test]
fn test_valid_job_permissions_read_only_variants() {
    let yaml = r#"
name: CI
permissions: read-all
jobs:
  inherit_root:
    runs-on: ubuntu-latest
  explicit_read_all:
    runs-on: ubuntu-latest
    permissions: read-all
  explicit_none:
    runs-on: ubuntu-latest
    permissions: none
  explicit_empty_map:
    runs-on: ubuntu-latest
    permissions: {}
  explicit_scoped_read:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: none
      issues: read
"#;
    assert_eq!(validate_str(yaml), Ok(()));
}

#[test]
fn test_invalid_job_permissions_write_all() {
    let yaml = r#"
name: Test Workflow
permissions: read-all
concurrency:
  group: ci-test
jobs:
  job_with_write_all:
    timeout-minutes: 10
    permissions: write-all
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
"#;
    let err = validate_str(yaml).expect_err("job write-all should fail");
    assert!(err.contains("job_with_write_all"), "error was: {}", err);
    assert!(err.contains("write-all"), "error was: {}", err);
    assert_eq!(
        err,
        "Job 'job_with_write_all' has invalid permissions 'write-all': job-level permissions must be read-only ('read-all' or 'none')."
    );
}

#[test]
fn test_invalid_job_permissions_contents_write() {
    let yaml = r#"
name: CI
permissions: read-all
jobs:
  safe_build:
    runs-on: ubuntu-latest
  escalated_publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
"#;
    let err = validate_str(yaml).expect_err("job packages: write should fail");
    assert!(err.contains("escalated_publish"), "error was: {}", err);
    assert!(err.contains("packages: write"), "error was: {}", err);
    assert_eq!(
        err,
        "Job 'escalated_publish' has invalid permission 'packages: write': job-level permissions must be 'read' or 'none'."
    );
}

#[test]
fn test_invalid_job_permissions_malformed_blocks() {
    let non_map_or_str = r#"
name: CI
permissions: read-all
jobs:
  bad_block:
    runs-on: ubuntu-latest
    permissions:
      - read
"#;
    let err1 = validate_str(non_map_or_str).expect_err("list permissions must fail");
    assert!(err1.contains("bad_block"), "error was: {}", err1);

    let non_str_scope = r#"
name: CI
permissions: read-all
jobs:
  bad_scope_val:
    runs-on: ubuntu-latest
    permissions:
      contents: 123
"#;
    let err2 = validate_str(non_str_scope).expect_err("numeric permission value must fail");
    assert!(err2.contains("bad_scope_val"), "error was: {}", err2);
    assert!(err2.contains("contents"), "error was: {}", err2);
}

fn pages_deploy_exceptions() -> Vec<JobPermissionException> {
    vec![JobPermissionException {
        workflow: ".github/workflows/pages.yml".to_string(),
        job: "deploy".to_string(),
        allow_write: vec!["pages".to_string(), "id-token".to_string()],
        reason: Some("GitHub Pages deployment".to_string()),
    }]
}

#[test]
fn test_job_permission_exception_scoped_enforcement() {
    let exceptions = pages_deploy_exceptions();
    let compliant_deploy = r#"
name: Deploy
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
  deploy:
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
"#;
    assert!(
        validate_str_with_exceptions(
            compliant_deploy,
            Some(Path::new(".github/workflows/pages.yml")),
            &exceptions,
        )
        .is_ok()
    );

    // Cross-platform backslash and ./ path normalization
    assert!(
        validate_str_with_exceptions(
            compliant_deploy,
            Some(Path::new(".\\.github\\workflows\\pages.yml")),
            &exceptions,
        )
        .is_ok()
    );

    // Same job in a non-exempted workflow must fail
    let wrong_workflow_err = validate_str_with_exceptions(
        compliant_deploy,
        Some(Path::new(".github/workflows/bypass.yml")),
        &exceptions,
    )
    .expect_err("non-exempted workflow must fail");
    assert!(wrong_workflow_err.contains("deploy"));
}

#[test]
fn test_job_permission_exception_rejects_unexempted_jobs_and_scopes() {
    let exceptions = pages_deploy_exceptions();
    let wrong_job = r#"
name: Deploy
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      pages: write
"#;
    let wrong_job_err = validate_str_with_exceptions(
        wrong_job,
        Some(Path::new(".github/workflows/pages.yml")),
        &exceptions,
    )
    .expect_err("unexempted job must fail");
    assert!(wrong_job_err.contains("build"));
    assert!(wrong_job_err.contains("pages: write"));

    let extra_scope = r#"
name: Deploy
permissions: read-all
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      pages: write
      contents: write
"#;
    let extra_scope_err = validate_str_with_exceptions(
        extra_scope,
        Some(Path::new(".github/workflows/pages.yml")),
        &exceptions,
    )
    .expect_err("unlisted write scope must fail");
    assert!(extra_scope_err.contains("deploy"));
    assert!(extra_scope_err.contains("contents: write"));

    let write_all_on_exempt = r#"
name: Deploy
permissions: read-all
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions: write-all
"#;
    assert!(
        validate_str_with_exceptions(
            write_all_on_exempt,
            Some(Path::new(".github/workflows/pages.yml")),
            &exceptions,
        )
        .is_err()
    );
}

#[test]
fn test_lint_workflows_in_dir_and_repo_whitelist() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let repo_workflows = crate::manifest_cache::resolve_workspace_path(".github/workflows");
    let whitelist = find_whitelist_path().expect("verification_whitelist.toml should exist");
    assert!(
        lint_workflows_in_dir(&repo_workflows, Some(&whitelist)),
        "Repository workflows should pass with verification_whitelist.toml"
    );

    let id = WORKFLOW_TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let temp_root = env::temp_dir().join(format!(
        "oxidize_workflow_linter_test_{}_{}",
        std::process::id(),
        id
    ));
    let temp_wf_dir = temp_root.join(".github").join("workflows");
    fs::create_dir_all(&temp_wf_dir).unwrap();

    let bypass_yml = r#"name: Test Workflow
permissions: read-all
concurrency:
  group: ci-test
jobs:
  job_with_write_all:
    timeout-minutes: 10
    permissions: write-all
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
"#;
    let bypass_path = temp_wf_dir.join("bypass.yml");
    fs::write(&bypass_path, bypass_yml).unwrap();
    assert!(
        !lint_workflows_in_dir(&temp_wf_dir, Some(&whitelist)),
        "Workflow with job-level write-all must fail lint_workflows_in_dir"
    );

    let _ = fs::remove_file(&bypass_path);
    let _ = fs::remove_dir(&temp_wf_dir);
    let _ = fs::remove_dir(temp_root.join(".github"));
    let _ = fs::remove_dir(&temp_root);
}

fn validate_run_expr_str(content: &str) -> Result<(), String> {
    let yaml: Value =
        serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
    validate_run_script_expressions(&yaml)
}

#[test]
fn test_rejects_github_expressions_in_run_scripts() {
    let ref_name_yaml = r#"
jobs:
  publish-release:
    steps:
      - name: Publish GitHub Release
        run: gh release create "${{ github.ref_name }}"
"#;
    let err1 = validate_run_expr_str(ref_name_yaml).expect_err("github.ref_name must fail");
    assert!(err1.contains("publish-release") && err1.contains("github.ref_name"));

    let head_ref_yaml = r#"
jobs:
  pr-check:
    steps:
      - name: Inspect branch
        run: echo "Branch is ${{ github.head_ref }}"
"#;
    let err2 = validate_run_expr_str(head_ref_yaml).expect_err("github.head_ref must fail");
    assert!(err2.contains("pr-check") && err2.contains("github.head_ref"));

    let event_yaml = r#"
jobs:
  audit:
    steps:
      - name: Title check
        run: echo "${{ github.event.pull_request.title }}"
"#;
    let err3 = validate_run_expr_str(event_yaml).expect_err("github.event.* must fail");
    assert!(err3.contains("audit") && err3.contains("github.event.pull_request.title"));
}

#[test]
fn test_rejects_whitespace_bracket_multiline_and_compound_github_expressions() {
    let bracket_and_ws = r#"
jobs:
  check:
    steps:
      - name: Multiline check
        run: |
          echo "Target: ${{ matrix.target }}"
          echo "Ref: ${{  github['head_ref']  }}"
"#;
    let err1 = validate_run_expr_str(bracket_and_ws).expect_err("bracket github expr must fail");
    assert!(err1.contains("Multiline check") && err1.contains("github['head_ref']"));

    let compound_expr = r#"
jobs:
  verify:
    steps:
      - run: |
          git diff "${{ github.event.before || github.sha }}"
"#;
    let err2 = validate_run_expr_str(compound_expr).expect_err("compound github expr must fail");
    assert!(err2.contains("verify") && err2.contains("github.event.before || github.sha"));

    let spacing_variation =
        "jobs:\n  ci:\n    steps:\n      - run: echo \"${{  github.ref_name }}\"\n";
    assert!(validate_run_expr_str(spacing_variation).is_err());
}

#[test]
fn test_allows_env_bindings_and_static_matrix_or_env_run_expressions() {
    let safe_workflow = r#"
name: Safe CI
permissions: read-all
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
jobs:
  test-and-release:
    runs-on: ${{ matrix.os }}
    timeout-minutes: 15
    steps:
      - name: Checkout
        uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
      - name: Matrix and env interpolations
        env:
          CARGO_PACKAGER_VERSION: "0.11.8"
          BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}
          RELEASE_TAG: ${{ github.ref_name }}
        run: |
          if [ -n "${{ matrix.feature }}" ]; then
            cargo run -p xtask -- test-features ${{ matrix.feature }}
          fi
          echo "Installing @${{ env.CARGO_PACKAGER_VERSION }} for $RELEASE_TAG ($BASE_SHA)"
          if [ "${{ matrix.target == 'github-runner' }}" = "true" ]; then
            echo "Safe string literal containing github"
          fi
"#;
    assert_eq!(validate_run_expr_str(safe_workflow), Ok(()));
}

#[test]
fn test_lint_workflows_in_dir_rejects_run_github_expression() {
    let id = WORKFLOW_TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let temp_dir = env::temp_dir().join(format!("oxidize_wf_expr_{}_{}", std::process::id(), id));
    fs::create_dir_all(&temp_dir).unwrap();
    let bad_wf = r#"name: Unsafe Release
permissions: read-all
concurrency:
  group: release
jobs:
  release:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
      - name: Unsafe tag interpolation
        run: gh release create "${{ github.ref_name }}"
"#;
    let wf_path = temp_dir.join("unsafe.yml");
    fs::write(&wf_path, bad_wf).unwrap();
    assert!(!lint_workflows_in_dir(&temp_dir, None));
    let _ = fs::remove_file(&wf_path);
    let _ = fs::remove_dir(&temp_dir);
}
