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

fn validate_action_in_dir_str(content: &str, repo_root: &Path) -> Result<(), String> {
    let yaml: Value =
        serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {}", e))?;
    validate_action_pinning_in_dir(&yaml, repo_root)
}

fn make_temp_workspace(tag: &str) -> PathBuf {
    crate::utils::test_utils::create_temp_test_dir(&format!("oxidize_pinning_{tag}"))
}

fn write_fixture(root: &Path, rel: &str, content: &str) {
    let full = root.join(rel);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(full, content).unwrap();
}

#[test]
fn test_valid_action_pinning_local_action() {
    let temp = make_temp_workspace("local_action");
    write_fixture(
        &temp,
        ".github/actions/local-test/action.yml",
        "name: Local Test\nruns:\n  using: composite\n  steps:\n    - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683\n",
    );
    let yaml = "jobs:\n  test:\n    steps:\n      - uses: ./.github/actions/local-test\n";
    assert_eq!(validate_action_in_dir_str(yaml, &temp), Ok(()));

    {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&temp).unwrap();
        let res = validate_action_str(yaml);
        std::env::set_current_dir(prev).unwrap();
        assert_eq!(res, Ok(()));
    }

    let missing = "jobs:\n  test:\n    steps:\n      - uses: ./.github/actions/missing-action\n";
    assert!(validate_action_in_dir_str(missing, &temp).is_err());
    let _ = fs::remove_dir_all(&temp);
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

#[test]
fn test_job_uses_external_reusable_workflow_pinning() {
    let pinned = "jobs:\n  reusable:\n    uses: example-org/canary-repo/.github/workflows/reusable-build.yml@11bd71901bbe5b1630ceea73d27597364c9af683\n";
    assert_eq!(validate_action_str(pinned), Ok(()));

    let unpinned_branch = "jobs:\n  external-reusable:\n    uses: example-org/canary-repo/.github/workflows/reusable-build.yml@main\n";
    assert!(validate_action_str(unpinned_branch).is_err());

    let unpinned_tag = "jobs:\n  external-reusable:\n    uses: example-org/canary-repo/.github/workflows/reusable-build.yml@v2\n";
    assert!(validate_action_str(unpinned_tag).is_err());

    let missing_wf_path = "jobs:\n  external-reusable:\n    uses: example-org/canary-repo@11bd71901bbe5b1630ceea73d27597364c9af683\n";
    assert!(validate_action_str(missing_wf_path).is_err());

    let empty_before_at =
        "jobs:\n  external-reusable:\n    uses: \"@11bd71901bbe5b1630ceea73d27597364c9af683\"\n";
    assert!(validate_action_str(empty_before_at).is_err());
}

#[test]
fn test_job_uses_local_reusable_workflow_resolution() {
    let temp = make_temp_workspace("local_reusable");
    write_fixture(
        &temp,
        ".github/workflows/reusable.yml",
        "permissions: read-all\nconcurrency: reusable-ci\non: workflow_call\njobs:\n  build:\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    steps:\n      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683\n",
    );
    let valid_caller = "jobs:\n  call-local:\n    uses: ./.github/workflows/reusable.yml\n";
    assert_eq!(validate_action_in_dir_str(valid_caller, &temp), Ok(()));

    let missing_caller = "jobs:\n  call-missing:\n    uses: ./.github/workflows/missing.yml\n";
    assert!(validate_action_in_dir_str(missing_caller, &temp).is_err());

    write_fixture(&temp, "scripts/outside.yml", "on: workflow_call\n");
    let wrong_dir_caller = "jobs:\n  call-outside:\n    uses: ./scripts/outside.yml\n";
    assert!(validate_action_in_dir_str(wrong_dir_caller, &temp).is_err());

    write_fixture(
        &temp,
        ".github/workflows/unpinned_called.yaml",
        "permissions: read-all\nconcurrency: reusable-ci\non: workflow_call\njobs:\n  inner:\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    steps:\n      - uses: actions/checkout@v4\n",
    );
    let unpinned_caller =
        "jobs:\n  call-unpinned:\n    uses: ./.github/workflows/unpinned_called.yaml\n";
    assert!(validate_action_in_dir_str(unpinned_caller, &temp).is_err());

    let _ = fs::remove_dir_all(&temp);
}

#[test]
fn test_nested_and_cyclic_composite_actions() {
    let temp = make_temp_workspace("nested_composite");
    write_fixture(
        &temp,
        ".github/actions/outer/action.yml",
        "name: Outer\nruns:\n  using: composite\n  steps:\n    - uses: ./.github/actions/inner\n",
    );
    write_fixture(
        &temp,
        ".github/actions/inner/action.yaml",
        "name: Inner\nruns:\n  using: composite\n  steps:\n    - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683\n",
    );
    let wf = "jobs:\n  build:\n    steps:\n      - uses: ./.github/actions/outer\n      - uses: ./.github/actions/inner\n";
    assert_eq!(validate_action_in_dir_str(wf, &temp), Ok(()));

    write_fixture(
        &temp,
        ".github/actions/inner/action.yaml",
        "name: Inner Unpinned\nruns:\n  using: composite\n  steps:\n    - uses: actions/checkout@v4\n",
    );
    assert!(validate_action_in_dir_str(wf, &temp).is_err());

    write_fixture(
        &temp,
        ".github/actions/cycle-a/action.yml",
        "name: Cycle A\nruns:\n  using: composite\n  steps:\n    - uses: ./.github/actions/cycle-b\n",
    );
    write_fixture(
        &temp,
        ".github/actions/cycle-b/action.yaml",
        "name: Cycle B\nruns:\n  using: composite\n  steps:\n    - uses: ./.github/actions/cycle-a\n",
    );
    let cyclic_wf = "jobs:\n  build:\n    steps:\n      - uses: ./.github/actions/cycle-a\n";
    let err = validate_action_in_dir_str(cyclic_wf, &temp).expect_err("cyclic actions must fail");
    assert!(err.contains("Cyclic"), "unexpected error: {}", err);

    let _ = fs::remove_dir_all(&temp);
}

#[test]
fn test_rejects_dotdot_path_traversal_and_docker_unpinned() {
    let temp = make_temp_workspace("traversal_docker");
    let step_dotdot = "jobs:\n  test:\n    steps:\n      - uses: ./.github/actions/../../escape\n";
    assert!(validate_action_in_dir_str(step_dotdot, &temp).is_err());

    let parent_rel = "jobs:\n  test:\n    steps:\n      - uses: ../outside/action\n";
    assert!(validate_action_in_dir_str(parent_rel, &temp).is_err());

    let job_dotdot = "jobs:\n  test:\n    uses: ./.github/workflows/../../escape.yml\n";
    assert!(validate_action_in_dir_str(job_dotdot, &temp).is_err());

    let unpinned_docker = "jobs:\n  test:\n    steps:\n      - uses: docker://alpine:3.19\n";
    assert!(validate_action_in_dir_str(unpinned_docker, &temp).is_err());

    let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let pinned_docker = format!(
        "jobs:\n  test:\n    steps:\n      - uses: docker://alpine@sha256:{}\n",
        digest
    );
    assert_eq!(validate_action_in_dir_str(&pinned_docker, &temp), Ok(()));

    write_fixture(
        &temp,
        ".github/actions/bad-docker/action.yml",
        "name: Bad Docker\nruns:\n  using: docker\n  image: docker://alpine:3.19\n",
    );
    let bad_docker_wf = "jobs:\n  test:\n    steps:\n      - uses: ./.github/actions/bad-docker\n";
    assert!(validate_action_in_dir_str(bad_docker_wf, &temp).is_err());

    let _ = fs::remove_dir_all(&temp);
}

#[test]
fn test_dockerfile_continuations_tabs_mounts_stages_and_utf8() {
    let temp = make_temp_workspace("dockerfile_pinning");
    let wf = "name: DF\npermissions: read-all\nconcurrency: df\non: push\njobs:\n  t:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n    steps:\n      - uses: ./.github/actions/df-act\n";
    write_fixture(&temp, ".github/workflows/df.yml", wf);
    write_fixture(
        &temp,
        ".github/actions/df-act/action.yml",
        "name: DF Action\nruns:\n  using: docker\n  image: Dockerfile\n",
    );
    let dig = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    for bad_df in [
        "FROM scratch\nFROM\texample.invalid/canary-image:unpinned-tab\n".to_string(),
        "FROM scratch\nFROM\\\n example.invalid/canary-image:unpinned-continuation\n".to_string(),
        "FROM scratch\nCOPY --from=example.invalid/canary-image:unpinned-copy /canary /canary\n"
            .to_string(),
        format!(
            "FROM alpine@sha256:{dig}\nRUN --mount=type=bind,from=example.invalid/canary-image:unpinned-mount,source=/canary,target=/mnt true\n"
        ),
        "LABEL maintainer=\"none\"\n".to_string(),
        format!("ARG IMG=alpine@sha256:{dig}\nFROM $IMG\n"),
        "FROM scratch\nCOPY --from=$IMG /a /b\n".to_string(),
        "FROM scratch\nRUN --mount=type=bind,from=$IMG,target=/mnt true\n".to_string(),
        "FROM scratch\nCOPY --from= /a /b\n".to_string(),
        "FROM scratch\nRUN --mount=type=bind,from=,target=/mnt true\n".to_string(),
        "FROM scratch\nCOPY --from=1 /a /b\nFROM scratch\n".to_string(),
        "FROM scratch\nFROM scratch\nCOPY --from=2 /a /b\n".to_string(),
        "FROM scratch\nRUN --mount=type=bind,from=1,target=/mnt true\n".to_string(),
        "# escape=`\nFROM scratch\nFROM`\n example.invalid/canary:unpinned\n".to_string(),
        "FROM \\\n  # comment inside continuation\n  example.invalid/canary:unpinned\n".to_string(),
        "FROM scratch\nONBUILD COPY --from=example.invalid/canary:unpinned /a /b\n".to_string(),
    ] {
        write_fixture(&temp, ".github/actions/df-act/Dockerfile", &bad_df);
        assert!(
            validate_action_in_dir_str(wf, &temp).is_err(),
            "accepted bad Dockerfile: {bad_df}"
        );
        assert!(
            !lint_workflows_in_dir(&temp.join(".github/workflows"), None),
            "lint_workflows_in_dir accepted: {bad_df}"
        );
    }

    for good_df in [
        "FROM scratch\nLABEL \\\n  xy=\"π\"\n".to_string(),
        format!(
            "# escape=`\nFROM`\n  # inline comment\n\talpine@sha256:{dig} AS Builder\nFROM scratch\nCOPY --from=builder /a /b\nCOPY --from=0 /a /b\nCOPY --from=scratch /a /b\nCOPY --from=alpine@sha256:{dig} /a /b\nRUN --mount=type=cache,target=/cache --mount=type=bind,from=0,source=/a,target=/mnt true\nRUN --mount=type=bind,from=Builder,source=/a,target=/mnt true\nRUN --mount=type=bind,from=alpine@sha256:{dig},source=/a,target=/mnt true\n"
        ),
    ] {
        write_fixture(&temp, ".github/actions/df-act/Dockerfile", &good_df);
        assert_eq!(validate_action_in_dir_str(wf, &temp), Ok(()));
        assert!(lint_workflows_in_dir(&temp.join(".github/workflows"), None));
    }

    let _ = fs::remove_dir_all(&temp);
}

#[test]
#[cfg(unix)]
fn test_rejects_symlink_escape_outside_repo_root() {
    let outer = make_temp_workspace("symlink_outer");
    let repo = outer.join("repo");
    let outside = outer.join("outside_action");
    fs::create_dir_all(repo.join(".github/actions")).unwrap();
    fs::create_dir_all(repo.join(".github/workflows")).unwrap();
    fs::create_dir_all(&outside).unwrap();
    fs::write(
        outside.join("action.yml"),
        "name: Outside\nruns:\n  using: composite\n  steps:\n    - run: echo hi\n      shell: bash\n",
    )
    .unwrap();
    std::os::unix::fs::symlink(&outside, repo.join(".github/actions/escaped")).unwrap();

    let wf = "jobs:\n  test:\n    steps:\n      - uses: ./.github/actions/escaped\n";
    assert!(validate_action_in_dir_str(wf, &repo).is_err());
    assert!(!lint_workflows_in_dir(
        &repo.join(".github/workflows"),
        None
    ));

    let _ = fs::remove_dir_all(&outer);
}

#[test]
fn test_lint_workflows_audits_standalone_actions_and_canary_cases() {
    let temp1 = make_temp_workspace("canary_reusable");
    write_fixture(
        &temp1,
        ".github/workflows/reusable_canary.yml",
        "name: Canary Reusable Workflow\npermissions: read-all\nconcurrency: canary-reusable-group\non: push\njobs:\n  external-reusable:\n    uses: example-org/canary-repo/.github/workflows/reusable-build.yml@main\n",
    );
    assert!(!lint_workflows_in_dir(
        &temp1.join(".github/workflows"),
        None
    ));
    let _ = fs::remove_dir_all(&temp1);

    let temp2 = make_temp_workspace("canary_composite");
    write_fixture(
        &temp2,
        ".github/actions/canary-composite/action.yml",
        "name: Canary Composite Action\nruns:\n  using: \"composite\"\n  steps:\n    - uses: actions/checkout@v4\n",
    );
    // Fails even as a standalone .github/actions manifest before any workflow references it
    assert!(!lint_workflows_in_dir(
        &temp2.join(".github/workflows"),
        None
    ));

    write_fixture(
        &temp2,
        ".github/workflows/composite_canary.yml",
        "name: Canary Composite\npermissions: read-all\nconcurrency: canary-composite-group\non: push\njobs:\n  build:\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    steps:\n      - uses: ./.github/actions/canary-composite\n",
    );
    assert!(!lint_workflows_in_dir(
        &temp2.join(".github/workflows"),
        None
    ));
    let _ = fs::remove_dir_all(&temp2);
}

#[test]
fn test_rejects_composite_inputs_and_indirect_env_script_injections() {
    let temp = make_temp_workspace("canary_script_injections");
    let wf_dir = temp.join(".github/workflows");
    write_fixture(
        &temp,
        ".github/actions/canary-composite/action.yml",
        "runs:\n  using: composite\n  steps:\n    - shell: bash\n      run: printf 'canary-composite:%s\\n' \"${{ github.event.pull_request.title }}\"\n",
    );
    assert!(!lint_workflows_in_dir(&wf_dir, None));
    let _ = fs::remove_dir_all(&temp);
    for bad_wf in [
        "jobs:\n  d:\n    steps:\n      - run: printf 'canary-input:%s\\n' \"${{ inputs.canary_branch }}\"\n",
        "env:\n  WORKFLOW_PR_TITLE: ${{ github.event.pull_request.title }}\njobs:\n  check:\n    env:\n      JOB_HEAD_REF: ${{ github.head_ref }}\n    steps:\n      - run: printf 'canary-env:%s:%s\\n' \"${{ env.WORKFLOW_PR_TITLE }}\" \"${{ env.JOB_HEAD_REF }}\"\n",
        "jobs:\n  u:\n    steps:\n      - run: echo \"${{ env.UNMAPPED_KEY }}\"\n",
    ] {
        let yaml: Value = serde_saphyr::from_str(bad_wf).unwrap();
        assert!(validate_run_script_expressions(&yaml).is_err());
    }
}
