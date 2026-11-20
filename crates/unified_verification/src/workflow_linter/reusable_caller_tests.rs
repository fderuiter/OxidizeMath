use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static REUSABLE_TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_repo(tag: &str) -> PathBuf {
    let id = REUSABLE_TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "oxidize_reusable_caller_{}_{}_{}",
        tag,
        std::process::id(),
        id
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_file(root: &Path, rel: &str, content: &str) {
    let full = root.join(rel);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(full, content).unwrap();
}

#[test]
fn test_ignores_single_quoted_strings_and_dotted_inputs_or_github_properties() {
    let safe_yaml = r#"
jobs:
  verify:
    steps:
      - name: Safe literals and dotted step outputs
        env:
          INPUT_TARGET: ${{ inputs.target }}
          GITHUB_REF_NAME: ${{ github.ref_name }}
        run: |
          if [ "${{ matrix.mode == 'inputs' && matrix.kind != 'github' }}" = "true" ]; then
            echo "${{ 'it''s inputs.target and github.ref_name' }}"
          fi
          echo "${{ steps.meta.inputs }} ${{ steps.meta.github }} ${{ steps.meta.outputs.inputs }}"
          echo "$INPUT_TARGET $GITHUB_REF_NAME"
"#;
    let parsed: Value = serde_saphyr::from_str(safe_yaml).unwrap();
    assert_eq!(validate_run_script_expressions(&parsed), Ok(()));

    for unsafe_expr in [
        "echo \"${{ inputs.target }}\"",
        "echo \"${{ inputs['extra-files'] }}\"",
        "echo \"${{ steps[inputs.step_id].outputs.val }}\"",
        "echo \"${{ format('}}', inputs.untrusted) }}\"",
        "echo \"${{ format('}}', github.head_ref) }}\"",
    ] {
        let wf = format!("jobs:\n  bad:\n    steps:\n      - run: {unsafe_expr}\n");
        let yaml: Value = serde_saphyr::from_str(&wf).unwrap();
        assert!(
            validate_run_script_expressions(&yaml).is_err(),
            "expected rejection for: {unsafe_expr}"
        );
    }
}

#[test]
fn test_composite_action_scans_run_scripts_for_composite_and_steps_lists() {
    let repo = temp_repo("composite_steps");
    let wf_dir = repo.join(".github/workflows");

    // 1. Composite action binding inputs/github via env passes
    write_file(
        &repo,
        ".github/actions/safe-comp/action.yml",
        "name: Safe\nruns:\n  using: composite\n  steps:\n    - shell: bash\n      env:\n        IN_VAL: ${{ inputs.target }}\n        GH_SHA: ${{ github.sha }}\n      run: |\n        printf '%s:%s\\n' \"$IN_VAL\" \"$GH_SHA\"\n        echo \"${{ steps.meta.inputs }} ${{ 'inputs' }}\"\n",
    );
    assert!(lint_workflows_in_dir(&wf_dir, None));

    // 2. Direct ${{ inputs.* }} inside composite action fails
    write_file(
        &repo,
        ".github/actions/safe-comp/action.yml",
        "name: Unsafe Inputs\nruns:\n  using: composite\n  steps:\n    - name: Run target\n      shell: bash\n      run: cargo build --target ${{ inputs.target }}\n",
    );
    assert!(!lint_workflows_in_dir(&wf_dir, None));

    // 3. Non-composite action manifest that still defines a runs.steps list is also scanned
    write_file(
        &repo,
        ".github/actions/safe-comp/action.yml",
        "name: Steps Without Composite\nruns:\n  using: node20\n  main: index.js\n  steps:\n    - run: echo \"${{ inputs.extra }}\"\n",
    );
    let caller: Value = serde_saphyr::from_str(
        "jobs:\n  b:\n    steps:\n      - uses: ./.github/actions/safe-comp\n",
    )
    .unwrap();
    let err = validate_action_pinning_in_dir(&caller, &repo).unwrap_err();
    assert!(err.contains("inputs.extra"), "unexpected error: {err}");
    assert!(!lint_workflows_in_dir(&wf_dir, None));

    let _ = fs::remove_dir_all(&repo);
}

#[test]
fn test_job_uses_caller_exempts_and_rejects_timeout_minutes() {
    let valid_caller = r#"
name: Caller
permissions: read-all
concurrency: caller-ci
on: push
jobs:
  call-ext:
    uses: example-org/repo/.github/workflows/ci.yml@11bd71901bbe5b1630ceea73d27597364c9af683
  regular-job:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
"#;
    let valid_yaml: Value = serde_saphyr::from_str(valid_caller).unwrap();
    assert_eq!(validate_job_timeouts(&valid_yaml), Ok(()));

    let caller_with_timeout = r#"
jobs:
  call-ext:
    timeout-minutes: 10
    uses: example-org/repo/.github/workflows/ci.yml@11bd71901bbe5b1630ceea73d27597364c9af683
"#;
    let bad_yaml: Value = serde_saphyr::from_str(caller_with_timeout).unwrap();
    let err = validate_job_timeouts(&bad_yaml).expect_err("uses caller with timeout must fail");
    assert!(
        err.contains("call-ext") && err.contains("timeout-minutes"),
        "err: {err}"
    );

    let repo = temp_repo("caller_e2e");
    write_file(&repo, ".github/workflows/caller.yml", valid_caller);
    assert!(lint_workflows_in_dir(&repo.join(".github/workflows"), None));

    write_file(
        &repo,
        ".github/workflows/caller.yml",
        &format!(
            "name: Bad\npermissions: read-all\nconcurrency: c\non: push\n{caller_with_timeout}"
        ),
    );
    assert!(!lint_workflows_in_dir(
        &repo.join(".github/workflows"),
        None
    ));
    let _ = fs::remove_dir_all(&repo);
}

#[test]
fn test_local_reusable_workflow_enforces_full_policy_checks() {
    let repo = temp_repo("reusable_full_checks");
    let caller: Value =
        serde_saphyr::from_str("jobs:\n  call:\n    uses: ./.github/workflows/reusable.yml\n")
            .unwrap();

    for (label, bad_reusable, expected_substr) in [
        (
            "missing permissions",
            "concurrency: c\non: workflow_call\njobs:\n  j:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n    steps:\n      - run: echo ok\n",
            "permissions",
        ),
        (
            "missing concurrency",
            "permissions: read-all\non: workflow_call\njobs:\n  j:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n    steps:\n      - run: echo ok\n",
            "concurrency",
        ),
        (
            "missing timeout",
            "permissions: read-all\nconcurrency: c\non: workflow_call\njobs:\n  j:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo ok\n",
            "timeout-minutes",
        ),
        (
            "unpinned action",
            "permissions: read-all\nconcurrency: c\non: workflow_call\njobs:\n  j:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n    steps:\n      - uses: actions/checkout@v4\n",
            "40-character commit hash",
        ),
        (
            "unsafe github expr",
            "permissions: read-all\nconcurrency: c\non: workflow_call\njobs:\n  j:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n    steps:\n      - run: echo \"${{ github.head_ref }}\"\n",
            "github.head_ref",
        ),
    ] {
        write_file(&repo, ".github/workflows/reusable.yml", bad_reusable);
        let err = validate_action_pinning_in_dir(&caller, &repo)
            .expect_err(&format!("expected failure for {label}"));
        assert!(
            err.contains(expected_substr),
            "case '{label}' got unexpected error: {err}"
        );
    }

    let _ = fs::remove_dir_all(&repo);
}

#[test]
fn test_local_reusable_workflow_forwards_exceptions_and_detects_cycles() {
    let repo = temp_repo("reusable_exceptions_cycles");
    let reusable_deploy = "permissions: read-all\nconcurrency: deploy\non: workflow_call\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    permissions:\n      pages: write\n      id-token: write\n    steps:\n      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683\n";
    let caller_wf = "name: Caller\npermissions: read-all\nconcurrency: caller\non: push\njobs:\n  call-a:\n    uses: ./.github/workflows/reusable_deploy.yml\n  call-b:\n    uses: ./.github/workflows/reusable_deploy.yml\n";
    write_file(
        &repo,
        ".github/workflows/reusable_deploy.yml",
        reusable_deploy,
    );
    write_file(&repo, ".github/workflows/caller.yml", caller_wf);

    let caller_yaml: Value = serde_saphyr::from_str(caller_wf).unwrap();
    assert!(validate_action_pinning_in_dir(&caller_yaml, &repo).is_err());

    let whitelist = repo.join("verification_whitelist.toml");
    fs::write(
        &whitelist,
        "[[workflow_linter.exceptions]]\nworkflow = \".github/workflows/reusable_deploy.yml\"\njob = \"deploy\"\nallow_write = [\"pages\", \"id-token\"]\nreason = \"Pages reusable deploy\"\n",
    )
    .unwrap();
    let exceptions = load_workflow_exceptions(Some(&whitelist));
    assert_eq!(
        validate_action_pinning_with_exceptions(&caller_yaml, &repo, &exceptions),
        Ok(())
    );
    assert!(lint_workflows_in_dir(
        &repo.join(".github/workflows"),
        Some(&whitelist)
    ));

    // Mutual cycle: cycle_a.yml <-> cycle_b.yml must fail cleanly via PinningState
    write_file(
        &repo,
        ".github/workflows/cycle_a.yml",
        "permissions: read-all\nconcurrency: a\non: workflow_call\njobs:\n  to-b:\n    uses: ./.github/workflows/cycle_b.yml\n",
    );
    write_file(
        &repo,
        ".github/workflows/cycle_b.yml",
        "permissions: read-all\nconcurrency: b\non: workflow_call\njobs:\n  to-a:\n    uses: ./.github/workflows/cycle_a.yml\n",
    );
    let cyc_caller: Value =
        serde_saphyr::from_str("jobs:\n  start:\n    uses: ./.github/workflows/cycle_a.yml\n")
            .unwrap();
    let cyc_err = validate_action_pinning_in_dir(&cyc_caller, &repo).unwrap_err();
    assert!(cyc_err.contains("Cyclic"), "err: {cyc_err}");

    let _ = fs::remove_dir_all(&repo);
}
