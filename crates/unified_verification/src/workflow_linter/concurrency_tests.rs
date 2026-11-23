use super::*;

fn check_concurrency(content: &str) -> Result<(), String> {
    let yaml: Value =
        serde_saphyr::from_str(content).map_err(|e| format!("Invalid YAML syntax: {e}"))?;
    validate_concurrency_block(&yaml)
}

#[test]
fn test_accepts_pr_scoped_concurrency_with_run_id_or_sha_fallbacks() {
    for valid_yaml in [
        r#"
name: Rust CI
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name == 'pull_request' && format('pr-{0}', github.event.pull_request.number || github.ref) || github.run_id }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
on:
  push:
    branches: [ "main", "master" ]
    tags: [ "v*" ]
  pull_request:
    branches: [ "main", "master" ]
  merge_group:
    branches: [ "main", "master" ]
  workflow_dispatch:
"#,
        r#"
name: PR Scoped Parenthesized
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name == 'pull_request' && (github.event.pull_request.number || github.ref) || github.run_id }}
  cancel-in-progress: ${{ startsWith(github.event_name, 'pull_request') }}
on:
  push:
    branches: main
  pull_request:
    branches: main
"#,
        r#"
name: Commit SHA Scoped Push
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}-${{ github.sha }}
  cancel-in-progress: false
on:
  push:
    branches: [ "main", "release/*", "!release/*-wip" ]
"#,
    ] {
        assert_eq!(
            check_concurrency(valid_yaml),
            Ok(()),
            "expected valid workflow concurrency for:\n{valid_yaml}"
        );
    }
}

#[test]
fn test_rejects_unscoped_github_ref_and_ref_name_on_push_branches() {
    for bad_group in [
        "${{ github.workflow }}-${{ github.ref }}",
        "${{ github.workflow }}-${{ github.ref_name }}",
        "${{ github.workflow }}-${{ github['ref'] }}",
        "${{ github.workflow }}-${{ github['ref_name'] }}",
        "${{ github.workflow }}-${{ github.ref || github.run_id }}",
        "${{ github.workflow }}-${{ github.ref_name || github.sha }}",
        "${{ github.workflow }}-${{ github.event.pull_request.number }}",
    ] {
        let mapping_yaml = format!(
            "name: CI\nconcurrency:\n  group: {bad_group}\n  cancel-in-progress: false\non:\n  push:\n    branches: [\"main\", \"master\"]\n"
        );
        let err = check_concurrency(&mapping_yaml)
            .expect_err(&format!("expected failure for group: {bad_group}"));
        assert!(
            err.contains("concurrency"),
            "unexpected error for {bad_group}: {err}"
        );
    }

    let shorthand_ref = r#"
name: Shorthand Ref
concurrency: ${{ github.workflow }}-${{ github.ref }}
on:
  push:
    branches: [ "main" ]
"#;
    assert!(check_concurrency(shorthand_ref).is_err());
}

#[test]
fn test_rejects_cancel_in_progress_that_cancels_main_master_or_wildcard_branches() {
    for (label, cancel_rule, branches) in [
        ("bool true", "true", "[\"main\"]"),
        (
            "commit 159c2d9d leaving master cancelable",
            "${{ github.ref != 'refs/heads/main' && !startsWith(github.ref, 'refs/tags/') }}",
            "[\"main\", \"master\"]",
        ),
        (
            "missing main exclusion",
            "${{ github.ref_name != 'master' }}",
            "[\"main\", \"master\"]",
        ),
        (
            "disjunctive bypass",
            "${{ github.event_name == 'pull_request' || github.ref != 'refs/heads/main' }}",
            "[\"main\", \"master\"]",
        ),
        (
            "wildcard branch insufficiently excluded",
            "${{ github.ref != 'refs/heads/main' }}",
            "[\"main\", \"release/*\"]",
        ),
    ] {
        let wf = format!(
            "name: CI\nconcurrency:\n  group: ${{{{ github.workflow }}}}-${{{{ github.run_id }}}}\n  cancel-in-progress: {cancel_rule}\non:\n  push:\n    branches: {branches}\n"
        );
        let err = check_concurrency(&wf).expect_err(&format!("expected rejection for {label}"));
        assert!(
            err.contains("cancel-in-progress"),
            "case '{label}' got unexpected error: {err}"
        );
    }
}

#[test]
fn test_validates_job_level_pages_lock_and_rejects_invalid_job_concurrency() {
    let valid_pages = r#"
name: CI With Pages Lock
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name == 'pull_request' && (github.event.pull_request.number || github.ref) || github.run_id }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]
jobs:
  deploy-pages:
    if: github.ref == 'refs/heads/main' && (github.event_name == 'push' || github.event_name == 'workflow_dispatch')
    runs-on: ubuntu-latest
    timeout-minutes: 10
    concurrency:
      group: pages
      cancel-in-progress: false
  pr-preview:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    timeout-minutes: 10
    concurrency:
      group: preview-${{ github.ref }}
      cancel-in-progress: true
"#;
    assert_eq!(check_concurrency(valid_pages), Ok(()));

    for (label, bad_job_conc, expected_msg) in [
        (
            "pages canceling main",
            "group: pages\n      cancel-in-progress: true",
            "Job 'deploy-pages'",
        ),
        (
            "job unscoped github.ref on main",
            "group: pages-${{ github.ref }}\n      cancel-in-progress: false",
            "Job 'deploy-pages'",
        ),
        ("job missing group", "cancel-in-progress: false", "group"),
        ("job empty group", "group: \"   \"", "empty"),
        (
            "job invalid cancel type",
            "group: pages\n      cancel-in-progress: 123",
            "cancel-in-progress",
        ),
        (
            "job unknown key",
            "group: pages\n      extra-key: true",
            "extra-key",
        ),
    ] {
        let wf = format!(
            "name: Bad Job\nconcurrency:\n  group: ${{{{ github.workflow }}}}-${{{{ github.run_id }}}}\n  cancel-in-progress: false\non:\n  push:\n    branches: [\"main\"]\njobs:\n  deploy-pages:\n    runs-on: ubuntu-latest\n    concurrency:\n      {bad_job_conc}\n"
        );
        let err = check_concurrency(&wf).expect_err(&format!("expected failure for {label}"));
        assert!(
            err.contains(expected_msg),
            "case '{label}' got unexpected error: {err}"
        );
    }
}

#[test]
fn test_preserves_workflow_call_tag_only_and_repo_ci_workflow() {
    for non_branch_push in [
        "name: Reusable\nconcurrency: reusable-ci\non: workflow_call\n",
        "name: Tag Release\nconcurrency:\n  group: release-${{ github.ref }}\n  cancel-in-progress: false\non:\n  push:\n    tags: [\"v*\"]\n",
        "name: Dedicated Pages\nconcurrency:\n  group: pages\n  cancel-in-progress: false\non:\n  push:\n    branches: [\"main\"]\n",
    ] {
        assert_eq!(check_concurrency(non_branch_push), Ok(()));
    }

    let _lock = crate::DIR_LOCK.lock().unwrap();
    let ci_path = crate::manifest_cache::resolve_workspace_path(".github/workflows/ci.yml");
    let ci_content = fs::read_to_string(&ci_path).expect("ci.yml should be readable");
    assert_eq!(check_concurrency(&ci_content), Ok(()));
    assert!(
        lint_workflows(),
        "Repository workflows must pass lint_workflows()"
    );
}
