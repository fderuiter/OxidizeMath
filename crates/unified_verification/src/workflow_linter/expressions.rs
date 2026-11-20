use serde_json::Value;
use std::collections::HashSet;

/// Tracks `env:` keys statically bound to scalar literals without `${{` expressions.
#[derive(Clone, Debug, Default)]
pub(super) struct SafeEnvScope {
    safe_keys: HashSet<String>,
}

fn is_safe_env_scalar(val: &Value) -> bool {
    match val {
        Value::String(s) => !s.contains("${{"),
        Value::Number(_) | Value::Bool(_) => true,
        Value::Null | Value::Array(_) | Value::Object(_) => false,
    }
}

impl SafeEnvScope {
    /// Builds an initial `SafeEnvScope` from an optional YAML `env` mapping.
    pub(super) fn from_opt_value(
        env_val: Option<&Value>,
        owner_desc: &str,
    ) -> Result<Self, String> {
        Self::default().merged_with(env_val, owner_desc)
    }

    /// Merges an inner `env:` block over `self`, allowing inner bindings to shadow outer keys.
    pub(super) fn merged_with(
        &self,
        env_val: Option<&Value>,
        owner_desc: &str,
    ) -> Result<Self, String> {
        let mut next = self.clone();
        let map = match env_val {
            None | Some(Value::Null) => return Ok(next),
            Some(Value::Object(map)) => map,
            Some(_) => return Err(format!("{owner_desc} 'env' block must be a mapping.")),
        };
        for (key, val) in map {
            next.safe_keys.retain(|k| !k.eq_ignore_ascii_case(key));
            let has_tainted_collision = map
                .iter()
                .any(|(k2, v2)| k2.eq_ignore_ascii_case(key) && !is_safe_env_scalar(v2));
            if is_safe_env_scalar(val) && !has_tainted_collision {
                next.safe_keys.insert(key.clone());
            }
        }
        Ok(next)
    }

    /// Returns `true` if `key` resolves to a statically bound scalar literal in this scope.
    pub(super) fn is_safe(&self, key: &str) -> bool {
        self.safe_keys.contains(key)
    }
}

fn skip_ascii_ws(bytes: &[u8], mut idx: usize) -> usize {
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    idx
}

/// Advances past a single-quoted GitHub Actions string literal (`'...'` with `''` escapes).
fn skip_single_quoted(bytes: &[u8], start: usize) -> usize {
    let mut idx = start + 1;
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            if idx + 1 < bytes.len() && bytes[idx + 1] == b'\'' {
                idx += 2;
                continue;
            }
            return idx + 1;
        }
        idx += 1;
    }
    bytes.len()
}

/// Decodes a closed single-quoted GitHub Actions string literal (`'...'` with `''` escapes).
fn parse_single_quoted_literal(expr: &str, start: usize) -> Option<(String, usize)> {
    let bytes = expr.as_bytes();
    if bytes.get(start) != Some(&b'\'') {
        return None;
    }
    let mut out = String::new();
    let mut seg_start = start + 1;
    let mut idx = start + 1;
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            out.push_str(&expr[seg_start..idx]);
            if bytes.get(idx + 1) == Some(&b'\'') {
                out.push('\'');
                idx += 2;
                seg_start = idx;
            } else {
                return Some((out, idx + 1));
            }
        } else {
            idx += 1;
        }
    }
    None
}

/// Extracts the next `${{ ... }}` expression body and remaining slice from `input`.
fn next_gha_expression(input: &str) -> Option<(&str, &str)> {
    let start = input.find("${{")?;
    let rest = &input[start + 3..];
    let bytes = rest.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            idx = skip_single_quoted(bytes, idx);
        } else if idx + 1 < bytes.len() && bytes[idx] == b'}' && bytes[idx + 1] == b'}' {
            return Some((&rest[..idx], &rest[idx + 2..]));
        } else {
            idx += 1;
        }
    }
    Some((rest, ""))
}

/// Resolves `env.<KEY>` or `env['<KEY>']` against `safe_env`, returning the end byte index if safe.
fn resolve_safe_env_access(expr: &str, after_env: usize, safe_env: &SafeEnvScope) -> Option<usize> {
    let bytes = expr.as_bytes();
    let pos = skip_ascii_ws(bytes, after_env);
    match bytes.get(pos).copied() {
        Some(b'.') => {
            let key_start = skip_ascii_ws(bytes, pos + 1);
            let first = *bytes.get(key_start)?;
            if !(first.is_ascii_alphabetic() || first == b'_') {
                return None;
            }
            let mut key_end = key_start + 1;
            while key_end < bytes.len()
                && (bytes[key_end].is_ascii_alphanumeric()
                    || bytes[key_end] == b'_'
                    || bytes[key_end] == b'-')
            {
                key_end += 1;
            }
            safe_env
                .is_safe(&expr[key_start..key_end])
                .then_some(key_end)
        }
        Some(b'[') => {
            let quote_pos = skip_ascii_ws(bytes, pos + 1);
            let (key, after_quote) = parse_single_quoted_literal(expr, quote_pos)?;
            let close_pos = skip_ascii_ws(bytes, after_quote);
            if bytes.get(close_pos) != Some(&b']') {
                return None;
            }
            safe_env.is_safe(&key).then_some(close_pos + 1)
        }
        _ => None,
    }
}

/// Returns `true` if `expr` references `github`, `inputs`, or a non-static/unmapped `env` entry.
fn is_unsafe_expression(expr: &str, safe_env: &SafeEnvScope) -> bool {
    let bytes = expr.as_bytes();
    let mut idx = 0;
    let mut prev_non_ws: Option<u8> = None;

    while idx < bytes.len() {
        let byte = bytes[idx];
        if byte == b'\'' {
            idx = skip_single_quoted(bytes, idx);
            prev_non_ws = Some(b'\'');
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            let ident_start = idx;
            while idx < bytes.len()
                && (bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'_' || bytes[idx] == b'-')
            {
                idx += 1;
            }
            if prev_non_ws != Some(b'.') {
                let ident = &expr[ident_start..idx];
                if ident.eq_ignore_ascii_case("github") || ident.eq_ignore_ascii_case("inputs") {
                    return true;
                }
                if ident.eq_ignore_ascii_case("env") {
                    let Some(next_idx) = resolve_safe_env_access(expr, idx, safe_env) else {
                        return true;
                    };
                    idx = next_idx;
                }
            }
            prev_non_ws = Some(b'a');
        } else {
            if !byte.is_ascii_whitespace() {
                prev_non_ws = Some(byte);
            }
            idx += 1;
        }
    }

    false
}

/// Finds the first `${{ ... }}` expression in `script` that reads untrusted or non-static context.
pub(super) fn find_unsafe_expression_in_script(
    script: &str,
    safe_env: &SafeEnvScope,
) -> Option<String> {
    let mut remaining = script;
    while let Some((expr, rest)) = next_gha_expression(remaining) {
        if is_unsafe_expression(expr, safe_env) {
            return Some(expr.trim().to_string());
        }
        remaining = rest;
    }
    None
}

/// Validates `run` scripts across `steps` using `base_env` merged with each step's `env:` block.
pub(super) fn validate_steps_run_expressions(
    steps: &[Value],
    base_env: &SafeEnvScope,
    owner: &str,
) -> Result<(), String> {
    for (step_idx, step) in steps.iter().enumerate() {
        let Some(run_val) = step.get("run") else {
            continue;
        };
        let Some(run_script) = run_val.as_str() else {
            return Err(format!(
                "{owner} contains a step with an invalid 'run' script."
            ));
        };
        let default_step = format!("step #{}", step_idx + 1);
        let step_name = step
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(&default_step);
        let step_ctx = format!("{owner} step '{step_name}'");
        let step_env = base_env.merged_with(step.get("env"), &step_ctx)?;
        if let Some(offending_expr) = find_unsafe_expression_in_script(run_script, &step_env) {
            return Err(format!(
                "{owner} step '{step_name}' interpolates '${{{{ {offending_expr} }}}}' directly in a 'run' script; pass dynamic 'github', 'inputs', or 'env' values via shell environment variables instead."
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{
        lint_workflows_in_dir, validate_action_pinning_in_dir, validate_run_script_expressions,
    };
    use serde_json::Value;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn check_wf(yaml_str: &str) -> Result<(), String> {
        let yaml: Value = serde_saphyr::from_str(yaml_str).unwrap();
        validate_run_script_expressions(&yaml)
    }

    #[test]
    fn test_env_scope_shadowing_scalar_types_and_bracket_syntax() {
        let safe_shadow = r#"
env:
  SHARED_KEY: ${{ github.event.pull_request.title }}
  STATIC_NUM: 42
  STATIC_BOOL: true
  QUOTED_KEY: "1.2.3"
jobs:
  build:
    env:
      JOB_TAINTED: ${{ inputs.tag }}
    steps:
      - name: Override tainted with literal
        env:
          SHARED_KEY: "safe-override"
          JOB_TAINTED: "safe-job-override"
        run: |
          echo "${{ env.SHARED_KEY }} ${{ env['JOB_TAINTED'] }}"
          echo "${{ env.STATIC_NUM }} ${{ env.STATIC_BOOL }} ${{ env[ 'QUOTED_KEY' ] }}"
"#;
        assert_eq!(check_wf(safe_shadow), Ok(()));

        let tainted_shadow = r#"
env:
  SAFE_ROOT: "0.11.8"
jobs:
  build:
    steps:
      - name: Override safe with tainted
        env:
          SAFE_ROOT: ${{ github.head_ref }}
        run: echo "${{ env.SAFE_ROOT }}"
"#;
        let err = check_wf(tainted_shadow).expect_err("tainted inner shadow must fail");
        assert!(err.contains("env.SAFE_ROOT"), "err: {err}");

        let case_shadow = r#"
env:
  SAFE_ROOT: "0.11.8"
jobs:
  build:
    env:
      safe_root: ${{ github.head_ref }}
    steps:
      - run: echo "${{ env.SAFE_ROOT }}"
"#;
        assert!(check_wf(case_shadow).is_err());
    }

    #[test]
    fn test_rejects_dynamic_brackets_unmapped_env_and_case_variants() {
        for bad_run in [
            "echo \"${{ inputs.canary_branch }}\"",
            "echo \"${{ Inputs['canary_branch'] }}\"",
            "echo \"${{ GITHUB.event.pull_request.title }}\"",
            "echo \"${{ env.UNMAPPED_FROM_GITHUB_ENV }}\"",
            "echo \"${{ env['UNMAPPED_BRACKET'] }}\"",
            "echo \"${{ ENV.UNMAPPED }}\"",
            "echo \"${{ env[matrix.key] }}\"",
            "echo \"${{ env['SAFE' || 'OTHER'] }}\"",
            "echo \"${{ toJSON(env) }}\"",
            "echo \"${{ env.SAFE || env.UNMAPPED }}\"",
            "echo \"${{ env.SAFE && inputs.flag }}\"",
        ] {
            let wf =
                format!("env:\n  SAFE: \"1.0\"\njobs:\n  t:\n    steps:\n      - run: {bad_run}\n");
            assert!(check_wf(&wf).is_err(), "unexpectedly allowed: {bad_run}");
        }
    }

    #[test]
    fn test_composite_action_and_reusable_workflow_defense_in_depth() {
        static ID: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "oxidize_expr_def_{}_{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join(".github/actions/comp")).unwrap();
        fs::create_dir_all(root.join(".github/workflows")).unwrap();

        fs::write(
            root.join(".github/actions/comp/action.yml"),
            "name: Comp\nruns:\n  using: composite\n  steps:\n    - shell: bash\n      env:\n        SAFE_V: \"2.0\"\n      run: echo \"${{ env.SAFE_V }}\"\n",
        )
        .unwrap();
        let caller = "name: C\npermissions: read-all\nconcurrency: c\non: push\njobs:\n  j:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n    steps:\n      - uses: ./.github/actions/comp\n";
        fs::write(root.join(".github/workflows/caller.yml"), caller).unwrap();
        assert!(lint_workflows_in_dir(&root.join(".github/workflows"), None));

        // Tainted env inside composite action fails both standalone and via caller
        fs::write(
            root.join(".github/actions/comp/action.yml"),
            "name: Comp\nruns:\n  using: composite\n  steps:\n    - shell: bash\n      env:\n        BAD_V: ${{ inputs.user_arg }}\n      run: echo \"${{ env.BAD_V }}\"\n",
        )
        .unwrap();
        let caller_yaml: Value = serde_saphyr::from_str(caller).unwrap();
        assert!(validate_action_pinning_in_dir(&caller_yaml, &root).is_err());
        assert!(!lint_workflows_in_dir(
            &root.join(".github/workflows"),
            None
        ));

        // Local reusable workflow with unsafe run script fails validate_action_pinning_in_dir directly
        fs::write(
            root.join(".github/workflows/reusable.yml"),
            "permissions: read-all\nconcurrency: c\non: workflow_call\njobs:\n  inner:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n    steps:\n      - run: echo \"${{ inputs.ref }}\"\n",
        )
        .unwrap();
        let reuse_caller: Value =
            serde_saphyr::from_str("jobs:\n  call:\n    uses: ./.github/workflows/reusable.yml\n")
                .unwrap();
        let reuse_err = validate_action_pinning_in_dir(&reuse_caller, &root).unwrap_err();
        assert!(reuse_err.contains("inputs.ref"), "err: {reuse_err}");

        let _ = fs::remove_dir_all(&root);
    }
}
