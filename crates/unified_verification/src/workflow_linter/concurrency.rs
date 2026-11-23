use super::concurrency_eval::{
    ExprRefs, PushSimContext, Token, eval_tokens, next_closed_gha_expr, parse_ident_or_context,
    tokenize_expr,
};
use super::expressions::skip_single_quoted;
use serde_json::Value;

fn check_all_gha_exprs_closed(mut s: &str) -> Result<(), ()> {
    while let Some((_, _, rest)) = next_closed_gha_expr(s)? {
        s = rest;
    }
    Ok(())
}

fn scan_fallback_refs(expr: &str) -> ExprRefs {
    let bytes = expr.as_bytes();
    let (mut idx, mut refs) = (0, ExprRefs::default());
    while idx < bytes.len() {
        if bytes[idx] == b'\'' {
            idx = skip_single_quoted(bytes, idx);
        } else if let Ok((Token::Context(segs), next)) = parse_ident_or_context(expr, idx) {
            refs.record_context_path(&segs);
            idx = next;
        } else {
            idx += 1;
        }
    }
    refs
}

fn analyze_and_render_template(
    raw: &str,
    ctx: Option<&PushSimContext<'_>>,
) -> Result<(String, ExprRefs), ()> {
    let (mut remaining, mut out, mut total_refs) = (raw, String::new(), ExprRefs::default());
    while let Some((prefix, expr, rest)) = next_closed_gha_expr(remaining)? {
        out.push_str(prefix);
        if let Ok((tokens, refs)) = tokenize_expr(expr) {
            total_refs.merge(refs);
            if let Some(sim_ctx) = ctx {
                out.push_str(&eval_tokens(&tokens, sim_ctx)?.as_gha_str());
            }
        } else if ctx.is_some() {
            return Err(());
        } else {
            total_refs.merge(scan_fallback_refs(expr));
        }
        remaining = rest;
    }
    out.push_str(remaining);
    Ok((out, total_refs))
}

fn eval_condition(raw: &str, ctx: &PushSimContext<'_>) -> Result<bool, ()> {
    let trimmed = raw.trim();
    if let Some((prefix, expr, rest)) = next_closed_gha_expr(trimmed)? {
        if prefix.is_empty() && rest.is_empty() {
            let (tokens, _) = tokenize_expr(expr)?;
            return Ok(eval_tokens(&tokens, ctx)?.is_truthy());
        }
        let (rendered, _) = analyze_and_render_template(trimmed, Some(ctx))?;
        return match rendered.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(true),
            "false" | "" => Ok(false),
            _ => Err(()),
        };
    }
    let (tokens, _) = tokenize_expr(trimmed)?;
    Ok(eval_tokens(&tokens, ctx)?.is_truthy())
}

fn extract_push_branches(yaml: &Value) -> Result<Vec<String>, String> {
    let Some(Value::Object(on_map)) = yaml.get("on") else {
        return Ok(Vec::new());
    };
    let Some(Value::Object(push_map)) = on_map.get("push") else {
        return Ok(Vec::new());
    };
    let Some(branches_val) = push_map.get("branches") else {
        return Ok(Vec::new());
    };
    match branches_val {
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Err(
                    "Workflow 'on.push.branches' cannot contain an empty pattern.".to_string(),
                );
            }
            Ok((!trimmed.starts_with('!'))
                .then(|| trimmed.to_string())
                .into_iter()
                .collect())
        }
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                let Some(s) = item.as_str().map(str::trim).filter(|s| !s.is_empty()) else {
                    return Err(
                        "Workflow 'on.push.branches' entries must be non-empty strings."
                            .to_string(),
                    );
                };
                if !s.starts_with('!') {
                    out.push(s.to_string());
                }
            }
            Ok(out)
        }
        _ => {
            Err("Workflow 'on.push.branches' must be a string or sequence of strings.".to_string())
        }
    }
}

fn sample_branches(pattern: &str) -> Vec<String> {
    if pattern.contains('*') || pattern.contains('?') {
        ["a", "b"]
            .into_iter()
            .map(|tag| {
                pattern
                    .replace("**", &format!("nested/branch-{tag}"))
                    .replace('*', &format!("canary-{tag}"))
                    .replace('?', tag)
            })
            .collect()
    } else {
        vec![pattern.to_string()]
    }
}

fn validate_cancel_type(cancel_val: &Value, owner: &str) -> Result<(), String> {
    match cancel_val {
        Value::Bool(_) => Ok(()),
        Value::String(s) if !s.trim().is_empty() => check_all_gha_exprs_closed(s).map_err(|()| {
            format!(
                "{owner} 'concurrency' cancel-in-progress contains an unclosed '${{{{' expression."
            )
        }),
        _ => Err(format!(
            "{owner} 'concurrency' cancel-in-progress must be a boolean or non-empty expression string."
        )),
    }
}

fn parse_concurrency_mapping<'a>(
    map: &'a serde_json::Map<String, Value>,
    owner: &str,
    is_root: bool,
) -> Result<(&'a str, Option<&'a Value>), String> {
    for key in map.keys() {
        if key != "group" && key != "cancel-in-progress" {
            return Err(format!(
                "{owner} 'concurrency' block contains unknown key '{key}'."
            ));
        }
    }
    let group_val = map.get("group").ok_or_else(|| {
        if is_root {
            "Workflow 'concurrency' block must specify a 'group'.".to_string()
        } else {
            format!("{owner} 'concurrency' block must specify a 'group'.")
        }
    })?;
    let Value::String(group) = group_val else {
        return Err(if is_root {
            "Workflow 'concurrency' group must be a string.".to_string()
        } else {
            format!("{owner} 'concurrency' group must be a string.")
        });
    };
    if group.trim().is_empty() {
        return Err(if is_root {
            "Workflow 'concurrency' group cannot be empty.".to_string()
        } else {
            format!("{owner} 'concurrency' group cannot be empty.")
        });
    }
    check_all_gha_exprs_closed(group).map_err(|()| {
        format!("{owner} 'concurrency' group contains an unclosed '${{{{' expression.")
    })?;
    let cancel = map.get("cancel-in-progress");
    if let Some(cancel_val) = cancel {
        validate_cancel_type(cancel_val, owner)?;
    }
    Ok((group.as_str(), cancel))
}

fn parse_concurrency_spec<'a>(
    concurrency: &'a Value,
    owner: &str,
    is_root: bool,
) -> Result<(&'a str, Option<&'a Value>), String> {
    match concurrency {
        Value::String(s) => {
            if s.trim().is_empty() {
                return Err(if is_root {
                    "Workflow 'concurrency' block cannot be empty.".to_string()
                } else {
                    format!("{owner} 'concurrency' block cannot be empty.")
                });
            }
            check_all_gha_exprs_closed(s).map_err(|()| {
                format!("{owner} 'concurrency' contains an unclosed '${{{{' expression.")
            })?;
            Ok((s.as_str(), None))
        }
        Value::Object(map) => parse_concurrency_mapping(map, owner, is_root),
        _ => Err(if is_root {
            "Workflow 'concurrency' block must be a valid string or mapping.".to_string()
        } else {
            format!("{owner} 'concurrency' block must be a valid string or mapping.")
        }),
    }
}

fn verify_run_unique_group(
    group: &str,
    refs: ExprRefs,
    pattern: &str,
    sample: &str,
    owner: &str,
) -> Result<(), String> {
    if !refs.uses_branch_ref && !refs.uses_pr_only_key {
        return Ok(());
    }
    let ctx_a = PushSimContext {
        branch: sample,
        run_id: "1001",
        sha: "1111111111111111111111111111111111111111",
    };
    let ctx_b = PushSimContext {
        branch: sample,
        run_id: "1002",
        sha: "2222222222222222222222222222222222222222",
    };
    let (Ok((g_a, _)), Ok((g_b, _))) = (
        analyze_and_render_template(group, Some(&ctx_a)),
        analyze_and_render_template(group, Some(&ctx_b)),
    ) else {
        return Err(format!(
            "{owner} 'concurrency' group '{group}' could not be verified as run-unique for push branch '{pattern}'."
        ));
    };
    if g_a == g_b {
        return Err(format!(
            "{owner} 'concurrency' group '{group}' collapses concurrent pushes on branch '{pattern}' into shared group '{g_a}'."
        ));
    }
    Ok(())
}

fn validate_cancel_on_branch(
    cancel: Option<&Value>,
    pattern: &str,
    sample_branch: &str,
    owner: &str,
) -> Result<(), String> {
    let Some(cancel_val) = cancel else {
        return Ok(());
    };
    match cancel_val {
        Value::Bool(false) => Ok(()),
        Value::Bool(true) => Err(format!(
            "{owner} 'concurrency' cancel-in-progress cannot be 'true' when workflow triggers on push branch '{pattern}'."
        )),
        Value::String(expr) => {
            let ctx = PushSimContext {
                branch: sample_branch,
                run_id: "1001",
                sha: "1111111111111111111111111111111111111111",
            };
            match eval_condition(expr, &ctx) {
                Ok(false) => Ok(()),
                Ok(true) => Err(format!(
                    "{owner} 'concurrency' cancel-in-progress '{expr}' cancels in-progress runs on push branch '{pattern}'."
                )),
                Err(()) => Err(format!(
                    "{owner} 'concurrency' cancel-in-progress '{expr}' cannot be proven false on push branch '{pattern}'."
                )),
            }
        }
        _ => Err(format!(
            "{owner} 'concurrency' cancel-in-progress must be a boolean or non-empty expression string."
        )),
    }
}

fn validate_push_branch_scope(
    group: &str,
    cancel: Option<&Value>,
    branches: &[String],
    owner: &str,
) -> Result<(), String> {
    if branches.is_empty() {
        return Ok(());
    }
    let (_, refs) = analyze_and_render_template(group, None).map_err(|()| {
        format!("{owner} 'concurrency' group contains an invalid GitHub Actions expression.")
    })?;
    if refs.uses_branch_ref && !refs.uses_unique_run_key {
        return Err(format!(
            "{owner} 'concurrency' group '{group}' uses 'github.ref' or 'github.ref_name' on 'on.push.branches' without 'github.run_id' or 'github.sha'."
        ));
    }
    if refs.uses_pr_only_key && !refs.uses_unique_run_key {
        return Err(format!(
            "{owner} 'concurrency' group '{group}' scopes to pull requests on 'on.push.branches' without a 'github.run_id' or 'github.sha' fallback."
        ));
    }
    for pat in branches {
        for sample in sample_branches(pat) {
            verify_run_unique_group(group, refs, pat, &sample, owner)?;
            validate_cancel_on_branch(cancel, pat, &sample, owner)?;
        }
    }
    Ok(())
}

fn active_push_branches_for_job(
    job_map: &serde_json::Map<String, Value>,
    branches: &[String],
) -> Vec<String> {
    let Some(if_val) = job_map.get("if") else {
        return branches.to_vec();
    };
    match if_val {
        Value::Bool(false) => Vec::new(),
        Value::Bool(true) => branches.to_vec(),
        Value::String(if_expr) => branches
            .iter()
            .filter(|pat| {
                sample_branches(pat).iter().any(|sample| {
                    let ctx = PushSimContext {
                        branch: sample,
                        run_id: "1001",
                        sha: "1111111111111111111111111111111111111111",
                    };
                    eval_condition(if_expr, &ctx).unwrap_or(true)
                })
            })
            .cloned()
            .collect(),
        _ => branches.to_vec(),
    }
}

pub(super) fn validate_concurrency_block(yaml: &Value) -> Result<(), String> {
    let concurrency = yaml
        .get("concurrency")
        .ok_or_else(|| "Workflow is missing a global 'concurrency' block.".to_string())?;
    let (root_group, root_cancel) = parse_concurrency_spec(concurrency, "Workflow", true)?;
    let push_branches = extract_push_branches(yaml)?;
    validate_push_branch_scope(root_group, root_cancel, &push_branches, "Workflow")?;

    let jobs = match yaml.get("jobs") {
        Some(Value::Object(map)) => map,
        Some(_) => return Err("Workflow 'jobs' block must be a mapping.".to_string()),
        None => return Ok(()),
    };
    for (job_name, job_val) in jobs {
        let Some(job_map) = job_val.as_object() else {
            return Err(format!("Job '{job_name}' must be a mapping."));
        };
        let Some(job_conc) = job_map.get("concurrency") else {
            continue;
        };
        let owner = format!("Job '{job_name}'");
        let (job_group, job_cancel) = parse_concurrency_spec(job_conc, &owner, false)?;
        let job_branches = active_push_branches_for_job(job_map, &push_branches);
        validate_push_branch_scope(job_group, job_cancel, &job_branches, &owner)?;
    }
    Ok(())
}
