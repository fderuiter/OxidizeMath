#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::ParameterConstraint;
use serde_json::{Map, Number, Value};
use std::collections::HashMap;

const STEP_KEYS: &[&str] = &[
    "dt",
    "time_step",
    "timestep",
    "step_size",
    "stepsize",
    "delta_t",
];
const UI_ONLY_KEYS: &[&str] = &[
    "camera",
    "snapshot_manager",
    "callout_manager",
    "plot_states",
    "fixed_callout_offsets",
];

fn is_non_finite_str(s: &str) -> bool {
    matches!(
        s.trim(),
        "NaN"
            | "nan"
            | "+NaN"
            | "-NaN"
            | "Infinity"
            | "+Infinity"
            | "-Infinity"
            | "inf"
            | "+inf"
            | "-inf"
    )
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Wraps bare non-standard `NaN` / `Infinity` tokens in quotes so `serde_json` can parse them.
fn quote_non_finite_tokens(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(raw.len() + 16);
    let mut in_quotes = false;
    let mut escaped = false;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if in_quotes {
            out.push(b as char);
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_quotes = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_quotes = true;
            out.push('"');
            i += 1;
            continue;
        }

        let prev_ident = i > 0 && is_ident_byte(bytes[i - 1]);
        if !prev_ident {
            let rest = &raw[i..];
            let matched = [
                "-Infinity",
                "+Infinity",
                "Infinity",
                "-NaN",
                "+NaN",
                "NaN",
                "-inf",
                "+inf",
                "inf",
                "nan",
            ]
            .into_iter()
            .find(|tok| {
                rest.starts_with(tok)
                    && (i + tok.len() >= bytes.len() || !is_ident_byte(bytes[i + tok.len()]))
            });
            if let Some(tok) = matched {
                out.push('"');
                out.push_str(tok);
                out.push('"');
                i += tok.len();
                continue;
            }
        }

        out.push(b as char);
        i += 1;
    }
    out
}

fn parse_state_value(raw: &str) -> Option<Value> {
    serde_json::from_str(raw)
        .ok()
        .or_else(|| serde_json::from_str(&quote_non_finite_tokens(raw)).ok())
}

fn is_diverged_enum_value(val: &Value) -> bool {
    match val {
        Value::String(s) => s == "Diverged",
        Value::Object(map) => map.len() == 1 && map.contains_key("Diverged"),
        _ => false,
    }
}

fn is_non_finite_value(val: &Value, base: Option<&Value>, has_constraint: bool) -> bool {
    match val {
        Value::Number(n) => n.as_f64().is_none_or(|f| !f.is_finite()),
        Value::String(s) => is_non_finite_str(s),
        Value::Null => has_constraint || base.is_some_and(Value::is_number),
        _ => false,
    }
}

fn fallback_number(
    base: Option<&Value>,
    constraint: Option<&ParameterConstraint>,
    is_step: bool,
) -> Value {
    if let Some(b @ Value::Number(n)) = base {
        if n.as_f64().is_some_and(f64::is_finite) {
            return b.clone();
        }
    }
    if let Some(c) = constraint {
        if c.min.is_finite() && c.max.is_finite() && c.min <= c.max {
            let v = if is_step {
                c.default
                    .filter(|d| d.is_finite() && *d > 0.0)
                    .map_or(c.min.max(1e-6), |d| d.clamp(c.min, c.max))
            } else {
                c.effective_default().clamp(c.min, c.max)
            };
            if let Some(num) = Number::from_f64(v) {
                return Value::Number(num);
            }
        }
    }
    if is_step {
        serde_json::json!(0.01)
    } else {
        serde_json::json!(0.0)
    }
}

fn contract_value(
    raw: f64,
    base_f64: Option<f64>,
    constraint: Option<&ParameterConstraint>,
    is_step: bool,
    allow_contract: bool,
) -> f64 {
    let mut next = raw;
    if let Some(c) = constraint {
        if c.min.is_finite() && c.max.is_finite() && c.min <= c.max {
            next = next.clamp(c.min, c.max);
        }
    }
    if is_step {
        if next <= 0.0 {
            next = base_f64
                .filter(|&b| b > 0.0)
                .or_else(|| constraint.map(|c| c.min.max(1e-6)))
                .unwrap_or(0.01);
        }
        if let Some(b) = base_f64.filter(|&b| b > 0.0) {
            next = next.min(b);
        }
    } else if allow_contract {
        if let Some(b) = base_f64 {
            if (next - b).abs() > 1e-12 {
                if constraint.is_none() {
                    let span = b.abs().max(1.0) * 10.0;
                    let low = if b >= 0.0 && next >= 0.0 {
                        0.0
                    } else {
                        b - span
                    };
                    next = next.clamp(low, b + span);
                }
                next = b + 0.5 * (next - b);
                if let Some(c) = constraint {
                    if c.min.is_finite() && c.max.is_finite() && c.min <= c.max {
                        next = next.clamp(c.min, c.max);
                    }
                }
            }
        }
    }
    next
}

fn clamp_scalar_number(
    key: &str,
    num: &Number,
    base: Option<&Value>,
    constraint: Option<&ParameterConstraint>,
    allow_contract: bool,
) -> Option<Value> {
    let is_step = STEP_KEYS.contains(&key);
    let Some(raw) = num.as_f64().filter(|f| f.is_finite()) else {
        return Some(fallback_number(base, constraint, is_step));
    };
    let base_f64 = base.and_then(Value::as_f64).filter(|f| f.is_finite());
    let keep_int = (num.is_i64() || num.is_u64())
        && constraint.is_none()
        && base
            .and_then(Value::as_number)
            .is_none_or(|bn| bn.is_i64() || bn.is_u64());

    let next = contract_value(raw, base_f64, constraint, is_step, allow_contract);
    if keep_int {
        let orig_i = num.as_i64().unwrap_or(raw as i64);
        let mut next_i = next.round() as i64;
        if base.and_then(Value::as_i64).is_some_and(|bi| bi >= 1) && orig_i >= 1 {
            next_i = next_i.max(1);
        }
        (next_i != orig_i).then(|| Value::Number(next_i.into()))
    } else {
        ((next - raw).abs() > 1e-12)
            .then(|| Number::from_f64(next).map(Value::Number))
            .flatten()
    }
}

fn sanitize_nested_container(
    root: &mut Value,
    base_root: Option<&Value>,
    allow_contract: bool,
) -> bool {
    const MAX_VISITS: usize = 4096;
    let mut stack: Vec<(&mut Value, Option<&Value>, bool, &str)> =
        vec![(root, base_root, allow_contract, "")];
    let mut visits = 0;
    let mut changed = false;

    while let Some((cur, cur_base, contract, key)) = stack.pop() {
        if visits >= MAX_VISITS {
            break;
        }
        visits += 1;

        if is_diverged_enum_value(cur) {
            *cur = Value::String("Ready".to_string());
            changed = true;
            continue;
        }
        if is_non_finite_value(cur, cur_base, false) {
            *cur = fallback_number(cur_base, None, STEP_KEYS.contains(&key));
            changed = true;
            continue;
        }
        match cur {
            Value::Number(ref n) => {
                if let Some(updated) = clamp_scalar_number(key, n, cur_base, None, contract) {
                    *cur = updated;
                    changed = true;
                }
            }
            Value::Array(items) => {
                let base_arr = cur_base.and_then(Value::as_array);
                for (idx, item) in items.iter_mut().enumerate() {
                    stack.push((item, base_arr.and_then(|a| a.get(idx)), contract, ""));
                }
            }
            Value::Object(map) => {
                let base_map = cur_base.and_then(Value::as_object);
                for (k, v) in map.iter_mut() {
                    if k == "diverged" && *v == Value::Bool(true) {
                        *v = Value::Bool(false);
                        changed = true;
                        continue;
                    }
                    let child_contract = contract && !UI_ONLY_KEYS.contains(&k.as_str());
                    let b_val = base_map.and_then(|m| m.get(k));
                    stack.push((v, b_val, child_contract, k.as_str()));
                }
            }
            _ => {}
        }
    }
    changed
}

fn clamp_object_fields(
    map: &mut Map<String, Value>,
    base_map: Option<&Map<String, Value>>,
    constraints: &HashMap<String, ParameterConstraint>,
) -> bool {
    let mut changed = false;
    for (key, val) in map.iter_mut() {
        let base_val = base_map.and_then(|m| m.get(key));
        let constraint = constraints.get(key);
        let is_step = STEP_KEYS.contains(&key.as_str());

        if is_diverged_enum_value(val) {
            *val = Value::String("Ready".to_string());
            changed = true;
        } else if key == "diverged" && *val == Value::Bool(true) {
            *val = Value::Bool(false);
            changed = true;
        } else if is_non_finite_value(val, base_val, constraint.is_some()) {
            *val = fallback_number(base_val, constraint, is_step);
            changed = true;
        } else if let Value::Number(ref n) = *val {
            if let Some(updated) = clamp_scalar_number(key, n, base_val, constraint, true) {
                *val = updated;
                changed = true;
            }
        } else if matches!(val, Value::Array(_) | Value::Object(_)) {
            let allow_contract = !UI_ONLY_KEYS.contains(&key.as_str());
            changed |= sanitize_nested_container(val, base_val, allow_contract);
        }
    }
    changed
}

/// Clamps serialized JSON tool parameters using `ParameterConstraint` bounds, caps step-size
/// fields (`dt`) at baseline, contracts drifted numeric fields toward `baseline_json`, replaces
/// non-finite values with baseline values, and resets `Diverged` enum states to `"Ready"`.
///
/// Returns `Some(clamped_json)` if at least one field was modified, or `None` otherwise.
pub fn clamp_state_json(
    state_json: &str,
    baseline_json: Option<&str>,
    constraints: &HashMap<String, ParameterConstraint>,
) -> Option<String> {
    let mut state_val = parse_state_value(state_json)?;
    let base_val = baseline_json.and_then(parse_state_value);
    let Value::Object(ref mut map) = state_val else {
        return None;
    };
    let base_map = base_val.as_ref().and_then(Value::as_object);

    if clamp_object_fields(map, base_map, constraints) {
        serde_json::to_string(&state_val).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_state_json_sanitizes_non_finite_caps_dt_and_resets_diverged() {
        let base =
            r#"{"sigma":10.0,"rho":28.0,"beta":2.5,"dt":0.01,"speed":5,"execution_state":"Ready"}"#;
        let dirty = r#"{"sigma":"NaN","rho":null,"beta":Infinity,"dt":0.2,"speed":15,"execution_state":{"Diverged":{"step":3}}}"#;
        let constraints = HashMap::from([(
            "dt".to_string(),
            ParameterConstraint {
                min: 0.001,
                max: 0.05,
                step: 0.001,
                default: Some(0.01),
            },
        )]);

        let clamped = clamp_state_json(dirty, Some(base), &constraints).expect("must clamp");
        let parsed: Value = serde_json::from_str(&clamped).unwrap();
        assert_eq!(parsed["sigma"].as_f64(), Some(10.0));
        assert_eq!(parsed["rho"].as_f64(), Some(28.0));
        assert_eq!(parsed["beta"].as_f64(), Some(2.5));
        assert_eq!(parsed["dt"].as_f64(), Some(0.01));
        assert_eq!(parsed["speed"].as_i64(), Some(10));
        assert_eq!(parsed["execution_state"], serde_json::json!("Ready"));
        assert!(clamp_state_json(base, Some(base), &constraints).is_none());
    }

    #[test]
    fn test_clamp_state_json_uses_calibrated_constraint_defaults_without_baseline() {
        let dirty = r#"{"g_l":"NaN","i_ext":null,"dt":"Infinity"}"#;
        let constraints = HashMap::from([
            (
                "g_l".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 5.0,
                    step: 0.1,
                    default: Some(0.3),
                },
            ),
            (
                "i_ext".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 50.0,
                    step: 1.0,
                    default: None,
                },
            ),
            (
                "dt".to_string(),
                ParameterConstraint {
                    min: 0.001,
                    max: 0.1,
                    step: 0.001,
                    default: Some(0.02),
                },
            ),
        ]);

        let clamped = clamp_state_json(dirty, None, &constraints).expect("must clamp");
        let parsed: Value = serde_json::from_str(&clamped).unwrap();
        assert_eq!(parsed["g_l"].as_f64(), Some(0.3));
        assert_eq!(parsed["i_ext"].as_f64(), Some(25.0));
        assert_eq!(parsed["dt"].as_f64(), Some(0.02));
    }
}
