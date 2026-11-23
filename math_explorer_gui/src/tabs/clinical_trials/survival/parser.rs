//! Single-arm, multi-arm CSV/TSV, and optional separate Control-arm cohort parser for [`super::SurvivalAnalysisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::applied::clinical_trials::survival_analysis::Observation;
use math_explorer::applied::clinical_trials::types::SurvivalTime;

pub(crate) type ArmBucket = (String, Vec<Observation>);

fn parse_event_flag(line_idx: usize, event_str: &str) -> Result<bool, String> {
    match event_str.parse::<u8>() {
        Ok(1) => Ok(true),
        Ok(0) => Ok(false),
        _ if event_str.eq_ignore_ascii_case("true") || event_str.eq_ignore_ascii_case("event") => {
            Ok(true)
        }
        _ if event_str.eq_ignore_ascii_case("false")
            || event_str.eq_ignore_ascii_case("censored") =>
        {
            Ok(false)
        }
        _ => Err(format!(
            "Line {}: Event must be 1 (event) or 0 (censored)",
            line_idx + 1
        )),
    }
}

fn is_header_line(parts: &[&str]) -> bool {
    parts.iter().any(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "time" | "survival_time" | "arm" | "group" | "cohort"
        )
    })
}

fn split_fields(line: &str) -> Vec<&str> {
    if line.contains('\t') {
        line.split('\t').collect()
    } else {
        line.split(',').collect()
    }
}

fn parse_survival_row(
    line_idx: usize,
    parts: &[&str],
    allow_three_col: bool,
) -> Result<(String, Observation), String> {
    let (arm_name, time_str, event_str) = match parts.len() {
        2 => ("Cohort".to_string(), parts[0].trim(), parts[1].trim()),
        3 if allow_three_col => parse_three_col_fields(line_idx, parts)?,
        _ if allow_three_col => {
            return Err(format!(
                "Line {}: Expected 'time, event' or 'arm, time, event'",
                line_idx + 1
            ));
        }
        _ => {
            return Err(format!("Line {}: Expected 'time, event'", line_idx + 1));
        }
    };

    let time_val = time_str
        .parse::<f64>()
        .map_err(|_| format!("Line {}: Invalid time '{}'", line_idx + 1, time_str))?;
    let event_occurred = parse_event_flag(line_idx, event_str)?;
    let time = SurvivalTime::new(time_val).map_err(|e| format!("Line {}: {}", line_idx + 1, e))?;

    Ok((
        arm_name,
        Observation {
            time,
            event_occurred,
        },
    ))
}

fn parse_three_col_fields<'a>(
    line_idx: usize,
    parts: &[&'a str],
) -> Result<(String, &'a str, &'a str), String> {
    let p0 = parts[0].trim();
    let p1 = parts[1].trim();
    let p2 = parts[2].trim();
    if p1.parse::<f64>().is_ok() && parse_event_flag(line_idx, p2).is_ok() {
        if p0.is_empty() {
            return Err(format!("Line {}: Arm label cannot be empty", line_idx + 1));
        }
        Ok((p0.to_string(), p1, p2))
    } else if p0.parse::<f64>().is_ok() && parse_event_flag(line_idx, p1).is_ok() {
        if p2.is_empty() {
            return Err(format!("Line {}: Arm label cannot be empty", line_idx + 1));
        }
        Ok((p2.to_string(), p0, p1))
    } else {
        Err(format!(
            "Line {}: Invalid 3-column survival row (expected 'arm, time, event')",
            line_idx + 1
        ))
    }
}

fn push_arm_obs(buckets: &mut Vec<ArmBucket>, arm: String, obs: Observation) {
    if let Some((_, bucket)) = buckets
        .iter_mut()
        .find(|(existing, _)| existing.eq_ignore_ascii_case(&arm))
    {
        bucket.push(obs);
    } else {
        buckets.push((arm, vec![obs]));
    }
}

/// Parses primary `input_text` plus optional separate `control_text` into validated arm buckets.
pub(crate) fn parse_survival_cohorts(
    input_text: &str,
    control_text: &str,
) -> Result<(Vec<Observation>, Vec<ArmBucket>), String> {
    let mut all_obs = Vec::new();
    let mut arm_buckets: Vec<ArmBucket> = Vec::new();
    parse_block_into(input_text, true, "", &mut all_obs, &mut arm_buckets)?;

    if !control_text.trim().is_empty() {
        for (arm_name, _) in &mut arm_buckets {
            if arm_name == "Cohort" {
                *arm_name = "Treatment".to_string();
            }
        }
        parse_block_into(
            control_text,
            false,
            "Control arm ",
            &mut all_obs,
            &mut arm_buckets,
        )?;
    }

    Ok((all_obs, arm_buckets))
}

fn parse_block_into(
    raw_text: &str,
    allow_three_col: bool,
    err_prefix: &str,
    all_obs: &mut Vec<Observation>,
    arm_buckets: &mut Vec<ArmBucket>,
) -> Result<(), String> {
    let cleaned = crate::web_io::strip_utf8_bom(raw_text);
    let mut saw_data_row = false;

    for (line_idx, raw_line) in cleaned.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts = split_fields(line);
        if !saw_data_row && is_header_line(&parts) {
            saw_data_row = true;
            continue;
        }
        saw_data_row = true;

        if all_obs.len() >= math_commons::registry::MAX_ITERATIONS {
            return Err(format!(
                "{err_prefix}Line {}: Maximum of {} observations exceeded",
                line_idx + 1,
                math_commons::registry::MAX_ITERATIONS
            ));
        }

        let (parsed_arm, obs) = parse_survival_row(line_idx, &parts, allow_three_col)
            .map_err(|e| format!("{err_prefix}{e}"))?;
        let target_arm = if allow_three_col {
            parsed_arm
        } else {
            "Control".to_string()
        };
        all_obs.push(obs);
        push_arm_obs(arm_buckets, target_arm, obs);
    }
    Ok(())
}

// [cite:clinical_trials_statistics]
