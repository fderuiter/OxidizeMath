//! Parsing, two-arm Kaplan-Meier evaluation, step/censor coordinate builders,
//! and local file I/O helpers for [`super::survival::SurvivalAnalysisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::mpsc::{channel, Receiver};

use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, strip_utf8_bom, OpenDialogConfig,
    SaveAsDialogConfig,
};
use math_explorer::applied::clinical_trials::survival_analysis::{
    kaplan_meier, log_rank_test, try_estimate_hazard_ratio, LogRankResult, Observation, TimePoint,
};
use math_explorer::applied::clinical_trials::types::SurvivalTime;

const SURVIVAL_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Survival Cohort (*.csv, *.tsv, *.txt)",
    extensions: &["csv", "tsv", "txt"],
    accept: ".csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain",
    read_error_prefix: "Failed to read survival file",
};

const RISK_TABLE_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "Kaplan-Meier Risk Table (*.csv)",
    extensions: &["csv"],
    fallback_filename: "kaplan_meier_risk_table.csv",
    write_error_prefix: "Failed to save risk table CSV",
};

/// Sample two-arm oncology cohort (`time, event, arm`) for quick workbench exploration.
pub const TWO_ARM_SAMPLE_TEXT: &str = "time,event,arm\n\
4.0,1,Treatment\n\
7.0,0,Treatment\n\
11.0,1,Treatment\n\
15.0,0,Treatment\n\
19.0,1,Treatment\n\
24.0,0,Treatment\n\
28.0,0,Treatment\n\
33.0,1,Treatment\n\
2.0,1,Control\n\
5.0,1,Control\n\
6.0,1,Control\n\
9.0,0,Control\n\
12.0,1,Control\n\
14.0,1,Control\n\
17.0,1,Control\n\
21.0,0,Control";

/// Evaluated Kaplan-Meier curve and raw observations for one study arm.
#[derive(Debug, Clone)]
pub struct ArmSurvivalSummary {
    /// Display name of the study arm (`"Cohort"` for single-arm data).
    pub label: String,
    /// Validated `Observation` list for this arm.
    pub observations: Vec<Observation>,
    /// Product-limit `TimePoint` trajectory including `n_at_risk`, `n_events`, and `n_censored`.
    pub curve: Vec<TimePoint>,
}

/// Complete single-arm or two-arm survival evaluation result.
#[derive(Debug, Clone)]
pub struct ParsedSurvivalCohort {
    /// Combined observations across all arms (preserving legacy total observation count).
    pub all_observations: Vec<Observation>,
    /// Per-arm summaries (`len == 1` for two-column inputs, `len == 2` for two-arm inputs).
    pub arms: Vec<ArmSurvivalSummary>,
    /// Incidence-density hazard ratio `HR` when two arms are present.
    pub hazard_ratio: Option<Result<f64, String>>,
    /// Mantel-Haenszel two-arm log-rank test summary when two arms are present.
    pub log_rank: Option<Result<LogRankResult, String>>,
}

fn is_survival_header(first_col: &str) -> bool {
    let t = first_col.trim().trim_matches('"');
    t.eq_ignore_ascii_case("time")
        || t.eq_ignore_ascii_case("duration")
        || t.eq_ignore_ascii_case("t")
}

fn parse_event_flag(line_no: usize, event_str: &str) -> Result<bool, String> {
    match event_str.parse::<u8>() {
        Ok(1) => Ok(true),
        Ok(0) => Ok(false),
        _ if event_str.eq_ignore_ascii_case("true") => Ok(true),
        _ if event_str.eq_ignore_ascii_case("false") => Ok(false),
        _ => Err(format!(
            "Line {line_no}: Event must be 1 (event) or 0 (censored)"
        )),
    }
}

fn parse_survival_row(line_no: usize, line: &str) -> Result<(Observation, Option<String>), String> {
    let delimiter = if line.contains('\t') { '\t' } else { ',' };
    let parts: Vec<&str> = line.split(delimiter).collect();
    if parts.len() != 2 && parts.len() != 3 {
        return Err(format!(
            "Line {line_no}: Expected 'time, event' or 'time, event, arm'"
        ));
    }
    let time_str = parts[0].trim().trim_matches('"');
    let event_str = parts[1].trim().trim_matches('"');
    let time_val = time_str
        .parse::<f64>()
        .map_err(|_| format!("Line {line_no}: Invalid time '{time_str}'"))?;
    let event_occurred = parse_event_flag(line_no, event_str)?;
    let time = SurvivalTime::new(time_val).map_err(|e| format!("Line {line_no}: {e}"))?;
    let arm_opt = if parts.len() == 3 {
        let arm = parts[2].trim().trim_matches('"');
        if arm.is_empty() {
            return Err(format!("Line {line_no}: Arm label cannot be empty"));
        }
        Some(arm.to_string())
    } else {
        None
    };
    Ok((
        Observation {
            time,
            event_occurred,
        },
        arm_opt,
    ))
}

fn push_into_arm_bucket(
    line_no: usize,
    buckets: &mut Vec<(String, Vec<Observation>)>,
    arm_name: String,
    obs: Observation,
) -> Result<(), String> {
    if let Some((_, list)) = buckets
        .iter_mut()
        .find(|(existing, _)| existing.eq_ignore_ascii_case(&arm_name))
    {
        list.push(obs);
        return Ok(());
    }
    if buckets.len() >= 2 {
        return Err(format!(
            "Line {line_no}: At most 2 survival arms are supported, found unexpected arm '{arm_name}'"
        ));
    }
    buckets.push((arm_name, vec![obs]));
    Ok(())
}

fn order_two_arms_treatment_first(buckets: &mut [(String, Vec<Observation>)]) {
    if buckets.len() == 2 {
        let second_is_trt = buckets[1].0.eq_ignore_ascii_case("treatment")
            || buckets[1].0.eq_ignore_ascii_case("treated")
            || buckets[1].0 == "1";
        let first_is_trt = buckets[0].0.eq_ignore_ascii_case("treatment")
            || buckets[0].0.eq_ignore_ascii_case("treated")
            || buckets[0].0 == "1";
        if second_is_trt && !first_is_trt {
            buckets.swap(0, 1);
        }
    }
}

/// Parses legacy two-column (`time, event`) or two-arm (`time, event, arm`) text/CSV/TSV.
pub fn parse_survival_cohort(raw: &str) -> Result<ParsedSurvivalCohort, String> {
    let cleaned = strip_utf8_bom(raw);
    let mut all_observations = Vec::new();
    let mut arm_buckets: Vec<(String, Vec<Observation>)> = Vec::new();
    let mut expected_has_arm: Option<bool> = None;
    let mut saw_first = false;

    for (line_idx, raw_line) in cleaned.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let line_no = line_idx + 1;
        if !saw_first {
            saw_first = true;
            let delimiter = if line.contains('\t') { '\t' } else { ',' };
            if let Some(first_col) = line.split(delimiter).next() {
                if is_survival_header(first_col) {
                    continue;
                }
            }
        }
        if all_observations.len() >= math_commons::registry::MAX_ITERATIONS {
            return Err(format!(
                "Line {line_no}: Maximum of {} observations exceeded",
                math_commons::registry::MAX_ITERATIONS
            ));
        }
        let (obs, arm_opt) = parse_survival_row(line_no, line)?;
        let has_arm = arm_opt.is_some();
        if let Some(expected) = expected_has_arm {
            if expected != has_arm {
                return Err(format!(
                    "Line {line_no}: Cannot mix 2-column and 3-column survival rows"
                ));
            }
        } else {
            expected_has_arm = Some(has_arm);
        }
        all_observations.push(obs);
        if let Some(arm_name) = arm_opt {
            push_into_arm_bucket(line_no, &mut arm_buckets, arm_name, obs)?;
        }
    }
    Ok(finalize_survival_cohort(
        all_observations,
        arm_buckets,
        expected_has_arm == Some(true),
    ))
}

fn finalize_survival_cohort(
    all_observations: Vec<Observation>,
    mut arm_buckets: Vec<(String, Vec<Observation>)>,
    has_arms: bool,
) -> ParsedSurvivalCohort {
    if has_arms {
        order_two_arms_treatment_first(&mut arm_buckets);
        let arms: Vec<ArmSurvivalSummary> = arm_buckets
            .into_iter()
            .map(|(label, observations)| {
                let curve = kaplan_meier(&observations);
                ArmSurvivalSummary {
                    label,
                    observations,
                    curve,
                }
            })
            .collect();
        let (hazard_ratio, log_rank) = if arms.len() == 2 {
            let hr = try_estimate_hazard_ratio(&arms[0].observations, &arms[1].observations)
                .map_err(|e| e.to_string());
            let lr = log_rank_test(&arms[0].observations, &arms[1].observations)
                .map_err(|e| e.to_string());
            (Some(hr), Some(lr))
        } else {
            (None, None)
        };
        ParsedSurvivalCohort {
            all_observations,
            arms,
            hazard_ratio,
            log_rank,
        }
    } else {
        let curve = kaplan_meier(&all_observations);
        let arms = vec![ArmSurvivalSummary {
            label: "Cohort".to_string(),
            observations: all_observations.clone(),
            curve,
        }];
        ParsedSurvivalCohort {
            all_observations,
            arms,
            hazard_ratio: None,
            log_rank: None,
        }
    }
}

/// Builds step-function polyline coordinates from a Kaplan-Meier `TimePoint` slice.
#[must_use]
pub fn build_km_step_points(curve: &[TimePoint]) -> Vec<[f64; 2]> {
    let mut plot_points = Vec::with_capacity(curve.len() * 2 + 2);
    plot_points.push([0.0, 1.0]);
    for point in curve {
        if let Some(last) = plot_points.last() {
            plot_points.push([point.time, last[1]]);
        }
        plot_points.push([point.time, point.survival_probability]);
    }
    if let Some(last) = plot_points.last() {
        plot_points.push([last[0] * 1.05, last[1]]);
    }
    plot_points
}

/// Extracts censored observation coordinates `(t, S(t))` for censor tick markers.
#[must_use]
pub fn build_censor_points(curve: &[TimePoint]) -> Vec<[f64; 2]> {
    curve
        .iter()
        .filter(|pt| pt.n_censored > 0)
        .map(|pt| [pt.time, pt.survival_probability])
        .collect()
}

/// Formats all arm `TimePoint` records (`n_at_risk`, `n_events`, `n_censored`) as CSV.
#[must_use]
pub fn format_risk_table_csv(arms: &[ArmSurvivalSummary]) -> String {
    use std::fmt::Write;
    let mut out =
        String::from("arm,time,survival_probability,n_at_risk,n_events,n_censored\n");
    for arm in arms {
        let safe_arm = arm.label.replace(',', ";");
        for pt in &arm.curve {
            let _ = writeln!(
                &mut out,
                "{},{:.6},{:.6},{},{},{}",
                safe_arm,
                pt.time,
                pt.survival_probability,
                pt.n_at_risk,
                pt.n_events,
                pt.n_censored
            );
        }
    }
    out
}

/// Non-blocking file open/save channel state for survival cohorts and risk-table CSV exports.
#[derive(Default)]
pub struct SurvivalDialogIo {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    /// Latest I/O banner message (`is_error`, `message`).
    pub status: Option<(bool, String)>,
}

impl SurvivalDialogIo {
    /// Opens a local CSV/TSV survival cohort file picker.
    pub fn trigger_open(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, SURVIVAL_OPEN_CONFIG, |path| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "survival.csv".to_string())
        });
    }

    /// Saves the `TimePoint` risk table to a local CSV file or browser download.
    pub fn trigger_save_risk_table(&mut self, csv_content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(
            tx,
            RISK_TABLE_SAVE_CONFIG,
            "kaplan_meier_risk_table.csv",
            csv_content,
            |path, fallback| {
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| fallback.to_string())
            },
        );
    }

    /// Polls channels and returns newly loaded cohort text if ready.
    pub fn poll(&mut self) -> Option<String> {
        let mut loaded_text = None;
        if let Some(rx) = &self.open_rx {
            if let Ok(res) = rx.try_recv() {
                self.open_rx = None;
                match res {
                    Ok((name, text)) => {
                        self.status = Some((false, format!("Loaded {name}")));
                        loaded_text = Some(text);
                    }
                    Err(err) => self.status = Some((true, err)),
                }
            }
        }
        if let Some(rx) = &self.save_rx {
            if let Ok(res) = rx.try_recv() {
                self.save_rx = None;
                match res {
                    Ok(name) => self.status = Some((false, format!("Saved {name}"))),
                    Err(err) => self.status = Some((true, err)),
                }
            }
        }
        loaded_text
    }
}

// [cite:clinical_trials_statistics]
