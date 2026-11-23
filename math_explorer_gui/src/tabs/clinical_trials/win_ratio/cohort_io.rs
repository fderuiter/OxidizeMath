//! Local two-arm cohort `.csv` and `.json` parsing, strict validation, and composite evaluation.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::applied::clinical_trials::analysis::{calculate_risk_metrics, RiskMetrics};
use math_explorer::applied::clinical_trials::survival_analysis::{
    kaplan_meier, try_estimate_hazard_ratio, Observation, TimePoint,
};
use math_explorer::applied::clinical_trials::types::ContingencyTable;
use math_explorer::applied::win_ratio::censoring::{
    CensoredOutcome, RightCensoredComparator, TierCount,
};
use math_explorer::applied::win_ratio::pair_comparison::WinRatioAnalysis;
use math_explorer::applied::win_ratio::power_sweep::MAX_ARM_SIZE;
use math_explorer::applied::win_ratio::u_statistic::UnmatchedWinRatioSummary;

/// Maximum permitted raw file byte length for local cohort uploads (1 MiB).
pub const MAX_COHORT_FILE_BYTES: usize = 1_048_576;

/// Validated two-tier composite patient record loaded from local `.csv` or `.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CohortPatient {
    /// Tier 1 (fatal) time-to-event or follow-up censoring time.
    pub fatal: CensoredOutcome,
    /// Tier 2 (non-fatal) time-to-event or follow-up censoring time.
    pub nonfatal: CensoredOutcome,
}

/// Validated two-arm patient cohort and its computed composite, survival, and binary risk metrics.
#[derive(Debug, Clone)]
pub struct LoadedCohortAnalysis {
    /// Source leaf filename for UI display and takeaway metadata.
    pub source_name: String,
    /// Validated treatment-arm patient records.
    pub treatment: Vec<CohortPatient>,
    /// Validated control-arm patient records.
    pub control: Vec<CohortPatient>,
    /// Unmatched U-statistic win ratio summary.
    pub summary: UnmatchedWinRatioSummary,
    /// Hierarchical `[Fatal, Non-Fatal]` tier counts.
    pub tiers: [TierCount; 2],
    /// Treatment Tier-1 Kaplan-Meier curve.
    pub trt_km: Vec<TimePoint>,
    /// Control Tier-1 Kaplan-Meier curve.
    pub ctrl_km: Vec<TimePoint>,
    /// Tier-1 incidence-rate hazard ratio (`Err` when zero events occur in an arm).
    pub fatal_hazard_ratio: Result<f64, String>,
    /// Composite any-event binary risk metrics (`RR`, `OR`, and 95% CIs).
    pub risk_metrics: Result<RiskMetrics, String>,
    /// Composite event risk difference ($p_{\text{trt}} - p_{\text{ctrl}}$).
    pub risk_difference: f64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonPatientRow {
    #[serde(default)]
    arm: Option<String>,
    fatal_time: f64,
    fatal_event: serde_json::Value,
    nonfatal_time: f64,
    nonfatal_event: serde_json::Value,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonTwoArmCohort {
    treatment: Vec<JsonPatientRow>,
    control: Vec<JsonPatientRow>,
}

fn parse_bool_token(raw: &str, context: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Ok(true),
        "0" | "false" | "no" => Ok(false),
        other => Err(format!(
            "{context}: invalid event flag '{other}' (expected 0/1 or false/true)"
        )),
    }
}

fn parse_json_event(val: &serde_json::Value, context: &str) -> Result<bool, String> {
    match val {
        serde_json::Value::Bool(b) => Ok(*b),
        serde_json::Value::Number(n) if n.as_u64() == Some(1) => Ok(true),
        serde_json::Value::Number(n) if n.as_u64() == Some(0) => Ok(false),
        serde_json::Value::String(s) => parse_bool_token(s, context),
        _ => Err(format!(
            "{context}: event indicator must be 0, 1, or boolean"
        )),
    }
}

fn build_patient(
    fatal_time: f64,
    fatal_event: bool,
    nonfatal_time: f64,
    nonfatal_event: bool,
    ctx: &str,
) -> Result<CohortPatient, String> {
    let fatal =
        CensoredOutcome::new(fatal_time, fatal_event).map_err(|e| format!("{ctx} fatal: {e}"))?;
    let nonfatal = CensoredOutcome::new(nonfatal_time, nonfatal_event)
        .map_err(|e| format!("{ctx} nonfatal: {e}"))?;
    Ok(CohortPatient { fatal, nonfatal })
}

fn is_treatment_arm(raw: &str, ctx: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "treatment" | "trt" | "active" | "1" => Ok(true),
        "control" | "ctrl" | "placebo" | "0" => Ok(false),
        other => Err(format!(
            "{ctx}: unrecognized arm '{other}' (expected treatment or control)"
        )),
    }
}

/// Parses a two-arm composite cohort from CSV (`arm, fatal_time, fatal_event, nonfatal_time, nonfatal_event`).
pub fn parse_cohort_csv(content: &str) -> Result<(Vec<CohortPatient>, Vec<CohortPatient>), String> {
    let clean = crate::web_io::strip_utf8_bom(content);
    let mut trt = Vec::new();
    let mut ctrl = Vec::new();
    for (idx, raw_line) in clean.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if idx == 0 && line.to_ascii_lowercase().contains("fatal") {
            continue;
        }
        let ctx = format!("CSV line {}", idx + 1);
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 5 {
            return Err(format!(
                "{ctx}: expected 5 columns 'arm, fatal_time, fatal_event, nonfatal_time, nonfatal_event'"
            ));
        }
        let is_trt = is_treatment_arm(cols[0], &ctx)?;
        let ft = cols[1]
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("{ctx}: invalid fatal_time '{}'", cols[1].trim()))?;
        let fe = parse_bool_token(cols[2], &ctx)?;
        let nt = cols[3]
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("{ctx}: invalid nonfatal_time '{}'", cols[3].trim()))?;
        let ne = parse_bool_token(cols[4], &ctx)?;
        let patient = build_patient(ft, fe, nt, ne, &ctx)?;
        push_bounded_arm(is_trt, patient, &mut trt, &mut ctrl)?;
    }
    validate_arm_counts(&trt, &ctrl)?;
    Ok((trt, ctrl))
}

fn push_bounded_arm(
    is_trt: bool,
    patient: CohortPatient,
    trt: &mut Vec<CohortPatient>,
    ctrl: &mut Vec<CohortPatient>,
) -> Result<(), String> {
    let target = if is_trt { trt } else { ctrl };
    if target.len() >= MAX_ARM_SIZE {
        return Err(format!(
            "Cohort exceeds maximum of {MAX_ARM_SIZE} patients per arm"
        ));
    }
    target.push(patient);
    Ok(())
}

fn validate_arm_counts(trt: &[CohortPatient], ctrl: &[CohortPatient]) -> Result<(), String> {
    if trt.len() < 2 || ctrl.len() < 2 {
        return Err(format!(
            "Both arms require at least 2 patients (got {} treatment, {} control)",
            trt.len(),
            ctrl.len()
        ));
    }
    Ok(())
}

/// Parses a two-arm composite cohort from JSON (`{"treatment":[...],"control":[...]}` or row array).
pub fn parse_cohort_json(
    content: &str,
) -> Result<(Vec<CohortPatient>, Vec<CohortPatient>), String> {
    let clean = crate::web_io::strip_utf8_bom(content).trim();
    if clean.starts_with('[') {
        let rows: Vec<JsonPatientRow> =
            serde_json::from_str(clean).map_err(|e| format!("Invalid JSON cohort array: {e}"))?;
        let (mut trt, mut ctrl) = (Vec::new(), Vec::new());
        for (idx, r) in rows.iter().enumerate() {
            let ctx = format!("JSON row {idx}");
            let arm_str = r
                .arm
                .as_deref()
                .ok_or_else(|| format!("{ctx}: missing 'arm' field"))?;
            let is_trt = is_treatment_arm(arm_str, &ctx)?;
            let fe = parse_json_event(&r.fatal_event, &ctx)?;
            let ne = parse_json_event(&r.nonfatal_event, &ctx)?;
            let patient = build_patient(r.fatal_time, fe, r.nonfatal_time, ne, &ctx)?;
            push_bounded_arm(is_trt, patient, &mut trt, &mut ctrl)?;
        }
        validate_arm_counts(&trt, &ctrl)?;
        return Ok((trt, ctrl));
    }

    let grouped: JsonTwoArmCohort =
        serde_json::from_str(clean).map_err(|e| format!("Invalid JSON cohort object: {e}"))?;
    let (mut trt, mut ctrl) = (Vec::new(), Vec::new());
    for (idx, r) in grouped.treatment.iter().enumerate() {
        let ctx = format!("JSON treatment[{idx}]");
        let fe = parse_json_event(&r.fatal_event, &ctx)?;
        let ne = parse_json_event(&r.nonfatal_event, &ctx)?;
        push_bounded_arm(
            true,
            build_patient(r.fatal_time, fe, r.nonfatal_time, ne, &ctx)?,
            &mut trt,
            &mut ctrl,
        )?;
    }
    for (idx, r) in grouped.control.iter().enumerate() {
        let ctx = format!("JSON control[{idx}]");
        let fe = parse_json_event(&r.fatal_event, &ctx)?;
        let ne = parse_json_event(&r.nonfatal_event, &ctx)?;
        push_bounded_arm(
            false,
            build_patient(r.fatal_time, fe, r.nonfatal_time, ne, &ctx)?,
            &mut trt,
            &mut ctrl,
        )?;
    }
    validate_arm_counts(&trt, &ctrl)?;
    Ok((trt, ctrl))
}

/// Parses either `.json` or `.csv` local cohort text and evaluates all composite and secondary metrics.
pub fn load_and_analyze_cohort(
    source_name: &str,
    content: &str,
    sig_alpha: f64,
    horizon: Option<f64>,
) -> Result<LoadedCohortAnalysis, String> {
    if content.len() > MAX_COHORT_FILE_BYTES {
        return Err(format!(
            "Cohort file exceeds maximum size of {MAX_COHORT_FILE_BYTES} bytes"
        ));
    }
    let trimmed = crate::web_io::strip_utf8_bom(content).trim();
    if trimmed.is_empty() {
        return Err("Cohort file is empty".to_string());
    }
    let is_json = source_name.to_ascii_lowercase().ends_with(".json")
        || trimmed.starts_with('{')
        || trimmed.starts_with('[');
    let (treatment, control) = if is_json {
        parse_cohort_json(trimmed)?
    } else {
        parse_cohort_csv(trimmed)?
    };
    evaluate_loaded_cohort(
        source_name.to_string(),
        treatment,
        control,
        sig_alpha,
        horizon,
    )
}

/// Re-evaluates an already parsed two-arm cohort under updated significance $\alpha$ or censoring horizon.
pub fn evaluate_loaded_cohort(
    source_name: String,
    treatment: Vec<CohortPatient>,
    control: Vec<CohortPatient>,
    sig_alpha: f64,
    horizon: Option<f64>,
) -> Result<LoadedCohortAnalysis, String> {
    let cmp = match horizon {
        Some(h) => RightCensoredComparator::with_horizon(h).map_err(|e| e.to_string())?,
        None => RightCensoredComparator::new(),
    };
    let analysis = WinRatioAnalysis::new()
        .add_strategy(Box::new(cmp))
        .add_strategy(Box::new(cmp));
    let trt_tiers: Vec<[CensoredOutcome; 2]> =
        treatment.iter().map(|p| [p.fatal, p.nonfatal]).collect();
    let ctrl_tiers: Vec<[CensoredOutcome; 2]> =
        control.iter().map(|p| [p.fatal, p.nonfatal]).collect();
    let (summary, tier_vec) = analysis
        .evaluate_unmatched(&trt_tiers, &ctrl_tiers, sig_alpha)
        .ok_or_else(|| "Cohort produced zero untied comparisons across all tiers".to_string())?;
    let tiers = [
        tier_vec.first().copied().unwrap_or_default(),
        tier_vec.get(1).copied().unwrap_or_default(),
    ];

    let trt_fatal_obs: Vec<Observation> = treatment.iter().map(|p| p.fatal.into()).collect();
    let ctrl_fatal_obs: Vec<Observation> = control.iter().map(|p| p.fatal.into()).collect();
    let trt_km = kaplan_meier(&trt_fatal_obs);
    let ctrl_km = kaplan_meier(&ctrl_fatal_obs);
    let fatal_hazard_ratio =
        try_estimate_hazard_ratio(&trt_fatal_obs, &ctrl_fatal_obs).map_err(|e| e.to_string());

    let (risk_metrics, risk_difference) =
        compute_binary_composite_risk(&treatment, &control, sig_alpha, horizon);

    Ok(LoadedCohortAnalysis {
        source_name,
        treatment,
        control,
        summary,
        tiers,
        trt_km,
        ctrl_km,
        fatal_hazard_ratio,
        risk_metrics,
        risk_difference,
    })
}

fn compute_binary_composite_risk(
    treatment: &[CohortPatient],
    control: &[CohortPatient],
    sig_alpha: f64,
    horizon: Option<f64>,
) -> (Result<RiskMetrics, String>, f64) {
    let count_events = |arm: &[CohortPatient]| -> u32 {
        arm.iter()
            .filter(|p| p.fatal.censored_at(horizon).1 || p.nonfatal.censored_at(horizon).1)
            .count() as u32
    };
    let trt_ev = count_events(treatment);
    let ctrl_ev = count_events(control);
    let trt_n = treatment.len() as u32;
    let ctrl_n = control.len() as u32;
    let risk_diff =
        (f64::from(trt_ev) / f64::from(trt_n)) - (f64::from(ctrl_ev) / f64::from(ctrl_n));
    let metrics = ContingencyTable::new(trt_ev, trt_n - trt_ev, ctrl_ev, ctrl_n - ctrl_ev)
        .and_then(|tbl| calculate_risk_metrics(&tbl, sig_alpha))
        .map_err(|e| e.to_string());
    (metrics, risk_diff)
}

/// Built-in deterministic sample cardiovascular two-arm CSV cohort for instant interactive exploration.
pub const SAMPLE_COHORT_CSV: &str = "\
arm,fatal_time,fatal_event,nonfatal_time,nonfatal_event
treatment,4.8,0,4.8,0
treatment,4.6,0,3.2,1
treatment,4.9,0,4.9,0
treatment,3.9,1,2.4,1
treatment,4.7,0,4.1,1
treatment,5.0,0,5.0,0
treatment,4.2,0,2.9,1
treatment,4.9,0,4.9,0
control,1.4,1,0.8,1
control,2.1,1,1.2,1
control,4.5,0,1.6,1
control,2.8,1,1.9,1
control,4.4,0,2.2,1
control,3.3,1,1.5,1
control,4.7,0,4.7,0
control,1.9,1,1.1,1
";
