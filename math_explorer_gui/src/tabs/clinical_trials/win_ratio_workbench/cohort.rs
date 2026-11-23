//! Cohort CSV/TSV parser, comparator configurations, and input validation for `WinRatioTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::web_io::strip_utf8_bom;
use math_commons::registry::MAX_ITERATIONS;
use math_explorer::applied::clinical_trials::design::Group;
use math_explorer::applied::win_ratio::bmi::calculate_bmi;

/// Maximum threshold magnitude permitted for [`ComparatorKind::Threshold`].
pub const MAX_THRESHOLD: f64 = 10_000.0;
/// Minimum Weibull scale parameter (`lambda1`, `lambda2`).
pub const MIN_WEIBULL_LAMBDA: f64 = 0.01;
/// Maximum Weibull scale parameter (`lambda1`, `lambda2`).
pub const MAX_WEIBULL_LAMBDA: f64 = 2.0;
/// Minimum Weibull correlation shape parameter (`alpha`).
pub const MIN_WEIBULL_ALPHA: f64 = 0.10;
/// Maximum Weibull correlation shape parameter (`alpha <= 1.0`).
pub const MAX_WEIBULL_ALPHA: f64 = 1.0;
/// Minimum treatment hazard multiplier (`theta`).
pub const MIN_WEIBULL_THETA: f64 = 0.10;
/// Maximum treatment hazard multiplier (`theta`).
pub const MAX_WEIBULL_THETA: f64 = 5.0;
/// Minimum censoring horizon `c_max` for `PR(c)` curves.
pub const MIN_HORIZON_C: f64 = 1.0;
/// Maximum censoring horizon `c_max` for `PR(c)` curves.
pub const MAX_HORIZON_C: f64 = 50.0;
/// Minimum sample count along `c in [c_min, c_max]`.
pub const MIN_PRC_SAMPLES: usize = 16;
/// Default sample count along `c in [c_min, c_max]`.
pub const DEFAULT_PRC_SAMPLES: usize = 48;
/// Maximum sample count along `c in [c_min, c_max]`.
pub const MAX_PRC_SAMPLES: usize = 128;

/// Outcome comparator strategy for a single composite hierarchy tier.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ComparatorKind {
    /// Higher value wins (`HigherIsBetter`, e.g., survival days or quality-of-life score).
    HigherIsBetter,
    /// Lower value wins (`LowerIsBetter`, e.g., hospitalization count or adverse severity).
    LowerIsBetter,
    /// Higher value wins strictly when `a - b > threshold` (`ThresholdComparator`).
    Threshold {
        /// Minimum clinically important difference required to declare a win or loss.
        threshold: f64,
    },
}

impl ComparatorKind {
    /// Human-readable label for UI selectors and report cards.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::HigherIsBetter => "Higher is Better".to_string(),
            Self::LowerIsBetter => "Lower is Better".to_string(),
            Self::Threshold { threshold } => format!("Threshold (Δ > {threshold:.2})"),
        }
    }

    /// Checks finiteness and non-negative bounds for threshold comparators.
    #[must_use]
    pub fn is_valid(self) -> bool {
        match self {
            Self::HigherIsBetter | Self::LowerIsBetter => true,
            Self::Threshold { threshold } => {
                threshold.is_finite() && (0.0..=MAX_THRESHOLD).contains(&threshold)
            }
        }
    }
}

/// Subject pairing scheme between `Treatment` and `Control` arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PairingMode {
    /// All-pairs Cartesian comparison (`UnmatchedPairing`).
    Unmatched,
    /// 1:1 index-matched pair comparison (`MatchedPairing`, requires equal arm sizes).
    Matched,
}

impl PairingMode {
    /// Human-readable pairing label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unmatched => "Unmatched (All-Pairs)",
            Self::Matched => "Matched Pairs (1:1)",
        }
    }
}

/// Active plot visualization mode in `WinRatioTool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WinRatioPlotMode {
    /// Parametric `BivariateWeibullModel` survival and `ProbabilityWinRatioContext` `PR(c)` curves.
    WeibullPrc,
    /// Empirical composite tier cumulative Win Ratio and win/loss attribution profile.
    EmpiricalHierarchy,
}

/// Parsed two-arm composite cohort ready for `WinRatioAnalysis` and `TTestIndependent`.
#[derive(Debug, Clone, Default)]
pub struct ParsedCompositeCohort {
    /// Per-subject prioritized outcomes `[tier1, tier2, tier3]` for the Treatment arm.
    pub treatment: Vec<Vec<f64>>,
    /// Per-subject prioritized outcomes `[tier1, tier2, tier3]` for the Control arm.
    pub control: Vec<Vec<f64>>,
    /// Derived BMI values (`kg/m^2`) when 5-column `(arm, t1, t2, weight_kg, height_m)` rows are supplied.
    pub derived_bmis: Vec<(Group, f64)>,
}

/// Default 12-patient cardiovascular composite trial cohort (`Arm, SurvivalDays, HospCount, QoLScore`).
pub const DEFAULT_COHORT_CSV: &str = "\
Treatment, 920.0, 0.0, 78.0
Treatment, 880.0, 1.0, 74.0
Treatment, 760.0, 1.0, 69.0
Treatment, 920.0, 0.0, 82.0
Treatment, 640.0, 2.0, 63.0
Treatment, 920.0, 1.0, 76.0
Control, 410.0, 3.0, 54.0
Control, 530.0, 2.0, 58.0
Control, 920.0, 2.0, 65.0
Control, 620.0, 2.0, 60.0
Control, 920.0, 1.0, 68.0
Control, 480.0, 3.0, 52.0";

fn parse_arm_token(raw: &str, line_no: usize) -> Result<Group, String> {
    let token = raw.trim();
    if token.eq_ignore_ascii_case("treatment")
        || token.eq_ignore_ascii_case("treat")
        || token.eq_ignore_ascii_case("tx")
        || token.eq_ignore_ascii_case("t")
        || token == "1"
    {
        Ok(Group::Treatment)
    } else if token.eq_ignore_ascii_case("control")
        || token.eq_ignore_ascii_case("ctrl")
        || token.eq_ignore_ascii_case("placebo")
        || token.eq_ignore_ascii_case("c")
        || token == "0"
    {
        Ok(Group::Control)
    } else {
        Err(format!(
            "Line {line_no}: Unknown arm '{token}' (expected Treatment or Control)"
        ))
    }
}

fn is_header_row(fields: &[&str]) -> bool {
    if fields.len() < 2 {
        return false;
    }
    let f0 = fields[0].trim();
    let f1 = fields[1].trim();
    f1.parse::<f64>().is_err()
        && (f0.eq_ignore_ascii_case("arm")
            || f0.eq_ignore_ascii_case("group")
            || f1.eq_ignore_ascii_case("tier1")
            || f1.eq_ignore_ascii_case("survival"))
}

fn parse_finite_f64(raw: &str, line_no: usize, label: &str) -> Result<f64, String> {
    let val = raw
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("Line {line_no}: Invalid {label} '{}'", raw.trim()))?;
    if !val.is_finite() {
        return Err(format!("Line {line_no}: Non-finite {label} '{val}'"));
    }
    Ok(val)
}

fn parse_cohort_row(
    fields: &[&str],
    line_no: usize,
    derive_bmi_tier3: bool,
) -> Result<(Group, Vec<f64>, Option<f64>), String> {
    if fields.len() < 3 || fields.len() > 5 {
        return Err(format!(
            "Line {line_no}: Expected 3 to 5 columns (Arm, Tier1, Tier2[, Tier3 | WeightKg, HeightM])"
        ));
    }
    let group = parse_arm_token(fields[0], line_no)?;
    let t1 = parse_finite_f64(fields[1], line_no, "Tier 1 value")?;
    let t2 = parse_finite_f64(fields[2], line_no, "Tier 2 value")?;

    if fields.len() == 5 || (derive_bmi_tier3 && fields.len() == 4) {
        if fields.len() < 5 {
            return Err(format!(
                "Line {line_no}: BMI derivation requires 5 columns (Arm, Tier1, Tier2, WeightKg, HeightM)"
            ));
        }
        let weight_kg = parse_finite_f64(fields[3], line_no, "weight_kg")?;
        let height_m = parse_finite_f64(fields[4], line_no, "height_m")?;
        if weight_kg <= 0.0 || height_m <= 0.0 {
            return Err(format!(
                "Line {line_no}: Weight and height must be strictly positive for BMI"
            ));
        }
        let bmi = calculate_bmi(weight_kg, height_m);
        return Ok((group, vec![t1, t2, bmi], Some(bmi)));
    }

    let mut outcomes = vec![t1, t2];
    if fields.len() == 4 {
        outcomes.push(parse_finite_f64(fields[3], line_no, "Tier 3 value")?);
    }
    Ok((group, outcomes, None))
}

/// Parses a comma- or tab-delimited composite cohort string, enforcing `MAX_ITERATIONS` (`5000`).
pub fn parse_composite_cohort(
    raw_text: &str,
    derive_bmi_tier3: bool,
) -> Result<ParsedCompositeCohort, String> {
    let cleaned = strip_utf8_bom(raw_text);
    let mut cohort = ParsedCompositeCohort::default();
    let mut total_rows = 0usize;
    let mut expected_cols: Option<usize> = None;
    let mut saw_data_row = false;

    for (idx, raw_line) in cleaned.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let delim = if line.contains('\t') { '\t' } else { ',' };
        let fields: Vec<&str> = line.split(delim).collect();
        if !saw_data_row && is_header_row(&fields) {
            saw_data_row = true;
            continue;
        }
        saw_data_row = true;
        if total_rows >= MAX_ITERATIONS {
            return Err(format!(
                "Line {}: Maximum of {} cohort rows exceeded",
                idx + 1,
                MAX_ITERATIONS
            ));
        }
        let (group, outcomes, bmi_opt) = parse_cohort_row(&fields, idx + 1, derive_bmi_tier3)?;
        if let Some(cols) = expected_cols {
            if outcomes.len() != cols {
                return Err(format!(
                    "Line {}: Inconsistent endpoint tier count (expected {}, got {})",
                    idx + 1,
                    cols,
                    outcomes.len()
                ));
            }
        } else {
            expected_cols = Some(outcomes.len());
        }
        if let Some(bmi) = bmi_opt {
            cohort.derived_bmis.push((group, bmi));
        }
        match group {
            Group::Treatment => cohort.treatment.push(outcomes),
            Group::Control => cohort.control.push(outcomes),
        }
        total_rows += 1;
    }

    if cohort.treatment.is_empty() || cohort.control.is_empty() {
        return Err("Cohort must include at least 1 Treatment and 1 Control subject".into());
    }
    Ok(cohort)
}

/// Starter preset JSON payload for composite empirical hierarchy analysis.
pub const COMPOSITE_STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Composite Endpoint Win Ratio",
    "domain": "clinical_trials",
    "timestamp": "2026-11-20T00:00:00Z"
  },
  "config": {
    "weibull_theta": 0.65,
    "weibull_lambda1": 0.10,
    "weibull_lambda2": 0.22,
    "weibull_alpha": 0.80,
    "horizon_c": 12.0,
    "prc_samples": 48,
    "pairing_mode": "Unmatched",
    "tier1_kind": "HigherIsBetter",
    "tier2_kind": "LowerIsBetter",
    "tier3_kind": { "Threshold": { "threshold": 5.0 } },
    "derive_bmi_tier3": false,
    "ttest_tier_index": 2,
    "ttest_alpha": 0.05,
    "treat_weight_kg": 73.0,
    "treat_height_m": 1.76,
    "ctrl_weight_kg": 81.0,
    "ctrl_height_m": 1.74,
    "plot_mode": "EmpiricalHierarchy",
    "cohort_text": "Treatment, 950.0, 0.0, 80.0\nTreatment, 890.0, 1.0, 75.0\nTreatment, 780.0, 1.0, 71.0\nTreatment, 950.0, 0.0, 84.0\nControl, 420.0, 3.0, 55.0\nControl, 540.0, 2.0, 59.0\nControl, 950.0, 2.0, 64.0\nControl, 610.0, 2.0, 61.0"
  }
}"#;

/// Starter preset JSON payload for parametric `BivariateWeibullModel` `PR(c)` exploration.
pub const WEIBULL_PRC_STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bivariate Weibull PR(c) Integral",
    "domain": "clinical_trials",
    "timestamp": "2026-11-20T00:00:00Z"
  },
  "config": {
    "weibull_theta": 0.55,
    "weibull_lambda1": 0.12,
    "weibull_lambda2": 0.25,
    "weibull_alpha": 0.75,
    "horizon_c": 16.0,
    "prc_samples": 64,
    "pairing_mode": "Unmatched",
    "tier1_kind": "HigherIsBetter",
    "tier2_kind": "LowerIsBetter",
    "tier3_kind": { "Threshold": { "threshold": 4.0 } },
    "derive_bmi_tier3": false,
    "ttest_tier_index": 2,
    "ttest_alpha": 0.05,
    "treat_weight_kg": 74.0,
    "treat_height_m": 1.75,
    "ctrl_weight_kg": 82.0,
    "ctrl_height_m": 1.74,
    "plot_mode": "WeibullPrc",
    "cohort_text": "Treatment, 920.0, 0.0, 78.0\nTreatment, 880.0, 1.0, 74.0\nTreatment, 760.0, 1.0, 69.0\nControl, 410.0, 3.0, 54.0\nControl, 530.0, 2.0, 58.0\nControl, 920.0, 2.0, 65.0"
  }
}"#;

// [cite:win_ratio]
// [cite:clinical_trials_statistics]
