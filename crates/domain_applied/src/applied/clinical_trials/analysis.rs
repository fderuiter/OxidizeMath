use super::types::{ClinicalTrialError, ContingencyTable};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct RiskMetrics {
    #[allow(missing_docs)]
    pub relative_risk: f64,
    #[allow(missing_docs)]
    pub odds_ratio: f64,
    #[allow(missing_docs)]
    pub rr_ci: (f64, f64),
    #[allow(missing_docs)]
    pub or_ci: (f64, f64),
}

/// Calculates Relative Risk (RR) and Odds Ratio (OR) with Confidence Intervals.
///
/// Uses the provided `ContingencyTable` which encapsulates the 2x2 matrix.
///
/// # Arguments
/// * `table` - The 2x2 contingency table.
/// * `alpha` - Significance level (e.g., 0.05 for 95% CI).
#[verified_engine::verified]
pub fn calculate_risk_metrics(
    table: &ContingencyTable,
    alpha: f64,
) -> Result<RiskMetrics, ClinicalTrialError> {
    let a = table.treatment_event as f64;
    let b = table.treatment_no_event as f64;
    let c = table.control_event as f64;
    let d = table.control_no_event as f64;

    // Check for zeros to avoid NaN/Infinity
    if a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Cell counts must be non-zero for simple RR/OR calculation.".to_string(),
        ));
    }

    // Relative Risk
    let risk_treatment = a / (a + b);
    let risk_control = c / (c + d);
    let rr = risk_treatment / risk_control;

    // Odds Ratio
    let odds_treatment = a / b;
    let odds_control = c / d;
    let or = odds_treatment / odds_control;

    // Confidence Intervals (using Normal distribution for log-transformed RR/OR)
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let z = normal.inverse_cdf(1.0 - alpha / 2.0);

    // SE for ln(RR)
    let se_ln_rr = ((1.0 / a) - (1.0 / (a + b)) + (1.0 / c) - (1.0 / (c + d))).sqrt();
    let ln_rr = rr.ln();
    let rr_lower = (ln_rr - z * se_ln_rr).exp();
    let rr_upper = (ln_rr + z * se_ln_rr).exp();

    // SE for ln(OR)
    let se_ln_or = ((1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d)).sqrt();
    let ln_or = or.ln();
    let or_lower = (ln_or - z * se_ln_or).exp();
    let or_upper = (ln_or + z * se_ln_or).exp();

    Ok(RiskMetrics {
        relative_risk: rr,
        odds_ratio: or,
        rr_ci: (rr_lower, rr_upper),
        or_ci: (or_lower, or_upper),
    })
}
