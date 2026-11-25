use super::types::{ClinicalTrialError, ContingencyTable};
use statrs::distribution::{ContinuousCDF, Normal};

/// Point estimates and Wald $(1 - \alpha)$ confidence intervals for $2 \times 2$ binary outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskMetrics {
    /// Treatment-to-control incidence ratio $\mathrm{RR} = [a/(a+b)] / [c/(c+d)]$.
    pub relative_risk: f64,
    /// Cross-product odds ratio $\mathrm{OR} = (ad) / (bc)$.
    pub odds_ratio: f64,
    /// Two-sided log-scale $(1 - \alpha)$ confidence interval `(lower, upper)` for `relative_risk`.
    pub rr_ci: (f64, f64),
    /// Two-sided Woolf log-scale $(1 - \alpha)$ confidence interval `(lower, upper)` for `odds_ratio`.
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

/// Prognostic probability calibration and discrimination summary for clinical risk models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrognosticCalibrationMetrics {
    /// Mean squared risk error $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$.
    pub brier_score: f64,
    /// Murphy reliability (calibration error) component.
    pub reliability: f64,
    /// Murphy resolution (case-separation variance) component.
    pub resolution: f64,
    /// Marginal event prevalence variance $\bar{y}(1 - \bar{y})$.
    pub uncertainty: f64,
    /// Symmetrically clipped binary dataset log-loss.
    pub log_loss: f64,
    /// Expected Calibration Error (`ECE`) across `num_bins` equal-width risk strata.
    pub ece: f64,
    /// Maximum Calibration Error (`MCE`) across non-empty risk strata.
    pub mce: f64,
    /// Concordance statistic (`ROC-AUC`) for prognostic discrimination.
    pub roc_auc: f64,
    /// Average-precision (`Step` `PR-AUC`) across risk thresholds.
    pub pr_auc: f64,
}

fn map_clinical_math_err(err: math_commons::error::MathError) -> ClinicalTrialError {
    ClinicalTrialError::InvalidData(err.to_string())
}

/// Evaluates proper scoring rules, Murphy decomposition, `ECE`/`MCE`, and `ROC`/`PR-AUC`
/// for prognostic patient event probabilities in `[0.0, 1.0]`.
#[verified_engine::verified]
pub fn evaluate_prognostic_calibration(
    predicted_risks: &[f64],
    events: &[bool],
    num_bins: usize,
) -> Result<PrognosticCalibrationMetrics, ClinicalTrialError> {
    use math_commons::primitives::UnitInterval;
    use pure_math::pure_math::statistics::calibration::{
        BinningStrategy, DEFAULT_LOG_LOSS_EPSILON, PrIntegration, binary_log_loss,
        calibration_summary, murphy_decomposition, pr_auc, roc_auc,
    };

    if predicted_risks.len() != events.len() {
        return Err(ClinicalTrialError::InvalidData(format!(
            "Risk count ({}) must match event count ({})",
            predicted_risks.len(),
            events.len()
        )));
    }
    if predicted_risks.is_empty() {
        return Err(ClinicalTrialError::InsufficientSampleSize {
            required: 2,
            actual: 0,
        });
    }

    let mut unit_risks = Vec::with_capacity(predicted_risks.len());
    for &r in predicted_risks {
        let ui = UnitInterval::new(r).map_err(map_clinical_math_err)?;
        unit_risks.push(ui);
    }

    let decomp = murphy_decomposition(&unit_risks, events).map_err(map_clinical_math_err)?;
    let nll = binary_log_loss(&unit_risks, events, DEFAULT_LOG_LOSS_EPSILON)
        .map_err(map_clinical_math_err)?;
    let summary = calibration_summary(&unit_risks, events, num_bins, BinningStrategy::Uniform)
        .map_err(map_clinical_math_err)?;
    let c_stat = roc_auc(predicted_risks, events).map_err(map_clinical_math_err)?;
    let ap = pr_auc(predicted_risks, events, PrIntegration::Step).map_err(map_clinical_math_err)?;

    Ok(PrognosticCalibrationMetrics {
        brier_score: decomp.brier_score,
        reliability: decomp.reliability,
        resolution: decomp.resolution,
        uncertainty: decomp.uncertainty,
        log_loss: nll,
        ece: summary.ece,
        mce: summary.mce,
        roc_auc: c_stat,
        pr_auc: ap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_evaluate_prognostic_calibration() {
        let risks = [0.08, 0.12, 0.25, 0.72, 0.88, 0.93];
        let events = [false, false, false, true, true, true];
        let metrics =
            evaluate_prognostic_calibration(&risks, &events, 3).expect("valid clinical risks");
        assert!(metrics.brier_score < 0.05);
        assert!((metrics.roc_auc - 1.0).abs() < 1e-9);
        assert!(
            (metrics.brier_score
                - (metrics.reliability - metrics.resolution + metrics.uncertainty))
                .abs()
                < math_commons::registry::TOLERANCE_STANDARD
        );
        assert!(evaluate_prognostic_calibration(&[1.2], &[true], 2).is_err());
    }
}
