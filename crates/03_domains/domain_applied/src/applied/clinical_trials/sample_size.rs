use super::types::ClinicalTrialError;
use statrs::distribution::{ContinuousCDF, Normal};

/// Calculates the required sample size per group for a two-sample t-test (comparing means).
/// Based on the formula: n = 2 * sigma^2 * (Z_{1-alpha/2} + Z_{power})^2 / delta^2
///
/// # Arguments
/// * `alpha` - Type I error rate (e.g., 0.05).
/// * `power` - Statistical power (1 - beta) (e.g., 0.80 or 0.90).
/// * `delta` - The minimum difference the researchers hope to see (Effect Size).
/// * `sigma` - Standard deviation in the population.
///
/// Returns the number of patients required *per group*.
#[verified_engine::verified]
pub fn calculate_sample_size_means(
    alpha: f64,
    power: f64,
    delta: f64,
    sigma: f64,
) -> Result<usize, ClinicalTrialError> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Alpha must be between 0 and 1".to_string(),
        ));
    }
    if power <= 0.0 || power >= 1.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Power must be between 0 and 1".to_string(),
        ));
    }
    if delta == 0.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Effect size (delta) cannot be zero".to_string(),
        ));
    }

    let normal =
        Normal::new(0.0, 1.0).map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    // Z value for alpha (two-tailed)
    let z_alpha = normal.inverse_cdf(1.0 - alpha / 2.0);

    // Z value for beta (one-tailed for power)
    // Power = 1 - beta. We need Z_{1-beta} which is inverse_cdf(power)
    let z_beta = normal.inverse_cdf(power);

    let numerator = 2.0 * sigma.powi(2) * (z_alpha + z_beta).powi(2);
    let denominator = delta.powi(2);

    let n = numerator / denominator;
    Ok(n.ceil() as usize)
}

/// Calculates the required sample size per group for comparing two proportions (Chi-square/Z-test).
/// Based on the formula: n = (p1(1-p1) + p2(1-p2)) * (Z_{1-alpha/2} + Z_{power})^2 / (p1 - p2)^2
///
/// # Arguments
/// * `alpha` - Type I error rate.
/// * `power` - Statistical power.
/// * `p1` - Expected proportion in group 1 (e.g., control).
/// * `p2` - Expected proportion in group 2 (e.g., treatment).
#[verified_engine::verified]
pub fn calculate_sample_size_proportions(
    alpha: f64,
    power: f64,
    p1: f64,
    p2: f64,
) -> Result<usize, ClinicalTrialError> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Alpha must be between 0 and 1".to_string(),
        ));
    }
    if power <= 0.0 || power >= 1.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Power must be between 0 and 1".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&p1) || !(0.0..=1.0).contains(&p2) {
        return Err(ClinicalTrialError::InvalidData(
            "Proportions must be between 0 and 1".to_string(),
        ));
    }
    if (p1 - p2).abs() < math_commons::registry::TOLERANCE_STANDARD {
        return Err(ClinicalTrialError::InvalidData(
            "Proportions cannot be equal".to_string(),
        ));
    }

    let normal =
        Normal::new(0.0, 1.0).map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;

    let z_alpha = normal.inverse_cdf(1.0 - alpha / 2.0);
    let z_beta = normal.inverse_cdf(power);

    let variance_term = p1 * (1.0 - p1) + p2 * (1.0 - p2);
    let z_term = (z_alpha + z_beta).powi(2);
    let effect_sq = (p1 - p2).powi(2);

    let n = (variance_term * z_term) / effect_sq;
    Ok(n.ceil() as usize)
}

/// Calculates the achieved two-sided statistical power for a two-sample t-test (comparing means)
/// at a given per-group sample size `n_per_group`.
#[verified_engine::verified]
pub fn calculate_power_means(
    alpha: f64,
    n_per_group: usize,
    delta: f64,
    sigma: f64,
) -> Result<f64, ClinicalTrialError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Alpha must be between 0 and 1".to_string(),
        ));
    }
    if n_per_group == 0 {
        return Err(ClinicalTrialError::InvalidData(
            "Sample size per group must be positive".to_string(),
        ));
    }
    if !delta.is_finite() || delta == 0.0 || !sigma.is_finite() || sigma <= 0.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Effect size must be non-zero finite and sigma must be positive finite".to_string(),
        ));
    }
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let z_alpha = normal.inverse_cdf(1.0 - alpha / 2.0);
    let noncentrality = delta.abs() * (n_per_group as f64 / (2.0 * sigma * sigma)).sqrt();
    let power = normal.cdf(noncentrality - z_alpha) + normal.cdf(-noncentrality - z_alpha);
    Ok(power.clamp(0.0, 1.0))
}

/// Calculates the achieved two-sided statistical power for comparing two proportions
/// at a given per-group sample size `n_per_group`.
#[verified_engine::verified]
pub fn calculate_power_proportions(
    alpha: f64,
    n_per_group: usize,
    p1: f64,
    p2: f64,
) -> Result<f64, ClinicalTrialError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Alpha must be between 0 and 1".to_string(),
        ));
    }
    if n_per_group == 0 {
        return Err(ClinicalTrialError::InvalidData(
            "Sample size per group must be positive".to_string(),
        ));
    }
    if !p1.is_finite()
        || !p2.is_finite()
        || !(0.0..=1.0).contains(&p1)
        || !(0.0..=1.0).contains(&p2)
    {
        return Err(ClinicalTrialError::InvalidData(
            "Proportions must be between 0 and 1".to_string(),
        ));
    }
    let variance_term = p1 * (1.0 - p1) + p2 * (1.0 - p2);
    if (p1 - p2).abs() < math_commons::registry::TOLERANCE_STANDARD || variance_term <= 0.0 {
        return Err(ClinicalTrialError::InvalidData(
            "Proportions must differ and have positive variance".to_string(),
        ));
    }
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| ClinicalTrialError::StatisticalError(e.to_string()))?;
    let z_alpha = normal.inverse_cdf(1.0 - alpha / 2.0);
    let noncentrality = (p1 - p2).abs() * ((n_per_group as f64) / variance_term).sqrt();
    let power = normal.cdf(noncentrality - z_alpha) + normal.cdf(-noncentrality - z_alpha);
    Ok(power.clamp(0.0, 1.0))
}
