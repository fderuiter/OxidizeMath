use crate::error::HighEnergyError;

/// Calculates the Li & Ma Significance (sigma).
///
/// # Arguments
/// * `n_on` - Counts on source.
/// * `n_off` - Background counts.
/// * `alpha` - Ratio of exposure times (t_on / t_off).
#[verified_engine::verified]
pub fn li_ma_significance(n_on: f64, n_off: f64, alpha: f64) -> Result<f64, HighEnergyError> {
    if n_on < 0.0 || n_off < 0.0 || alpha <= 0.0 {
        return Err(HighEnergyError::InvalidStatisticsParams {
            reason: "Counts must be non-negative and alpha positive".to_string(),
        });
    }

    let term1 = if n_on > 0.0 {
        let ratio = (1.0 + alpha) / alpha * (n_on / (n_on + n_off));
        n_on * ratio.ln()
    } else {
        0.0
    };

    let term2 = if n_off > 0.0 {
        let ratio = (1.0 + alpha) * (n_off / (n_on + n_off));
        n_off * ratio.ln()
    } else {
        0.0
    };

    let sum = term1 + term2;
    if sum < 0.0 {
        // Should not happen for valid inputs where n_on/n_off reflect an excess,
        // but numerically possible or if deficit.
        // Formula has sqrt.
        return Err(HighEnergyError::Math(
            math_commons::error::MathError::NumericalError {
                reason: "Negative argument for sqrt in Li & Ma",
            },
        ));
    }

    Ok(2.0f64.sqrt() * sum.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_statistics_li_ma() {
        // Example: Non=10, Noff=10, alpha=1.
        // term1 = 10 * ln(2 * 10/20) = 10 * ln(1) = 0.
        // term2 = 10 * ln(2 * 10/20) = 0.
        // S = 0.
        let s = li_ma_significance(10.0, 10.0, 1.0).unwrap();
        assert_relative_eq!(s, 0.0);

        // Example: Non=20, Noff=10, alpha=1.
        // term1: 20 * ln(2 * 20/30) = 20 * ln(4/3) = 20 * 0.28768 = 5.75
        // term2: 10 * ln(2 * 10/30) = 10 * ln(2/3) = 10 * -0.4054 = -4.05
        // sum = 1.70. S = sqrt(2 * 1.70) = sqrt(3.4) ~ 1.84.
        let s2 = li_ma_significance(20.0, 10.0, 1.0).unwrap();
        assert!(s2 > 1.8 && s2 < 1.9);
    }
}
