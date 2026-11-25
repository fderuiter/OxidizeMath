//! Normal-approximation confidence intervals and hypothesis-test helpers for Win Ratio counts.

use statrs::distribution::{ContinuousCDF, Normal};

/// Calculates the raw Win Ratio `WR = N_wins / N_losses` (`f64::INFINITY` if `losses == 0`).
#[verified_engine::verified]
pub fn calculate_win_ratio(wins: i32, losses: i32) -> f64 {
    if losses == 0 {
        f64::INFINITY
    } else {
        wins as f64 / losses as f64
    }
}

/// Statistical results for a Win Ratio analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WinRatioStats {
    /// The calculated win ratio ($N_W / N_L$).
    pub win_ratio: f64,
    /// Lower bound of the 95% Confidence Interval.
    pub ci_low: f64,
    /// Upper bound of the 95% Confidence Interval.
    pub ci_high: f64,
    /// Two-sided P-value testing the null hypothesis that $WR = 1$.
    pub p_value: f64,
}

/// Fallibly calculates 95% Confidence Intervals and P-values for the Win Ratio, rejecting negative counts.
#[verified_engine::verified]
pub fn try_calculate_statistics(
    wins: i32,
    losses: i32,
) -> Result<Option<WinRatioStats>, math_commons::error::MathError> {
    if wins < 0 || losses < 0 {
        return Err(math_commons::error::MathError::OutOfBounds {
            value: f64::from(wins.min(losses)),
            min: 0.0,
            max: f64::INFINITY,
        });
    }
    Ok(calculate_statistics(wins, losses))
}

/// Calculates 95% Confidence Intervals and P-values for the Win Ratio (`None` if total pairs is zero).
///
/// Uses the normal approximation with standard error $SE = \sqrt{p(1-p)/N}$ where $p$ is the win proportion.
#[verified_engine::verified]
pub fn calculate_statistics(wins: i32, losses: i32) -> Option<WinRatioStats> {
    let total_pairs = wins.checked_add(losses)?;
    if wins < 0 || losses < 0 || total_pairs <= 0 {
        return None;
    }

    let p_win = wins as f64 / total_pairs as f64;
    let se = (p_win * (1.0 - p_win) / total_pairs as f64).sqrt();

    if se == 0.0 {
        let win_ratio = calculate_win_ratio(wins, losses);
        return Some(WinRatioStats {
            win_ratio,
            ci_low: if p_win == 1.0 { win_ratio } else { 0.0 },
            ci_high: if p_win == 1.0 {
                f64::INFINITY
            } else {
                win_ratio
            },
            p_value: 0.0,
        });
    }

    let ci_low_p = (p_win - 1.96 * se).max(0.0);
    let ci_high_p = (p_win + 1.96 * se).min(1.0);

    let ci_low = if ci_low_p == 1.0 {
        f64::INFINITY
    } else {
        ci_low_p / (1.0 - ci_low_p)
    };

    let ci_high = if ci_high_p == 1.0 {
        f64::INFINITY
    } else {
        ci_high_p / (1.0 - ci_high_p)
    };

    let z_score = (p_win - 0.5) / se;
    let normal = Normal::new(0.0, 1.0).ok()?;
    let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));

    Some(WinRatioStats {
        win_ratio: calculate_win_ratio(wins, losses),
        ci_low,
        ci_high,
        p_value,
    })
}
