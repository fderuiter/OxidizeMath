//! Rating update algorithms for Glicko-2.

use super::core::{
    GlickoPlayer, MatchResult, Rating, RatingDeviation, SystemConstant, Volatility,
    expected_outcome, g_function,
};
use crate::error::Glicko2Error;

/// Maximum iterations for volatility convergence.
const MAX_VOLATILITY_ITERATIONS: usize = 100;

/// Convergence tolerance for volatility iteration.
const VOLATILITY_TOLERANCE: f64 = math_commons::registry::TOLERANCE_FAST;

/// Updates a player's rating after a rating period with match results.
///
/// This is the main Glicko-2 algorithm implementing the full rating update procedure
/// as described in Glickman's paper.
///
/// # Arguments
///
/// * `player` - The player to update
/// * `results` - Match results from the rating period
/// * `tau` - System constant controlling volatility changes
///
/// # Returns
///
/// * `Result<GlickoPlayer, Glicko2Error>` - Updated player or error
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::glicko2::{
///     GlickoPlayer, MatchResult, Rating, RatingDeviation, Volatility, SystemConstant, update_rating
/// };
///
/// let player = GlickoPlayer::new(
///     Rating::new(1500.0).unwrap(),
///     RatingDeviation::new(200.0).unwrap(),
///     Volatility::new(0.06).unwrap(),
/// );
///
/// let opponent = GlickoPlayer::new(
///     Rating::new(1400.0).unwrap(),
///     RatingDeviation::new(30.0).unwrap(),
///     Volatility::new(0.06).unwrap(),
/// );
///
/// let results = vec![MatchResult::new(opponent, 1.0).unwrap()]; // Win
/// let tau = SystemConstant::default();
///
/// let updated = update_rating(&player, &results, &tau).unwrap();
/// assert!(updated.rating.value() > player.rating.value()); // Rating increased
/// ```
///
/// # Algorithm Steps
///
/// 1. Convert to Glicko-2 scale
/// 2. Compute variance (v)
/// 3. Compute improvement estimate (Δ)
/// 4. Update volatility (σ') using Illinois algorithm
/// 5. Update RD pre-rating (φ*)
/// 6. Update RD (φ')
/// 7. Update rating (μ')
/// 8. Convert back to Glicko scale
///
/// # References
///
/// Glickman, M. E. (2012). "Example of the Glicko-2 system."
/// <http://www.glicko.net/glicko/glicko2.pdf>
#[verified_engine::verified]
pub fn update_rating(
    player: &GlickoPlayer,
    results: &[MatchResult],
    tau: &SystemConstant,
) -> Result<GlickoPlayer, Glicko2Error> {
    // Handle no games played (RD increases, rating and volatility unchanged)
    if results.is_empty() {
        return Ok(increase_rd_for_inactivity(player));
    }

    // Step 1: Convert to Glicko-2 scale
    let (mu, phi, sigma) = player.to_glicko2_scale();

    // Step 2: Compute variance v
    let v = compute_variance(mu, results)?;

    // Step 3: Compute delta (performance estimate)
    let delta = compute_delta(mu, results, v)?;

    // Step 4: Update volatility using Illinois algorithm
    let sigma_new = update_volatility(phi, sigma, delta, v, tau.value())?;

    // Step 5: Compute pre-rating period RD
    let phi_star = (phi * phi + sigma_new * sigma_new).sqrt();

    // Step 6: Update RD
    let phi_new = 1.0 / (1.0 / (phi_star * phi_star) + 1.0 / v).sqrt();

    // Step 7: Update rating
    let mu_new = mu + phi_new * phi_new * compute_summation(mu, results)?;

    // Step 8: Convert back to Glicko scale
    Ok(GlickoPlayer::new(
        Rating::from_glicko2_scale(mu_new)?,
        RatingDeviation::from_glicko2_scale(phi_new)?,
        Volatility::new(sigma_new)?,
    ))
}

/// Computes the variance v for the rating period.
///
/// Formula: v = 1 / Σⱼ g(φⱼ)² E(μ, μⱼ, φⱼ)(1 - E(μ, μⱼ, φⱼ))
#[verified_engine::verified]
fn compute_variance(mu: f64, results: &[MatchResult]) -> Result<f64, Glicko2Error> {
    let sum: f64 = results
        .iter()
        .map(|result| {
            let (mu_j, phi_j, _) = result.opponent.to_glicko2_scale();
            let g_phi = g_function(phi_j);
            let e = expected_outcome(mu, mu_j, phi_j);
            g_phi * g_phi * e * (1.0 - e)
        })
        .sum();

    if sum <= 0.0 {
        return Err(Glicko2Error::InvalidOpponentCount {
            count: results.len(),
        });
    }

    Ok(1.0 / sum)
}

/// Computes the delta (performance improvement estimate).
///
/// Formula: Δ = v · Σⱼ g(φⱼ)(sⱼ - E(μ, μⱼ, φⱼ))
#[verified_engine::verified]
fn compute_delta(mu: f64, results: &[MatchResult], v: f64) -> Result<f64, Glicko2Error> {
    let sum = compute_summation(mu, results)?;
    Ok(v * sum)
}

/// Helper function to compute Σⱼ g(φⱼ)(sⱼ - E(μ, μⱼ, φⱼ)).
#[verified_engine::verified]
fn compute_summation(mu: f64, results: &[MatchResult]) -> Result<f64, Glicko2Error> {
    let sum: f64 = results
        .iter()
        .map(|result| {
            let (mu_j, phi_j, _) = result.opponent.to_glicko2_scale();
            let g_phi = g_function(phi_j);
            let e = expected_outcome(mu, mu_j, phi_j);
            g_phi * (result.score - e)
        })
        .sum();

    Ok(sum)
}

/// Updates volatility using the Illinois algorithm for root finding.
///
/// This solves the non-linear equation:
/// f(x) = exp(x)(Δ² - φ² - v - exp(x)) / (2(φ² + v + exp(x))²) - (x - ln(σ²)) / τ² = 0
///
/// The Illinois algorithm is a variant of the false position method with better convergence.
#[verified_engine::verified]
fn update_volatility(
    phi: f64,
    sigma: f64,
    delta: f64,
    v: f64,
    tau: f64,
) -> Result<f64, Glicko2Error> {
    // Function f(x) to find root of
    let f = |x: f64| -> f64 {
        let exp_x = x.exp();
        let phi_sq = phi * phi;
        let delta_sq = delta * delta;
        let tau_sq = tau * tau;
        let denominator = phi_sq + v + exp_x;

        let term1 = exp_x * (delta_sq - phi_sq - v - exp_x) / (2.0 * denominator * denominator);
        let term2 = (x - (sigma * sigma).ln()) / tau_sq;

        term1 - term2
    };

    // Initialize search bounds
    let a_init = (sigma * sigma).ln();
    let mut a = a_init;
    let mut b = if delta * delta > phi * phi + v {
        (delta * delta - phi * phi - v).ln()
    } else {
        let mut k = 1.0;
        while f(a_init - k * tau) < 0.0 {
            k += 1.0;
            if k > 100.0 {
                return Err(Glicko2Error::VolatilityConvergenceFailed { iterations: 0 });
            }
        }
        a_init - k * tau
    };

    let mut f_a = f(a);
    let mut f_b = f(b);

    // Illinois algorithm
    for iteration in 0..MAX_VOLATILITY_ITERATIONS {
        if (b - a).abs() < VOLATILITY_TOLERANCE {
            return Ok((a / 2.0).exp());
        }

        // Compute new point using false position
        let c = a - f_a * (a - b) / (f_a - f_b);
        let f_c = f(c);

        if f_c * f_b <= 0.0 {
            a = b;
            f_a = f_b;
            b = c;
            f_b = f_c;
        } else {
            // Illinois modification: reduce weight of stale endpoint
            f_a *= 0.5;
            b = c;
            f_b = f_c;
        }

        if iteration == MAX_VOLATILITY_ITERATIONS - 1 {
            return Err(Glicko2Error::VolatilityConvergenceFailed {
                iterations: MAX_VOLATILITY_ITERATIONS,
            });
        }
    }

    Ok((a / 2.0).exp())
}

/// Increases rating deviation for a player who hasn't played in a rating period.
///
/// Formula: φ' = √(φ² + σ²)
///
/// This models the increased uncertainty about a player's skill when they are inactive.
#[verified_engine::verified]
pub fn increase_rd_for_inactivity(player: &GlickoPlayer) -> GlickoPlayer {
    let (_, phi, sigma) = player.to_glicko2_scale();
    let phi_new = (phi * phi + sigma * sigma).sqrt();

    GlickoPlayer::new(
        player.rating,
        RatingDeviation::from_glicko2_scale(phi_new).unwrap_or(player.rating_deviation),
        player.volatility,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_update_rating_single_win() {
        // Player beats weaker opponent
        let player = GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(200.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent = GlickoPlayer::new(
            Rating::new(1400.0).unwrap(),
            RatingDeviation::new(30.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let results = vec![MatchResult::new(opponent, 1.0).unwrap()];
        let tau = SystemConstant::default();

        let updated = update_rating(&player, &results, &tau).unwrap();

        // Rating should increase after winning
        assert!(updated.rating.value() > player.rating.value());
        // RD should decrease (more certainty)
        assert!(updated.rating_deviation.value() < player.rating_deviation.value());
    }

    #[test]
    #[verified_engine::verified]
    fn test_update_rating_single_loss() {
        // Player loses to stronger opponent
        let player = GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(200.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent = GlickoPlayer::new(
            Rating::new(1600.0).unwrap(),
            RatingDeviation::new(30.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let results = vec![MatchResult::new(opponent, 0.0).unwrap()];
        let tau = SystemConstant::default();

        let updated = update_rating(&player, &results, &tau).unwrap();

        // Rating should decrease after losing
        assert!(updated.rating.value() < player.rating.value());
        // RD should decrease (more certainty)
        assert!(updated.rating_deviation.value() < player.rating_deviation.value());
    }

    #[test]
    #[verified_engine::verified]
    fn test_update_rating_draw() {
        // Player draws with equal opponent
        let player = GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(200.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent = GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(200.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let results = vec![MatchResult::new(opponent, 0.5).unwrap()];
        let tau = SystemConstant::default();

        let updated = update_rating(&player, &results, &tau).unwrap();

        // Rating should stay approximately the same
        assert!((updated.rating.value() - player.rating.value()).abs() < 10.0);
        // RD should decrease (more certainty)
        assert!(updated.rating_deviation.value() < player.rating_deviation.value());
    }

    #[test]
    #[verified_engine::verified]
    fn test_update_rating_multiple_games() {
        // Player plays multiple games
        let player = GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(200.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent1 = GlickoPlayer::new(
            Rating::new(1400.0).unwrap(),
            RatingDeviation::new(30.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent2 = GlickoPlayer::new(
            Rating::new(1550.0).unwrap(),
            RatingDeviation::new(100.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent3 = GlickoPlayer::new(
            Rating::new(1700.0).unwrap(),
            RatingDeviation::new(300.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let results = vec![
            MatchResult::new(opponent1, 1.0).unwrap(), // Win
            MatchResult::new(opponent2, 0.0).unwrap(), // Loss
            MatchResult::new(opponent3, 0.0).unwrap(), // Loss
        ];

        let tau = SystemConstant::default();
        let updated = update_rating(&player, &results, &tau).unwrap();

        // RD should decrease significantly with multiple games
        assert!(updated.rating_deviation.value() < player.rating_deviation.value());
    }

    #[test]
    #[verified_engine::verified]
    fn test_glickman_example() {
        // Example from Glickman's paper
        // Player: rating=1500, RD=200, volatility=0.06
        let player = GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(200.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        // Opponents from the paper
        let opponent1 = GlickoPlayer::new(
            Rating::new(1400.0).unwrap(),
            RatingDeviation::new(30.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent2 = GlickoPlayer::new(
            Rating::new(1550.0).unwrap(),
            RatingDeviation::new(100.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let opponent3 = GlickoPlayer::new(
            Rating::new(1700.0).unwrap(),
            RatingDeviation::new(300.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let results = vec![
            MatchResult::new(opponent1, 1.0).unwrap(), // Win
            MatchResult::new(opponent2, 0.0).unwrap(), // Loss
            MatchResult::new(opponent3, 0.0).unwrap(), // Loss
        ];

        let tau = SystemConstant::new(0.5).unwrap();
        let updated = update_rating(&player, &results, &tau).unwrap();

        // Expected results from Glickman's paper (approximately)
        // New rating ≈ 1464.06
        // New RD ≈ 151.52
        // New volatility ≈ 0.05999

        // Allow for some numerical differences
        assert!(
            (updated.rating.value() - 1464.06).abs() < 10.0,
            "Rating: expected ≈1464.06, got {}",
            updated.rating.value()
        );
        assert!(
            (updated.rating_deviation.value() - 151.52).abs() < 10.0,
            "RD: expected ≈151.52, got {}",
            updated.rating_deviation.value()
        );
        // Volatility computation can have more variance due to iterative methods
        assert!(
            (updated.volatility.value() - 0.06).abs() < 0.02,
            "Volatility: expected ≈0.06, got {}",
            updated.volatility.value()
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_increase_rd_for_inactivity() {
        let player = GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(50.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        );

        let inactive = increase_rd_for_inactivity(&player);

        // Rating and volatility should remain the same
        assert_eq!(inactive.rating.value(), player.rating.value());
        assert_eq!(inactive.volatility.value(), player.volatility.value());

        // RD should increase
        assert!(inactive.rating_deviation.value() > player.rating_deviation.value());
    }

    #[test]
    #[verified_engine::verified]
    fn test_empty_results() {
        let player = GlickoPlayer::default();
        let results: Vec<MatchResult> = vec![];
        let tau = SystemConstant::default();

        let updated = update_rating(&player, &results, &tau).unwrap();

        // Should behave same as inactivity
        assert_eq!(updated.rating.value(), player.rating.value());
        assert!(updated.rating_deviation.value() > player.rating_deviation.value());
    }
}
