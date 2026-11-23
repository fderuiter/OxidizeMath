use crate::error::GameTheoryError;
use engine_prob::math::{Z_95_TWO_SIDED, mean_and_sample_variance};
use engine_prob::variance_reduction::{estimate_antithetic_uniform, estimate_control_variate};
use engine_prob::workspace::OwnedSolverWorkspace;
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::roots::{AnalysisError, Bisection, RootFinder};
use rand::Rng;
pub use rand::distributions::Distribution as RandDistribution;
pub use statrs::distribution::{self, Continuous, ContinuousCDF};
pub use statrs::statistics::Distribution;

/// Variance-reduction mode for Myerson optimal auction Monte Carlo simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuctionVarianceReduction {
    /// Standard independent and identically distributed Monte Carlo draws.
    None,
    /// Coupled antithetic uniform draws $(U, 1 - U)$ mapped via `inverse_cdf`.
    Antithetic,
    /// Optimal linear control variate using the max uniform order statistic $\max_i U_i$
    /// with analytical expectation $\frac{n}{n + 1}$.
    ControlVariate,
}

/// Summary of an optimal auction Monte Carlo revenue simulation with standard error and 95% CI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RevenueSimulationReport {
    /// Estimated expected optimal auction revenue $\mathbb{E}[\max(0, J(V_1), \dots, J(V_n))]$.
    pub mean_revenue: f64,
    /// Monte Carlo standard error of `mean_revenue`.
    pub std_error: f64,
    /// Lower 95% confidence bound (clamped at `0.0` since optimal reserve revenue is non-negative).
    pub lower_95: f64,
    /// Upper 95% confidence bound.
    pub upper_95: f64,
    /// Total auction evaluations performed.
    pub n_simulations: usize,
}

/// Represents a distribution of bidder valuations.
///
/// In Mechanism Design, particularly for auction theory, we often need to analyze properties
/// of the distribution of valuations $v$ drawn from a cumulative distribution function $F(v)$
/// with probability density function $f(v)$.
pub trait ValuationDistribution:
    Continuous<f64, f64> + ContinuousCDF<f64, f64> + Distribution<f64> + RandDistribution<f64>
{
    /// Computes the **Virtual Valuation** $J(v)$ according to Myerson's Lemma.
    ///
    /// $$ J(v) = v - \frac{1 - F(v)}{f(v)} $$
    ///
    /// In the context of optimal auction design (revenue maximization), the auctioneer
    /// treats the virtual valuation $J(v)$ as the "real" value they extract from the bidder.
    /// An optimal auction awards the item to the bidder with the highest virtual valuation,
    /// provided it is non-negative.
    ///
    /// The condition $J(v) \geq 0$ defines the optimal reserve price.
    #[verified_engine::verified]
    fn virtual_valuation(&self, v: f64) -> f64 {
        if !v.is_finite() {
            return f64::NAN;
        }
        let pdf = self.pdf(v);
        let cdf = self.cdf(v);
        if !pdf.is_finite() || !cdf.is_finite() {
            return f64::NAN;
        }
        if pdf.abs() < math_commons::registry::TOLERANCE_STANDARD {
            v
        } else {
            v - (1.0 - cdf) / pdf
        }
    }
}

// Blanket implementation for any continuous distribution from statrs
impl<D: Continuous<f64, f64> + ContinuousCDF<f64, f64> + Distribution<f64> + RandDistribution<f64>>
    ValuationDistribution for D
{
}

/// Calculates the **Optimal Reserve Price** for a single-item auction.
///
/// According to Myerson (1981), for a "regular" distribution (where $J(v)$ is strictly increasing),
/// the revenue-maximizing reserve price $r^*$ is the value such that the virtual valuation is zero:
///
/// $$ J(r^*) = r^* - \frac{1 - F(r^*)}{f(r^*)} = 0 $$
///
/// This function finds the root of $J(r) = 0$ using the bisection method within the given bounds.
///
/// # Parameters
/// - `dist`: The probability distribution of bidder valuations.
/// - `lower_bound`: The lower bound for the search (e.g., min possible valuation).
/// - `upper_bound`: The upper bound for the search (e.g., max possible valuation).
///
/// # Example
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::game_theory::mechanism_design::optimal_reserve_price;
/// use math_explorer::applied::game_theory::mechanism_design::distribution::Uniform;
///
/// // For Uniform(0, 100), J(v) = 2v - 100.
/// // J(r) = 0 => 2r = 100 => r = 50.
/// let dist = Uniform::new(0.0, 100.0).unwrap();
/// let r_star = optimal_reserve_price(&dist, 0.0, 100.0);
/// assert!((r_star - 50.0).abs() < 1e-4);
/// ```
#[verified_engine::verified]
pub fn optimal_reserve_price<D: ValuationDistribution>(
    dist: &D,
    lower_bound: f64,
    upper_bound: f64,
) -> f64 {
    if !lower_bound.is_finite() || !upper_bound.is_finite() {
        return 0.0;
    }
    let solver = Bisection::default();
    match optimal_reserve_price_with_solver(dist, lower_bound, upper_bound, &solver) {
        Ok(root) => root,
        Err(AnalysisError::ConvergenceError(best_guess)) => best_guess,
        Err(
            AnalysisError::RootNotBracketed { .. }
            | AnalysisError::DerivativeTooSmall { .. }
            | AnalysisError::InvalidBracket { .. },
        ) => {
            let j_low = dist.virtual_valuation(lower_bound);
            if j_low > 0.0 {
                return lower_bound;
            }
            let j_high = dist.virtual_valuation(upper_bound);
            if j_high < 0.0 {
                return upper_bound;
            }
            (lower_bound + upper_bound) / 2.0
        }
    }
}

/// Calculates the **Optimal Reserve Price** for a single-item auction using a custom solver.
#[verified_engine::verified]
pub fn optimal_reserve_price_with_solver<D: ValuationDistribution, S: RootFinder>(
    dist: &D,
    lower_bound: f64,
    upper_bound: f64,
    solver: &S,
) -> Result<f64, AnalysisError> {
    solver.find_root(|v| dist.virtual_valuation(v), lower_bound, upper_bound)
}

#[verified_engine::verified]
fn validate_auction_counts<D: ValuationDistribution>(
    dist: &D,
    n_bidders: usize,
    n_simulations: usize,
) -> Result<(), GameTheoryError> {
    if n_bidders == 0 {
        return Err(GameTheoryError::InvalidParameter {
            name: "n_bidders".to_string(),
            value: 0.0,
        });
    }
    if n_simulations == 0 {
        return Err(GameTheoryError::InvalidParameter {
            name: "n_simulations".to_string(),
            value: 0.0,
        });
    }
    let median_v = dist.inverse_cdf(0.5);
    let median_j = dist.virtual_valuation(median_v);
    if !median_v.is_finite() || !median_j.is_finite() {
        return Err(GameTheoryError::InvalidParameter {
            name: "valuation_distribution".to_string(),
            value: f64::NAN,
        });
    }
    Ok(())
}

#[verified_engine::verified]
fn eval_auction_from_uniforms<D: ValuationDistribution>(dist: &D, uniforms: &[f64]) -> f64 {
    let mut max_virtual_val = 0.0;
    for &u in uniforms {
        let v = dist.inverse_cdf(u.clamp(1e-15, 1.0 - 1e-15));
        let j = dist.virtual_valuation(v);
        if j.is_finite() && j > max_virtual_val {
            max_virtual_val = j;
        }
    }
    max_virtual_val
}

/// Fallible Myerson optimal revenue simulation reporting mean, standard error, and 95% CI.
#[verified_engine::verified]
pub fn try_simulate_optimal_revenue<D: ValuationDistribution>(
    dist: &D,
    n_bidders: usize,
    n_simulations: usize,
) -> Result<RevenueSimulationReport, GameTheoryError> {
    let mut rng = OxidizeRng::default();
    try_simulate_optimal_revenue_with_rng(dist, n_bidders, n_simulations, &mut rng)
}

/// Fallible Myerson optimal revenue simulation using a caller-supplied RNG.
#[verified_engine::verified]
pub fn try_simulate_optimal_revenue_with_rng<D: ValuationDistribution, R: Rng + ?Sized>(
    dist: &D,
    n_bidders: usize,
    n_simulations: usize,
    rng: &mut R,
) -> Result<RevenueSimulationReport, GameTheoryError> {
    validate_auction_counts(dist, n_bidders, n_simulations)?;
    let mut revenues = Vec::with_capacity(n_simulations);
    for _ in 0..n_simulations {
        let mut max_virtual_val = 0.0;
        for _ in 0..n_bidders {
            let v = dist.sample(rng);
            let j = dist.virtual_valuation(v);
            if !v.is_finite() || !j.is_finite() {
                return Err(GameTheoryError::InvalidParameter {
                    name: "simulated_valuation".to_string(),
                    value: f64::NAN,
                });
            }
            if j > max_virtual_val {
                max_virtual_val = j;
            }
        }
        revenues.push(max_virtual_val);
    }

    let (mean_revenue, sample_var) =
        mean_and_sample_variance(&revenues).map_err(|_| GameTheoryError::InvalidParameter {
            name: "simulated_revenues".to_string(),
            value: f64::NAN,
        })?;
    let std_error = if n_simulations > 1 {
        (sample_var / (n_simulations as f64)).sqrt()
    } else {
        0.0
    };
    let half = Z_95_TWO_SIDED * std_error;
    Ok(RevenueSimulationReport {
        mean_revenue,
        std_error,
        lower_95: (mean_revenue - half).max(0.0),
        upper_95: mean_revenue + half,
        n_simulations,
    })
}

/// Fallible variance-reduced Myerson optimal revenue estimator backed by [`engine_prob::variance_reduction`].
///
/// # Example
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::game_theory::mechanism_design::{
///     AuctionVarianceReduction, distribution::Uniform, try_simulate_optimal_revenue_variance_reduced,
/// };
/// use math_explorer::rng::OxidizeRng;
///
/// let dist = Uniform::new(0.0, 1.0).unwrap();
/// let mut rng = OxidizeRng::new(42);
/// let rep = try_simulate_optimal_revenue_variance_reduced(
///     &dist,
///     2,
///     4_000,
///     AuctionVarianceReduction::Antithetic,
///     &mut rng,
/// )
/// .unwrap();
/// // Exact 2-bidder Uniform(0, 1) Myerson optimal revenue is 5/12 = 0.416666...
/// assert!((rep.mean_revenue - 5.0 / 12.0).abs() < 0.02);
/// assert!(rep.std_error > 0.0);
/// ```
#[verified_engine::verified]
pub fn try_simulate_optimal_revenue_variance_reduced<D: ValuationDistribution>(
    dist: &D,
    n_bidders: usize,
    n_simulations: usize,
    mode: AuctionVarianceReduction,
    rng: &mut OxidizeRng,
) -> Result<RevenueSimulationReport, GameTheoryError> {
    if mode == AuctionVarianceReduction::None {
        return try_simulate_optimal_revenue_with_rng(dist, n_bidders, n_simulations, rng);
    }
    validate_auction_counts(dist, n_bidders, n_simulations)?;

    if mode == AuctionVarianceReduction::Antithetic {
        let n_pairs = n_simulations.div_ceil(2);
        let mut owned = OwnedSolverWorkspace::with_capacity(2 * n_bidders, 0);
        let mut ws = owned.as_workspace();
        let est = estimate_antithetic_uniform(n_bidders, n_pairs, rng, &mut ws, |u| {
            eval_auction_from_uniforms(dist, u)
        })
        .map_err(|_| GameTheoryError::InvalidParameter {
            name: "antithetic_revenue".to_string(),
            value: f64::NAN,
        })?;
        return Ok(RevenueSimulationReport {
            mean_revenue: est.mean.max(0.0),
            std_error: est.std_error,
            lower_95: est.lower_95.max(0.0),
            upper_95: est.upper_95,
            n_simulations: est.n_evaluations,
        });
    }

    let control_mean = (n_bidders as f64) / ((n_bidders + 1) as f64);
    let mut owned = OwnedSolverWorkspace::with_capacity(2 * n_simulations, 0);
    let mut ws = owned.as_workspace();
    let est = estimate_control_variate(n_simulations, control_mean, rng, &mut ws, |r| {
        let mut max_j = 0.0;
        let mut max_u = 0.0_f64;
        for _ in 0..n_bidders {
            let u = r.next_f64_open01();
            max_u = max_u.max(u);
            let v = dist.inverse_cdf(u);
            let j = dist.virtual_valuation(v);
            if j.is_finite() && j > max_j {
                max_j = j;
            }
        }
        (max_j, max_u)
    })
    .map_err(|_| GameTheoryError::InvalidParameter {
        name: "control_variate_revenue".to_string(),
        value: f64::NAN,
    })?;
    Ok(RevenueSimulationReport {
        mean_revenue: est.mean.max(0.0),
        std_error: est.std_error,
        lower_95: est.lower_95.max(0.0),
        upper_95: est.upper_95,
        n_simulations: est.n_samples,
    })
}

/// Estimates the expected revenue of an optimal auction with `n_bidders`
/// via Monte Carlo simulation.
///
/// Returns `0.0` safely without panicking or producing `NaN` when `n_bidders == 0` or
/// `n_simulations == 0`.
#[verified_engine::verified]
pub fn simulate_optimal_revenue<D: ValuationDistribution>(
    dist: &D,
    n_bidders: usize,
    n_simulations: usize,
) -> f64 {
    let mut rng = OxidizeRng::default();
    simulate_optimal_revenue_with_rng(dist, n_bidders, n_simulations, &mut rng)
}

/// Same as `simulate_optimal_revenue` but allows injecting a custom RNG
/// for deterministic testing.
#[verified_engine::verified]
pub fn simulate_optimal_revenue_with_rng<D: ValuationDistribution, R: Rng + ?Sized>(
    dist: &D,
    n_bidders: usize,
    n_simulations: usize,
    rng: &mut R,
) -> f64 {
    if n_bidders == 0 || n_simulations == 0 {
        return 0.0;
    }
    let mut total_revenue = 0.0;
    let mut valid_sims = 0usize;

    for _ in 0..n_simulations {
        let mut max_virtual_val = 0.0;
        let mut any_nan = false;
        for _ in 0..n_bidders {
            let v = dist.sample(rng);
            let j = dist.virtual_valuation(v);
            if !v.is_finite() || !j.is_finite() {
                any_nan = true;
                break;
            }
            if j > max_virtual_val {
                max_virtual_val = j;
            }
        }
        if !any_nan {
            total_revenue += max_virtual_val;
            valid_sims += 1;
        }
    }

    if valid_sims == 0 {
        0.0
    } else {
        total_revenue / (valid_sims as f64)
    }
}

#[cfg(test)]
#[path = "mechanism_design_tests.rs"]
mod tests;
