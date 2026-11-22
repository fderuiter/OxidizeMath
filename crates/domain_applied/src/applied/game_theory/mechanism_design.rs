use pure_math::pure_math::analysis::roots::{AnalysisError, Bisection, RootFinder};
use rand::Rng;
pub use rand::distributions::Distribution as RandDistribution;
pub use statrs::distribution::{self, Continuous, ContinuousCDF};
pub use statrs::statistics::Distribution;

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
        let pdf = self.pdf(v);
        let cdf = self.cdf(v);
        if pdf.abs() < math_commons::registry::TOLERANCE_STANDARD {
            // Handle edge case where density is 0 (shouldn't happen in support)
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
    let solver = Bisection::default();
    match optimal_reserve_price_with_solver(dist, lower_bound, upper_bound, &solver) {
        Ok(root) => root,
        Err(AnalysisError::ConvergenceError(best_guess)) => best_guess,
        Err(AnalysisError::InvalidParameters(_)) => {
            // Fallback for unbracketed roots to match legacy behavior
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

/// Estimates the expected revenue of an optimal auction with `n_bidders`
/// via Monte Carlo simulation.
///
/// The revenue of the optimal auction is given by:
/// $$ \text{Revenue} = \mathbb{E} \left[ \max(0, J(v_1), \dots, J(v_n)) \right] $$
///
/// This simulation draws random valuations for $n$ bidders, calculates their virtual valuations,
/// and averages the maximum non-negative virtual valuation over `n_simulations`.
#[verified_engine::verified]
pub fn simulate_optimal_revenue<D: ValuationDistribution>(
    dist: &D,
    n_bidders: usize,
    n_simulations: usize,
) -> f64 {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
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
    let mut total_revenue = 0.0;

    for _ in 0..n_simulations {
        let mut max_virtual_val = 0.0;
        for _ in 0..n_bidders {
            let v = dist.sample(rng);
            let j = dist.virtual_valuation(v);
            if j > max_virtual_val {
                max_virtual_val = j;
            }
        }
        total_revenue += max_virtual_val;
    }

    total_revenue / (n_simulations as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::Uniform;

    #[test]
    #[verified_engine::verified]
    fn test_virtual_valuation_uniform() {
        // For Uniform(0, 1): f(v) = 1, F(v) = v.
        // J(v) = v - (1 - v)/1 = v - 1 + v = 2v - 1.
        let dist = Uniform::new(0.0, 1.0).unwrap();

        let v = 0.5;
        let j = dist.virtual_valuation(v);
        assert!((j - (2.0 * v - 1.0)).abs() < math_commons::registry::TOLERANCE_STANDARD); // J(0.5) = 0.

        let v = 0.8;
        let j = dist.virtual_valuation(v);
        assert!((j - (2.0 * v - 1.0)).abs() < math_commons::registry::TOLERANCE_STANDARD); // J(0.8) = 0.6.
    }

    #[test]
    #[verified_engine::verified]
    fn test_optimal_reserve_uniform() {
        // For Uniform(0, 1), J(r) = 2r - 1 = 0 => r = 0.5.
        let dist = Uniform::new(0.0, 1.0).unwrap();
        let r_star = optimal_reserve_price(&dist, 0.0, 1.0);
        assert!((r_star - 0.5).abs() < 1e-4);
    }

    #[test]
    #[verified_engine::verified]
    fn test_revenue_simulation() {
        // 1 bidder, Uniform(0, 1). Optimal auction sets reserve r=0.5.
        // Revenue = E[max(0, J(v))]. J(v) = 2v-1.
        // If v < 0.5, J < 0, contribution 0.
        // If v >= 0.5, contribution 2v-1.
        // Integral_{0.5}^1 (2v - 1) * 1 dv = [v^2 - v]_{0.5}^1
        // = (1 - 1) - (0.25 - 0.5) = 0 - (-0.25) = 0.25.

        let dist = Uniform::new(0.0, 1.0).unwrap();
        let revenue = simulate_optimal_revenue(&dist, 1, 10_000);
        assert!((revenue - 0.25).abs() < 0.02); // MC error margin
    }

    #[test]
    #[verified_engine::verified]
    fn test_optimal_reserve_unbracketed_fallback() {
        // Uniform(0, 100). J(v) = 2v - 100. Root at 50.
        let dist = Uniform::new(0.0, 100.0).unwrap();

        // Search range [60, 100]. J(60)=20, J(100)=100. Both > 0.
        // Should return lower bound (60) as J is increasing and positive.
        let r = optimal_reserve_price(&dist, 60.0, 100.0);
        assert!((r - 60.0).abs() < 1e-4);

        // Search range [0, 40]. J(0)=-100, J(40)=-20. Both < 0.
        // Should return upper bound (40) as J is increasing and negative.
        let r = optimal_reserve_price(&dist, 0.0, 40.0);
        assert!((r - 40.0).abs() < 1e-4);
    }

    #[verified_engine::verified]
    fn compose_auction_and_payoff_check<D: ValuationDistribution>(
        dist: &D,
        rows: usize,
        cols: usize,
        lower_bound: f64,
        upper_bound: f64,
    ) -> Result<f64, crate::error::AppliedError> {
        if rows != cols {
            Err(crate::error::GameTheoryError::NonSquarePayoffMatrix { rows, cols })?;
        }
        let solver = Bisection::default();
        let reserve = optimal_reserve_price_with_solver(dist, lower_bound, upper_bound, &solver)?;
        Ok(reserve)
    }

    #[test]
    #[verified_engine::verified]
    fn test_mechanism_design_composes_with_applied_error_and_diagnostic_bus() {
        use diagnostics::DiagnosticBus;
        use std::error::Error as StdError;
        use std::sync::{Arc, Mutex};

        let dist = Uniform::new(0.0, 100.0).unwrap();
        let ok_reserve = compose_auction_and_payoff_check(&dist, 2, 2, 0.0, 100.0).unwrap();
        assert!((ok_reserve - 50.0).abs() < 1e-4);

        let game_err = compose_auction_and_payoff_check(&dist, 2, 3, 0.0, 100.0).unwrap_err();
        assert!(game_err.source().is_some());

        let analysis_err = compose_auction_and_payoff_check(&dist, 2, 2, 60.0, 100.0).unwrap_err();
        assert!(analysis_err.source().is_some());

        let bus = DiagnosticBus::new();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&captured);
        bus.register_listener(move |diag| {
            sink.lock().unwrap().push((
                diag.metadata.get("source").cloned(),
                diag.metadata.get("umbrella_type").cloned(),
            ));
        });
        bus.emit_diagnostic_with_source("mechanism_design", &analysis_err);
        let events = captured.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0.as_deref(), Some("mechanism_design"));
        assert_eq!(events[0].1.as_deref(), Some("AppliedError"));
    }
}
