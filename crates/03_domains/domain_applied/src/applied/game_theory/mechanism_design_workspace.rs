//! Preallocated evaluation workspace and summary metrics for Myerson optimal-reserve
//! and first-price Bayes-Nash equilibrium bid-shading verification.

use super::super::simulate_optimal_revenue_with_rng;
use super::{AuctionDistEval, AuctionRuleMode};
use crate::applied::game_theory::equilibrium::{ScalarBestResponse, is_fixed_point};
use nalgebra::DVector;
use oxidize_core::rng::OxidizeRng;

/// Summary metrics computed by [`AuctionEvalWorkspace`] with zero dynamic heap allocations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AuctionSummaryMetrics {
    /// Myerson optimal reserve price $r^*$.
    pub optimal_reserve: f64,
    /// Analytical expected seller revenue at active reserve $r$.
    pub expected_revenue: f64,
    /// Analytical expected seller revenue at optimal reserve $r^*$.
    pub optimal_revenue: f64,
    /// Monte Carlo simulated optimal revenue.
    pub mc_optimal_revenue: f64,
    /// Expected social welfare under reserve $r$.
    pub social_welfare: f64,
    /// Efficient (`r = v_min`) social welfare benchmark.
    pub efficient_welfare: f64,
    /// Probability that at least one bidder exceeds reserve $r$.
    pub trade_probability: f64,
    /// Equilibrium bid $b^*(v)$ at `focal_v`.
    pub equilibrium_bid: f64,
    /// Truthful Vickrey bid at `focal_v`.
    pub truthful_bid: f64,
    /// Interim expected utility at `candidate_bid`.
    pub candidate_utility: f64,
    /// Interim expected utility at `equilibrium_bid`.
    pub equilibrium_utility: f64,
    /// Whether `candidate_bid` satisfies [`is_fixed_point`] under [`ScalarBestResponse`].
    pub is_equilibrium_fixed_point: bool,
}

/// Preallocated workspace verifying Bayes-Nash fixed points and Monte Carlo Myerson revenue.
pub struct AuctionEvalWorkspace {
    candidate_vec: DVector<f64>,
    rng: OxidizeRng,
}

impl Default for AuctionEvalWorkspace {
    fn default() -> Self {
        Self {
            candidate_vec: DVector::from_element(1, 0.0),
            rng: OxidizeRng::from_seed(42),
        }
    }
}

/// Scalar mechanism and bidding inputs for [`AuctionEvalWorkspace::evaluate_analytical`].
#[derive(Clone, Copy, Debug)]
pub struct AuctionEvalInputs {
    /// Payment and bidding rule (`SecondPrice` or `FirstPrice`).
    pub rule: AuctionRuleMode,
    /// Seller reserve price $r$.
    pub reserve: f64,
    /// Number of symmetric bidders $n$.
    pub n_bidders: usize,
    /// Focal bidder valuation $v$.
    pub focal_v: f64,
    /// Candidate submitted bid $b(v)$.
    pub candidate_bid: f64,
    /// Fixed-point tolerance around $b^*(v)$.
    pub fixed_point_tol: f64,
}

impl AuctionEvalWorkspace {
    /// Evaluates analytical Myerson revenue, shading, and [`is_fixed_point`] with `0` allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use domain_applied::__doctest_facade as math_explorer;
    /// use math_explorer::applied::game_theory::mechanism_design::{
    ///     AuctionDistEval, AuctionDistributionKind, AuctionEvalInputs, AuctionEvalWorkspace,
    ///     AuctionRuleMode,
    /// };
    ///
    /// let dist = AuctionDistEval::from_params(
    ///     AuctionDistributionKind::Uniform,
    ///     0.0,
    ///     100.0,
    ///     0.1,
    ///     2.0,
    ///     2.0,
    /// );
    /// let mut ws = AuctionEvalWorkspace::default();
    /// let summary = ws.evaluate_analytical(
    ///     &dist,
    ///     AuctionEvalInputs {
    ///         rule: AuctionRuleMode::FirstPrice,
    ///         reserve: 50.0,
    ///         n_bidders: 2,
    ///         focal_v: 100.0,
    ///         candidate_bid: 62.5,
    ///         fixed_point_tol: 1e-4,
    ///     },
    /// );
    /// assert!((summary.equilibrium_bid - 62.5).abs() < 1e-6);
    /// assert!(summary.is_equilibrium_fixed_point);
    /// ```
    #[verified_engine::verified]
    pub fn evaluate_analytical(
        &mut self,
        dist: &AuctionDistEval,
        inp: AuctionEvalInputs,
    ) -> AuctionSummaryMetrics {
        let optimal_reserve = dist.optimal_reserve();
        let (expected_revenue, social_welfare, efficient_welfare, trade_probability) =
            dist.expected_revenue_and_welfare(inp.reserve, inp.n_bidders);
        let (optimal_revenue, _, _, _) =
            dist.expected_revenue_and_welfare(optimal_reserve, inp.n_bidders);
        let truthful_bid = if inp.focal_v >= inp.reserve {
            inp.focal_v
        } else {
            0.0
        };
        let equilibrium_bid = match inp.rule {
            AuctionRuleMode::SecondPrice => truthful_bid,
            AuctionRuleMode::FirstPrice => {
                dist.first_price_equilibrium_bid(inp.focal_v, inp.reserve, inp.n_bidders)
            }
        };
        let (candidate_utility, equilibrium_utility) = match inp.rule {
            AuctionRuleMode::SecondPrice => {
                let eq_u = (inp.focal_v - inp.reserve).max(0.0)
                    * dist
                        .cdf(inp.focal_v)
                        .powi((inp.n_bidders.max(1) - 1) as i32);
                (eq_u, eq_u)
            }
            AuctionRuleMode::FirstPrice => (
                dist.first_price_interim_utility(
                    inp.candidate_bid,
                    inp.focal_v,
                    inp.reserve,
                    inp.n_bidders,
                ),
                dist.first_price_interim_utility(
                    equilibrium_bid,
                    inp.focal_v,
                    inp.reserve,
                    inp.n_bidders,
                ),
            ),
        };

        self.candidate_vec[0] = inp.candidate_bid;
        let br = ScalarBestResponse::new(|_b| equilibrium_bid, inp.fixed_point_tol.max(1e-9));
        let is_equilibrium_fixed_point = is_fixed_point(&br, &self.candidate_vec);

        AuctionSummaryMetrics {
            optimal_reserve,
            expected_revenue,
            optimal_revenue,
            mc_optimal_revenue: optimal_revenue,
            social_welfare,
            efficient_welfare,
            trade_probability,
            equilibrium_bid,
            truthful_bid,
            candidate_utility,
            equilibrium_utility,
            is_equilibrium_fixed_point,
        }
    }

    /// Runs deterministic [`simulate_optimal_revenue_with_rng`] over `mc_samples` draws.
    #[verified_engine::verified]
    pub fn simulate_mc_optimal_revenue(
        &mut self,
        dist: &AuctionDistEval,
        n_bidders: usize,
        mc_samples: usize,
        seed: u64,
    ) -> f64 {
        self.rng = OxidizeRng::from_seed(seed);
        let samples = mc_samples.clamp(100, 25_000);
        let n = n_bidders.clamp(1, 64);
        match dist {
            AuctionDistEval::Uniform { dist, .. } => {
                simulate_optimal_revenue_with_rng(dist, n, samples, &mut self.rng)
            }
            AuctionDistEval::Exp { dist, .. } => {
                simulate_optimal_revenue_with_rng(dist, n, samples, &mut self.rng)
            }
            AuctionDistEval::Beta { dist, .. } => {
                simulate_optimal_revenue_with_rng(dist, n, samples, &mut self.rng)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::applied::game_theory::mechanism_design::AuctionDistributionKind;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_auction_eval_workspace_zero_alloc_first_price_shading() {
        let dist = AuctionDistEval::from_params(
            AuctionDistributionKind::Uniform,
            0.0,
            100.0,
            0.1,
            2.0,
            2.0,
        );
        let mut ws = AuctionEvalWorkspace::default();
        let _guard = crate::VERIFIED_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (summary, _) = VerifiedEngine::run_verified(|| {
            ws.evaluate_analytical(
                &dist,
                AuctionEvalInputs {
                    rule: AuctionRuleMode::FirstPrice,
                    reserve: 50.0,
                    n_bidders: 2,
                    focal_v: 100.0,
                    candidate_bid: 62.5,
                    fixed_point_tol: 1e-5,
                },
            )
        });
        assert!((summary.optimal_reserve - 50.0).abs() < 1e-3);
        assert!((summary.equilibrium_bid - 62.5).abs() < 1e-6);
        assert!(summary.is_equilibrium_fixed_point);
    }
}
