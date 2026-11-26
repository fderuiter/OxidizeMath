//! Zero-allocation Myerson mechanism design and Bayes-Nash bid-shading evaluators.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::presets::{AuctionDistributionKind, AuctionPlotView, AuctionRuleMode};
use math_explorer::applied::game_theory::equilibrium::{is_fixed_point, ScalarBestResponse};
use math_explorer::applied::game_theory::mechanism_design::distribution::{Beta, Exp, Uniform};
use math_explorer::applied::game_theory::mechanism_design::{
    optimal_reserve_price, simulate_optimal_revenue_with_rng, ContinuousCDF, ValuationDistribution,
};
use nalgebra::DVector;
use oxidize_core::rng::OxidizeRng;

/// 16-point Gauss-Legendre abscissas on `[-1.0, 1.0]`.
const GL16_NODES: [f64; 16] = [
    -0.989_400_934_991_649_9,
    -0.944_575_023_073_232_6,
    -0.865_631_202_387_831_8,
    -0.755_404_408_355_003,
    -0.617_876_244_402_643_8,
    -0.458_016_777_657_227_37,
    -0.281_603_550_779_258_9,
    -0.095_012_509_837_637_44,
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_37,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];

/// 16-point Gauss-Legendre weights on `[-1.0, 1.0]`.
const GL16_WEIGHTS: [f64; 16] = [
    0.027_152_459_411_754_095,
    0.062_253_523_938_647_894,
    0.095_158_511_682_492_78,
    0.124_628_971_255_533_87,
    0.149_595_988_816_576_73,
    0.169_156_519_395_002_54,
    0.182_603_415_044_923_58,
    0.189_450_610_455_068_5,
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_58,
    0.169_156_519_395_002_54,
    0.149_595_988_816_576_73,
    0.124_628_971_255_533_87,
    0.095_158_511_682_492_78,
    0.062_253_523_938_647_894,
    0.027_152_459_411_754_095,
];

/// Pre-validated stack-resident valuation distribution for zero-allocation evaluation.
#[derive(Clone, Debug)]
pub enum AuctionDistEval {
    /// Validated `Uniform(min, max)` distribution.
    Uniform {
        dist: Uniform,
        v_min: f64,
        v_max: f64,
    },
    /// Validated `Exp(rate)` distribution with plot and quadrature tail horizons.
    Exp {
        dist: Exp,
        v_plot_max: f64,
        v_quad_max: f64,
    },
    /// Validated `Beta(alpha, beta)` distribution on `[0.0, 1.0]`.
    Beta { dist: Beta },
}

impl AuctionDistEval {
    /// Builds a validated [`AuctionDistEval`] from user parameters.
    #[must_use]
    pub fn from_params(
        kind: AuctionDistributionKind,
        uniform_min: f64,
        uniform_max: f64,
        exp_rate: f64,
        beta_alpha: f64,
        beta_beta: f64,
    ) -> Self {
        match kind {
            AuctionDistributionKind::Uniform => {
                let lo = if uniform_min.is_finite() {
                    uniform_min.clamp(0.0, 499.0)
                } else {
                    0.0
                };
                let hi = if uniform_max.is_finite() {
                    uniform_max.clamp(lo + 1.0, 500.0)
                } else {
                    (lo + 100.0).min(500.0)
                };
                let dist = Uniform::new(lo, hi)
                    .unwrap_or_else(|_| Uniform::new(0.0, 100.0).expect("valid fallback uniform"));
                Self::Uniform {
                    dist,
                    v_min: lo,
                    v_max: hi,
                }
            }
            AuctionDistributionKind::Exp => {
                let rate = if exp_rate.is_finite() {
                    exp_rate.clamp(0.05, 5.0)
                } else {
                    0.1
                };
                let dist =
                    Exp::new(rate).unwrap_or_else(|_| Exp::new(0.1).expect("valid fallback exp"));
                Self::Exp {
                    dist,
                    v_plot_max: 6.0 / rate,
                    v_quad_max: 20.0 / rate,
                }
            }
            AuctionDistributionKind::Beta => {
                let a = if beta_alpha.is_finite() {
                    beta_alpha.clamp(0.5, 8.0)
                } else {
                    2.0
                };
                let b = if beta_beta.is_finite() {
                    beta_beta.clamp(0.5, 8.0)
                } else {
                    2.0
                };
                let dist = Beta::new(a, b)
                    .unwrap_or_else(|_| Beta::new(2.0, 2.0).expect("valid fallback beta"));
                Self::Beta { dist }
            }
        }
    }

    /// Lower and upper plot bounds `[v_min, v_plot_max]`.
    #[inline]
    #[must_use]
    pub const fn plot_bounds(&self) -> (f64, f64) {
        match *self {
            Self::Uniform { v_min, v_max, .. } => (v_min, v_max),
            Self::Exp { v_plot_max, .. } => (0.0, v_plot_max),
            Self::Beta { .. } => (0.0, 1.0),
        }
    }

    /// Upper integration support `v_quad_max` used for Gauss-Legendre tail integrals.
    #[inline]
    #[must_use]
    pub const fn quad_upper(&self) -> f64 {
        match *self {
            Self::Uniform { v_max, .. } => v_max,
            Self::Exp { v_quad_max, .. } => v_quad_max,
            Self::Beta { .. } => 1.0,
        }
    }

    /// Evaluates cumulative distribution $F(v)$ in `[0.0, 1.0]`.
    #[inline]
    #[must_use]
    pub fn cdf(&self, v: f64) -> f64 {
        match self {
            Self::Uniform { dist, v_min, v_max } => {
                if v <= *v_min {
                    0.0
                } else if v >= *v_max {
                    1.0
                } else {
                    dist.cdf(v).clamp(0.0, 1.0)
                }
            }
            Self::Exp { dist, .. } => {
                if v <= 0.0 {
                    0.0
                } else {
                    dist.cdf(v).clamp(0.0, 1.0)
                }
            }
            Self::Beta { dist, .. } => {
                if v <= 0.0 {
                    0.0
                } else if v >= 1.0 {
                    1.0
                } else {
                    dist.cdf(v).clamp(0.0, 1.0)
                }
            }
        }
    }

    /// Evaluates Myerson's virtual valuation $J(v) = v - \frac{1 - F(v)}{f(v)}$.
    #[inline]
    #[must_use]
    pub fn virtual_valuation(&self, v: f64) -> f64 {
        match self {
            Self::Uniform { dist, .. } => dist.virtual_valuation(v),
            Self::Exp { dist, .. } => dist.virtual_valuation(v.max(0.0)),
            Self::Beta { dist, .. } => dist.virtual_valuation(v.clamp(1e-4, 1.0 - 1e-4)),
        }
    }

    /// Computes Myerson's optimal reserve price $r^*$ via [`optimal_reserve_price`].
    #[must_use]
    pub fn optimal_reserve(&self) -> f64 {
        match self {
            Self::Uniform { dist, v_min, v_max } => optimal_reserve_price(dist, *v_min, *v_max),
            Self::Exp {
                dist, v_plot_max, ..
            } => optimal_reserve_price(dist, 0.0, *v_plot_max),
            Self::Beta { dist, .. } => optimal_reserve_price(dist, 1e-4, 1.0 - 1e-4),
        }
    }

    /// Evaluates `(revenue, social_welfare, efficient_welfare, trade_prob)` in `O(1)` quadrature.
    #[must_use]
    pub fn expected_revenue_and_welfare(
        &self,
        reserve: f64,
        n_bidders: usize,
    ) -> (f64, f64, f64, f64) {
        let n = n_bidders.clamp(1, 64);
        let n_i32 = n as i32;
        let n_f64 = n as f64;
        let (v_min, _) = self.plot_bounds();
        let v_hi = self.quad_upper();
        let r = reserve.clamp(v_min, v_hi);
        let f_r = self.cdf(r);
        let f_r_n = f_r.powi(n_i32);
        let trade_prob = (1.0 - f_r_n).clamp(0.0, 1.0);

        let rev_tail = integrate_gl32_stack(r, v_hi, |v| {
            let fv = self.cdf(v);
            let fv_n1 = fv.powi(n_i32 - 1);
            let fv_n = fv_n1 * fv;
            1.0 - fv_n - n_f64 * fv_n1 * (1.0 - fv)
        });
        let wel_tail = integrate_gl32_stack(r, v_hi, |v| 1.0 - self.cdf(v).powi(n_i32));
        let eff_tail = integrate_gl32_stack(v_min, v_hi, |v| 1.0 - self.cdf(v).powi(n_i32));

        let revenue = (r * trade_prob + rev_tail).max(0.0);
        let welfare = (r * trade_prob + wel_tail).max(0.0);
        let eff_welfare = (v_min + eff_tail).max(welfare);
        (revenue, welfare, eff_welfare, trade_prob)
    }

    /// Evaluates symmetric First-Price Bayes-Nash equilibrium bid $b^*(v; r, n)$.
    #[must_use]
    pub fn first_price_equilibrium_bid(&self, v: f64, reserve: f64, n_bidders: usize) -> f64 {
        let (v_min, _) = self.plot_bounds();
        let r = reserve.max(v_min);
        if v < r {
            return 0.0;
        }
        let n = n_bidders.clamp(1, 64);
        if n <= 1 {
            return r;
        }
        let g_v = self.cdf(v).powi((n - 1) as i32);
        if g_v <= 1e-12 {
            return r;
        }
        let integral = integrate_gl32_stack(r, v, |x| self.cdf(x).powi((n - 1) as i32));
        (v - integral / g_v).clamp(r, v)
    }

    /// Interim expected utility $U_i(b \mid v)$ when $n - 1$ rivals bid $b^*(\cdot; r, n)$.
    #[must_use]
    pub fn first_price_interim_utility(
        &self,
        candidate_bid: f64,
        v: f64,
        reserve: f64,
        n_bidders: usize,
    ) -> f64 {
        let (v_min, _) = self.plot_bounds();
        let r = reserve.max(v_min);
        if candidate_bid < r {
            return 0.0;
        }
        let n = n_bidders.clamp(1, 64);
        if n <= 1 {
            return v - candidate_bid;
        }
        let v_hi = self.quad_upper();
        let b_max = self.first_price_equilibrium_bid(v_hi, r, n);
        if candidate_bid >= b_max {
            return v - candidate_bid;
        }
        let mut lo = r;
        let mut hi = v_hi;
        for _ in 0..36 {
            let mid = 0.5 * (lo + hi);
            if self.first_price_equilibrium_bid(mid, r, n) < candidate_bid {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let z = 0.5 * (lo + hi);
        let win_prob = self.cdf(z).powi((n - 1) as i32);
        (v - candidate_bid) * win_prob
    }
}

#[inline]
fn integrate_gl16_panel<F: Fn(f64) -> f64>(a: f64, b: f64, f: &F) -> f64 {
    if b <= a {
        return 0.0;
    }
    let half = 0.5 * (b - a);
    let mid = 0.5 * (a + b);
    let mut sum = 0.0;
    for i in 0..16 {
        sum += GL16_WEIGHTS[i] * f(mid + half * GL16_NODES[i]);
    }
    half * sum
}

#[inline]
fn integrate_gl32_stack<F: Fn(f64) -> f64>(a: f64, b: f64, f: F) -> f64 {
    if b <= a {
        return 0.0;
    }
    let split = a + 0.35 * (b - a);
    integrate_gl16_panel(a, split, &f) + integrate_gl16_panel(split, b, &f)
}

/// Summary metrics computed by [`AuctionEvalWorkspace`] with zero dynamic heap allocations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AuctionSummaryMetrics {
    pub optimal_reserve: f64,
    pub expected_revenue: f64,
    pub optimal_revenue: f64,
    pub mc_optimal_revenue: f64,
    pub social_welfare: f64,
    pub efficient_welfare: f64,
    pub trade_probability: f64,
    pub equilibrium_bid: f64,
    pub truthful_bid: f64,
    pub candidate_utility: f64,
    pub equilibrium_utility: f64,
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
