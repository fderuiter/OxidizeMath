//! Zero-allocation Gauss-Legendre Myerson revenue quadrature and Bayes-Nash bid-shading evaluators.

use super::distribution::{Beta, Exp, Uniform};
use super::{ContinuousCDF, ValuationDistribution, optimal_reserve_price};
use crate::error::GameTheoryError;
use serde::{Deserialize, Serialize};

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

/// Supported bidder valuation distribution families.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuctionDistributionKind {
    /// Continuous uniform distribution $U(v_{\min}, v_{\max})$.
    #[default]
    Uniform,
    /// Exponential distribution $\text{Exp}(\lambda)$ on $[0, \infty)$.
    Exp,
    /// Beta distribution $\text{Beta}(\alpha, \beta)$ on $[0, 1]$.
    Beta,
}

impl AuctionDistributionKind {
    /// Human-readable distribution label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Uniform => "Uniform(min, max)",
            Self::Exp => "Exponential(λ)",
            Self::Beta => "Beta(α, β)",
        }
    }
}

/// Auction payment and bidding rule mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuctionRuleMode {
    /// Second-price sealed-bid (Vickrey) auction with truthful dominant strategy $b(v) = v$.
    #[default]
    SecondPrice,
    /// First-price sealed-bid auction with symmetric Bayes-Nash shading $b^*(v; r, n)$.
    FirstPrice,
}

impl AuctionRuleMode {
    /// Human-readable auction format label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SecondPrice => "Second-Price (Vickrey)",
            Self::FirstPrice => "First-Price Sealed-Bid",
        }
    }
}

/// Pre-validated stack-resident valuation distribution for zero-allocation evaluation.
#[derive(Clone, Debug)]
pub enum AuctionDistEval {
    /// Validated `Uniform(min, max)` distribution.
    Uniform {
        /// Underlying `statrs` uniform distribution.
        dist: Uniform,
        /// Lower support bound $v_{\min}$.
        v_min: f64,
        /// Upper support bound $v_{\max}$.
        v_max: f64,
    },
    /// Validated `Exp(rate)` distribution with plot and quadrature tail horizons.
    Exp {
        /// Underlying `statrs` exponential distribution.
        dist: Exp,
        /// Display upper horizon $6/\lambda$.
        v_plot_max: f64,
        /// Quadrature upper horizon $20/\lambda$.
        v_quad_max: f64,
    },
    /// Validated `Beta(alpha, beta)` distribution on `[0.0, 1.0]`.
    Beta {
        /// Underlying `statrs` Beta distribution.
        dist: Beta,
    },
}

impl AuctionDistEval {
    /// Fallibly constructs an [`AuctionDistEval`] after clamping parameters to valid support.
    #[verified_engine::verified]
    pub fn try_from_params(
        kind: AuctionDistributionKind,
        uniform_min: f64,
        uniform_max: f64,
        exp_rate: f64,
        beta_alpha: f64,
        beta_beta: f64,
    ) -> Result<Self, GameTheoryError> {
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
                let dist = Uniform::new(lo, hi).map_err(|_| GameTheoryError::InvalidParameter {
                    name: "uniform_bounds".to_string(),
                    value: hi - lo,
                })?;
                Ok(Self::Uniform {
                    dist,
                    v_min: lo,
                    v_max: hi,
                })
            }
            AuctionDistributionKind::Exp => {
                let rate = if exp_rate.is_finite() {
                    exp_rate.clamp(0.05, 5.0)
                } else {
                    0.1
                };
                let dist = Exp::new(rate).map_err(|_| GameTheoryError::InvalidParameter {
                    name: "exp_rate".to_string(),
                    value: rate,
                })?;
                Ok(Self::Exp {
                    dist,
                    v_plot_max: 6.0 / rate,
                    v_quad_max: 20.0 / rate,
                })
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
                let dist = Beta::new(a, b).map_err(|_| GameTheoryError::InvalidParameter {
                    name: "beta_shape".to_string(),
                    value: a,
                })?;
                Ok(Self::Beta { dist })
            }
        }
    }

    /// Builds a validated [`AuctionDistEval`] from user parameters.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_params(
        kind: AuctionDistributionKind,
        uniform_min: f64,
        uniform_max: f64,
        exp_rate: f64,
        beta_alpha: f64,
        beta_beta: f64,
    ) -> Self {
        Self::try_from_params(
            kind,
            uniform_min,
            uniform_max,
            exp_rate,
            beta_alpha,
            beta_beta,
        )
        .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Lower and upper plot bounds `[v_min, v_plot_max]`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
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
    #[verified_engine::verified]
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
    #[verified_engine::verified]
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
    #[verified_engine::verified]
    pub fn virtual_valuation(&self, v: f64) -> f64 {
        match self {
            Self::Uniform { dist, .. } => dist.virtual_valuation(v),
            Self::Exp { dist, .. } => dist.virtual_valuation(v.max(0.0)),
            Self::Beta { dist, .. } => dist.virtual_valuation(v.clamp(1e-4, 1.0 - 1e-4)),
        }
    }

    /// Computes Myerson's optimal reserve price $r^*$ via [`optimal_reserve_price`].
    #[must_use]
    #[verified_engine::verified]
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
    #[verified_engine::verified]
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
        let f_r_n = self.cdf(r).powi(n_i32);
        let trade_prob = (1.0 - f_r_n).clamp(0.0, 1.0);

        let rev_tail = integrate_gl32_stack(r, v_hi, |v| {
            let fv = self.cdf(v);
            let fv_n1 = fv.powi(n_i32 - 1);
            1.0 - fv_n1 * fv - n_f64 * fv_n1 * (1.0 - fv)
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
    #[verified_engine::verified]
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
    #[verified_engine::verified]
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
        if candidate_bid >= self.first_price_equilibrium_bid(v_hi, r, n) {
            return v - candidate_bid;
        }
        let (mut lo, mut hi) = (r, v_hi);
        for _ in 0..36 {
            let mid = 0.5 * (lo + hi);
            if self.first_price_equilibrium_bid(mid, r, n) < candidate_bid {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let win_prob = self.cdf(0.5 * (lo + hi)).powi((n - 1) as i32);
        (v - candidate_bid) * win_prob
    }
}

#[inline]
#[verified_engine::verified]
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
#[verified_engine::verified]
fn integrate_gl32_stack<F: Fn(f64) -> f64>(a: f64, b: f64, f: F) -> f64 {
    if b <= a {
        return 0.0;
    }
    let split = a + 0.35 * (b - a);
    integrate_gl16_panel(a, split, &f) + integrate_gl16_panel(split, b, &f)
}

#[path = "mechanism_design_workspace.rs"]
mod workspace;
pub use workspace::*;
