//! # Risk of Ruin, Lundberg Bounds & Zero-Allocation Bankroll Ensembles
//!
//! Hybrid analytical and deterministic Monte Carlo solvency engine covering both fixed-stake
//! absorption and proportional-Kelly finite-drawdown barriers:
//!
//! - **Gambler's Ruin ([`gamblers_ruin_with_target`], [`gamblers_ruin_infinite`])**: Exact
//!   two-barrier and infinite-horizon absorption probabilities for fixed-stake binary wagers,
//!   including overflow-free sub-fair ($p < 0.5$) and fair-game ($p = 0.5$) limits.
//! - **Lundberg Adjustment Coefficient ([`solve_lundberg_ruin`])**: Solves $E[\exp(-r X)] = 1$
//!   across heterogeneous fixed-stake bet portfolios via [`crate::pure_math::analysis::roots::Bisection`]
//!   and safeguarded [`crate::pure_math::analysis::roots::NewtonRaphson`], returning both the
//!   Cramér-Lundberg upper bound and the Brownian diffusion approximation.
//! - **Proportional-Kelly Drawdown Barriers ([`kelly_barrier_risk`])**: Computes asymptotic
//!   drawdown barrier probabilities via $(B_{\text{floor}} / W_0)^{2 \mu / \sigma^2}$ diffusion
//!   approximations and exact log-return Lundberg exponents $E[(1 + f R)^{-\theta}] = 1$.
//! - **Monte Carlo Ensemble ([`BankrollEnsembleWorkspace`])**: Preallocated zero-allocation
//!   [`oxidize_core::rng::OxidizeRng`] simulator reporting finite-horizon barrier ruin,
//!   95% [`WilsonInterval`] bounds, and pathwise [`DrawdownQuantiles`].
//!
//! ## Quick Start
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{Odds, UnitInterval, kelly_fraction};
//! use math_explorer::pure_math::statistics::ruin::{
//!     BankrollEnsembleWorkspace, ProportionalPortfolioBet, kelly_barrier_risk,
//! };
//! use math_explorer::rng::{BASE_SEED, OxidizeRng};
//!
//! let p = UnitInterval::new(0.55).unwrap();
//! let odds = Odds::new(2.0).unwrap();
//! let full_kelly = kelly_fraction(&p, &odds).unwrap();
//!
//! // Full Kelly has an exact 50% asymptotic bound of hitting a 50% wealth floor:
//! let barrier = kelly_barrier_risk(&p, &odds, &full_kelly, 0.50).unwrap();
//! assert!((barrier.lundberg_barrier_bound - 0.50).abs() < 1e-6);
//!
//! // Finite-horizon 200-bet ensemble with zero heap allocations during simulation:
//! let mut ws = BankrollEnsembleWorkspace::with_capacity(2000, 0).unwrap();
//! let mut rng = OxidizeRng::from_seed(BASE_SEED);
//! let bet = ProportionalPortfolioBet::unit_weight(p, odds, full_kelly).unwrap();
//! let report = ws
//!     .simulate_proportional(100.0, 50.0, &bet, 200, &mut rng)
//!     .unwrap();
//! assert!(report.ruin_probability > 0.30 && report.ruin_probability < 0.45);
//! ```

pub mod analytical;
pub mod barrier;
pub mod ensemble;

pub use analytical::{
    GamblersRuinResult, LundbergSolution, PortfolioBet, gamblers_ruin_infinite,
    gamblers_ruin_with_target, solve_lundberg_ruin, solve_single_bet_lundberg,
};
pub use barrier::{
    KellyBarrierRisk, ProportionalPortfolioBet, kelly_barrier_risk, portfolio_kelly_barrier_risk,
};
pub use ensemble::{
    BankrollEnsembleWorkspace, DrawdownQuantiles, EnsembleSolvencyReport, WilsonInterval,
    wilson_interval_95, wilson_score_interval,
};

// [cite:clinical_trials_statistics]

#[cfg(test)]
mod tests;
