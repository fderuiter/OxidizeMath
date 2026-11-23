//! # Zero-Allocation Market Pricing, Devigging & Cross-Book Execution
//!
//! Slice-based routines for removing bookmaker overround ("vig") from multi-outcome markets,
//! pricing `+EV` wagers against sharp consensus lines, and sizing cross-book arbitrage,
//! subset Dutching, spread/total middles, and live/futures hedges without heap allocations.
//!
//! ## Core Capabilities
//!
//! - **Vig & Overround Telemetry**: [`booksum`], [`overround()`], [`turnover_margin`], and
//!   [`market_overround`].
//! - **Fair-Probability Devigging**:
//!   - [`DevigMethod::Multiplicative`] — proportional scaling (`p_i = pi_i / S`).
//!   - [`DevigMethod::Additive`] — equal-margin subtraction with simplex boundary diagnostics
//!     ([`SimplexViolationDiagnostic`]) when extreme longshots turn negative.
//!   - [`DevigMethod::Power`] — constant-exponent longshot-bias calibration (`sum(pi_i^k) = 1`)
//!     solved via [`crate::pure_math::analysis::roots::Brent`].
//!   - [`DevigMethod::Shin`] — Shin (1991, 1993) insider-trading equilibrium solved analytically
//!     for 2-way markets and via rationalized `Brent` root-finding for `N`-way slates.
//! - **Expected Value & Kelly Integration**: [`evaluate_outcome_ev`] and
//!   [`compare_outcome_ev_all_methods`] pair devigged [`crate::pure_math::statistics::kelly::UnitInterval`]
//!   probabilities with [`crate::pure_math::statistics::kelly::kelly_fraction`].
//! - **Cross-Market Execution**: [`evaluate_arbitrage_into`], [`dutch_outcomes_into`],
//!   [`dutch_subset_into`], [`evaluate_middle`], and [`evaluate_hedge`].
//!
//! ## Example: Shin vs Multiplicative Devigging on a Longshot
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::Odds;
//! use math_explorer::pure_math::statistics::market_pricing::{
//!     DevigMethod, compare_outcome_ev_all_methods,
//! };
//!
//! fn main() -> math_explorer::error::Result<()> {
//!     let sharp = [Odds::from_american(-500.0)?, Odds::from_american(350.0)?];
//!     let soft_underdog = Odds::from_american(400.0)?;
//!
//!     let report = compare_outcome_ev_all_methods(&sharp, 1, soft_underdog)?;
//!     // Naive multiplicative devigging hallucinated +5.26% EV, whereas Shin recognizes -2.78% EV.
//!     assert!(report.for_method(DevigMethod::Multiplicative).has_edge);
//!     assert!(!report.for_method(DevigMethod::Shin).has_edge);
//!     Ok(())
//! }
//! ```

pub mod devig;
pub mod dutching;
pub mod ev;
pub mod execution;
pub mod overround;

/// Maximum number of mutually exclusive outcomes supported in zero-allocation stack buffers.
pub const MAX_MARKET_OUTCOMES: usize = 64;

pub use crate::error::MarketPricingError;
pub use devig::{
    DevigMetadata, DevigMethod, SimplexViolationDiagnostic, devig_additive_detailed_into,
    devig_additive_into, devig_into, devig_multiplicative_into, devig_power_into, devig_shin_into,
    fair_decimal_odds_into, fair_odds_from_probabilities_into,
};
pub use dutching::{
    ArbitrageReport, DutchingMode, DutchingReport, dutch_outcomes_into, dutch_subset_into,
    evaluate_arbitrage_into, evaluate_two_way_arbitrage, is_arbitrage,
};
pub use ev::{
    AllMethodsEvReport, EvEvaluation, compare_outcome_ev_all_methods, evaluate_ev_from_probability,
    evaluate_outcome_ev,
};
pub use execution::{
    HedgeMode, HedgeReport, MiddleReport, evaluate_hedge, evaluate_middle,
    evaluate_middle_custom_stakes,
};
pub use overround::{
    MarketOverround, booksum, implied_probabilities_into, market_overround, overround,
    turnover_margin,
};

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
