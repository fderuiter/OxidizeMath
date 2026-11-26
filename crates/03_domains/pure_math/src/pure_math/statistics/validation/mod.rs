//! # Post-Bet Pricing Validation, Calibration & Yield Significance
//!
//! Zero-allocation statistical validation routines separating pre-settlement pricing skill
//! (Closing Line Value and proper scoring rules) from short-run Bernoulli outcome variance:
//!
//! - **Closing Line Value ([`clv`])**: Single-bet and single-pass Welford ledger CLV summaries
//!   (EV percentage, implied-probability edge, log-odds shift) plus two-way and multi-way
//!   closing-market vigorish removal per Kaunitz, Zhong, and Kreiner (2017).
//! - **Model Calibration ([`calibration`])**: Brier score and open-interval domain-guarded
//!   binary log-loss benchmarking model probabilities against devigged closing lines.
//! - **Yield Significance ([`significance`])**: Exact heterogeneous-odds Bernoulli profit
//!   variance, realized yield z-scores, one-sided and two-sided p-values, and required
//!   95%-significance sample sizes.
//!
//! ## Quick Start
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{Odds, UnitInterval};
//! use math_explorer::pure_math::statistics::validation::{
//!     SettledBet, evaluate_yield_significance, single_bet_clv, summarize_clv,
//! };
//!
//! let placed = Odds::new(2.10).unwrap();
//! let close = UnitInterval::new(0.50).unwrap();
//! let single = single_bet_clv(&placed, &close).unwrap();
//! assert!((single.ev_pct - 5.0).abs() < 1e-9);
//!
//! let ledger = [
//!     SettledBet::new(Odds::new(2.10).unwrap(), UnitInterval::new(0.50).unwrap(), 100.0, false).unwrap(),
//!     SettledBet::new(Odds::new(2.25).unwrap(), UnitInterval::new(0.48).unwrap(), 150.0, true).unwrap(),
//!     SettledBet::new(Odds::new(1.95).unwrap(), UnitInterval::new(0.54).unwrap(), 100.0, false).unwrap(),
//! ];
//! let clv_report = summarize_clv(&ledger).unwrap();
//! let yield_report = evaluate_yield_significance(&ledger).unwrap();
//! assert!(clv_report.unweighted_mean_clv > 0.0);
//! assert!(yield_report.p_value_two_sided > 0.05);
//! ```

pub mod calibration;
pub mod clv;
pub mod significance;

pub use calibration::{
    CalibrationComparison, binary_log_loss, brier_score, compare_calibration,
    settled_closing_calibration,
};
pub use clv::{
    ClvMetrics, ClvSummary, SettledBet, devig_multi_way, devig_multi_way_into, devig_two_way,
    single_bet_clv, single_bet_clv_two_way, summarize_clv,
};
pub use significance::{
    SampleSize95, YieldSignificance, Z_95_ONE_SIDED, Z_95_TWO_SIDED,
    evaluate_yield_against_closing, evaluate_yield_significance, required_bets_for_95_significance,
};

// [cite:clinical_trials_statistics]

#[cfg(test)]
mod tests;
