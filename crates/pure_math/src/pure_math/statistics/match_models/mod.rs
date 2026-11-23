//! # Calibrated 2D Score Grids & Quantitative Football Match Models
//!
//! Bounded, zero-heap-allocation 2D score probability matrices (`ScoreGrid`) supporting:
//! - **Independent Maher (1982) Poisson** ([`poisson`])
//! - **Karlis-Ntzoufras (2003) Bivariate Poisson** ([`bivariate_poisson`])
//! - **Dixon-Coles (1997) Low-Score Correlation** ([`dixon_coles`])
//! - **Time-Decayed Maximum Likelihood Calibration** ([`calibration`])
//! - **Exact Match Market Pricing** ([`markets`]): `1X2`, Draw-No-Bet, BTTS, Asian Handicap,
//!   and Over/Under totals across whole, half, and split quarter lines.
//!
//! ## Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::match_models::{
//!     DixonColesParams, IndependentPoissonParams, asian_handicap, one_x_two, over_under,
//! };
//!
//! let indep = IndependentPoissonParams::new(1.35, 1.15).unwrap().default_grid().unwrap();
//! let m_indep = one_x_two(&indep);
//! assert!((m_indep.home_win - 0.413382).abs() < 1e-6);
//! assert!((m_indep.draw - 0.268397).abs() < 1e-6);
//! assert!((over_under(&indep, 2.5).unwrap().under - 0.543813).abs() < 1e-6);
//! assert!((asian_handicap(&indep, -1.0).unwrap().push - 0.222207).abs() < 1e-6);
//!
//! // Dixon-Coles negative rho boosts low-scoring draw mass by +12.35% relative to independent Poisson
//! let dc = DixonColesParams::new(1.35, 1.15, -0.13).unwrap().default_grid().unwrap();
//! assert!((one_x_two(&dc).draw - 0.301531).abs() < 1e-6);
//! ```

pub mod bivariate_poisson;
pub mod calibration;
pub mod dixon_coles;
pub mod grid;
pub mod markets;
pub mod poisson;

pub use bivariate_poisson::{
    BivariatePoissonParams, bivariate_poisson_grid, bivariate_poisson_pmf,
};
pub use calibration::{
    DixonColesMleConfig, FittedDixonColes, HistoricalMatch, fit_dixon_coles_mle,
    fit_dixon_coles_with_optimizer,
};
pub use dixon_coles::{
    DixonColesParams, dixon_coles_grid, dixon_coles_rho_bounds, dixon_coles_tau,
};
pub use grid::{DEFAULT_MAX_GOALS, GRID_CAPACITY, GRID_STRIDE, MAX_GOALS_LIMIT, ScoreGrid};
pub use markets::{
    AsianHandicapQuote, BttsQuote, DrawNoBetQuote, OneXTwoQuote, OverUnderQuote, asian_handicap,
    btts, draw_no_bet, one_x_two, over_under,
};
pub use poisson::{
    IndependentPoissonParams, independent_poisson_grid, log_factorial, poisson_log_pmf, poisson_pmf,
};

// [cite:clinical_trials_statistics]

#[cfg(test)]
mod tests;
