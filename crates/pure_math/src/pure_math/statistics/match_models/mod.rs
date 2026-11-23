//! # Match Score Matrix & Derivative Market Models
//!
//! Zero-allocation score-grid pricing for association football, hockey, and low-scoring team sports:
//!
//! - **Independent Poisson**: $P(X = x, Y = y) = \text{Pois}(x; \lambda_H)\,\text{Pois}(y; \lambda_A)$
//! - **Dixon-Coles (1997)**: Low-score dependence factor $\tau_{\lambda_H,\lambda_A}(x, y; \rho)$
//!   correcting $(0, 0), (1, 0), (0, 1), (1, 1)$ draw/underdispersion anomalies
//! - **Karlis-Ntzoufras (2003) Bivariate Poisson**: Shared latent tempo intensity $\lambda_3 \ge 0$
//!
//! All score grids are stored row-major and indexed via
//! [`math_commons::math_kernel::types::flatten_2d_index`].
//!
//! ## Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{Odds, kelly_fraction};
//! use math_explorer::pure_math::statistics::match_models::{
//!     DixonColesRho, dixon_coles_matrix_into, match_1x2_probabilities,
//! };
//! use math_explorer::pure_math::statistics::zip_regression::PoissonRate;
//!
//! let mut grid = [0.0_f64; 81]; // 0..=8 goals per side
//! dixon_coles_matrix_into(
//!     PoissonRate::new(1.65).unwrap(),
//!     PoissonRate::new(1.15).unwrap(),
//!     DixonColesRho::new(-0.12).unwrap(),
//!     8,
//!     &mut grid,
//! )
//! .unwrap();
//!
//! let outcomes = match_1x2_probabilities(&grid, 8).unwrap();
//! let stake = kelly_fraction(&outcomes.home_win, &Odds::new(2.15).unwrap()).unwrap();
//! assert!(stake.value() > 0.0);
//! ```

pub mod markets;
pub mod score_matrix;

pub use markets::{
    MatchOutcomeProbs, asian_handicap_home_cover, both_teams_to_score_probability,
    match_1x2_probabilities, over_under_probabilities,
};
pub use score_matrix::{
    DixonColesRho, SharedGoalRate, bivariate_poisson_matrix_into, bivariate_poisson_pmf,
    dixon_coles_matrix_into, dixon_coles_tau, independent_poisson_matrix_into,
};

/// Default match-model family selector for UI and pricing pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScoreModelKind {
    /// Independent Poisson marginal goal rates.
    IndependentPoisson,
    /// Dixon & Coles (1997) low-score $\tau$-adjusted Poisson model.
    DixonColes,
    /// Karlis & Ntzoufras (2003) shared-rate bivariate Poisson model.
    BivariatePoisson,
}

#[cfg(test)]
mod tests;
