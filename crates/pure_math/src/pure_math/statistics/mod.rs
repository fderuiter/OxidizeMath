//! # Statistical Modeling & Competitive Analysis
//!
//! A comprehensive suite of statistical tools designed for competitive modeling,
//! risk analysis, and advanced data science.
//!
//! Unlike standard statistics libraries that focus on hypothesis testing, this module
//! emphasizes **generative modeling**, **ranking systems**, and **decision theory**.
//!
//! ## Core Domains
//!
//! ###  Competitive Ranking
//! *   **[Glicko-2](glicko2)**: The gold standard for player rating systems (Chess, CS:GO).
//!     Tracks rating volatility and deviation to handle inactive players correctly.
//!
//! ###  Stochastic Processes
//! *   **[Markov Chains](markov)**: Discrete-Time (DTMC), Continuous-Time (CTMC), and Hidden Markov Models (HMM).
//!     Essential for modeling state transitions in sports (e.g., possession value) or finance.
//! *   **[Ornstein-Uhlenbeck](ou_process)**: Mean-reverting processes for modeling "momentum" or "hot hands"
//!     where performance fluctuates around a long-term average.
//!
//! ###  Risk & Decision Theory
//! *   **[Kelly Criterion](kelly)**: Optimal capital allocation strategy to maximize logarithmic wealth growth.
//! *   **[Risk of Ruin & Solvency](ruin)**: Exact Gambler's Ruin, Cramér-Lundberg adjustment coefficients,
//!     proportional-Kelly drawdown barrier bounds, and zero-allocation Monte Carlo bankroll ensembles.
//! *   **[Edge Validation & Calibration](validation)**: Closing Line Value (CLV), market vigorish removal,
//!     Brier/log-loss proper scoring rules, and realized yield statistical significance.
//! *   **[Copulas](copula)**: sophisticated dependency modeling (e.g., Gaussian Copula) for correlated events,
//!     critical for pricing derivatives or "Same Game Parlays".
//!
//! ###  Advanced Data Analysis
//! *   **[Topological Data Analysis (TDA)](tda)**: Uses persistent homology to find structural features (holes, voids)
//!     in high-dimensional point clouds.
//! *   **[ZIP Regression](zip_regression)**: Zero-Inflated Poisson models for count data with excess zeros
//!     (e.g., goals scored by a defensive player).
//! *   **[Regression](regression)**: Standard multivariate linear regression.
//!
//! ##  Quick Start: Optimal Betting with Kelly Criterion
//!
//! Calculate the optimal fraction of your bankroll to wager given an edge.
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{
//!     kelly_fraction, UnitInterval, Odds
//! };
//!
//! fn main() -> math_explorer::error::Result<()> {
//!     // Scenario: A coin toss that lands heads 55% of the time, paying 2.0 (even money).
//!     let probability = UnitInterval::new(0.55)?;
//!     let odds = Odds::new(2.0)?;
//!
//!     // Calculate full Kelly fraction
//!     let fraction = kelly_fraction(&probability, &odds)?;
//!
//!     // Kelly says: bet 10% of your bankroll
//!     // f = (bp - q) / b = (1.0*0.55 - 0.45) / 1.0 = 0.10
//!     assert!((fraction.value() - 0.10).abs() < math_explorer::registry::TOLERANCE_FAST);
//!     println!("Optimal bet size: {:.1}%", fraction.value() * 100.0);
//!
//!     Ok(())
//! }
//! ```
//!
//! ##  Deep Dive: Glicko-2 Rating Update
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::glicko2::{
//!     GlickoPlayer, Rating, RatingDeviation, Volatility, MatchResult, update_rating, SystemConstant
//! };
//!
//! fn main() -> math_explorer::error::Result<()> {
//!     // Initialize a player (Rating=1500, RD=350, Vol=0.06)
//!     let player = GlickoPlayer::new(
//!         Rating::new(1500.0)?,
//!         RatingDeviation::new(350.0)?,
//!         Volatility::new(0.06)?
//!     );
//!
//!     // Player competes against a strong opponent (Rating=1700, RD=300) and WINS
//!     let opponent = GlickoPlayer::new(
//!         Rating::new(1700.0)?,
//!         RatingDeviation::new(300.0)?,
//!         Volatility::new(0.06)?
//!     );
//!
//!     // 1.0 = Win
//!     let result = MatchResult::new(opponent, 1.0)?;
//!
//!     // Update ratings (tau=0.5 constraints volatility change)
//!     let tau = SystemConstant::new(0.5)?;
//!     let new_player = update_rating(&player, &[result], &tau)?;
//!
//!     // Rating should increase, RD should decrease (more certainty)
//!     println!("New Rating: {:.0}", new_player.rating.value());
//!     assert!(new_player.rating.value() > 1500.0);
//!     assert!(new_player.rating_deviation.value() < 350.0);
//!
//!     Ok(())
//! }
//! ```

pub mod copula;
pub mod glicko2;
pub mod kelly;
pub mod markov;
pub mod ou_process;
pub mod regression;
pub mod ruin;
pub mod tda;
pub mod validation;
pub mod zip_regression;

// [cite:clinical_trials_statistics]
