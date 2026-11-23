//! # Ornstein-Uhlenbeck Process for Momentum Modeling
//!
//! This module implements the Ornstein-Uhlenbeck (OU) process for modeling
//! mean-reverting performance dynamics in sports analytics.
//!
//! ## Overview
//!
//! Unlike discrete count models (ZIP), the OU process treats performance as a
//! **continuous, fluctuating signal** that naturally reverts to a long-term average.
//!
//! ## The SDE Formulation
//!
//! ```text
//! dX_t = θ(μ - X_t)dt + σdW_t
//! ```
//!
//! where:
//! - **X_t**: Current performance level
//! - **μ (mu)**: Long-term mean (true skill level)
//! - **θ (theta)**: Mean reversion rate (speed of reversion)
//! - **σ (sigma)**: Volatility (noise/randomness)
//! - **dW_t**: Wiener process (Brownian motion)
//!
//! ### Drift Term: θ(μ - X_t)dt
//!
//! The deterministic "pull" toward the mean:
//! - If X_t > μ (hot streak), drift is negative → pulls down
//! - If X_t < μ (slump), drift is positive → pulls up
//! - θ controls the strength of this pull
//!
//! ### Diffusion Term: σdW_t
//!
//! The random fluctuations that prevent straight-line movement.
//!
//! ## Euler-Maruyama Numerical Method
//!
//! Discrete-time approximation:
//! ```text
//! X_{t+Δt} = X_t + θ(μ - X_t)Δt + σ√(Δt)Z
//! ```
//! where Z ~ N(0,1)
//!
//! ## Sports Analytics Applications
//!
//! ### 1. Momentum Classification
//!
//! Players are classified by their mean reversion rate θ:
//!
//! - **θ > 2.0**: "Flash in the pan" - streaks revert instantly
//! - **0.5 < θ ≤ 2.0**: "Normal" mean reversion
//! - **θ ≤ 0.5**: "Heat check" player - momentum is sticky
//!
//! ### 2. Live Probability Pricing
//!
//! Use Monte Carlo simulation to estimate comeback probabilities:
//! - Simulate thousands of paths to end of game
//! - Account for volatility (σ) to capture extreme outcomes
//! - High volatility → higher comeback probability
//!
//! ## Example: Shooting Percentage Dynamics
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::ou_process::{
//!     EulerMaruyama, OuParams, TimeStep
//! };
//! use math_explorer::rand::SeedableRng;
//! use math_explorer::rand::rngs::StdRng;
//!
//! // Player has 45% true shooting percentage
//! // Currently at 60% (hot streak)
//! // Model parameters:
//! let mu = 0.45;      // True skill
//! let theta = 1.0;    // Normal mean reversion
//! let sigma = 0.15;   // Moderate volatility
//!
//! let params = OuParams::from_values(mu, theta, sigma).unwrap();
//! let dt = TimeStep::new(0.01).unwrap();
//! let solver = EulerMaruyama::new(params, dt);
//!
//! // Simulate rest of game (100 possessions)
//! let mut rng = math_explorer::rng::OxidizeRng::default();
//! let trajectory = solver.simulate(0.60, 100, &mut rng);
//!
//! // Analyze final shooting percentage
//! let final_pct = trajectory.last().unwrap();
//! println!("Final shooting %: {:.1}%", final_pct * 100.0);
//! ```
//!
//! ## Example: Comeback Probability
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::ou_process::{OuAnalyzer, OuParams, TimeStep};
//! use math_explorer::rand::SeedableRng;
//! use math_explorer::rand::rngs::StdRng;
//!
//! // Team is down 10 points with high volatility
//! let params = OuParams::from_values(0.0, 1.0, 0.3).unwrap();  // High σ
//! let dt = TimeStep::new(0.01).unwrap();
//! let analyzer = OuAnalyzer::new(params, dt);
//!
//! let mut rng = math_explorer::rng::OxidizeRng::default();
//!
//! // What's the probability of a comeback?
//! let current_deficit = -10.0;
//! let target = 0.0;  // Tie or better
//! let time_remaining = 1.0;  // One quarter
//!
//! let comeback_prob = analyzer.comeback_probability(
//!     current_deficit,
//!     target,
//!     time_remaining,
//!     10000,  // Monte Carlo paths
//!     &mut rng
//! );
//!
//! println!("Comeback probability: {:.1}%", comeback_prob * 100.0);
//! ```
//!
//! ## References
//!
//! - Uhlenbeck, G. E., & Ornstein, L. S. (1930). "On the Theory of the Brownian Motion."
//!   *Physical Review*, 36(5), 823-841.
//! - Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of Stochastic Differential
//!   Equations*. Springer-Verlag.

pub mod analysis;
pub mod core;
/// Exact discrete Gaussian transition solver for the Ornstein-Uhlenbeck SDE.
pub mod exact_solver;
pub mod solver;

pub use analysis::{MomentumType, OuAnalyzer, PerformanceStats, estimate_ou_params};
pub use core::{LongTermMean, OuParams};
pub use exact_solver::ExactOuSolver;
pub use solver::{EulerMaruyama, TimeStep};

// [cite:clinical_trials_statistics]
