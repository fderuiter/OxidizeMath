//! # Glicko-2 Rating System
//!
//! This module implements the Glicko-2 rating system, an improvement over the
//! original Glicko and ELO rating systems for competitive games and sports.
//!
//! ## Overview
//!
//! Glicko-2 extends the original Glicko system by adding a **volatility** parameter
//! that measures the degree of expected fluctuation in a player's rating. This allows
//! the system to identify players whose performance is consistent versus those who
//! are erratic or still developing.
//!
//! ## Mathematical Framework
//!
//! ### Core Parameters
//!
//! Each player is characterized by three values:
//!
//! - **Rating (r)**: Skill estimate (default: 1500)
//! - **Rating Deviation (RD)**: Uncertainty in rating (default: 350 for new players)
//! - **Volatility (σ)**: Expected degree of rating fluctuation (default: 0.06)
//!
//! ### Glicko-2 Scale
//!
//! Internally, the algorithm uses a different scale:
//!
//! - **μ** = (r - 1500) / 173.7178
//! - **φ** = RD / 173.7178
//! - **σ** remains unchanged
//!
//! ### The Algorithm
//!
//! For a rating period with games against opponents j = 1,...,m:
//!
//! 1. **Compute g(φⱼ)**:
//!    ```text
//!    g(φⱼ) = 1 / √(1 + 3φⱼ²/π²)
//!    ```
//!
//! 2. **Compute expected outcome E**:
//!    ```text
//!    E(μ, μⱼ, φⱼ) = 1 / (1 + exp(-g(φⱼ)(μ - μⱼ)))
//!    ```
//!
//! 3. **Compute variance v**:
//!    ```text
//!    v = 1 / Σⱼ g(φⱼ)² E(μ, μⱼ, φⱼ)(1 - E(μ, μⱼ, φⱼ))
//!    ```
//!
//! 4. **Compute improvement estimate Δ**:
//!    ```text
//!    Δ = v · Σⱼ g(φⱼ)(sⱼ - E(μ, μⱼ, φⱼ))
//!    ```
//!    where sⱼ = 1 for win, 0.5 for draw, 0 for loss
//!
//! 5. **Update volatility σ'** by solving:
//!    ```text
//!    f(x) = exp(x)(Δ² - φ² - v - exp(x)) / (2(φ² + v + exp(x))²) - (x - ln(σ²)) / τ² = 0
//!    ```
//!    using the Illinois algorithm (modified false position method)
//!
//! 6. **Update RD**:
//!    ```text
//!    φ* = √(φ² + σ'²)
//!    φ' = 1 / √(1/φ*² + 1/v)
//!    ```
//!
//! 7. **Update rating**:
//!    ```text
//!    μ' = μ + φ'² Σⱼ g(φⱼ)(sⱼ - E(μ, μⱼ, φⱼ))
//!    ```
//!
//! ### System Constant τ
//!
//! The parameter τ constrains volatility changes:
//!
//! - Smaller τ (0.3-0.5): Conservative, prevents large volatility swings
//! - Larger τ (0.8-1.2): More responsive to unexpected results
//! - Recommended default: 0.5
//!
//! ## Applications
//!
//! Glicko-2 is used in:
//!
//! - **Chess**: Online platforms (e.g., Lichess, Chess.com)
//! - **Esports**: Competitive gaming rankings
//! - **Sports**: Player performance tracking
//! - **Machine Learning**: Model evaluation tournaments
//!
//! ## Example: Basic Usage
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::glicko2::{
//!     GlickoPlayer, Rating, RatingDeviation, Volatility,
//!     MatchResult, SystemConstant, update_rating
//! };
//!
//! // Create two players
//! let player = GlickoPlayer::new(
//!     Rating::new(1500.0).unwrap(),
//!     RatingDeviation::new(200.0).unwrap(),
//!     Volatility::new(0.06).unwrap(),
//! );
//!
//! let opponent = GlickoPlayer::new(
//!     Rating::new(1400.0).unwrap(),
//!     RatingDeviation::new(30.0).unwrap(),
//!     Volatility::new(0.06).unwrap(),
//! );
//!
//! // Record a win against the opponent
//! let results = vec![MatchResult::new(opponent, 1.0).unwrap()];
//!
//! // Update the player's rating
//! let tau = SystemConstant::default();
//! let updated = update_rating(&player, &results, &tau).unwrap();
//!
//! println!("Old rating: {}", player.rating.value());
//! println!("New rating: {}", updated.rating.value());
//! ```
//!
//! ## Example: Multiple Games
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::glicko2::{
//!     GlickoPlayer, Rating, RatingDeviation, Volatility,
//!     MatchResult, SystemConstant, update_rating
//! };
//!
//! let player = GlickoPlayer::default(); // Uses default values
//!
//! // Play against multiple opponents
//! let opponent1 = GlickoPlayer::new(
//!     Rating::new(1400.0).unwrap(),
//!     RatingDeviation::new(30.0).unwrap(),
//!     Volatility::new(0.06).unwrap(),
//! );
//!
//! let opponent2 = GlickoPlayer::new(
//!     Rating::new(1550.0).unwrap(),
//!     RatingDeviation::new(100.0).unwrap(),
//!     Volatility::new(0.06).unwrap(),
//! );
//!
//! let results = vec![
//!     MatchResult::new(opponent1, 1.0).unwrap(),  // Win
//!     MatchResult::new(opponent2, 0.5).unwrap(),  // Draw
//! ];
//!
//! let tau = SystemConstant::default();
//! let updated = update_rating(&player, &results, &tau).unwrap();
//!
//! println!("Rating: {}", updated.rating.value());
//! println!("RD: {}", updated.rating_deviation.value());
//! println!("Volatility: {}", updated.volatility.value());
//! ```
//!
//! ## Example: Handling Inactivity
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::glicko2::{
//!     GlickoPlayer, Rating, RatingDeviation, Volatility,
//!     update_rating, SystemConstant
//! };
//!
//! let player = GlickoPlayer::new(
//!     Rating::new(1600.0).unwrap(),
//!     RatingDeviation::new(50.0).unwrap(),
//!     Volatility::new(0.06).unwrap(),
//! );
//!
//! // No games played this period
//! let results = vec![];
//! let tau = SystemConstant::default();
//!
//! let updated = update_rating(&player, &results, &tau).unwrap();
//!
//! // Rating unchanged, but RD increases (more uncertainty)
//! assert_eq!(updated.rating.value(), player.rating.value());
//! assert!(updated.rating_deviation.value() > player.rating_deviation.value());
//! ```
//!
//! ## References
//!
//! - Glickman, M. E. (2012). *Example of the Glicko-2 system*.
//!   Retrieved from <http://www.glicko.net/glicko/glicko2.pdf>
//! - Glickman, M. E. (1999). *Parameter estimation in large dynamic paired comparison experiments*.
//!   Journal of the Royal Statistical Society: Series C, 48(3), 377-394.

#[doc(hidden)]
pub mod core;
pub mod rating;

// Re-export main types
pub use core::{
    GlickoPlayer, MatchResult, Rating, RatingDeviation, SystemConstant, Volatility,
    expected_outcome, g_function,
};
pub use rating::{increase_rd_for_inactivity, update_rating};

// [cite:clinical_trials_statistics]

#[cfg(test)]
mod tests;
