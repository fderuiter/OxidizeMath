//! # Kelly Criterion for Optimal Bet Sizing
//!
//! This module implements the Kelly Criterion, a formula for optimal bet sizing
//! that maximizes the expected logarithmic growth rate of wealth.
//!
//! ## Overview
//!
//! The Kelly Criterion answers the question: **"What fraction of my bankroll should I bet?"**
//! given known probabilities and odds. It provides a mathematically optimal strategy
//! that maximizes long-term wealth growth while avoiding ruin.
//!
//! ## Mathematical Framework
//!
//! ### The Kelly Formula
//!
//! For a binary outcome (win/loss) with:
//! - **p** = probability of winning
//! - **q** = 1 - p = probability of losing
//! - **b** = net profit multiplier (odds - 1)
//!
//! The optimal bet fraction is:
//!
//! ```text
//! f* = (bp - q) / b
//! ```
//!
//! ### Interpretation
//!
//! - **f* > 0**: Positive edge, bet f* of bankroll
//! - **f* = 0**: No edge, don't bet
//! - **f* < 0**: Negative edge, don't bet (or bet the other side)
//!
//! ### Expected Growth Rate
//!
//! The logarithmic growth rate for fraction f is:
//!
//! ```text
//! g(f) = p ln(1 + bf) + q ln(1 - f)
//! ```
//!
//! The Kelly fraction f* maximizes g(f).
//!
//! ### Fractional Kelly
//!
//! Many practitioners use **fractional Kelly** to reduce volatility:
//!
//! - **Quarter-Kelly** (f*/4): Very conservative, ~94% of growth with 6% of variance
//! - **Half-Kelly** (f*/2): Popular compromise, ~75% of growth with 25% of variance
//! - **Full Kelly** (f*): Maximum growth but high volatility
//!
//! ## Why Kelly Works
//!
//! The Kelly Criterion has several optimal properties:
//!
//! 1. **Maximizes long-term growth**: Geometric mean return is maximized
//! 2. **Minimizes time to goal**: Reaches any wealth goal in minimum expected time
//! 3. **Never risks ruin**: For f ≤ f*, probability of ruin is zero
//! 4. **Information-theoretic**: Related to Shannon entropy and channel capacity
//!
//! ## Applications
//!
//! - **Gambling**: Blackjack, sports betting, poker
//! - **Trading**: Position sizing in stocks, options, forex
//! - **Investing**: Portfolio allocation with edge estimates
//! - **Venture Capital**: Investment sizing with success probabilities
//!
//! ## Example: Basic Usage
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{
//!     kelly_fraction, expected_value, UnitInterval, Odds
//! };
//!
//! // You estimate 55% chance of winning at even money odds
//! let prob = UnitInterval::new(0.55).unwrap();
//! let odds = Odds::new(2.0).unwrap();
//!
//! // Check if there's an edge
//! let ev = expected_value(&prob, &odds);
//! println!("Expected value per $1 bet: ${:.3}", ev);
//!
//! // Compute optimal Kelly fraction
//! let kelly = kelly_fraction(&prob, &odds).unwrap();
//! println!("Optimal Kelly bet: {:.1}% of bankroll", kelly.value() * 100.0);
//!
//! // Calculate actual bet amount
//! let bankroll = 10000.0;
//! let bet_amount = kelly.bet_amount(bankroll).unwrap();
//! println!("Bet amount: ${:.2}", bet_amount);
//! ```
//!
//! ## Example: Fractional Kelly
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{
//!     kelly_fraction, fractional_kelly, expected_growth_rate,
//!     UnitInterval, Odds, variants
//! };
//!
//! let prob = UnitInterval::new(0.55).unwrap();
//! let odds = Odds::new(2.0).unwrap();
//!
//! // Compare growth rates for different strategies
//! let full = kelly_fraction(&prob, &odds).unwrap();
//! let half = variants::half_kelly(&prob, &odds).unwrap();
//! let quarter = variants::quarter_kelly(&prob, &odds).unwrap();
//!
//! println!("Full Kelly: {:.3}, growth: {:.4}",
//!     full.value(), expected_growth_rate(&prob, &odds, &full));
//! println!("Half Kelly: {:.3}, growth: {:.4}",
//!     half.value(), expected_growth_rate(&prob, &odds, &half));
//! println!("Quarter Kelly: {:.3}, growth: {:.4}",
//!     quarter.value(), expected_growth_rate(&prob, &odds, &quarter));
//! ```
//!
//! ## Example: Converting Odds Formats
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::Odds;
//!
//! // American odds (including -100 and +100 even-money boundaries)
//! let underdog = Odds::from_american(200.0).unwrap(); // +200
//! assert!((underdog.value() - 3.0).abs() < 1e-9);
//! assert!((underdog.to_american() - 200.0).abs() < 1e-9);
//!
//! let favorite = Odds::from_american(-150.0).unwrap(); // -150
//! assert_eq!(favorite.to_fractional(1000), (2, 3));
//!
//! // Fractional odds and implied probability
//! let odds_5_to_2 = Odds::from_fractional(5.0, 2.0).unwrap();
//! let from_prob = Odds::from_implied_probability(odds_5_to_2.implied_probability()).unwrap();
//! assert!((from_prob.value() - odds_5_to_2.value()).abs() < 1e-9);
//! ```
//!
//! ## Example: Realistic Sports Betting
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{
//!     kelly_fraction, expected_value, UnitInterval, Odds, variants
//! };
//!
//! // You've modeled a game and think Team A has 52% win probability
//! // The sportsbook offers -110 odds (American) on Team A
//! let your_prob = UnitInterval::new(0.52).unwrap();
//! let odds = Odds::from_american(-110.0).unwrap();
//!
//! let ev = expected_value(&your_prob, &odds);
//! if ev > 0.0 {
//!     println!("You have a {:.2}% edge!", ev * 100.0);
//!
//!     let full_kelly = kelly_fraction(&your_prob, &odds).unwrap();
//!     let half_kelly = variants::half_kelly(&your_prob, &odds).unwrap();
//!
//!     println!("Full Kelly: {:.2}% of bankroll", full_kelly.value() * 100.0);
//!     println!("Half Kelly: {:.2}% of bankroll", half_kelly.value() * 100.0);
//!
//!     // Most professional bettors use fractional Kelly for risk management
//!     let recommended = half_kelly;
//!     let bankroll = 5000.0;
//!     println!("Recommended bet: ${:.2}",
//!         recommended.bet_amount(bankroll).unwrap());
//! } else {
//!     println!("No edge - don't bet!");
//! }
//! ```
//!
//! ## Important Caveats
//!
//! 1. **Requires accurate probabilities**: Garbage in, garbage out
//! 2. **High volatility**: Full Kelly can have large drawdowns (~50%)
//! 3. **Assumes accurate bankroll tracking**: Must update after each bet
//! 4. **Independent trials**: Assumes outcomes are independent
//! 5. **Unlimited betting**: Assumes you can always bet the exact fraction
//!
//! ## Risk Management
//!
//! Common strategies to manage Kelly risk:
//!
//! - **Use fractional Kelly** (1/2 or 1/4) for lower variance
//! - **Cap maximum bet** at some percentage (e.g., 5% of bankroll)
//! - **Use confidence intervals** on your probability estimates
//! - **Separate bankrolls** for different strategies/markets
//!
//! ## References
//!
//! - Kelly, J. L. (1956). *A New Interpretation of Information Rate*.
//!   Bell System Technical Journal, 35(4), 917-926.
//! - Thorp, E. O. (1969). *Optimal Gambling Systems for Favorable Games*.
//!   Review of the International Statistical Institute, 37(3), 273-293.
//! - Poundstone, W. (2005). *Fortune's Formula: The Untold Story of the
//!   Scientific Betting System That Beat the Casinos and Wall Street*.
//!   Hill and Wang.

pub mod core;
pub mod criterion;

// Re-export main types and functions
pub use core::{BankrollFraction, Odds, UnitInterval};
pub use criterion::{
    expected_growth_rate, expected_value, fractional_kelly, kelly_fraction, variants,
};

// [cite:clinical_trials_statistics]
