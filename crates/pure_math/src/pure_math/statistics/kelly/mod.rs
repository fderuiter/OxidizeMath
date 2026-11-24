//! # Kelly Criterion & Multi-State Convex Portfolio Sizing
//!
//! Optimal bankroll allocation maximizing expected logarithmic wealth growth across
//! isolated binary wagers, three-state push spreads, five-state Asian quarter-lines,
//! mutually exclusive `N`-way markets, simultaneous independent `K`-game slates, and
//! Busseti-Ryu-Boyd martingale drawdown constraints.
//!
//! ## Formulations
//!
//! - **Binary Kelly (`core`, `criterion`)**:
//!   `f* = (b * p - q) / b`, `g(f) = p * ln(1 + b * f) + q * ln(1 - f)`.
//! - **Three-State Push Kelly (`push`)**:
//!   `EV = b * p_win - p_loss`, `f* = (b * p_win - p_loss) / (b * (1 - p_push))`.
//! - **Five-State Asian Quarter-Line Kelly (`quarter_line`)**:
//!   Payoffs `{+b, +b/2, 0, -1/2, -1}` solved via bracketed [`crate::pure_math::analysis::roots::Bisection`]
//!   on the strictly decreasing log-growth derivative `g'(f) = 0`.
//! - **Mutually Exclusive Markets (`market`)**:
//!   Exact Smoczynski-Tomkins (2010) active-set pruning with reserve rate
//!   `r(S) = (1 - sum_{i in S} p_i) / (1 - sum_{i in S} 1 / O_i)`.
//! - **Simultaneous Independent Slates (`slate`, `slate_eval`)**:
//!   Maximizes `E[ln(1 + sum_i f_i X_i)]` over `f_i >= 0, sum_i f_i <= 1` via restarted
//!   FISTA / Barzilai-Borwein steps using exact state convolution (`K <= 10`) and `O(Q * K)`
//!   one-dimensional Frullani / Gamma Laplace quadrature (`K > 10`).
//! - **Martingale Drawdown Risk Bounds (`drawdown`)**:
//!   Enforces `E[R^(-lambda)] <= 1` with `lambda = ln(beta) / ln(alpha)` so
//!   `P(inf_{t >= 1} W_t / W_0 <= alpha) <= beta` across single wagers and concurrent slates.
//! - **Multi-Asset Covariance Kelly & Tail Risk (`portfolio`, `tail_risk`)**:
//!   Solves unconstrained Cholesky covariance Kelly `w* = Sigma^-1 (mu - r * 1)`, diagonal
//!   independent Kelly, and Duchi-projected long-only Kelly alongside empirical, parametric
//!   log-normal, and Cholesky Monte Carlo `95%`/`99%` `VaR` and `CVaR`.
//!
//! ## Continuous Bankruptcy vs. Finite Drawdown Barriers
//!
//! Under infinitely divisible stakes, $\lim_{t \to \infty} P(W_t = 0) = 0$ for $0 < f \le f^*$.
//! However, hitting a finite wealth floor $B_{\text{floor}} < W_0$ (e.g., a 50% drawdown or
//! minimum-bet table limit) has strictly positive probability bounded by the log-return Lundberg
//! exponent $(B_{\text{floor}} / W_0)^{\theta^*}$ and Brownian diffusion
//! $(B_{\text{floor}} / W_0)^{2\mu/\sigma^2}$ (see
//! [`crate::pure_math::statistics::ruin::kelly_barrier_risk`]). At Full Kelly ($f = f^*$),
//! $\theta^* = 1$ so the asymptotic probability of halving one's bankroll is **50%** (~38% over
//! 200 even-money bets at $p = 0.55$). Use [`drawdown`] or [`ruin`] to enforce explicit drawdown
//! budgets.
//!
//! ## Example: Binary, Push, Market, and Drawdown-Constrained Kelly
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::kelly::{
//!     DrawdownConstraint, MarketOutcome, MarketScratch, Odds, PushWager, UnitInterval,
//!     drawdown_constrained_kelly, kelly_fraction, push_expected_value, push_kelly_fraction,
//!     solve_mutually_exclusive_market,
//! };
//!
//! // 1. Classic binary wager (55% at even money => 10% Kelly, 2.69% at alpha=0.70, beta=0.10)
//! let prob = UnitInterval::new(0.55).unwrap();
//! let even_odds = Odds::from_net_odds(1.0).unwrap();
//! assert!((kelly_fraction(&prob, &even_odds).unwrap().value() - 0.10).abs() < 1e-12);
//! let dc = DrawdownConstraint::new(0.70, 0.10).unwrap();
//! assert!((drawdown_constrained_kelly(&prob, &even_odds, &dc).unwrap().value() - 0.0269).abs() < 1e-4);
//!
//! // 2. Whole-number -3 pushable spread at -110 (50% win, 10% push, 40% loss)
//! let spread_odds = Odds::from_american(-110.0).unwrap();
//! let spread = PushWager::new(0.50, 0.10, 0.40, spread_odds).unwrap();
//! assert!((push_expected_value(&spread) - 3.0 / 55.0).abs() < 1e-12);
//! assert!((push_kelly_fraction(&spread).unwrap().value() - 1.0 / 15.0).abs() < 1e-12);
//!
//! // 3. 3-way mutually exclusive market via Smoczynski-Tomkins active-set pruning
//! let market = [
//!     MarketOutcome::new(UnitInterval::new(0.50).unwrap(), Odds::new(2.20).unwrap()),
//!     MarketOutcome::new(UnitInterval::new(0.28).unwrap(), Odds::new(4.00).unwrap()),
//!     MarketOutcome::new(UnitInterval::new(0.22).unwrap(), Odds::new(3.20).unwrap()),
//! ];
//! let mut scratch = MarketScratch::new();
//! let alloc = solve_mutually_exclusive_market(&market, &mut scratch).unwrap();
//! assert_eq!(alloc.stakes()[2], 0.0);
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
//! ## References
//!
//! - Kelly, J. L. (1956). *A New Interpretation of Information Rate*. Bell System Technical Journal, 35(4), 917-926.
//! - Smoczynski, P., & Tomkins, D. (2010). *An Explicit Formula for Optimally Betting on Horse Races*.
//!   Journal of Applied Probability, 47(2), 341-349.
//! - Busseti, E., Ryu, E. K., & Boyd, S. (2016). *Risk-Constrained Kelly Gambling*.
//!   Journal of Investing, 25(3), 118-134 (arXiv:1603.06183).

pub mod core;
pub mod criterion;
pub mod drawdown;
pub mod market;
pub mod portfolio;
pub mod push;
pub mod quarter_line;
pub mod slate;
pub mod slate_eval;
pub mod tail_risk;

#[cfg(test)]
mod portfolio_tests;
#[cfg(test)]
mod tests;

pub use super::{ruin, validation};
pub use core::{
    BankrollFraction, DrawdownConstraint, EXACT_CONVOLUTION_MAX_WAGERS, LAPLACE_QUADRATURE_NODES,
    MAX_MARKET_OUTCOMES, MAX_SLATE_WAGERS, Odds, PROBABILITY_SUM_TOLERANCE, UnitInterval,
};
pub use criterion::{
    expected_growth_rate, expected_value, fractional_kelly, kelly_fraction, variants,
};
pub use drawdown::{
    drawdown_constrained_kelly, drawdown_constrained_push_kelly,
    drawdown_constrained_quarter_line_kelly, solve_drawdown_constrained_slate,
    solve_drawdown_constrained_slate_with_mode,
};
pub use market::{MarketAllocation, MarketOutcome, MarketScratch, solve_mutually_exclusive_market};
pub use portfolio::{
    PortfolioKellyMetrics, PortfolioKellySolution, PortfolioKellyWorkspace,
    evaluate_portfolio_kelly_metrics, independent_kelly_into, independent_kelly_weights,
    long_only_kelly_projected_gradient, solve_long_only_kelly_into, solve_unconstrained_kelly_into,
    unconstrained_covariance_kelly,
};
pub use push::{
    PushOutcome, PushWager, push_expected_growth_rate, push_expected_value, push_kelly_fraction,
};
pub use quarter_line::{
    QuarterLineOutcome, QuarterLineWager, quarter_line_expected_growth_rate,
    quarter_line_expected_value, quarter_line_growth_derivative, quarter_line_kelly_fraction,
};
pub use slate::{SlateSolution, solve_simultaneous_slate, solve_simultaneous_slate_with_mode};
pub use slate_eval::{MAX_CONVOLUTION_STATES, SlateEvalMode, SlateScratch, SlateWager};
pub use tail_risk::{
    PortfolioHorizonSpec, TailRiskEstimate, TailRiskMetrics, empirical_tail_risk,
    empirical_tail_risk_in_place, empirical_var_cvar_in_place, parametric_lognormal_tail_risk,
    simulate_portfolio_tail_risk_into, simulated_portfolio_tail_risk,
};

// [cite:clinical_trials_statistics]
