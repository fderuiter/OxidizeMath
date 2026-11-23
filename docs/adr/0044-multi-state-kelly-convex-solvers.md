# ADR 0044: Zero-Allocation Multi-State, Portfolio, and Drawdown-Constrained Kelly Convex Solvers

## Status
Accepted (extends [ADR 0035](0035-kelly-net-odds-seeded-rng-and-paradox-labs.md))

## Context
Previously, `pure_math::statistics::kelly` modeled only binary two-outcome wagers (`+b` win, `-1` loss) with heuristic fractional (`half_kelly`, `quarter_kelly`) sizing, and `math_explorer_gui/src/tabs/financial_math/bet_size.rs` carried `#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]` suppressions inside a monolithic file. Quantitative wagering and portfolio allocation require exact expected log-wealth maximization over integer push spreads (`+b, 0, -1`), Asian quarter-line split handicaps (`+b, +b/2, 0, -1/2, -1`), mutually exclusive `N`-way markets, simultaneous `K`-game independent slates, and Busseti-Ryu-Boyd martingale drawdown constraints—all executing with zero heap allocations inside `VerifiedEngine::run_verified`.

## Decision
1. **Modular Zero-Allocation Solvers in `crates/pure_math`**:
   - Decompose [`crates/pure_math/src/pure_math/statistics/kelly/`](../../crates/pure_math/src/pure_math/statistics/kelly/mod.rs) into focused submodules (`core`, `push`, `quarter_line`, `market`, `slate_eval`, `slate`, `drawdown`, `tests`) while keeping all legacy binary APIs unbroken.
   - Implement closed-form `push_kelly_fraction` ($f^* = (b p_{\text{win}} - p_{\text{loss}}) / (b (1 - p_{\text{push}}))$) and bracketed 1D root-finding `quarter_line_kelly_fraction` reusing `Bisection` from `pure_math::analysis::roots`.
   - Implement Smoczynski-Tomkins (2010) active-set allocation `solve_mutually_exclusive_market` with caller-supplied `MarketScratch` (`MAX_MARKET_OUTCOMES = 32`).
   - Implement projected FISTA + Barzilai-Borwein simplex solver `solve_simultaneous_slate` (`MAX_SLATE_WAGERS = 64`) using exact state convolution for $K \le 10$ and 128-node 1D Frullani/Gamma Laplace quadrature ($O(Q \cdot K)$) for $K > 10$ via caller-supplied `SlateScratch`.
   - Implement Busseti, Ryu, and Boyd (2016) martingale drawdown solvers (`DrawdownConstraint` with $\lambda = \ln(\beta) / \ln(\alpha)$, $\mathbb{E}[R^{-\lambda}] \le 1$) across binary, push, quarter-line, and simultaneous slate models.
2. **Decomposed Financial Math GUI Tools**:
   - Split `bet_size` and `bankroll_growth` into directory submodules (`state`, `simulation`, `ui`, `mod`, `tests`), removing `too_many_lines` and `cognitive_complexity` suppressions while preserving exact JSON backward compatibility for existing `InteractiveTool` presets and URL states.
   - Extend `BankrollEnsembleLab` and `BankrollGrowthTool` with push settlement, quarter-line and simultaneous slate trajectories, and drawdown-constrained curves.

## Consequences
- **Positive**: Delivers exact convex Kelly allocations across single multi-state wagers, mutually exclusive markets, and simultaneous slates up to 64 wagers in `< 1 ms` with zero runtime heap allocations under `VerifiedAllocator`.
- **Neutral**: Dependent multi-leg parlays with arbitrary empirical correlation matrices remain out of scope for the independent-slate quadrature solver.
