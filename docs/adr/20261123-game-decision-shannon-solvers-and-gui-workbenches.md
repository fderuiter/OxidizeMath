# ADR 20261123: Preallocated Game, Decision, and Shannon Solvers with Integrated GUI Workbenches

## Status

Accepted (2026-11-23)

## Context

Previously, [`crates/domain_applied/src/applied/game_theory`](../../crates/domain_applied/src/applied/game_theory/mod.rs) covered replicator ODEs, spatial evolutionary dynamics, Glicko-2 ratings, Myerson reserve-price calculation, and 1D Mean-Field Games, while [`crates/pure_math/src/pure_math/algorithmic_information`](../../crates/pure_math/src/pure_math/algorithmic_information/mod.rs) covered only FRACTRAN and Kolmogorov upper-bound proxies. Neither finite normal-form Nash solvers, coalitional Shapley/Core diagnostics, sequential decision-tree/value-of-information/bandit solvers, nor Shannon entropy and channel-capacity algorithms were available, and `mechanism_design` and `mean_field` lacked interactive workbenches in `math_explorer_gui`. In addition, `ReplicatorDynamicsTool` locked its strategy dimension to `3x3`, and `math_explorer`'s `applied` feature must remain decoupled from `domain_ai`.

## Decision

1. **Zero-Allocation Normal-Form & Cooperative Game Solvers in `domain_applied::game_theory`:**
   - Added [`normal_form`](../../crates/domain_applied/src/applied/game_theory/normal_form/mod.rs) implementing strict/weak iterated elimination of dominated strategies (`DominanceWorkspace`), primal-dual Simplex minimax for zero-sum matrix games (`MinimaxWorkspace`), exact $8\times 8$ Gaussian-elimination support enumeration (`support_enumeration`), and lexicographically perturbed Lemke-Howson pivoting (`LemkeHowsonWorkspace`).
   - Added [`cooperative.rs`](../../crates/domain_applied/src/applied/game_theory/cooperative.rs) over `u16` coalition bitmasks ($n \le 16$) providing exact `f64` Shapley values, superadditivity, convexity, and Core excess diagnostics.

2. **Decision Trees, Value of Information, and Seeded Bandits in `domain_applied::decision_theory`:**
   - Added [`decision_theory`](../../crates/domain_applied/src/applied/decision_theory/mod.rs) implementing `CARA` and `CRRA` utility models and certainty equivalents, arena-indexed finite DAG backward induction (`backward_induction`), `EVPI` and `EVSI` Bayesian preposterior valuation (`VoiWorkspace`), and seeded `Ucb1Bandit` and `ThompsonBandit` policies driven by `oxidize_core::rng::OxidizeRng` without depending on `domain_ai`.
   - Wired `DecisionTheoryError` into `AppliedError` and `math_explorer::error::Error` (`#[cfg(feature = "applied")]`), and documented the module in [`crates/domain_applied/README.md`](../../crates/domain_applied/README.md) and [`traceability.toml`](../../traceability.toml).

3. **Shannon Entropy & Blahut-Arimoto Channel Capacity in `pure_math::algorithmic_information::shannon`:**
   - Added [`shannon`](../../crates/pure_math/src/pure_math/algorithmic_information/shannon/mod.rs) providing base-2 categorical entropy, Rényi entropy ($\alpha \ge 0$, $\alpha \to 1$, $\alpha = \infty$), `KL` and `JSD` divergences, scratch-buffered `joint_mutual_information`, closed-form `BSC`, `BEC`, and `AWGN` Shannon-Hartley capacities, and iterative `blahut_arimoto_capacity` (`BlahutArimotoWorkspace`).
   - Wired `ShannonError` into `PureMathError` and `math_explorer::error::Error` (`#[cfg(feature = "pure_math")]`), and extended [`papers/algorithmic_information_rust.tex`](../../papers/algorithmic_information_rust.tex) and [`traceability.toml`](../../traceability.toml).

4. **Dynamic $2..=8$ Replicator Resize and Five Integrated GUI Workbenches:**
   - Extended [`ReplicatorDynamicsTool`](../../math_explorer_gui/src/tabs/game_theory/replicator.rs) with interactive $2..=8$ strategy resize controls preserving existing payoffs, labels, normalized populations, and compiled symbolic formulas while keeping `ReplicatorPresetConfig` compatibility intact.
   - Added `NashShapleyTool`, `DecisionBanditTool`, `OptimalAuctionTool`, and `MeanFieldGameTool` inside [`GameTheoryTab`](../../math_explorer_gui/src/tabs/game_theory/mod.rs) and `ShannonChannelTool` inside [`AlgorithmicInfoTab`](../../math_explorer_gui/src/tabs/algorithmic_information/mod.rs), preserving the 28 top-level `ExplorerTab` structure.

## Consequences

- All new `domain_applied` and `pure_math` hot-path solvers execute with zero dynamic heap allocations under `VerifiedAllocator`.
- `math_explorer` preserves clean feature isolation between `applied`, `pure_math`, and `ai` while exposing uniform umbrella error conversions.
- All 94 registered `InteractiveTool` workbenches and 163 `StarterPresetMetadata` cards satisfy lossless URL state sharing, preset schema validation, deferred PNG/SVG plot export, and the 3 MB WASM bundle budget.
