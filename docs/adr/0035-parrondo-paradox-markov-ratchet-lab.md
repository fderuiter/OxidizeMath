# ADR 0035: Parrondo's Paradox Discrete-Time Markov Ratchet Solver and Financial Math Lab

## Status

Accepted

- **Date:** 2026-11-21

## Context

`pure_math::statistics` already provides discrete-time finite-state `MarkovChain` solvers (`crates/pure_math/src/pure_math/statistics/markov/dtmc/mod.rs`) and Kelly bankroll-sizing primitives (`crates/pure_math/src/pure_math/statistics/kelly/criterion.rs`), while `FinancialMathTab` (`math_explorer_gui/src/tabs/financial_math/`) exposes single-game Kelly calculators (`BankrollGrowthTool` and `BetSizeCalculatorTool`).

However, two architectural constraints shaped adding a Harmer-Abbott / Parrondo's paradox Brownian ratchet laboratory:

1. **File-length and periodicity constraints in `dtmc/mod.rs`:** `crates/pure_math/src/pure_math/statistics/markov/dtmc/mod.rs` is already near the repository's 500-line ceiling (`494` lines), and `MarkovChain::stationary_distribution` uses power iteration $P^T \pi = \pi$, which requires aperiodic transition operators when parity or periodic schedules are evaluated.
2. **Zero-allocation ensemble stepping and Kelly `NoEdge` synthesis:** Isolated Parrondo games $A$ and $B$ exhibit strictly negative expected step drift under coin bias $\epsilon > 0$ (triggering `KellyError::NoEdge`), whereas alternating them randomly ($P_{\text{mix}} = \tfrac{1}{2}(P_A + P_B)$) or periodically ($P_{ABB} = P_A P_B P_B$) breaks detailed balance on $\mathbb{Z}/M\mathbb{Z}$, reduces occupancy of the losing $W \equiv 0 \pmod M$ trap state, and generates positive expected wealth growth without adding external crate dependencies or heap allocations inside hot simulation loops.

## Decision

1. **Dedicated `pure_math::statistics::markov::parrondo` Submodule:**
   - Implement [`crates/pure_math/src/pure_math/statistics/markov/parrondo.rs`](../../crates/pure_math/src/pure_math/statistics/markov/parrondo.rs) and [`crates/pure_math/src/pure_math/statistics/markov/parrondo/ensemble.rs`](../../crates/pure_math/src/pure_math/statistics/markov/parrondo/ensemble.rs), re-exported from [`crates/pure_math/src/pure_math/statistics/markov/mod.rs`](../../crates/pure_math/src/pure_math/statistics/markov/mod.rs) and reachable through `math_explorer::pure_math::statistics::markov::parrondo` while leaving `dtmc/mod.rs` and `kelly/criterion.rs` untouched.
   - Restrict capital moduli to odd $M \in \{3, 5, 7, 9\}$ to prevent bipartite parity traps, and compute exact stationary distributions (`pi_a`, `pi_b`, `pi_mix`, `pi_schedule`) via lazy damping $\tilde{P} = \tfrac{1}{2}(I + P)$ atop `MarkovChain::stationary_distribution` (preserving exact left eigenvectors since $\pi^T \tilde{P} = \pi^T \iff \pi^T P = \pi^T$).
   - Quantify broken detailed balance via maximum pairwise probability-flux residuals $\max_{i,j} |\pi_i P_{ij} - \pi_j P_{ji}|$, and map the `(p_A, epsilon)` paradox region via `scan_parrondo_phase_region`.

2. **Zero-Allocation `OxidizeRng` Ensemble Workspace:**
   - Provide `ParrondoEnsemble` preallocating walker wealth (`Vec<i64>`), trajectory (`Vec<f64>`), and stack residue-occupancy (`[u64; MAX_MODULUS]`) buffers up front.
   - Annotate hot-path methods (`reset`, `step_game`, `step_mixture`, `simulate_pure_game`, `simulate_mixture`, `simulate_schedule`) with `#[verified_engine::verified]`, using Euclidean modular arithmetic (`i64::rem_euclid`) for negative wealth states and enforcing `0` heap allocations under `VerifiedAllocator` in [`crates/pure_math/tests/test_parrondo.rs`](../../crates/pure_math/tests/test_parrondo.rs).

3. **`ParrondoParadoxTool` in `FinancialMathTab`:**
   - Implement `ParrondoParadoxTool` (`id: "ParrondoParadoxTool"`, `display_name: "Parrondo's Paradox"`, starter preset `id: "parrondo_paradox_starter"`) under [`math_explorer_gui/src/tabs/financial_math/parrondo/mod.rs`](../../math_explorer_gui/src/tabs/financial_math/parrondo/mod.rs) with `PaneSidePanel` controls, `kelly_fraction` `NoEdge` vs positive-edge badges, borrowed `PlotPoints` trajectory slices, modulo-$M$ trap-occupancy `BarChart` views, and single-coordinate-space offscreen dataset/figure exports (`ExportCapabilities::plot()`).

## Verification Commands

```bash
cargo test -p pure_math --test test_parrondo
cargo test -p math_explorer_gui parrondo
cargo run -p xtask -- regenerate-baseline
cargo run -p xtask -- lint-all
```

## Consequences

- **Positive:** Users can compare isolated `KellyError::NoEdge` losing games directly against positive-drift Markov ratchets (`P_mix`, `P_ABB`, `P_AABB`, `P_ABAB`, and custom bounded schedules) with exact stationary distributions and deterministic `OxidizeRng` ensembles across native and `wasm32-unknown-unknown` targets.
- **Positive:** `dtmc/mod.rs` and existing Kelly tools remain untouched under the 500-line ceiling, zero new external crates are introduced, and public API baseline parity is preserved via `xtask regenerate-baseline`.
- **Neutral:** Registered `InteractiveTool`, `StarterPresetMetadata`, and plot-export tool counts in `math_explorer_gui` contract suites advance from `55 -> 56`, `59 -> 60`, and `32 -> 33`, respectively.
