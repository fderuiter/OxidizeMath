# ADR 0039: Auto-Discovered Paradox Lab Tab for Stochastic Ratchets and Network Congestion

## Status
Accepted (extends [ADR 0001](0001-dynamic-workspace-discovery.md), [ADR 0008](0008-finite-state-solver-guard.md), and [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md))

## Context
Several verified mathematical engines across `pure_math` and `domain_applied`—specifically discrete-time `MarkovChain`, time-indexed `TransitionTensor`, continuous-time `GillespieSolver`, Nash fixed-point `BestResponseCorrespondence`, and coupled HJB/Fokker-Planck `FixedPointSolver`—previously had zero interactive callers in `math_explorer_gui`, restricting users to standalone CLI examples (`markov_demo` and `bench_mfg`).

## Decision
1. **Auto-Discovered `ParadoxLabTab` Under `applied`**:
   - Introduce `ParadoxLabTab` in `math_explorer_gui/src/tabs/paradox_lab/mod.rs` annotated with `// @explorer_feature = "applied"` and backed by `SimulationFramework::new("paradox_lab")`, overriding `academic_domain() -> "Applied Math"` and `url_slug() -> "paradox_lab"` so `build.rs` discovers the module without modifying `tabs/mod.rs` or `app/mod.rs`.
2. **Zero-Allocation `ParrondoParadoxTool` and `BraessNetworkParadoxTool`**:
   - Register `ParrondoParadoxTool` (`"Parrondo's Paradox"`) connecting `MarkovChain::stationary_distribution`, `TransitionTensor`, `OxidizeRng` ensembles, and pre-sized `GillespieSolver` buffers so `step_parrondo_zero_alloc` passes `VerifiedEngine::run_verified` with zero dynamic allocations.
   - Register `BraessNetworkParadoxTool` (`"Braess's Paradox"`) modeling the 4-node Wheatstone network (Wardrop equilibrium, Pigou social optimum, Pigouvian shortcut toll, Price of Anarchy), verifying Nash fixed points via `BoxSet`, `BestResponseCorrespondence`, and `is_fixed_point`, and solving continuum crowd congestion via `MFGConfigBuilder` + `FixedPointSolver` with explicit CFL clamping (`viscosity * dt / dx^2 < 0.25`).
3. **Documentation & Traceability Linkage**:
   - Update [`crates/pure_math/src/pure_math/statistics/markov/README.md`](../../crates/pure_math/src/pure_math/statistics/markov/README.md) to document interactive GUI exploration alongside `markov_demo`, and bind `paradox_lab` in [`traceability.toml`](../../traceability.toml).

## Consequences
- **Positive**: Exposes both stochastic ratchets and game-theoretic congestion/mean-field PDE solvers in native desktop and `wasm32-unknown-unknown` builds with shareable `#tab=paradox_lab` URLs, starter presets, `SemanticPalette` contrast, AccessKit keyboard navigation, and offline plot/dataset exports.
- **Neutral**: Inventory contract suites track 22 `ExplorerTab` modules (6 under Applied Math), 64 `InteractiveTool` entries, 76 `StarterPresetMetadata` cards, and 41 plot-exportable tools.
