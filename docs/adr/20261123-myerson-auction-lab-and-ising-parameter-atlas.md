# ADR 20261123: Myerson Mechanism-Design Auction Lab and Analytical Ising / Fermi Spatial Parameter Atlas

## Status

Accepted (2026-11-23)

## Context

1. **Dormant Mechanism-Design and Fixed-Point Equilibrium Kernels in `domain_applied`:** [`crates/domain_applied/src/applied/game_theory/mechanism_design.rs`](../../crates/03_domains/domain_applied/src/applied/game_theory/mechanism_design.rs) and [`crates/domain_applied/src/applied/game_theory/equilibrium.rs`](../../crates/03_domains/domain_applied/src/applied/game_theory/equilibrium.rs) provided verified `ValuationDistribution::virtual_valuation`, `optimal_reserve_price`, `simulate_optimal_revenue_with_rng`, and `Correspondence` / `is_fixed_point` primitives, while [`math_explorer_gui/src/tabs/game_theory/mod.rs`](../../apps/math_explorer_gui/src/tabs/game_theory/mod.rs) mounted only `Glicko2TournamentTool` and `ReplicatorDynamicsTool` (`0` GUI callers and `0` `game_theory` inverse challenges).
2. **Ising Boltzmann `|J|` Cancellation, Antiferromagnetic Diagnostics, and Spatial Dilemma Freezes:** [`math_explorer_gui/src/tabs/solid_state/ising/runner.rs`](../../apps/math_explorer_gui/src/tabs/solid_state/ising/runner.rs) and [`math_explorer_gui/src/tabs/solid_state/ising.rs`](../../apps/math_explorer_gui/src/tabs/solid_state/ising.rs) multiplied `temperature` by `|J|` before dividing by `KB`, cancelling `|J|` out of spin-spin Boltzmann acceptance factors while artificially rescaling Zeeman `h` terms. In addition, `IsingModelTool` lacked staggered magnetization diagnostics for $J < 0$, left all-`-1` `FermiGame` lattices permanently absorbed across preset switches, omitted painted droplet seeds from `IsingState`, and exposed no 2D `(T, S)` or `(T, H)` / `(T, J)` parameter atlas or `solid_state` inverse challenges.

## Decision

1. **Stack-Allocated `ScalarBestResponse` and Standalone `MyersonAuctionTool` (`game_theory`):**
   - Add stack-allocated `ScalarBestResponse<F>` implementing `Correspondence` in [`crates/domain_applied/src/applied/game_theory/equilibrium.rs`](../../crates/03_domains/domain_applied/src/applied/game_theory/equilibrium.rs) (`#[verified_engine::verified]`) so symmetric First-Price Sealed-Bid Bayes-Nash bid shading can be checked via `is_fixed_point` with `0` dynamic heap allocations.
   - Mount `MyersonAuctionTool` in [`math_explorer_gui/src/tabs/game_theory/auction.rs`](../../apps/math_explorer_gui/src/tabs/game_theory/auction.rs) (`math.rs`, `atlas.rs`, `challenge.rs`, `ui.rs`, `tests.rs`) supporting `Uniform`, `Exp`, and `Beta` valuations, plotting virtual valuation $J(v)$ with shaded $J(v) < 0$ exclusion zones, expected revenue versus social welfare / deadweight loss (`simulate_optimal_revenue_with_rng` via `OxidizeRng`), and First-Price vs Second-Price bid curves with `TraceSnapshotManager` (`<= 10` pinned traces) and `CalloutManager`.
   - Embed progressive zero-allocation `AuctionCartographer` over `(reserve_price, n_bidders)` and `(reserve_price, focal_valuation)` and register `2` `StarterPresetMetadata` cards plus `2` `ChallengeMetadata` cards (`myerson_welfare_constrained_reserve` and `first_price_shading_equilibrium`).

2. **Analytical `IsingCartographer`, Corrected Boltzmann Scaling, Fermi Mutation, and Nucleation Challenges (`solid_state`):**
   - Extend `RegimeCartographer` in [`math_explorer_gui/src/async_sim/atlas/mod.rs`](../../apps/math_explorer_gui/src/async_sim/atlas/mod.rs) with `extra_overlay_contours` rendered via `SemanticPalette` tokens, and add `O(1)` analytical `IsingCartographer` in [`math_explorer_gui/src/tabs/solid_state/ising/atlas.rs`](../../apps/math_explorer_gui/src/tabs/solid_state/ising/atlas.rs) covering `(T, S)` spatial dilemma survival (with `T = R`, `S = P`, and `2R + 2S = T` boundary contours exposing the `75.0%` Prisoner's Dilemma defector freeze) and `(temperature, h_field)` / `(temperature, j_coupling)` slices (with exact Onsager $T_c(J) = \frac{2|J|}{\ln(1 + \sqrt{2})}$).
   - Remove the `* |J|` temperature scaling bug in `IsingRunner` and `IsingModelTool`, add `SpinLattice::staggered_magnetization`, `SpinLattice::evolve_fermi_game_with_mutation` (`mutation_prob` clamped to `[0.0, 0.05]`), and `SpinLattice::reset_uniform_with_seed` in [`crates/domain_physics/src/physics/stat_mech/ising.rs`](../../crates/03_domains/domain_physics/src/physics/stat_mech/ising.rs), and auto-reseed absorbed `-1` `FermiGame` lattices on preset switches and `(T, S)` atlas teleports.
   - Persist `mutation_rate`, `defector_sea`, `droplet_seeds` (`<= 32` circular disks), and `atlas` slice selection in `IsingState` (`#[serde(deny_unknown_fields)]`, `<= 4,096` JSON bytes), and register `2` `solid_state` inverse challenges (`ising_critical_droplet_nucleation` and `fermi_prisoners_dilemma_nucleation`) evaluated by zero-allocation `IsingChallengeWorkspace`.
   - Guard `AstVisitor::inspect_fn_attr` in [`crates/oxidize_core/src/ast_visitor.rs`](../../crates/01_core/oxidize_core/src/ast_visitor.rs) so `#[doc = ...]` attributes extract `[cite:...]` citations without misparsing Rustdoc code examples containing `#[verified]` or `#[embed_theory(...)]` during build-time `traceability_manifest.json` generation.

## Consequences

- **Positive:** Mechanism design, Bayes-Nash bid shading, Onsager / Néel lattice transitions, and spatial evolutionary reciprocity become directly explorable and shareable via compact stateless `#challenge=<id>` and `#tab=...&state=...` URLs with `0` heap allocations inside `VerifiedEngine::run_verified`.
- **Trade-offs:** Persisted `droplet_seeds` in `IsingState` are capped at `32` circular disks (`<= 4` disks and `<= 380` `+1` sites in locked nucleation challenges) and `TraceSnapshotManager` branches are omitted from compact URL preset envelopes to stay strictly below the `4,096`-byte decoded JSON and `8,192`-byte URL fragment limits.

## Verification Commands

```bash
# Regenerate API baseline and run workspace tests
cargo run -p xtask -- regenerate-baseline
cargo test --workspace --lib --bins --tests --all-features
cargo test --doc --workspace --all-features

# Verify feature isolation, WASM compilation, and repository lint gates
cargo run -p xtask -- test-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features
cargo clippy --workspace --target wasm32-unknown-unknown --all-features -- -D warnings
cargo run -p xtask -- lint-all
```
