# ADR 0047: Zero-Allocation 2D Abelian Sandpile Engine, Recurrent Group Algebra, and Solid-State Studio

## Status
Accepted (extends [ADR 0040](0040-dual-backend-web-audio-sonification.md), [ADR 0040](0040-promote-zero-alloc-curve-solvers-to-math-commons.md), and [ADR 0042](0042-2d-chladni-plate-and-wave-tank-solver.md))

## Context
1. **Missing Bridge Between Statistical Mechanics and Finite Group Algebra**: `crates/domain_physics/src/physics/stat_mech` modeled equilibrium Ising lattices and ensemble thermodynamics, while `crates/pure_math/src/pure_math/algebra` provided abstract `Semigroup`, `Monoid`, `Group`, `AbelianGroup`, `Zn<N>`, `is_subgroup`, and `generate_left_coset` traits without a concrete spatial physical lattice realizing a non-trivial finite Abelian group or parameter-free self-organized criticality (`1/f` avalanches).
2. **Zero-Allocation & Exact Arithmetic Constraints**: Simulating 2D Bak-Tang-Wiesenfeld (BTW) cascades, Creutz recurrent identity fractals, Dhar burning recurrence waves, and power-law regressions interactively at 60 FPS under `VerifiedAllocator` requires pre-allocated integer wavefront ring buffers, deterministic `OxidizeRng` sampling, and fraction-free integer determinants without dynamic heap allocations.

## Decision
1. **Re-export `AbelianGroup` & Group Helpers in `pure_math::algebra`**:
   - Re-export `AbelianGroup`, `is_subgroup`, `is_normal_subgroup`, `generate_left_coset`, and `generate_right_coset` from [`crates/pure_math/src/pure_math/algebra/mod.rs`](../../crates/pure_math/src/pure_math/algebra/mod.rs) and implement `AbelianGroup` for `Zn<N>` in [`crates/pure_math/src/pure_math/algebra/group.rs`](../../crates/pure_math/src/pure_math/algebra/group.rs).
2. **Zero-Allocation 2D BTW Engine & Finite Sandpile Group in `domain_physics`**:
   - Add [`crates/domain_physics/src/physics/stat_mech/sandpile/mod.rs`](../../crates/domain_physics/src/physics/stat_mech/sandpile/mod.rs) (`SandpileGrid`) backed by power-of-two masked integer wavefront ring buffers, epoch-tracked avalanche area counters, `OxidizeRng` grain drops, forward/reverse lexicographic sweeps proving Abelian topple-order commutativity, Creutz's recurrent identity solver $e = (2 z_{\max} - (2 z_{\max})^\circ)^\circ$, zero-negative recurrent inversion $r^{-1} = (2 z_0 - r^\circ)^\circ$, Dhar's BFS burning recurrence test with per-cell wave depths, and fixed-capacity complementary survival histograms coupled to `math_commons::fit::FitModel::PowerLaw`.
   - Add [`crates/domain_physics/src/physics/stat_mech/sandpile/kirchhoff.rs`](../../crates/domain_physics/src/physics/stat_mech/sandpile/kirchhoff.rs) (`kirchhoff_determinant_bareiss`) computing exact 128-bit fraction-free Bareiss determinants of the reduced Kirchhoff Laplacian ($\det \Delta_{2\times 2} = 192$, $\det \Delta_{3\times 3} = 100{,}352$, $\det \Delta_{4\times 4} = 557{,}568{,}000$).
   - Add [`crates/domain_physics/src/physics/stat_mech/sandpile/algebra.rs`](../../crates/domain_physics/src/physics/stat_mech/sandpile/algebra.rs) (`SandpileElement<W, H>`) implementing `Semigroup`, `Monoid`, `Group`, and `AbelianGroup`, plus cyclic subgroup and `Zn<N>` homomorphism helpers interoperable with `is_subgroup` and `generate_left_coset`.
3. **Interactive `AbelianSandpileStudio` in `SolidStateTab`**:
   - Register `AbelianSandpileStudio` (`id: "AbelianSandpileStudio"`, `domain: "solid_state"`, `ExportCapabilities::plot()`) and three `StarterPresetMetadata` cards (*1/f Critical Avalanche Cascades*, *63×63 Recurrent Identity Crystal*, and *(r + e)° = r Abelian Group Proof*) under [`math_explorer_gui/src/tabs/solid_state/sandpile/`](../../math_explorer_gui/src/tabs/solid_state/sandpile/mod.rs).
   - Render 2D crystals via `flatten_2d_index` and shared perceptual/WCAG colormaps, overlay live log-log `FitModel::PowerLaw` regressions ($R^2 \ge 0.90$), and couple cascade wavefront telemetry to `-3.0 dBFS`-limited `SonificationFrame` seismic crackle audio.

## Consequences
- **Positive**: Bridges self-organized critical statistical mechanics and abstract finite Abelian group theory in a single zero-allocation, zero-dependency implementation verified under `VerifiedAllocator` on both native and `wasm32-unknown-unknown` targets.
- **Neutral**: Updates exact federated `ToolMetadata` (`75 -> 76`) and `StarterPresetMetadata` (`108 -> 111`) contract assertions in `math_explorer_gui` and records the new `pure_math` and `domain_physics` public exports in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json).
