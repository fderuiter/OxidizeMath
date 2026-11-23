# ADR 0046: 1D Quantum Potential Sculpting, Resonant Presets, and Zero-Allocation Region Probabilities

## Status
Accepted (extends [ADR 0027](0027-precompute-quantum-propagators-inplace-evolution.md), [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), and [ADR 0045](0045-spatial-brush-kit-and-field-sculpting.md))

## Context
1. [`crates/domain_physics/src/physics/quantum/hamiltonian.rs`](../../crates/domain_physics/src/physics/quantum/hamiltonian.rs) exposed `construct_1d_hamiltonian` and `gaussian_wavepacket`, while the GUI wavepacket sandbox lacked interactive freehand potential-brush sculpting, draggable wavepacket emitter handles, and centi-quantized 200-point URL state persistence.
2. Interactive freehand barrier sculpting on a 200-point grid requires continuous index interpolation across rapid pointer drags without heap allocations, bounded potential clamping ($V(x) \in [-20.0, 40.0]$), canonical resonant barrier/ramp profiles, and real-time conserved transmission/reflection partitioning ($P_T + P_R = 1$) while preserving NASA Power of 10 zero-allocation unitary stepping in [`evolve_state_in_place`](../../crates/domain_physics/src/physics/quantum/schrodinger.rs).

## Decision
1. **Verified Zero-Allocation 1D Potential Sculpting & Probability Kernels (`domain_physics`)**:
   - Add `MIN_SCULPT_POTENTIAL`, `MAX_SCULPT_POTENTIAL`, `PotentialPreset1D`, `apply_bresenham_potential_stroke`, `build_1d_potential_preset`, and `partition_region_probabilities` to [`crates/domain_physics/src/physics/quantum/hamiltonian.rs`](../../crates/domain_physics/src/physics/quantum/hamiltonian.rs) and re-export them from [`crates/domain_physics/src/physics/quantum/mod.rs`](../../crates/domain_physics/src/physics/quantum/mod.rs) (reachable via `math_explorer::physics::quantum`).
   - Guard `gaussian_wavepacket` against zero or non-finite packet widths $\sigma$.
2. **Interactive Quantum Sculpting Canvas & Quantized URL State (`math_explorer_gui`)**:
   - Extend `SimCommand` with `SculptStroke1D` and `SetWavepacket`, add `UnifiedCanvasContext` and `custom_series` hooks to [`math_explorer_gui/src/async_sim/unified/mod.rs`](../../math_explorer_gui/src/async_sim/unified/mod.rs), and persist centi-quantized 200-point `"custom_potential"` series within the 4,096-byte URL state ceiling via [`math_explorer_gui/src/url_state/codec.rs`](../../math_explorer_gui/src/url_state/codec.rs) and [`math_explorer_gui/src/tabs/quantum/wave_sim/persistence.rs`](../../math_explorer_gui/src/tabs/quantum/wave_sim/persistence.rs).
   - Integrate lazy per-frame propagator recomputation, draggable $(x_0, k_0, \sigma)$ emitter handles, Born & two-region $P_T / P_R$ readouts, tunneling and resonant double-well starter presets, and WCAG 2.2 AA keyboard reticle editing into [`math_explorer_gui/src/tabs/quantum/wave_sim.rs`](../../math_explorer_gui/src/tabs/quantum/wave_sim.rs).

## Consequences
- **Positive**: `apply_bresenham_potential_stroke` and `partition_region_probabilities` execute in place with zero dynamic heap allocations under `VerifiedAllocator`, while `WaveRunner` recomputes $e^{-i\hat{H}\Delta t/\hbar}$ at most once per frame when $V(x)$ or $\Delta t$ changes and keeps `evolve_state_in_place` zero-allocation.
- **Positive**: Custom sculpted 200-point quantum potentials round-trip losslessly through shareable URLs well below the 4 KiB decoded payload limit while omitting `custom_potential` for unsculpted presets.
- **Neutral**: [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) is updated via `xtask regenerate-baseline` to record the new public `domain_physics::physics::quantum` items.
