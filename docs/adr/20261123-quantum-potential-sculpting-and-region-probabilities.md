# ADR 20261123: Live 1D Quantum Potential Sculpting, Emitter Handles, and WCAG Reticle

## Status

Accepted

## Context

Following the migration of 1D Schrödinger evolution to zero-allocation `CrankNicolson1D` and `TridiagonalHamiltonian1D` (`ADR 20261123: Tridiagonal Crank-Nicolson 1D Quantum Propagation`) and 2D parameter atlases (`QuantumTunnelingCartographer`), the Quantum Wave Simulator (`math_explorer_gui/src/tabs/quantum/wave_sim/`) still lacked continuous Bresenham-interpolated barrier strokes, direct wavepacket emitter manipulation, and barrier-peak region probability splits:

1. **Discontinuous Fast-Pointer Potential Strokes & Redundant Propagator Rebuilds**: Fast pointer drags across the 1D spatial grid skipped intermediate grid cells unless interpolated with discrete Bresenham line stepping, and multi-segment brush frames rebuilt the Crank-Nicolson tridiagonal factors on every individual segment rather than coalescing Hamiltonian updates once per simulation frame.
2. **Wavepacket Reset Clobbering Sculpted Potentials**: Adjusting the initial wavepacket position $x_0$, wavenumber $k_0$, or width $\sigma$ previously triggered a full simulation reset that wiped out user-sculpted barriers $V(x)$, preventing rapid scattering experiments against custom double-well or step-barrier profiles.
3. **Missing Barrier-Peak Splits & Curated Tunneling Starter Presets**: Learners lacked real-time barrier-peak transmitted ($P_T$) and reflected ($P_R$) probability partitions alongside Born $T(t)/R(t)$ observables, curated starter gallery presets for tunneling and resonant double-barrier wells, and centi-quantized custom series support in `UnifiedSimTool`.
4. **Pointer-Only Canvas Controls & Coverage-Mode Plateau Bottlenecks**: Sculpting $V(x)$ needed WCAG 2.2 AA `GraphicsDocument` reticle announcements (`WCAG 2.2 AA 2.1.1 / 2.4.7 / 4.1.3`), and flat-potential plateau runs of $200{,}000$ points incurred scalar loop overhead under `-C instrument-coverage` in `math_commons` and `egui_plot` POI detection.

## Decision

1. **Zero-Allocation 1D Bresenham Sculpting, Presets, and Region Partitioning (`domain_physics`)**:
   - Add `MIN_SCULPT_POTENTIAL` (`-200.0`), `MAX_SCULPT_POTENTIAL` (`200.0`), `PotentialPreset1D` (`FreeParticle`, `HarmonicTrap`, `FiniteSquareBarrier`, `DoubleBarrierResonant`, `AsymmetricStep`), `apply_bresenham_potential_stroke`, `build_1d_potential_preset`, and `partition_region_probabilities` in `crates/domain_physics/src/physics/quantum/hamiltonian.rs` and re-export them from `domain_physics::physics::quantum`.
   - Enforce clamping to `[-200.0, 200.0]`, guard `gaussian_wavepacket` against non-positive $\sigma$, and verify zero heap allocations on `build_1d_potential_preset`, `apply_bresenham_potential_stroke`, and `partition_region_probabilities` under `VerifiedAllocator` in `crates/domain_physics/tests/test_quantum.rs`.
2. **Coalesced Dirty-Hamiltonian `CrankNicolson1D` Rebuilds & Live Emitter Handles (`math_explorer_gui`)**:
   - Extend `SimCommand` with `SculptStroke1D` and `SetWavepacket`, and add `UnifiedCanvasContext` plus centi-quantized `custom_series` synchronization to `UnifiedSimTool`.
   - Coalesce multiple `SculptStroke1D` and `SpatialBrushKit` commands via `WaveRunner::hamiltonian_dirty` so `CrankNicolson1D::update_potential` and `CrankNicolson1D::update_dt` execute at most once per simulation step, while preserving `sparse_potential` compatibility with `QuantumTunnelingCartographer`.
   - Render interactive plot-canvas emitter handles for packet center $x_0$ (diamond handle), momentum $k_0$ (phase-velocity arrow tip), and width $\sigma$ (square envelope handle) that re-initialize $\psi(x, 0)$ via `SetWavepacket` without erasing custom sculpted potentials, alongside real-time $P_T$, $P_R$, and norm conservation readouts and two new starter presets (`wave_sim_tunneling_barrier`, `wave_sim_resonant_double_well`).
3. **WCAG 2.2 AA Keyboard Reticle & Centi-Quantized URL Persistence**:
   - Wire `SpatialBrushKit` keyboard reticle sculpting (`Arrow` navigation, `Space`/`Enter` stamp, `Shift` $5\times$ coarse step, non-trapping `Tab` escape) to `attach_quantum_canvas_accessibility` (`Role::GraphicsDocument`) and polite AccessKit live announcements reporting reticle $x$, $V(x)$, $P_T$, and $P_R$.
   - Persist sculpted `sparse_potential` deltas on `WaveSimulatorTool` and centi-quantized (`0.01`-step, `CUSTOM_POTENTIAL_POINTS = 200`) custom series on `UnifiedSimTool` using prevalidated raw JSON preset envelopes (`serialize_compact_raw_preset_envelope` + `encode_share_fragment_with_hud_prevalidated`) compatible with both `#tab=` and `#ws=` share URLs.
4. **8-Wide Chunked Plateau Fast-Forward in `math_commons` & `egui_plot`**:
   - Upgrade `PointSeries::advance_plateau_run` in `crates/math_commons/src/series.rs` (including paired-slice overrides) and `PlateauSample::chunk8_same_plateau` in `egui_plot/src/poi/plateau.rs` to scan 8-sample chunks via `.as_chunks::<8>()`, delegating `SeriesCoords::advance_plateau_run` in `egui_plot/src/poi.rs` directly to `PointSeries`.

## Consequences

- Users can draw arbitrary 1D quantum barriers and wells at 60 FPS with zero heap allocations on the physics hot path, drag $(x_0, k_0, \sigma)$ emitter handles against custom potentials, inspect live $P_T / P_R$ tunneling splits, and share sculpted experiments via compact URL fragments.
- Full keyboard sculpting parity and screen-reader announcements satisfy WCAG 2.2 Level AA criteria without trapping focus.
- Coverage-instrumented 200k-point plateau POI benchmarks complete well below the 5 ms budget while reducing `egui_plot/src/poi.rs` line count below the 500-line modular gate.
