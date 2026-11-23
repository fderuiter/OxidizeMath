# ADR 0046: Quantum Tunneling and 2D Chladni Resonance Parameter Atlases

## Status

Accepted

## Context

`WaveSimulatorTool` (1D time-dependent Schrödinger wavepacket sandbox) and `ChladniWaveTankTool` (2D hyperbolic wave tank and Kirchhoff-Love biharmonic Chladni plate solver) previously forced users to search for narrow Fabry-Pérot tunneling peaks, Ramsauer-Townsend transparency bands, and high-$Q$ 2D plate/drumhead eigenmodes via 1D sliders (`step = 0.05` on $k_0, V_0, w$ and `step = 0.1 Hz` on `driver_frequency`). In addition:

1. Quasi-bound resonant wavepackets inside `DoubleBarrier` left probability trapped in the inter-barrier well while transmitted probability reached the Dirichlet guard wall, keeping `latched_transmission` stuck at `"Pending transit..."`.
2. Mid-drag slider motion in `WaveSimulatorTool` rebuilt the dense $N \times N$ Padé matrix exponential `time_evolution_operator` on every pointer frame.
3. `ChladniWaveTankTool` omitted `TheoryDescribable::theory_parameters()`, drove its `SquareChladniPlate` preset on the $y = 0.50$ nodal line of center-pinned odd modes, and displayed a hardcoded `Clamped f(1,1)` metric card regardless of active boundary (`Neumann`, `Dirichlet`, or circular `Bessel` drumhead).

All 2D parameter-space cartographers executed by `ParameterAtlasController` (`ADR 0035`) must evaluate `RegimeCartographer::evaluate_cell` with **zero dynamic heap allocations** under `VerifiedAllocator::lock_allocations()`.

## Decision

1. **Zero-Allocation 1D Transfer-Matrix Scattering (`crates/domain_physics/src/physics/quantum/scattering.rs`)**:
   - Implemented unimodular $2 \times 2$ state transfer matrices `TransferMatrix2x2` ($\mathrm{SL}(2, \mathbb{R})$) across oscillatory ($E > V$), turning-point ($E \approx V$), and evanescent ($E < V$, overflow-guarded at $\kappa \Delta x > 300$) slices.
   - Provided exact `rectangular_barrier_transmission`, `double_barrier_transmission`, `symmetric_double_barrier_transmission`, piecewise `piecewise_potential_transmission`, bytecode `compiled_barrier_transmission` (`CompiledExpr::eval_1d`), and 12-point `gauss_hermite_wavepacket_transmission`.
2. **`QuantumTunnelingCartographer`, Quasi-Bound Scatter Latching, and Slider Debouncing (`math_explorer_gui/src/tabs/quantum/wave_sim/`)**:
   - Integrated `ParameterAtlasController` over `("k0", "v0")` and `("k0", "width")` using `wave_parameter_constraints()`, displaying stationary $T(k_0)$, wavepacket $\langle T \rangle_\sigma$, and the $E_0 = V_0$ threshold contour.
   - Updated `WaveRunner::update_scatter_latch` to capture interpolated pre-wall transmission (`quasi_bound_latched`) when transmitted probability reaches `RIGHT_WALL_GUARD_X` before a quasi-bound cavity drains.
   - Debounced `sync_config_and_reset` behind `pending_slider_reset` while `pointer.primary_down()` is active so mid-drag motion incurs zero `time_evolution_operator` rebuilds while analytical transmission previews update live.
   - Registered `ChallengeMetadata` (`quantum_fabry_perot_resonance`) for sub-barrier double-barrier resonance tuning.
3. **Zero-Allocation 2D Kirchhoff-Love & Bessel Eigenmodes (`crates/pure_math/src/pure_math/analysis/pde/wave_2d/eigenmodes.rs`)**:
   - Added stack-only `bessel_jn`, `BESSEL_JN_ZEROS`, discrete-calibrated `square_plate_eigenfrequency_hz` (`Neumann`, `Clamped`, `Dirichlet`) and `circular_drumhead_eigenfrequency_hz`, pin-compatibility filtering (`PinConstraint2D`), `snap_to_mode_antinode`, and `lorentzian_modal_response`.
4. **`ChladniModalCartographer`, Antinodal Lock-In, Interactive Radial FFT, and Challenge (`math_explorer_gui/src/tabs/analysis/chladni_wave_tank/`)**:
   - Implemented `TheoryDescribable::theory_parameters()` (`0.001 Hz` `driver_frequency` step) and `ChladniModalCartographer` over `("driver_frequency", "plate_stiffness")`, `("driver_frequency", "wave_speed")`, and `("driver_x", "driver_y")` using precomputed unit wave/plate dispersion coefficients so `128x128` passes run in 24 stack iterations per cell with zero heap allocations.
   - Added antinodal click-to-lock on both the 2D Parameter Atlas and interactive radial 2D FFT eigenmode markers (`draw_radial_fft_plot`), locking `driver_frequency` within `<= 0.001 Hz` and snapping `(driver_x, driver_y)` off nodal lines onto a mode antinode with `TwinSlotLiveRegion` announcements.
   - Replaced hardcoded `Clamped f(1,1)` readouts with boundary-accurate `Neumann`, `Clamped`, `Dirichlet`, and `Bessel` eigenfrequency cards, aligned `SquareChladniPlate` and `CircularDrumhead` presets to discrete antinodal modes, and registered `ChallengeMetadata` (`chladni_antinodal_resonance`).

## Consequences

- **Positive**: Users can visually locate and click-to-lock sub-barrier quantum Fabry-Pérot resonances (`T > 0.99`) and high-$Q$ 2D Chladni/Bessel standing modes (`> 50x` steady-state wave energy gain over off-resonant/nodal driving) without frame hitches or heap allocations inside `evaluate_cell`.
- **Backward Compatibility**: `atlas_visible` and `atlas_axes` omit default values via `skip_serializing_if`, preserving existing URL share fragments, `StarterPresetMetadata` payloads, and `#[serde(deny_unknown_fields)]` validation within `ToolStateBounds`.
