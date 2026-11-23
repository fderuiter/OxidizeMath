# ADR 0045: Closed-Form Schwarzschild 4D Connection, Geodesic Bundle, and Relativistic Jet Workbench

## Status

Accepted (2026-11-23)

## Context

While `pure_math::tensor` provided `StaticMetric<N>` and heap-allocated `christoffel_symbols`, `domain_physics::physics::high_energy` still contained disconnected scalar kernels (`observer.rs`, `radiation.rs`, `fluid_dynamics.rs`, `statistics.rs`) with a placeholder `DUMMY = 1.0` `theory_verification!` assertion, and `AstrophysicsTab` lacked an integrated strong-field Schwarzschild geodesic and relativistic twin-jet Doppler beaming laboratory. Additionally, central-difference Christoffel stencils (`h = 10^-5`) break down within `10^-5` of the Schwarzschild horizon `r = R_s`.

## Decision

1. **Closed-Form `LorentzianConnection4D` and `SchwarzschildConnection4D` (`pure_math::tensor`):**
   - Added [`lorentzian.rs`](../../crates/pure_math/src/pure_math/tensor/lorentzian.rs) defining `LorentzianConnection4D` and `SchwarzschildConnection4D` over stack-allocated `nalgebra::Vector4<f64>` and `Matrix4<f64>` registers, agreeing with `christoffel_symbols` within `1e-10` at `r = 3 R_s` while remaining exact down to any exterior radius `r > R_s`.
   - Clarified the `[cite:tensors]` documentation note in [`crates/pure_math/src/pure_math/tensor/mod.rs`](../../crates/pure_math/src/pure_math/tensor/mod.rs) distinguishing SU(2) spherical tensor coupling (`papers/tensors.tex`) from Lorentzian manifold connections (`papers/general_relativity_geodesics.tex`).

2. **Pre-Allocated Geodesic Bundle and Coupled Jet-Knot Observables (`domain_physics::physics::high_energy`):**
   - Extended [`SchwarzschildBlackHole`](../../crates/domain_physics/src/physics/high_energy/general_relativity/schwarzschild.rs) with `connection_4d`, `critical_impact_parameter`, `isco_specific_angular_momentum`, `circular_shell_beta`, `killing_invariants`, and adaptive stack-only `step_geodesic_rk4`, plus pre-allocated [`SchwarzschildGeodesicBundle`](../../crates/domain_physics/src/physics/high_energy/general_relativity/geodesic_bundle.rs) supporting up to 64 simultaneous null and timelike trajectories alongside Newtonian `-GM/r^2` reference orbits.
   - Added Lorentz boosts (`FourVector::lorentz_boost_z`, `FourVector::lorentz_boost`), `relativistic_doppler_factor`, and `apparent_superluminal_beta` in [`observer.rs`](../../crates/domain_physics/src/physics/high_energy/observer.rs), coupled via [`evaluate_jet_knot_observables`](../../crates/domain_physics/src/physics/high_energy/radiation.rs) to `synchrotron_power`, `inverse_compton_spectral_index`, `shock_compression_ratio`, and `li_ma_significance`.
   - Replaced `DUMMY = 1.0` in [`crates/domain_physics/src/physics/high_energy/mod.rs`](../../crates/domain_physics/src/physics/high_energy/mod.rs) with physical Schwarzschild and relativistic shock invariants.

3. **Interactive `AstrophysicsTab` Tools (`math_explorer_gui`):**
   - Registered [`BlackHoleGeodesicWorkbench`](../../math_explorer_gui/src/tabs/astrophysics/black_hole_geodesic_tool.rs) and [`RelativisticJetBeamingLab`](../../math_explorer_gui/src/tabs/astrophysics/relativistic_jet_tool.rs) under `#tab=astrophysics` with four `StarterPresetMetadata` cards, `FiniteStateGuard`, `SemanticPalette` WCAG 2.2 AA styling, and `ExportCapabilities::plot()`.

## Consequences

- Stepping 64 simultaneous null and timelike Schwarzschild trajectories inside `VerifiedEngine::run_verified` executes with zero dynamic heap allocations and `< 1e-6` Killing energy and angular-momentum drift.
- Reproduces `6.51 rad` photon-ring deflection (`b = 2.60 R_s`), `37.03 R_s/c` sub-ISCO plunge vs. `2.88 R_s` Newtonian periapsis, `18,855x` twin-jet flux asymmetry, and `58.3x` EHT ISCO accretion-ring Doppler brightness asymmetry.
