# ADR 0037: Retire Obsolete Physics Compatibility Shims

## Status
Accepted (extends [ADR 0004](0004-remove-deprecated-domain-apis.md), [ADR 0013](0013-retire-legacy-bcs-gap-solver.md), [ADR 0016](0016-formal-beam-state-machine-guards.md), [ADR 0019](0019-purge-dead-code-bypasses.md), and [ADR 0030](0030-remove-autoencoder-weight-update-shims.md))

## Context
Following the strong-typing, strategy-pattern, and state-machine migrations in [ADR 0004](0004-remove-deprecated-domain-apis.md), [ADR 0013](0013-retire-legacy-bcs-gap-solver.md), and [ADR 0016](0016-formal-beam-state-machine-guards.md), `crates/domain_physics` still exported obsolete compatibility shims with zero production callers across the workspace:
1. **BCS Superconductivity (`solid_state/bcs.rs`)**: Retained a raw-float `coherence_factors(f64, f64)` wrapper with `#[allow(missing_docs)]` delegating to `coherence_factors_strong(ElectronVolts, ElectronVolts)`, called only by `test_bcs_probability_conservation`.
2. **Quantum Statistical Mechanics (`stat_mech/quantum_stats.rs`)**: Retained `ParticleType` (carrying `#[allow(missing_docs)]` on its variants) and `occupancy_probability`, called only by `test_fermion_limit` alongside the strategy-based `calculate_occupancy` API (`FermiDirac`, `BoseEinstein`, and `MaxwellBoltzmann`).
3. **Fluid Conservation Laws (`fluid_dynamics/conservation.rs`)**: Retained `navier_stokes_time_derivative` (called only by `test_navier_stokes_simple_couette` in [`crates/domain_physics/src/physics/fluid_dynamics/tests.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/tests.rs)) and uncalled `euler_time_derivative` (along with its otherwise unused `FluidError` import) wrapping `NavierStokes::acceleration` and `Euler::acceleration`.
4. **Respiratory Radar Gating (`medical/radar_gating/gating.rs`)**: Retained an unverified `GatingLogic::is_beam_on` shim delegating to `#[verified_engine::verified]` `GatingLogic::is_beam_active`.
5. **Nuclear Physics (`nuclear/mod.rs`)**: Retained `liquid_drop` (`binding_energy`, `binding_energy_per_nucleon`) and `shell_model` (`spin_orbit_coupling`) wrapper submodules around `LiquidDropModel` and `models::shell::spin_orbit_coupling`, called only by `test_iron_peak` and `test_spin_orbit`.
6. **Dwarf Galaxies Documentation (`astrophysics/galaxies.rs`)**: Module-level documentation still claimed that deprecated legacy functions were preserved after those functions had already been removed.

## Decision
1. **Delete Obsolete Physics Shims**:
   - Remove `coherence_factors` from [`crates/domain_physics/src/physics/solid_state/bcs.rs`](../../crates/domain_physics/src/physics/solid_state/bcs.rs).
   - Remove `ParticleType` and `occupancy_probability` from [`crates/domain_physics/src/physics/stat_mech/quantum_stats.rs`](../../crates/domain_physics/src/physics/stat_mech/quantum_stats.rs).
   - Remove `navier_stokes_time_derivative`, `euler_time_derivative`, and the unused `FluidError` import from [`crates/domain_physics/src/physics/fluid_dynamics/conservation.rs`](../../crates/domain_physics/src/physics/fluid_dynamics/conservation.rs).
   - Remove unverified `GatingLogic::is_beam_on` from [`crates/domain_physics/src/physics/medical/radar_gating/gating.rs`](../../crates/domain_physics/src/physics/medical/radar_gating/gating.rs), leaving every public method on `GatingLogic` annotated with `#[verified_engine::verified]`.
   - Remove the `liquid_drop` and `shell_model` wrapper submodules from [`crates/domain_physics/src/physics/nuclear/mod.rs`](../../crates/domain_physics/src/physics/nuclear/mod.rs).
2. **Migrate Legacy Unit Tests**:
   - Migrate `test_bcs_probability_conservation` to `coherence_factors_strong` with `ElectronVolts::new`.
   - Migrate `test_fermion_limit` to `calculate_occupancy` with `FermiDirac`.
   - Migrate `test_navier_stokes_simple_couette` to `NavierStokes.acceleration` with `SpatialGradients::new`.
   - Migrate `test_iron_peak` to `LiquidDropModel::new` and `test_spin_orbit` to `models::shell::spin_orbit_coupling`, preserving all assertions and numerical tolerances.
3. **Clean Stale Documentation & Regenerate Public API Baseline**:
   - Remove the stale legacy deprecation sentence from [`crates/domain_physics/src/physics/astrophysics/galaxies.rs`](../../crates/domain_physics/src/physics/astrophysics/galaxies.rs).
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) via `regenerate-baseline` to drop all 11 retired public signatures while keeping active exports (`LatticeBoltzmannD2Q9`, `BlochSimulator`, and `GatingLogic::is_beam_active`) intact.

## Consequences
- **Positive**: Eliminates eight obsolete wrapper functions, one legacy enum (`ParticleType`), two redundant wrapper submodules, four `#[allow(missing_docs)]` attributes, and an unverified `GatingLogic` method from `domain_physics` and `math_explorer::physics`, pruning 11 legacy signatures from `crates/unified_verification/api_baseline.json`.
- **Neutral**: Simulation behavior, verified numerical tolerances, error variants, and active physics APIs remain unchanged.
