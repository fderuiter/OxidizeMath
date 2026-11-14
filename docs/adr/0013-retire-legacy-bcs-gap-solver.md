# 13. Retire Legacy BCS Gap Solver Function

Date: 2026-11-14

## Status
Accepted

## Context
The BCS superconductivity module in `crates/domain_physics/src/physics/solid_state/bcs.rs` previously retained an obsolete float-based wrapper function `solve_gap_equation`. This wrapper function was marked as deprecated but persisted as dead code in the public API baseline (`crates/unified_verification/api_baseline.json`). Furthermore, unit tests in `bcs.rs` suppressed compiler deprecation warnings using `#[allow(deprecated)]` instead of interacting directly with modern, strongly typed solver types (`BcsGapSolver` and `IsotropicBCSModel`).

## Decision
We removed the deprecated `solve_gap_equation` wrapper from `crates/domain_physics/src/physics/solid_state/bcs.rs` and regenerated the public API baseline to eliminate warning suppressions and eliminate technical debt. Unit tests were modernized to assert directly against `BcsGapSolver` and `IsotropicBCSModel` with explicit handling of `SolidStateError` convergence variants.

### Migration Path for Downstream Consumers

Downstream library consumers migrating from `solve_gap_equation` should instantiate `IsotropicBCSModel` and `BcsGapSolver` directly:

```rust,ignore
use domain_physics::physics::solid_state::bcs::{BcsGapSolver, IsotropicBCSModel};
use domain_physics::physics::solid_state::types::ElectronVolts;
use domain_physics::error::SolidStateError;

// Legacy approach (removed):
// let delta_ev = solve_gap_equation(&energies, potential, debye, 1000)?;

// Modern strongly-typed approach:
let model = IsotropicBCSModel::new(energies, potential, debye)?;
let solver = BcsGapSolver::default()
    .with_max_iterations(1000)
    .with_tolerance(math_commons::registry::TOLERANCE_STANDARD)
    .with_mixing_param(0.5);

let initial_guess = ElectronVolts::new(0.01 * debye);
let delta = match solver.solve(&model, initial_guess) {
    Ok(delta) => delta,
    Err(SolidStateError::ConvergenceFailure(_, last_val)) => last_val,
    Err(e) => return Err(e),
};
let delta_ev = delta / ElectronVolts::new(1.0);
```

## Consequences
- **Positive:**
  - Removed obsolete wrapper and associated compiler deprecation warning suppressions (`#[allow(deprecated)]`).
  - Strengthened type safety by requiring explicit use of `ElectronVolts` domain units and structured `SolidStateError` handling.
  - Eliminated API baseline drift in `crates/unified_verification/api_baseline.json`.
- **Negative:**
  - Downstream consumers relying on `solve_gap_equation` will encounter a breaking change and must migrate to `BcsGapSolver` and `IsotropicBCSModel`.
