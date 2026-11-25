# ADR 20261124: Complete Checked Companion APIs and Eliminate Untyped Public Result Signatures

## Status
Accepted

## Context
[ADR 20261123: Checked Companion APIs, Typed Domain Errors, and Human-Readable Diagnostics](20261123-checked-apis-and-typed-domain-errors.md) standardized typed error enums across the domain crates, migrated bounded `math_commons` primitives away from `Result<_, &'static str>`, and added `try_*` / `checked_*` companions on panicking constructors, while [ADR 20261124: `#![no_std]` Visitor-Based `math_diagnostics`, Causal `LinalgError` Retention, and Umbrella `api_drift` Guards](20261124-no-std-math-diagnostics-and-causal-error-wiring.md) introduced stack-only `DiagnosticVisitor` traversal and un-gated `engine_pde` stencil constructors for `#![no_std]`.

Five untyped `Result` signatures (`String` and `&'static str` error payloads) remained in `crates/unified_verification/api_baseline.json` (`SculptedGridMdp::apply_cells_u8`, `MriVolume::new`, `CellMask2D::set_roles_slice`, `OrbitalQuantumNumbers::new`, and `parse_density_floors_toml`), alongside `#![no_std]` checked API gaps on `SolverWorkspace::take_slice`, `SemiImplicitFourierStepper::{new_1d, new_2d}`, and `BaoabLangevinSolver::new`. Downstream headless and `#![no_std]` callers cannot match structurally on `String` or `&'static str` errors, and `format!` error construction allocates on the heap on failure paths.

## Decision
1. **`#![no_std]` Checked Kernel & Solver Companions**:
   - Add stack-allocated `MathError::WorkspaceTooSmall { required: usize, available: usize }` (recording `required` and `available` in `math_diagnostics::Diagnostic::visit_fields` and deriving optional `serde`) and `SolverWorkspace::try_take_slice(&mut self, len: usize) -> Result<&'a mut [T], MathError>` alongside `SolverWorkspace::take_slice`.
   - Expose `SemiImplicitFourierStepper::{try_new_1d, try_new_2d}` (`engine_pde`) and `BaoabLangevinSolver::try_new` (`engine_diffeq`) unconditionally in `#![no_std]` builds returning `Result<Self, MathError>`.
2. **Zero Untyped Public `Result` Signatures**:
   - Migrate `SculptedGridMdp::apply_cells_u8` to `Result<(), AIError>` via stack-allocated `AIError::{CellMaskLengthMismatch, InvalidCellRoleTag}`.
   - Migrate `MriVolume::new` to `Result<Self, EngineeringError>` via stack-allocated `EngineeringError::{ZeroVolumeDimension, InvalidVoxelSize, NonFiniteOrigin, VolumeDimensionsOverflow}` and `DimensionMismatch`.
   - Migrate `CellMask2D::set_roles_slice` to `Result<(), ReactionDiffusionError>` via stack-allocated `ReactionDiffusionError::CellMaskLengthMismatch`.
   - Introduce `QuantumError` (`InvalidPrincipalQuantumNumber`, `InvalidOrbitalQuantumNumber`, `InvalidMagneticQuantumNumber`, `Math`) with co-located zero-allocation `Diagnostic::visit_fields` in `domain_physics::physics::quantum::hydrogenic`, migrate `OrbitalQuantumNumbers::new` to `Result<Self, QuantumError>`, and wire `QuantumError` into `PhysicsError`, `math_explorer::error`, and `math_explorer::prelude`.
   - Introduce `DensityFloorsError` in `oxidize_core::traceability` and migrate `parse_density_floors_toml` to `Result<Option<DensityFloors>, DensityFloorsError>` with `Display` formatting free of `{:?}` debug wrappers.

## Consequences
- **Positive**: Zero public library signatures in `crates/unified_verification/api_baseline.json` return `Result` with `String`, `&'static str`, or `()`, and `#![no_std]` solver callers can validate workspaces and physical parameters without panicking or allocating on the heap.
- **Positive**: Co-locating `QuantumError`'s `Diagnostic` implementation in `quantum/hydrogenic.rs` and isolating checked-API tests into dedicated integration test targets reduces churn in shared error umbrella and domain test files.
- **Negative / Trade-offs**: Changing the error types of `MriVolume::new`, `CellMask2D::set_roles_slice`, and `OrbitalQuantumNumbers::new` updates the public signature baseline (`crates/unified_verification/api_baseline.json`) for those Certified modules; GUI state loaders adapt via `.map_err(|err| err.to_string())` at the UI boundary.
