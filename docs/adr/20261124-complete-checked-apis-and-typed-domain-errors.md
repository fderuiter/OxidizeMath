# ADR 20261124: Complete Checked Companion APIs and Eliminate Untyped Public Result Signatures

## Status
Accepted

## Context
[ADR 20261123: Checked Companion APIs, Typed Domain Errors, and Human-Readable Diagnostics](20261123-checked-apis-and-typed-domain-errors.md) standardized typed error enums across the domain crates, migrated bounded `math_commons` primitives away from `Result<_, &'static str>`, and added `try_*` / `checked_*` companions on panicking constructors. Subsequent workspace features introduced five untyped `Result` signatures (`String` and `&'static str` error payloads) into `crates/unified_verification/api_baseline.json` (`SculptedGridMdp::apply_cells_u8`, `MriVolume::new`, `CellMask2D::set_roles_slice`, `OrbitalQuantumNumbers::new`, and `parse_density_floors_toml`) and left `#![no_std]` checked API gaps on `SolverWorkspace::take_slice`, `engine_pde` stencil/Fourier steppers, and `BaoabLangevinSolver`.

Downstream headless and `#![no_std]` callers cannot match structurally on `String` or `&'static str` errors, `format!` error construction allocates on the heap on hot failure paths, and `#[cfg(feature = "std")]` gates on `StencilOperators::try_new*` and `FusedStencilStepper::try_new*` prevented embedded callers from validating grid spacings with `MathError`.

## Decision
1. **`#![no_std]` Checked Kernel & Solver Companions**:
   - Add stack-allocated `MathError::WorkspaceTooSmall { required: usize, available: usize }` (with `Diagnostic::metadata` keys in `std` builds) and `SolverWorkspace::try_take_slice(&mut self, len: usize) -> Result<&'a mut [T], MathError>` alongside `SolverWorkspace::take_slice`.
   - Expose `StencilOperators::{try_new, try_new_2d, try_new_3d}`, `FusedStencilStepper::{try_new, try_new_2d, try_new_3d}`, `SemiImplicitFourierStepper::{try_new_1d, try_new_2d}`, and `BaoabLangevinSolver::try_new` unconditionally in `#![no_std]` builds returning `Result<Self, MathError>`.
2. **Zero Untyped Public `Result` Signatures**:
   - Migrate `SculptedGridMdp::apply_cells_u8` to `Result<(), AIError>` via stack-allocated `AIError::{CellMaskLengthMismatch, InvalidCellRoleTag}`.
   - Migrate `MriVolume::new` to `Result<Self, EngineeringError>` via stack-allocated `EngineeringError::{ZeroVolumeDimension, InvalidVoxelSize, NonFiniteOrigin, VolumeDimensionsOverflow}` and `DimensionMismatch`.
   - Migrate `CellMask2D::set_roles_slice` to `Result<(), ReactionDiffusionError>` via stack-allocated `ReactionDiffusionError::CellMaskLengthMismatch`.
   - Introduce `QuantumError` (`InvalidPrincipalQuantumNumber`, `InvalidOrbitalQuantumNumber`, `InvalidMagneticQuantumNumber`, `Math`) in `domain_physics::physics::quantum::hydrogenic`, migrate `OrbitalQuantumNumbers::new` to `Result<Self, QuantumError>`, and wire `QuantumError` into `Diagnostic`, `PhysicsError`, `math_explorer::error`, and `math_explorer::prelude`.
   - Introduce `DensityFloorsError` in `oxidize_core::traceability` and migrate `parse_density_floors_toml` to `Result<Option<DensityFloors>, DensityFloorsError>` with `Display` formatting free of `{:?}` debug wrappers.

## Consequences
- **Positive**: Zero public library signatures in `crates/unified_verification/api_baseline.json` or `crates/math_commons/src/primitives.rs` return `Result` with `String`, `&'static str`, or `()`, and `#![no_std]` solver callers can validate workspaces and physical parameters without panicking or allocating on the heap.
- **Positive**: `QuantumError` integrates cleanly into `PhysicsError` and `math_explorer::error::Error` with one-hop `?` conversion and `DiagnosticBus` metadata support.
- **Negative / Trade-offs**: Changing the error types of `MriVolume::new`, `CellMask2D::set_roles_slice`, and `OrbitalQuantumNumbers::new` updates the public signature baseline (`crates/unified_verification/api_baseline.json`) for those Certified modules; GUI state loaders adapt via `.map_err(|err| err.to_string())` at the UI boundary.
