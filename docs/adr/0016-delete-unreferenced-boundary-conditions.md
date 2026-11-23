# ADR 0016: Delete Unreferenced Boundary Conditions

## Status

Accepted

## Context
The boundary module in `crates/oxidize_core/src/boundary.rs` previously contained struct definitions and `BoundaryCondition` implementations for `PeriodicBoundary` and `BounceBackBoundary`. Neither struct had active callers anywhere across the workspace crates or applications. Furthermore, `BounceBackBoundary` and `PeriodicBoundary` were tracked in the public API baseline (`crates/unified_verification/api_baseline.json`), adding maintenance overhead and representing dead code.

Active callers in `crates/domain_biology` only utilize `NeumannBoundary` and the `BoundaryCondition` trait for finite-difference 2D diffusion boundary strategies.

## Decision
1. **Remove Dead Boundary Implementations**: Deleted `PeriodicBoundary` and `BounceBackBoundary` structs along with their `BoundaryCondition` trait implementations from `crates/oxidize_core/src/boundary.rs`.
2. **Preserve Active Boundary Logic**: Retained `BoundaryCondition` and `NeumannBoundary` intact for ongoing usage in `domain_biology` and other domain consumers.
3. **Synchronize API Baseline**: Updated `crates/unified_verification/api_baseline.json` to remove the obsolete signatures, ensuring the public API drift verification passes.
4. **Add Unit Testing**: Added comprehensive unit tests in `crates/oxidize_core/src/boundary.rs` asserting correct coordinate resolution and boundary clamping for interior points, negative coordinates, exceeding dimensions, and edge cases.

## Consequences
- **Positive**:
  - Eliminated unused structs and dead code from `oxidize_core`.
  - Reduced public API baseline surface area.
  - Added explicit test coverage for `NeumannBoundary` coordinate resolution and clamping edge cases.
- **Negative**:
  - Any external consumer relying on `PeriodicBoundary` or `BounceBackBoundary` directly from `oxidize_core` would encounter a breaking change, though no internal or workspace callers exist.
