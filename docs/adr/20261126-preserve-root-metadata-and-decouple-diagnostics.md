# ADR 20261126: Preserve Causal Root Diagnostic Metadata, Wire `math_rng` Diagnostics, and Decouple `std` `diagnostics` from Domain Crates

## Status

Accepted

## Context

Following [ADR 20261124](20261124-no-std-math-diagnostics-and-causal-error-wiring.md) and [ADR 20261125](20261125-metadata-driven-tier-migration.md), four diagnostic and dependency-hygiene gaps remained across the five-tier workspace:

1. **Causal Wrapper Metadata Clobbering**: `math_diagnostics::HashMapVisitor` unconditionally overwrote `"code"`, `"error_type"`, and `"description"` whenever an outer wrapper (`FitError`, `DiffeqError`, `PdeError`, `BatteryError`, `QuantumError`, etc.) visited its inner cause first and then recorded its own header fields. In addition, `MathError::visit_fields` did not emit `MathError::code()` (`MATH-001`..`MATH-007`), causing `DiffeqError::Math` and `PdeError::Math` to report only wrapper codes (`DIFFEQ-005` / `PDE-007`) and `FitError::Linalg(LinalgError::SingularMatrix)` to overwrite `"LINALG-004"` with `"MATH-FIT-006"`.
2. **Missing Narrow Scalar `DiagnosticVisitor` Hooks**: `DiagnosticVisitor` provided `record_usize`, `record_u64`, `record_i64`, `record_f64`, and `record_bool`, forcing errors with 32-bit or signed pointer-sized fields (such as `QuantumError` and `QmcError::InvalidBase`) to perform manual widening casts (`u64::from(*n)`, `i64::from(*m)`).
3. **Tier 0 `math_rng` Diagnostic and Facade Gap**: `math_rng::{RngError, QmcError}` implemented `core::error::Error` without `math_diagnostics::Diagnostic`, were not re-exported or convertible via `?` into `oxidizemath::error::Error`, and were excluded from `unified_verification`'s `api_drift` facade error parity checks (`facade_role = "none"`).
4. **Tier 3 Production Coupling to `std` `diagnostics`**: `pure_math` and all six `domain_*` crates still listed `crates/01_core/diagnostics` (`std` `RwLock`/`Arc` `DiagnosticBus`) in `[dependencies]` even though production domain code only implements or invokes `Diagnostic`, `DiagnosticVisitor`, and `Severity` from Tier 0 `math_diagnostics`.

## Decision

1. **Root Metadata Preservation in `HashMapVisitor` (`crates/00_foundation/math_diagnostics`)**:
   - Preserve the first-written `"code"` in `HashMapVisitor` and record subsequent outer wrapper `"code"` writes under `"wrapper_code"`.
   - Keep `metadata["error_type"]` and `metadata["description"]` set to the outermost error's type and message so `diagnostics::DiagnosticBus::emit_diagnostic` continues to attribute default event `source` to the immediate caller-facing error type, while saving the first-written values under `"root_error_type"` and `"root_description"` whenever overwritten.
   - Emit `visitor.record_str("code", self.code())` in `MathError::visit_fields` so wrapped `MathError` variants preserve `MATH-001`..`MATH-007`.

2. **SemVer-Compatible Typed Scalar Visitor Methods (`0.4.0`)**:
   - Extend `DiagnosticVisitor` with default-implemented `record_isize`, `record_u32`, `record_i32`, and `record_f32` methods delegating to `self.record_display(key, &value)`, satisfying `[semver_lock]` at `0.4.0` without breaking downstream visitor implementations.
   - Migrate `QuantumError::visit_fields` to `record_u32` and `record_i32`.

3. **`#![no_std]` `math_rng` Diagnostics and Facade Parity**:
   - Wire `math_diagnostics` (`default-features = false`, forwarding `serde` and `std`) into `crates/00_foundation/math_rng` and implement zero-allocation `Diagnostic` for `RngError` (`RNG-001`) and `QmcError` (`QMC-001`..`QMC-004`).
   - Re-export `QmcError` and `RngError` from `oxidizemath::error` (and `math_explorer::error`), add `Error::Qmc` and `Error::Rng` variants with `From` conversions and `Diagnostic` delegation, set `facade_role = "foundation"` on `math_rng`, and add `"math_rng"` to `DEFAULT_REACHABLE_PKGS` in `apps/unified_verification/src/api_drift/facade_context.rs`.

4. **Decouple `std` `diagnostics` from Tier 3 Production Graphs**:
   - Replace `diagnostics` with `math_diagnostics` in `[dependencies]` across `pure_math`, `domain_ai`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, and `domain_physics`.
   - Retain `diagnostics` strictly under `[dev-dependencies]` in the five domain crates whose test suites exercise `DiagnosticBus` (`pure_math`, `domain_applied`, `domain_biology`, `domain_epidemiology`, `domain_physics`), keeping `xtask` reverse-dependent test expansion intact while removing `std` telemetry synchronization from production dependency graphs.

## Consequences

- Multi-hop wrapped errors (such as `FitError::from(LinalgError::SingularMatrix)`, `DiffeqError::from(MathError::WorkspaceTooSmall { .. })`, and `PdeError::from(MathError::WorkspaceTooSmall { .. })`) preserve root diagnostic codes (`LINALG-004`, `MATH-005`) and `root_error_type` alongside `wrapper_code` and outer `error_type`.
- `RngError` and `QmcError` participate in zero-allocation `#![no_std]` diagnostic visitation, `?` conversion into `oxidizemath::error::Error`, and automated `api_drift` facade parity enforcement.
- `cargo tree -e normal` across `pure_math` and all six `domain_*` crates no longer pulls in `crates/01_core/diagnostics`.
