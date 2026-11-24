# Runtime Allocator Probes, Climate Scratch Workspaces, Checked Kernel Bounds, and Native Supply-Chain Security Gates

## Status

Accepted

## Context

OxidizeMath enforces NASA Power of 10 Rule 3 (no dynamic heap allocations inside verified simulation hot paths) through `VerifiedEngine::run_verified` and `VerifiedAllocator`. Prior to this decision, five defense-in-depth gaps remained across runtime harnesses, static analysis, numerical kernels, and CI supply-chain gates:

1. **Silent `System` Allocator Bypasses & Uncounted Bytes**: `VerifiedEngine::run_verified` toggled `MEMORY_LOCK` without verifying that `VerifiedAllocator` was actually installed as `#[global_allocator]` in the running test binary, allowing five math `--lib` harnesses and several integration binaries to pass even when allocating on the heap, while `math_explorer_gui/src/test_alloc.rs` exempted allocations below `64 KiB` and `ComplexityMetrics` did not record allocation counts or byte totals.
2. **Unannotated Allocating `#[verified]` Functions**: `#[verified]` lacked an explicit role attribute distinguishing one-time constructors, workspace builders, convenience wrappers, and error formatters from zero-allocation simulation kernels, and `batch-ast` neither flagged unannotated heap allocations in `#[verified]` functions nor rejected test harnesses invoking `run_verified` or `lock_allocations` without `#[global_allocator]`.
3. **Heap-Allocating `domain_climate` Solvers**: CERA forward, analytical backpropagation, 1D Earth Mover's Distance (EMD), and trainer steps allocated intermediate `DMatrix`, `DVector`, and sorting buffers per step.
4. **Unchecked Kernel Capacity Products & Missing Tier 5 Suites**: Select capacity calculations across `pure_math`, `domain_ai`, `domain_applied`, `domain_biology`, `domain_climate`, and `domain_physics` multiplied dimensions without `checked_mul` or divided by `resolution - 1` at `resolution <= 1`, and six of the eight math crates lacked dedicated `tests/security_*.rs` suites.
5. **Hardcoded Advisory Ignores & Unchecked Lockfile Invariants**: Ignored `RUSTSEC-*` informational advisories lived as a hardcoded constant in `vulnerabilities.rs` rather than audited entries in [`verification_whitelist.toml`](../../verification_whitelist.toml), and `xtask` / CI lacked a unified `check-security` gate verifying allowed registries, SPDX licenses, duplicate crate family ceilings, and Tier 5 suite coverage.

## Decision

1. **Runtime Allocator Probe & Exact `ComplexityMetrics` Counters**:
   - Extend [`crates/verified_engine/src/allocator.rs`](../../crates/verified_engine/src/allocator.rs) and [`crates/verified_engine/src/metrics.rs`](../../crates/verified_engine/src/metrics.rs) to record exact `allocations` and `allocated_bytes` on every `alloc`, `alloc_zeroed`, and `realloc` growth, and expose a one-time `OnceLock` probe (`ensure_allocator_probe`) invoked by `VerifiedEngine::run_verified` that fails fast in unit and integration test binaries lacking `VerifiedAllocator` while permitting `rustdoc` (`cargo test --doc`) executables to collect metrics cleanly.
   - Remove the `< 64 KiB` allocation filter from [`math_explorer_gui/src/test_alloc.rs`](../../math_explorer_gui/src/test_alloc.rs) and register `#[cfg(test)] #[global_allocator]` across all eight math `--lib` crates and integration binaries calling `run_verified`.
2. **`#[verified(alloc_ok = "reason")]` & Single-Pass AST Role Audit**:
   - Extend `#[verified]` in [`crates/verified_engine_macros/src/verified.rs`](../../crates/verified_engine_macros/src/verified.rs) to accept `#[verified(alloc_ok = "reason")]` with non-empty justification strings.
   - Add [`crates/unified_verification/src/verified_alloc_audit.rs`](../../crates/unified_verification/src/verified_alloc_audit.rs) to `batch-ast` and `batch-staged` so unannotated heap-allocating expressions in non-test `#[verified]` functions and test files calling `run_verified` or `lock_allocations` without `#[global_allocator]` fail static verification.
3. **Zero-Allocation `domain_climate` Scratch Workspaces**:
   - Introduce [`crates/domain_climate/src/climate/scratch.rs`](../../crates/domain_climate/src/climate/scratch.rs) (`EmdScratch`, `ConvLayerScratch`, `AutoencoderScratch`, `PredictorScratch`, `CeraScratch`, `CeraTrainingScratch`) and zero-allocation `*_into` entry points (`Autoencoder::forward_into`, `Autoencoder::backward_update_into`, `Predictor::forward_into`, `Predictor::backward_update_into`, `try_earth_movers_distance_into`, `Cera::predict_into`, `CeraTrainer::train_step_into`, `CeraTrainer::train_into`) while preserving existing public wrappers.
4. **Checked Capacity Bounds & Tier 5 `tests/security_*.rs` Suites**:
   - Enforce `checked_mul`, zero/minimum-dimension guards, and `NaN`/infinity rejection across capacity-sensitive kernels (returning `MathError::CapacityOverflow`, `ClimateError::DimensionOverflow`, or typed domain errors) and establish `tests/security_*.rs` suites equipped with `VerifiedAllocator` across all eight math crates (`math_commons`, `pure_math`, `domain_ai`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, `domain_physics`).
5. **Audited `[supply_chain]` Policy & `check-security` Gate**:
   - Move the seven informational `RUSTSEC-*` advisories from `vulnerabilities.rs` into `[[supply_chain.ignored_advisories]]` in [`verification_whitelist.toml`](../../verification_whitelist.toml) with package names and documented reasons alongside `allowed_registries`, `allowed_licenses`, and `max_duplicate_families = 34` (full elimination of the 34 duplicate crate families in `Cargo.lock` remains out of scope).
   - Add `check-security` to [`crates/unified_verification/src/supply_chain.rs`](../../crates/unified_verification/src/supply_chain.rs), [`apps/xtask/src/security.rs`](../../apps/xtask/src/security.rs), `xtask lint-all`, and `.github/workflows/ci.yml` (`pre-flight`).

```bash
cargo run -p xtask -- check-security
```

## Consequences

- **Positive**: Any heap allocation inside `VerifiedEngine::run_verified` immediately traps across all eight math crates, integration binaries, and GUI performance tests; `domain_climate` inference, EMD, and training loops execute with `allocations == 0` and `allocated_bytes == 0`; overflow and `NaN`/subnormal edge cases are regression-tested in every math crate; and unapproved registries, copyleft licenses, unwhitelisted advisories, or new duplicate crate families above `34` fail CI pre-flight.
- **Trade-offs**: Legitimate one-time constructors, builders, convenience wrappers, and string-formatting error helpers carrying `#[verified]` must declare `#[verified(alloc_ok = "...")]` so `batch-ast` can distinguish initialization/diagnostic helpers from zero-allocation hot kernels.
