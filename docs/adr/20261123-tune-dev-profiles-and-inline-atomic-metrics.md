# ADR 20261123: Root Dev/Test Dependency Optimization Profiles and Inlined Atomic Verification Fast Path

## Status
Accepted (extends [ADR 0027](0027-enforce-workspace-clippy-lints.md) and [ADR 0034](0034-scoped-pre-commit-and-single-parse-ast-verification.md))

## Context
1. **Unoptimized Third-Party Dependencies in `dev` and `test` Profiles**:
   - Root [`Cargo.toml`](../../Cargo.toml) previously configured only `[profile.release]` and omitted `[profile.dev]` and `[profile.test]`. Following [ADR 0034](0034-scoped-pre-commit-and-single-parse-ast-verification.md), local gates (`xtask lint-all`, `xtask lint-staged`) compile and invoke `target/debug/unified_verification` in debug mode to avoid release fat LTO, leaving heavy third-party dependencies (`syn`, `nalgebra`, `egui`, `eframe`, `rustfft`, `serde_json`, `toml`, `regex`) at `opt-level = 0` with full debug symbols (`~97.7 MiB` debug `unified_verification` binary and slower `batch-ast` parsing and `egui_plot` curve fitting).
2. **Thread-Local Reads in `#[verified]` Telemetry**:
   - [`crates/verified_engine_macros/src/verified.rs`](../../crates/verified_engine_macros/src/verified.rs) queries [`is_verification_enabled`](../../crates/verified_engine/src/metrics.rs) at `#[verified]` function entry (caching `__ve_active`) and inside `increment_calls`, `increment_arithmetic`, and `increment_memory_loads`. Without an inlined process-wide atomic guard, inactive calls still pay cross-crate call and thread-local `VERIFICATION_MODE` lookup overhead even when no thread has verification enabled (contrasting with [`VerifiedAllocator`](../../crates/verified_engine/src/allocator.rs), which guards `is_verification_enabled` behind atomic `MEMORY_LOCK`).

## Decision
1. **Workspace Root `dev` and `test` Profiles (`Cargo.toml` & `unified_verification`)**:
   - Configure `[profile.dev]` and `[profile.test]` in root [`Cargo.toml`](../../Cargo.toml) with `opt-level = 0` and `debug = "line-tables-only"` so first-party workspace crates retain fast incremental compilation, readable stack traces, and accurate `cargo llvm-cov` line coverage in [`crates/unified_verification/src/report.rs`](../../crates/unified_verification/src/report.rs).
   - Configure `[profile.dev.package."*"]` and `[profile.test.package."*"]` with `opt-level = 2` and `debug = false` so external third-party dependencies compile with optimizations and without debug symbol bloat, while leaving `[profile.release]` (`overflow-checks = true`, `lto = true`, `codegen-units = 1`, `strip = true`, `panic = "abort"`) untouched.
   - Extend `check_member_manifest` in [`crates/unified_verification/src/profile.rs`](../../crates/unified_verification/src/profile.rs) so `check-profiles` rejects member-level overrides of `[profile.dev]` and `[profile.test]` alongside `[profile.release]`.
2. **Idempotent `AtomicUsize` Guard and `#[inline(always)]` Short-Circuit (`verified_engine`)**:
   - Maintain a private process-wide `ACTIVE_VERIFIERS: AtomicUsize` counter in [`crates/verified_engine/src/metrics.rs`](../../crates/verified_engine/src/metrics.rs) without altering public signatures locked in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json).
   - Use `VERIFICATION_MODE.replace(...)` with `Ordering::SeqCst` transitions in `enable_verification` and `disable_verification` so `ACTIVE_VERIFIERS` increments and decrements idempotently per thread and stays exact across repeated toggles, worker threads, and `VerifiedEngine::run_verified` panic unwinds in [`crates/verified_engine/src/engine.rs`](../../crates/verified_engine/src/engine.rs).
   - Mark `is_verification_enabled` with `#[inline(always)]` and short-circuit via `ACTIVE_VERIFIERS.load(Ordering::Relaxed) != 0 && VERIFICATION_MODE.get()`, backed by unit tests in [`crates/verified_engine/src/metrics.rs`](../../crates/verified_engine/src/metrics.rs) and subprocess zero-allocation trap tests in [`crates/verified_engine/tests/allocator_trap.rs`](../../crates/verified_engine/tests/allocator_trap.rs).

## Verification Commands
```bash
cargo run -p xtask -- check-profiles
cargo test -p verified_engine --all-features
cargo test -p egui_plot -- --test-threads=1
cargo run -p xtask -- lint-all
```

## Consequences
- **Positive**: `target/debug/unified_verification` shrinks substantially while `syn` AST passes (`batch-ast`, `batch-static`, `batch-staged`), `cargo test`, `cargo llvm-cov`, and debug `math_explorer_gui` / `egui_plot` numerical loops execute with optimized third-party dependencies and a single inlined relaxed atomic check on inactive `#[verified]` paths.
- **Positive**: Public API signatures in `api_baseline.json`, `[profile.release]` safety settings, `VerifiedAllocator` heap-lock enforcement, and `ComplexityMetrics` telemetry assertions remain completely unchanged.
- **Neutral**: Initial cold compilation of third-party crates under `dev`/`test` performs `opt-level = 2` optimization once per dependency cache entry, amortized across all subsequent incremental workspace builds.
