# ADR 20261125: CI Pre-Flight Machete Metadata, Cargo Subcommand Tool Cache Probing, and WASM Unwind Test Guards

## Status
Accepted (extends [ADR 20261124](20261124-math-rng-static-qmc-views.md) and [ADR 20261123](20261123-tune-dev-profiles-and-inline-atomic-metrics.md))

## Context
Three cross-job CI verification failures surfaced across pre-flight, Linux coverage setup, and WebAssembly test execution:
1. **Doctest-Only `rand` Dependency in `oxidize_core`**: Following the Tier-0 RNG migration in [ADR 20261124](20261124-math-rng-static-qmc-views.md), [`crates/oxidize_core/src/rng.rs`](../../crates/oxidize_core/src/rng.rs) re-exports `math_rng` while [`crates/oxidize_core/README.md`](../../crates/oxidize_core/README.md) retains an executable `use rand::RngCore;` doctest and [`crates/oxidize_core/Cargo.toml`](../../crates/oxidize_core/Cargo.toml) preserves `rand` / `getrandom` feature unification. Because `cargo-machete` inspects `.rs` sources without scanning `README.md` doctests, `Pre-flight Checks` flagged `rand` in `crates/oxidize_core/Cargo.toml`.
2. **Cargo Subcommand `--version` Probing in `ci-tool-cache`**: [`.github/actions/ci-tool-cache/action.yml`](../../.github/actions/ci-tool-cache/action.yml) verified cached or newly installed binaries via `"$bin_path" --version`, which fails for strict Cargo subcommand binaries such as `cargo-llvm-cov` (`0.9.2`) that require `"$bin_path" llvm-cov --version`.
3. **Unwind-Only `catch_unwind` Test on `wasm32-unknown-unknown`**: [`crates/verified_engine/src/metrics.rs`](../../crates/verified_engine/src/metrics.rs) compiled `test_run_verified_panic_unwind_preserves_atomic_count` on `wasm32-unknown-unknown`, where `panic = "abort"` traps `unreachable` inside `std::panic::catch_unwind` during `xtask test-wasm`.

## Decision
1. Declare `[package.metadata.cargo-machete]` with `ignored = ["rand"]` in [`crates/oxidize_core/Cargo.toml`](../../crates/oxidize_core/Cargo.toml) so `cargo-machete` and `unified_verification` `test-policy-audit` agree on the doctest-referenced `rand` dependency.
2. Update [`.github/actions/ci-tool-cache/action.yml`](../../.github/actions/ci-tool-cache/action.yml) so `cargo-*` binaries probe `"$bin_path" "${BIN_NAME#cargo-}" --version` before falling back to `"$bin_path" --version`.
3. Gate `test_run_verified_panic_unwind_preserves_atomic_count` in [`crates/verified_engine/src/metrics.rs`](../../crates/verified_engine/src/metrics.rs) with `#[cfg(not(target_arch = "wasm32"))]`, and serialize `non_fatal_diagnostics_do_not_trip_crash_boundary` in [`math_explorer_gui/src/crash_boundary.rs`](../../math_explorer_gui/src/crash_boundary.rs) behind `TEST_MUTEX`.

## Consequences
- **Positive**: `cargo-machete`, `ci-tool-cache` binary verification (`cargo-llvm-cov`, `cargo-audit`, `trunk`, `wasm-bindgen`, `cargo-packager`), and full `cargo run -p xtask -- test-wasm` pass deterministically without altering public crate APIs or workspace feature unification.
- **Neutral**: Stack-unwinding `VerifiedEngine::run_verified` recovery remains verified on all unwinding host targets while `wasm32` unit suites execute all non-unwinding `verified_engine` metrics tests cleanly.
