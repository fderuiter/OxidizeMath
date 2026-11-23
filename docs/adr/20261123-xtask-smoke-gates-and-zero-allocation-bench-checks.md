# ADR 20261123: Add xtask Smoke Gates and Zero-Allocation Benchmark Checks

## Status

Accepted

## Context

Prior to this decision, the native CI workflow (`build-and-test` and `feature-matrix` in `.github/workflows/ci.yml`) and `apps/xtask` omitted Cargo benchmark targets (`--benches`, such as `math_explorer`'s `sorting_benchmark`) and never built or executed the 32 feature-gated `[[example]]` targets declared in `math_explorer/Cargo.toml`. In addition, three domain integration benchmark tests (`crates/domain_applied/tests/bench_isosurface.rs`, `crates/domain_biology/tests/bench_reaction_diffusion.rs`, and `crates/domain_physics/tests/bench_lbm_custom.rs`) printed `Instant::now()` wall-clock timings with zero assertions and without enforcing post-initialization zero-allocation guarantees under `VerifiedAllocator` and `VerifiedEngine::run_verified`.

## Decision

1. **`xtask check-benches` & Feature-Matrix Benchmark Checks**:
   - Introduce `cargo run -p xtask -- check-benches` (`apps/xtask/src/smoke.rs`) executing `cargo test --workspace --benches --all-features` on native targets, and configure `bench = false` on non-benchmarked binary crates (`math_explorer_gui` and `unified_verification`) so Cargo does not redundantly re-run binary unit suites already covered by `verify-suite` and `--lib --tests`.
   - Extend `cargo run -p xtask -- test-features` so each isolated domain feature leg also runs `cargo test -p math_explorer --benches` with matching feature flags.
   - Invoke `check-benches` on native legs (`matrix.target != 'wasm32-unknown-unknown'`) in `.github/workflows/ci.yml`.
2. **`xtask smoke-examples` & Bounded Smoke Scaling**:
   - Add `list-examples` to `unified_verification` so `apps/xtask` remains pure-`std` with zero external crate dependencies while dynamically reading every `[[example]]` target from `math_explorer/Cargo.toml`.
   - Introduce `cargo run -p xtask -- smoke-examples` to build all 32 `math_explorer` examples once and execute each binary inside an isolated RAII temporary working directory with `MATH_EXPLORER_SMOKE=1` and a per-example timeout, preventing generated files such as `torus.obj` (`isosurface_torus.rs`) from polluting the repository checkout.
   - Scale iteration counts and grid dimensions in heavy benchmark/demo examples when `MATH_EXPLORER_SMOKE=1` while preserving default full-scale parameters when unset.
3. **Deterministic Zero-Allocation Domain Benchmark Integration Tests**:
   - Install `VerifiedAllocator` as `#[global_allocator]` and wrap hot loops inside `VerifiedEngine::run_verified` across `bench_isosurface_performance` (using `IsosurfaceWorkspace::with_locked_capacity` and `extract_isosurface_into`), `bench_turing_2d_step`, and `bench_lbm_step`, accompanied by subprocess negative probes confirming `VerifiedAllocator` aborts on hot-loop heap allocation.
   - Assert finite physical/geometric invariants (vertex/normal/index counts, mass conservation, non-negative species concentrations) and positive `ComplexityMetrics` (`function_calls`, `arithmetic_ops`, `memory_loads`) rather than brittle wall-clock thresholds. In `bench_turing_2d_step`, enforcing finite species invariants exposed a silent 2D explicit Euler CFL violation (`dt = 0.01` with `D_v = 40.0, dx = dy = 1.0`, giving `2 * D_v * dt / dx^2 = 0.8 > 0.5`), corrected to a CFL-stable `dt = 0.005` (`0.4 <= 0.5`).
4. **Documentation & Rustdoc CLI Audits**:
   - Harden `unified_verification check-docs` to scan inline backtick `cargo` commands in `//!` and `///` doc comments, enforce `--features` or `--all-features` on `--workspace` and placeholder `-p <pkg>` `--example` invocations, and report `32/32` verified `math_explorer` examples.

## Consequences

- Regressions in Criterion benchmarks, runnable `math_explorer` examples, or documented quickstart/catalog `cargo run` commands fail fast in local `xtask` gates and native CI jobs without impacting `wasm32-unknown-unknown` steps.
- Unexpected heap allocations inside `extract_isosurface_into`, `TuringSystem::step`, or `LatticeBoltzmannD2Q9::step` deterministically abort their respective domain integration tests under `VerifiedAllocator`.
