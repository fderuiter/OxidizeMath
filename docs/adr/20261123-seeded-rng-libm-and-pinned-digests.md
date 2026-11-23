# ADR 20261123: Seeded RNG Threading, Pure-Rust `libm` Wrappers, and Pinned Trajectory Digests

## Status

Accepted (supersedes the per-layer `OxidizeRng::default()` initialization decision in [ADR 0031](0031-calibrated-climate-series-and-conv-rng-seeding.md))

## Context

Three reproducibility and verification gaps existed across stochastic math modules:

1. **Cross-Layer and Per-Step RNG Reseeding**: In [`crates/domain_climate/src/climate/autoencoder.rs`](../../crates/domain_climate/src/climate/autoencoder.rs), [`crates/domain_climate/src/climate/predictor.rs`](../../crates/domain_climate/src/climate/predictor.rs), and [`crates/domain_climate/src/climate/cera.rs`](../../crates/domain_climate/src/climate/cera.rs), `ConvLayer::new` reset to `OxidizeRng::default()` on every layer construction without `*_with_rng` constructors on `ConvLayer`, `Encoder`, `Decoder`, `Autoencoder`, `Predictor`, or `Cera`. Consequently, `Encoder::new` and `Decoder::new` produced identical `64x64` hidden kernels and `Predictor::new` cloned identical `128x128` weights across all three hidden layers. Similarly, legacy shims in `domain_epidemiology` (`NetworkEpidemicModel::initialize_geometric_graph`, `NetworkEpidemicModel::step`) and `domain_ai` (`sample_timestep`, `generate_noise`, `stratified_sampling`, `render_image`) reseeded on every invocation, and eight rustdoc examples imported non-portable `StdRng`.
2. **Host `libm` Drift in Stochastic Kernels**: Fourteen stochastic simulation and sampling modules called platform C `libm` transcendentals (`ln`, `ln_1p`, `exp`, `powf`, `sqrt`, `hypot`, `sin`, `cos`, `tan`), which are not guaranteed to round identically across Linux `glibc`, Windows `UCRT`, macOS `libSystem`, and `wasm32-unknown-unknown`.
3. **Missing Cross-Target Pinned Digest Suites and Gates**: Seven of eight stochastic crates lacked `tests/*_deterministic.rs` suites, `crates/oxidize_core/src/workspace_wasm.rs` invoked `cargo test --target wasm32-unknown-unknown --lib` without integration `--test` targets, and `entropy_guard` / `doc_audit` did not reject `StdRng` or unallowlisted `OxidizeRng::default()` calls.

## Decision

1. **Cross-Layer `OxidizeRng` Threading & Additive `*_with_rng` Constructors**:
   - Add `ConvLayer::new_with_rng`, `Encoder::new_with_rng`, `Decoder::new_with_rng`, `Autoencoder::new_with_rng`, `Predictor::new_with_rng`, `Cera::new_with_rng`, and `Cera::new_with_predictor_and_rng` while preserving all existing signatures in `crates/unified_verification/api_baseline.json`.
   - Thread one `OxidizeRng` across all layers in `Encoder::new`, `Decoder::new`, `Autoencoder::new`, `Predictor::new`, and `Cera::new` (with domain-separated `BASE_SEED` offsets for standalone `Decoder::new` and `Predictor::new`), guaranteeing pairwise-distinct kernel matrices across all encoder, decoder, and predictor layers.
   - Audit all domain and GUI `OxidizeRng::default()` call sites, migrate default-seed constructors to `OxidizeRng::from_seed(oxidize_core::rng::BASE_SEED)`, deprecate per-step reseeding shims, replace all eight `StdRng` doctests with `math_explorer::rng::OxidizeRng`, and remove `"StdRng"` from `STD_AND_REEXPORTED_TYPES` in [`crates/unified_verification/src/api_drift/facade.rs`](../../crates/unified_verification/src/api_drift/facade.rs).
2. **Pure-Rust `libm` (`soft_float`) Wrappers in `math_commons`**:
   - Reuse `libm = "0.2.16"` already present in `Cargo.lock` and expose zero-allocation `f64`/`f32` wrappers in [`crates/math_commons/src/math_kernel/soft_float.rs`](../../crates/math_commons/src/math_kernel/soft_float.rs).
   - Route transcendental and root operations across stochastic modules through `soft_float` while keeping non-stochastic ODE and PDE solvers on hardware float instructions and preserving zero heap allocations under `VerifiedAllocator`.
3. **Canonical IEEE-754 `TrajectoryDigest` and Cross-Platform `*_deterministic.rs` Suites**:
   - Add zero-dependency streaming SHA-256 `TrajectoryDigest` in [`crates/oxidize_core/src/digest.rs`](../../crates/oxidize_core/src/digest.rs) (re-exported via `oxidize_core::rng` and `math_explorer::rng`), normalizing `-0.0` to `+0.0`, all `NaN` bit patterns to canonical quiet `NaN` (`0x7ff8_0000_0000_0000` / `0x7fc0_0000`), and `usize` to 64-bit little-endian.
   - Add and upgrade `tests/*_deterministic.rs` suites with pinned 64-character SHA-256 hex digests across `oxidize_core`, `math_commons`, `pure_math`, `domain_ai`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, and `domain_physics`, plus multi-step `GillespieSolver` digest verification in [`crates/domain_epidemiology/src/epidemiology/stochastic.rs`](../../crates/domain_epidemiology/src/epidemiology/stochastic.rs).
4. **Static & `wasm32` Determinism Gates**:
   - Extend [`crates/unified_verification/src/entropy_guard.rs`](../../crates/unified_verification/src/entropy_guard.rs) and [`crates/unified_verification/src/doc_audit/facade_imports.rs`](../../crates/unified_verification/src/doc_audit/facade_imports.rs) to reject `StdRng` and unallowlisted `OxidizeRng::default()` calls (`default_rng_allowlist = ["crates/oxidize_core/src/rng.rs"]` in `verification_whitelist.toml`).
   - Extend [`crates/oxidize_core/src/workspace_wasm.rs`](../../crates/oxidize_core/src/workspace_wasm.rs), [`crates/unified_verification/src/batch.rs`](../../crates/unified_verification/src/batch.rs), and [`apps/xtask/src/gates.rs`](../../apps/xtask/src/gates.rs) to verify `*_deterministic.rs` coverage and pinned SHA-256 digests across every stochastic crate and execute all `*_deterministic.rs` integration suites on `wasm32-unknown-unknown`.

## Consequences

- **Positive**: Native Linux, Windows, macOS, and `wasm32-unknown-unknown` builds execute the same `*_deterministic.rs` suites and assert bit-identical pinned SHA-256 trajectory digests.
- **Positive**: Multi-layer CERA models initialize independent kernels across every encoder, decoder, and predictor layer while supporting caller-injected `OxidizeRng` streams for Monte Carlo ensembles.
- **Positive**: Regressions introducing `StdRng`, unallowlisted `OxidizeRng::default()`, or unpinned stochastic suites fail fast during `batch-static`, `lint-all`, and `test-wasm`.
