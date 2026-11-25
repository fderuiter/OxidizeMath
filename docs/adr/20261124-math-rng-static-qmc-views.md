# ADR 20261124: Tier 0 `math_rng` Crate, Stream-Split ChaCha8 Samplers, and 64D Static QMC Views

## Status
Accepted (2026-11-24)

## Context

Prior to this decision, deterministic pseudorandom number generation and Quasi-Monte Carlo (QMC) low-discrepancy sequences were split across `std`-coupled crates with four architectural gaps:

1. **Blocked `#![no_std]` engines:** `OxidizeRng` lived in `crates/oxidize_core`, which pulls `std` file-system and mesh utilities, preventing `#![no_std]` crates (`math_commons`, `engine_diffeq`, `engine_pde`) from sharing `OxidizeRng` directly in production APIs.
2. **Ad-hoc XOR seed offsets & manual 53-bit float conversions:** GUI workbenches derived sub-streams via `seed ^ HEX` bit-twiddling instead of ChaCha8's native 64-bit stream nonce (`ChaCha8Rng::set_stream`), while 13 production modules duplicated manual `(rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)` conversions and rejection-sampling loops.
3. **32D/32-bit Sobol ceiling & unscrambled high-dimensional Halton collinearity:** `pure_math::statistics::qmc` capped `SobolSequence` at 32 dimensions and 32-bit direction precision, lacked a base-2 Niederreiter `(t, s)`-sequence, and emitted unscrambled radical-inverse coordinates in higher Halton dimensions (`p_d >= 293`), causing severe diagonal collinearity ($r > 0.99$).
4. **Embedded `.rodata` budget:** Zero-allocation `#![no_std]` callers require borrowed-slice QMC views (`&mut [u64]`, `&mut [f64]`) backed by compact static direction/permutation tables strictly below `16 KiB`.

## Decision

1. **Create Tier 0 `#![no_std]` crate `crates/00_foundation/math_rng`:**
   - Remove `features = ["std"]` from `[workspace.dependencies].rand_chacha` so `math_rng` compiles cleanly with `--no-default-features` on host and `wasm32-unknown-unknown` with optional `alloc` and `serde` features and zero OS entropy calls.
   - Provide `OxidizeRng` (`BASE_SEED = 0x0001_D1CE_CAFE_BABE`) with `SeedableRng`, `with_stream`, `split_stream`, `set_stream`, `stream`, `jump_words`, `set_word_pos`, and `word_pos`.
   - Provide generic `RngScalar` (`f32`/`f64`) samplers (`sample_closed_open`, `sample_open`, `sample_symmetric_open`, `sample_standard_normal`, `sample_standard_normal_f64`), unbiased rejection-sampled `sample_bounded_u64` / `sample_bounded_usize`, and `fill_cranley_patterson_shifts`.

2. **64D 53-bit Borrowed-Slice QMC Views (`< 16 KiB` `.rodata`):**
   - Implement `SobolView<'a>`, `HaltonView<'a>`, and `NiederreiterView<'a>` operating over caller-supplied slices with `O(d)` Gray-code stepping, `seek`, stateless 53-bit nested hash Owen scrambling (`owen_scramble_53`), and Cranley-Patterson shifts (`~14.2 KiB` combined `.rodata` across 64D Joe-Kuo polynomials/initial numbers, 64D base-2 Niederreiter generator matrices, and 64D primes + packed Faure permutations).
   - Provide `alloc`-gated `SobolSequence`, `HaltonSequence`, and `NiederreiterSequence` wrappers.

3. **Workspace Re-Exports, QMC Delegation, and Call-Site Cleanup:**
   - Re-export `math_rng::*` (including `RngCore`, `SeedableRng`, and `CryptoRng`) in `oxidize_core::rng`, drop the redundant direct `rand` dependency from `crates/oxidize_core/Cargo.toml` and `crates/markdown_tests/Cargo.toml`, and surface `BASE_SEED`, `OxidizeRng`, `RngScalar`, `SobolView`, `HaltonView`, and `NiederreiterView` in `math_explorer::prelude`.
   - Link `math_rng` directly into `math_commons`, `engine_diffeq`, `engine_pde`, `oxidize_core`, `pure_math`, and `markdown_tests`.
   - Upgrade `pure_math::statistics::qmc`, `MonteCarloWorkspace` (`estimate_qmc_niederreiter`), and `MultivariateIntegrator` (`integrate_qmc_niederreiter`) to 64D/53-bit precision, and replace all `seed ^ HEX` hacks and manual 53-bit float / bounded-index conversions across `crates/` and `math_explorer_gui/`.

## Consequences

- **Positive:** `#![no_std]` engines (`math_commons`, `engine_diffeq`, `engine_pde`) and `std` crates share a single deterministic `OxidizeRng` and sampler kernel with zero dynamic heap allocations under `VerifiedAllocator`.
- **Positive:** Sobol and Niederreiter QMC scale to 64 dimensions at full 53-bit IEEE-754 mantissa resolution inside `< 16 KiB` `.rodata`, while Faure + hash-Owen scrambled Halton eliminates high-dimensional pair collinearity.
- **Neutral:** Workspace member count advances from `23` to `24` crates while preserving full `oxidize_core::rng` source compatibility.
