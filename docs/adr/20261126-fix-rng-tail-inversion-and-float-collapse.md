# ADR 20261126: Fix `math_rng` Probit Tail Inversion, `ln` Truncation, and Open-Unit `f32` Collapse

## Status
Accepted (2026-11-26)

## Context

Bit-exact IEEE-754 verification of the Tier 0 `#![no_std]` scalar samplers introduced in [ADR 20261124](20261124-math-rng-static-qmc-views.md) and their downstream `math_commons` / `engine_diffeq` callers uncovered four numerical defects:

1. **Gaussian tail sign inversion (`acklam_probit_f64`):** `crates/00_foundation/math_rng/src/scalar.rs` multiplied Acklam's already-negative tail rational polynomial $C(q)/D(q)$ by `-1.0` on $p < 0.02425$ and `+1.0` on $p > 0.97575$, flipping the sign of `4.85%` of standard normal draws with `3.946-sigma` discontinuities at both tail boundaries.
2. **Logarithm series truncation & `1e-15` dyadic clamping (`ln_open_unit_f64`):** Truncating the odd-harmonic series on unreduced $m \in [1, 2)$ lost up to `9,269` ULPs near $u = 0.5^-$, while clamping `f64` open-unit draws to `[1e-15, 1 - 1e-15]` collapsed 18 lower dyadic bins.
3. **Upper-half `f32` midpoint collapse (`<f32 as RngScalar>::sample_open`):** Computing `((bits as f32) + 0.5) * 2^-24` in `binary32` rounded every odd 24-bit integer in $[2^{23}, 2^{24})$ to its even neighbor via IEEE-754 round-ties-to-even, collapsing `50%` of `[0.5, 1.0)` draws into `4,194,305` distinct values.
4. **Narrowing `1.0_f32` leak into Gillespie SSA (`sample_open_unit<f32>`):** Casting `sample_open::<f64, _>` via `T::from_f64` in `crates/02_engines/math_commons/src/math_kernel/types.rs` rounded $2^{28} + 1$ upper 53-bit `f64` draws to `1.0_f32`, causing `ssa_sample_event<f32>` in `crates/02_engines/engine_diffeq/src/jump.rs` to evaluate `-(1.0_f32.ln()) / total == -0.0_f32`.

## Decision

1. **Wichura AS 241 (`PPND16`) Probit & 1-ULP `#![no_std]` Logarithm (`math_rng::scalar`):**
   - Fold mantissas $m \ge \sqrt{2}$ via `m *= 0.5; exp += 1` so $m \in [1/\sqrt{2}, \sqrt{2})$, evaluate exact Sterbenz $f = m - 1$ with minimax polynomial in $s = f / (2 + f)$ and split $\ln 2 = \text{LN2\_HI} + \text{LN2\_LO}$ to achieve $\le 1$-ULP precision across `(0, 1)`.
   - Replace the broken Acklam rational branch with Wichura's Algorithm AS 241 three-region degree-7 minimax rational polynomials (`|q| <= 0.425`, $s = \sqrt{-\ln r} \le 5.0$, and $s > 5.0$ for $r = \min(p, 1 - p)$) with proper tail signs (`-mag` for $p < 0.5$, `+mag` for $p > 0.5$).
2. **Exact IEEE-754 Open-Unit Endpoints & Bijective `2^24` `f32` Grid (`math_scalars` & `math_rng::scalar`):**
   - Set `F64_MIN_ABOVE_ZERO = 2^-54` and `F64_MAX_BELOW_ONE = 1.0 - 2^-53` in `math_scalars::traits::RealField::open_unit_from_f64`, and `F32_MIN_ABOVE_ZERO = 2^-25` in `math_scalars::ieee_f32`.
   - Map `bits == 0` to `2^-25_f32` (`0x3300_0000`) and `1..2^24` to `(bits as f32) * 2^-24` (`<= 0x3f7f_ffff`) in `<f32 as RealField>::uniform_open_from_u64` (backing `<f32 as RngScalar>::sample_open`), preserving all `16,777,216` strictly increasing `f32` draws while keeping `<f64 as RngScalar>::sample_open` and QMC digests unchanged.
3. **`TypeId`-Routed `sample_open_unit<f32>`, Strict SSA Positivity, and Doc Parity:**
   - Route `T = f32` in `math_commons::math_kernel::types::sample_open_unit` directly to `<f32 as RngScalar>::sample_open` via `core::any::TypeId`, enforce `debug_assert!(u > T::ZERO && u < T::ONE)`, and assert `tau > T::ZERO` in `engine_diffeq::jump::ssa_sample_event`.
   - Update stale Box-Muller docstrings in `crates/00_foundation/math_rng/README.md` and `apps/math_explorer_gui/src/tabs/ai/closed_loop_control/plants.rs` to inverse-CDF probit documentation.
4. **Cross-Platform Native & Coverage CI Determinism (`math_scalars::series`, `math_explorer_gui`):**
   - Route `T = f64` and `T = f32` in `<[T; 2] as Sample2D<T>>::advance_slice_plateau` and `<(T, T) as Sample2D<T>>::advance_slice_plateau` via `core::any::TypeId` to concrete IEEE-754 slice helpers so `[profile.test]` (`opt-level = 0` with `-C instrument-coverage`) avoids `600,000` uninlined generic `PartialEq`/`PartialOrd` trait calls across `200,000`-point `egui_plot` plateaus (`< 5ms` budget).
   - Deduplicate `logistic_lyapunov` evaluation in `BifurcationDiagram::recompute_default_coupled` and stride attractor scatter points into a preallocated `4,096`-point preview buffer while an animated slider or background sweep generation is in flight (restoring full-resolution borrowed slices once settled), keeping unoptimized/coverage-instrumented UI sweep frames well under the `16.67ms` frame budget.
   - Write GUI test PNG artifacts via `std::env::temp_dir()` in `sandpile::tests` and `knot_braid_studio::screenshot_tests` when `/tmp` is absent on Windows runners.

## Consequences

- **Positive:** Standard normal draws in `math_rng` and `closed_loop_control` are strictly monotonic to full `f64` precision across `[2^-54, 1 - 2^-53]` (`[-8.292, +8.210]`) with `0`-ULP `math_scalars::sqrt_f64` and zero heap allocations.
- **Positive:** `<f32 as RngScalar>::sample_open` and `sample_open_unit<f32>` emit `2^24` distinct values strictly inside `(0, 1)`, guaranteeing strictly positive Gillespie SSA waiting times `tau > 0.0_f32`.
- **Neutral:** Public signatures in `api_baseline.json`, `f64` Box-Muller SDE streams in `math_commons`, and the cross-target QMC digest (`0x59d1_6243_5f14_5673`) remain unchanged.
