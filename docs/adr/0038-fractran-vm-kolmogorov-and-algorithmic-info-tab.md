# ADR 0038: Zero-Allocation Prime-Exponent FRACTRAN VM, Kolmogorov Bounds, and Auto-Discovered `AlgorithmicInfoTab`

## Status
Accepted (extends [ADR 0004](0004-unified-verification-dependency-audit.md) and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/pure_math/src/pure_math/algorithmic_information`](../../crates/pure_math/src/pure_math/algorithmic_information/mod.rs) previously left three foundational components from [`papers/algorithmic_information_rust.tex`](../../papers/algorithmic_information_rust.tex) incomplete:
1. **Passive FRACTRAN Types**: [`fractran.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/fractran.rs) defined passive `FractranProgram` and `FractranState` structs without a stepping engine, halting predicate, cycle detector, or canonical Conway/Minsky constructors. Naive arbitrary-precision `BigInt` multiplication and division on every transition incurs heap churn incompatible with [`VerifiedAllocator`](../../crates/verified_engine/src/allocator.rs).
2. **Stubbed Complexity and Dyadic Geometry Primitives**: [`kolmogorov.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/kolmogorov.rs) unconditionally returned `0` from `kolmogorov_complexity` and omitted `prefix_kolmogorov_approx`, `k_floor_r`, Lempel-Ziv (`LZ76`), and Block Decomposition (`BDM`) upper bounds, while [`geometry.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/geometry.rs) omitted `Point2D`, `Line`, `DyadicPoint2D`, `distance`, and `project_point_onto_line`.
3. **Missing Interactive Explorer Surface**: `math_explorer_gui` had no `ExplorerTab` or `InteractiveTool` surfacing FRACTRAN Minsky registers or trajectory compressibility across desktop and `wasm32-unknown-unknown` builds.

## Decision
1. **Zero-Allocation Prime-Exponent FRACTRAN Virtual Machine**:
   - Pre-factor ordered positive fraction rules over the first 64 primes ($p_0 = 2 \dots p_{63} = 311$) into stack-inline signed delta vectors (`[i32; 64]`) inside [`FractranProgram`](../../crates/pure_math/src/pure_math/algorithmic_information/fractran.rs), preserving `FractranProgram::new(Vec<Rational>) -> Self` while adding validated `try_new`, `try_from_pairs`, and `parse` constructors.
   - Implement `#[verified_engine::verified]` `FractranProgram::step_exponents` and stack-only Floyd tortoise-and-hare `detect_cycle` so transitions execute in place with zero dynamic heap allocation under `VerifiedEngine::run_verified`, enforcing a `1_000_000` exponent cap via `u32::checked_add` and returning `Copy` `FractranError` diagnostics.
   - Provide canonical `primegame`, `adder`, `multiplier`, and `fibonacci` constructors alongside incremental Floyd cycle tracking in `FractranState`.
2. **Kolmogorov Estimators and Exact/Dyadic 2D Geometry**:
   - Implement `prefix_kolmogorov_approx`, rounded `kolmogorov_complexity`, dyadic point complexity `k_floor_r`, `lz76_phrases`, `lz76_complexity`, `bdm_complexity`, and `shannon_entropy_bits` in [`kolmogorov.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/kolmogorov.rs).
   - Implement `Point2D`, `Line`, `DyadicPoint2D`, `distance_squared`, `distance`, and `project_point_onto_line` in [`geometry.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/geometry.rs) using `dashu`-backed `Rational` arithmetic while preserving `DyadicRational::new(i64, u32) -> Self`.
3. **Auto-Discovered `AlgorithmicInfoTab` and `FractranMachineTool`**:
   - Add [`math_explorer_gui/src/tabs/algorithmic_information/mod.rs`](../../math_explorer_gui/src/tabs/algorithmic_information/mod.rs) annotated with `// @explorer_feature = "pure_math"` so [`math_explorer_gui/build.rs`](../../math_explorer_gui/build.rs) discovers `AlgorithmicInfoTab` (`academic_domain = "Pure Math"`, `url_slug = "algorithmic_information"`) without modifying `math_explorer_gui/src/tabs/mod.rs`.
   - Split [`FractranMachineTool`](../../math_explorer_gui/src/tabs/algorithmic_information/state.rs) into focused submodules (`state`, `controls`, `plots`, `presets`) using `PlaybackTransport::in_thread`, capping live playback at `256` steps per frame, auto-pausing upon Floyd cycle detection, and validating URL/JSON presets through `ToolStateBounds` and `FiniteStateGuard`.

## Consequences
- **Positive**: FRACTRAN register stepping and cycle detection run in bounded stack memory with zero `BigInt` heap churn under `VerifiedAllocator`, enabling smooth real-time visualization of Conway's `PRIMEGAME` and Minsky register programs on both desktop and `wasm32`.
- **Positive**: Existing public signatures in `pure_math::algorithmic_information` remain backward compatible while `math_explorer` facade re-exports, `#tab=algorithmic_information` share links, starter gallery cards, and figure/dataset exports work end-to-end.
- **Neutral**: Prime registers are intentionally bounded to the first 64 primes ($p \le 311$) and exponent `1_000_000`, returning explicit `FractranError` values for out-of-range inputs rather than silently allocating unbounded heap vectors.
