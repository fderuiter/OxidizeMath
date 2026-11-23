# ADR 0036: Delete Dead Theory Accessors, Description Macro, and `derive(Theory)` Cast Codegen

## Status
Accepted (extends [ADR 0019](0019-purge-dead-code-bypasses.md), [ADR 0022](0022-cleanup-verification-macros-dead-code.md), [ADR 0027](0027-enforce-workspace-clippy-lints.md), and [ADR 0034](0034-harden-verification-macros-and-facade-reexports.md))

## Context
1. **Uncalled Reflective Parameter Mutators and Wrapper Macro**: Following removal of the legacy reflective parameter UI subsystem, [`crates/scientific_metadata/src/theory.rs`](../../crates/scientific_metadata/src/theory.rs) still exported `TheoryDescribable::get_parameter`, `TheoryDescribable::set_parameter`, and `theory_description!` with zero production callers across all 20 workspace crates. Interactive UI presets and recovery instead clamp serialized state via `TheoryDescribable::theory_parameters` and `serde_json` per [ADR 0022](0022-unify-preset-state-serialization.md) in [`math_explorer_gui/src/framework/clamp.rs`](../../math_explorer_gui/src/framework/clamp.rs) and [`math_explorer_gui/src/framework/state_guard/rollback.rs`](../../math_explorer_gui/src/framework/state_guard/rollback.rs).
2. **Dead Macro Codegen and Cast Lint Suppressions**: [`crates/verified_engine_macros/src/theory.rs`](../../crates/verified_engine_macros/src/theory.rs) emitted `get_parameter` and `set_parameter` methods containing `as f64` and `as #field_ty` casts wrapped in `#[allow(clippy::unnecessary_cast)]` across all 13 `#[derive(Theory)]` domain models, solely retaining the casts as an indirect compile-time check against `#[theory(min, max, step)]` annotations on non-numeric struct fields.

## Decision
1. **Delete Dead Metadata Contracts**:
   - Remove `TheoryDescribable::get_parameter`, `TheoryDescribable::set_parameter`, `theory_description!`, and `pub use crate::theory_description` from [`crates/scientific_metadata/src/theory.rs`](../../crates/scientific_metadata/src/theory.rs) while preserving `theory_description`, `phonetic_description`, `theory_citation`, `available_descriptions`, `theory_parameters`, `theory_latex_equation`, `ParameterConstraint`, and facade re-exports in [`crates/verified_engine/src/lib.rs`](../../crates/verified_engine/src/lib.rs) and [`math_explorer/src/lib.rs`](../../math_explorer/src/lib.rs).
2. **Replace Cast Codegen with Zero-Cast Spanned Numeric Trait Bounds**:
   - Remove `get_parameter`, `set_parameter`, `as` casts, and `#[allow(clippy::unnecessary_cast)]` from `#[derive(Theory)]` in [`crates/verified_engine_macros/src/theory.rs`](../../crates/verified_engine_macros/src/theory.rs).
   - When bounded fields are present, emit a local `TheoryNumeric` trait implemented for primitive signed integers (`i8`..`i128`, `isize`), unsigned integers (`u8`..`u128`, `usize`), and floats (`f32`, `f64`), checked via `assert_theory_numeric::<#field_ty>(&self.#field_ident)` spanned to `field.ty.span()`.
3. **Align Tests, Documentation, and API Baseline**:
   - Update [`crates/verified_engine_macros/tests/compile_diagnostics.rs`](../../crates/verified_engine_macros/tests/compile_diagnostics.rs), [`crates/verified_engine/tests/macro_integration.rs`](../../crates/verified_engine/tests/macro_integration.rs), [`math_explorer/tests/facade_macros.rs`](../../math_explorer/tests/facade_macros.rs), [`crates/scientific_metadata/README.md`](../../crates/scientific_metadata/README.md), and [`crates/verified_engine_macros/README.md`](../../crates/verified_engine_macros/README.md) to assert `theory_parameters` and the `TheoryNumeric` diagnostic at exact field-type spans.
   - Regenerate [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) to remove the deleted signatures.

## Consequences
- **Positive**: Shrinks `TheoryDescribable` and `#[derive(Theory)]` expansions across all 13 production models while eliminating generated `as` casts and `#[allow(clippy::unnecessary_cast)]` suppressions.
- **Positive**: Preserves exact `field.ty.span()` `rustc` diagnostics when `#[theory(min, max, step)]` is applied to non-numeric fields without panics or lossy float-to-integer casts.
- **Neutral**: Regenerates [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) for `crates/scientific_metadata/src/theory.rs` while keeping all domain models and `serde_json` state-clamping pipelines unchanged.
