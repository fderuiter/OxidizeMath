# ADR 0013: Disambiguate Theory Verification Macro Exports

## Status
Accepted

## Context
Both `pure_math` (in `crates/pure_math/src/theory_macro.rs`) and `scientific_metadata` (in `crates/scientific_metadata/src/theory.rs`) exported a macro named `theory_verification!`. In `pure_math`, `theory_verification!` binds unit tests to traceability registration checks against academic literature. In `scientific_metadata`, `theory_verification!` was an accessor macro for retrieving `theory_description()` on types implementing `TheoryDescribable`.

When workspace crates or downstream consumers imported both `pure_math` and `scientific_metadata`, the identical macro name caused compiler macro resolution ambiguity failures.

## Decision
1. Renamed `theory_verification!` to `theory_description!` in `crates/scientific_metadata/src/theory.rs` to accurately reflect its role in retrieving verified theory descriptions.
2. Introduced a deprecated `theory_verification!` macro alias in `crates/scientific_metadata/src/theory.rs` marked with `#[deprecated]` to maintain backwards compatibility for existing callers while advising migration to `theory_description!`.
3. Kept `theory_verification!` solely in `pure_math` as the primary verification macro for mathematical models and traceability registration.
4. Added unit tests in `crates/scientific_metadata/src/theory.rs` asserting proper macro expansion for `theory_description!` and the deprecated compatibility alias.

## Consequences
- **Positive:** Crates importing both `pure_math` and `scientific_metadata` resolve `theory_verification!` unambiguously to `pure_math`.
- **Positive:** Downstream consumers referencing `scientific_metadata::theory_verification!` continue to compile with an informative deprecation warning guiding them to `theory_description!`.
- **Positive:** Macro naming aligns directly with the `TheoryDescribable::theory_description` method signature.
