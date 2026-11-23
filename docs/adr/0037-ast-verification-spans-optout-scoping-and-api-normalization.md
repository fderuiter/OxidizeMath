# ADR 0037: AST Verification Spans, Opt-Out Scoping, and API Signature Normalization

## Status

Accepted

## Context

Static AST verification gates across `oxidize_core` and `unified_verification` exhibited four correctness and diagnostic gaps:

1. **Opt-out leakage and substring false positives in `oxidize_core`**: `ModuleVerificationVisitor` stored a single boolean `opted_out` flag across both crate/file inner `#![...]` attributes and item-level outer `#[...]` attributes without resetting after visiting an item. Consequently, placing `#[cfg_attr(any(), verified(opt_out = "gui_tool"))]` on a single struct suppressed verification checks for all subsequent functions in the same file. In addition, `theory_verification!` vacuous-bypass detection used `tokens.contains("0.0")` substring matching across `stochastic_test` token strings, falsely flagging legitimate tolerances and bounds such as `0.05` or `10.0`.
2. **Missing `file:line` spans, item names, per-file densities, and remediation guidance**: `CloneDetectorVisitor`, `DeadCodeVisitor`, `ClippyBypassVisitor`, and `TraceabilityEngine` emitted bare file paths or module names without `syn::spanned::Spanned` line numbers, enclosing item/function names, per-file assertion densities, or actionable fix hints (`flatten_2d_index`, `colormap`, `#[cfg(test)]`, `verification_whitelist.toml`, `traceability.toml`, and `opt_out`).
3. **Inverted `check-api-drift` diff headers and missing CLI/ADR instructions**: `compare_file_items` printed `+` and `-` signature diff lines before the `API Drift detected in <path>:` header, and failure output omitted explicit `cargo run -p xtask -- regenerate-baseline` and `docs/adr/` guidance.
4. **Implementation leakage in `ApiVisitor` signatures**: `ApiVisitor` serialized private parameter binding names (`x: f64` vs `value: f64`, `mut self` vs `self`, destructured tuple patterns, and `fn(arg: T)` parameter labels) as well as `const` and `static` right-hand initializer expressions (`= #expr`) into `crates/unified_verification/api_baseline.json`, causing non-breaking internal refactorings to fail `check-api-drift`.

## Decision

1. **Scope opt-outs accurately and parse `stochastic_test` AST initializers in `oxidize_core`**:
   - Separate `file_opted_out` (set only by `syn::AttrStyle::Inner(_)`) from item-scoped opt-outs in `ModuleVerificationVisitor`, evaluating item opt-outs per `ItemFn` / `ImplItemFn` without leaking state to sibling items.
   - Standardize file-wide GUI opt-outs onto inner `#![cfg_attr(any(), verified(opt_out = "gui_tool"))]` attributes in `math_explorer_gui` (`lattice_boltzmann.rs`, `hodgkin_huxley.rs`, and `morphogenesis/tool.rs`).
   - Parse `stochastic_test = { ... }` blocks with `syn` to flag `let` bindings initialized to numeric zero (`0`, `0.0`, `-0.0`) rather than matching `"0.0"` substrings, allowing `0.05` and `10.0` bounds to pass.
2. **Emit `file:line` spans, enclosing identifiers, per-file densities, and hints across AST gates**:
   - Record `proc_macro2` / `syn::spanned::Spanned` 1-based line numbers across `CloneDetectorVisitor`, `DeadCodeVisitor`, `ClippyBypassVisitor`, and `TraceabilityReport` (with `#[serde(default)]` backward compatibility for `traceability_manifest.json`), falling back cleanly to `path` without `:0` when macro-generated spans report line `0`.
   - Include enclosing function/item names, unlinked `theory_verification!` module keys, per-file verified assertion densities, and targeted remediation hints in gate diagnostics.
3. **Order `check-api-drift` headers first and normalize `ApiVisitor` signatures**:
   - Emit `API Drift detected in <path>:` prior to `  + ` and `  - ` diff lines and print `cargo run -p xtask -- regenerate-baseline` plus `docs/adr/` instructions on drift failure.
   - Normalize `FnArg::Typed` parameter patterns to `_` (preserving `self`, `&self`, `&mut self`, tuple types, references, and generic bounds), strip `BareFnArg` parameter names, omit `= #expr` initializers from `const` and `static` signatures while preserving `facade.rs` AST signature type validation, and regenerate `crates/unified_verification/api_baseline.json`.

## Consequences

- Item-level `opt_out` attributes only exempt the annotated item, while file-wide GUI opt-outs explicitly use inner `#![...]` syntax.
- `stochastic_test` tolerances containing `"0.0"` substrings (`0.05`, `10.0`) no longer trigger false-positive vacuous bypass errors, whereas true zero initializers (`0.0`, `0`) remain blocked.
- Developers and CI logs receive exact `file:line` locations, item names, per-file assertion densities, and copy-pasteable remediation guidance across clone detection, policy audits, traceability checks, and API drift reports.
- Internal parameter renames and constant/static expression adjustments no longer churn `api_baseline.json`, while type, receiver-reference, re-export, and bound changes continue to fail `check-api-drift`.
