# ADR 20261123: Unify AST Verification Metrics, Ratcheted Debt Caps, and Read-Only Suite Outputs

## Status

Accepted (completes [ADR 0011](0011-ast-citation-binding-and-cli-check-enforcement.md), [ADR 0033](0033-wasm32-browser-tests-and-playwright-ci-smoke-gate.md), and [ADR 0040](0040-ast-verification-spans-optout-scoping-and-api-normalization.md); complements [ADR 20261123](20261123-unify-verify-suite-ast-cache-and-check-mode.md))

## Context

1. **Divergent Regex/Substring Metrics & Comment Spoofing**: `crates/unified_verification/src/report.rs` previously relied on regular expressions and substring checks (`#[verified`, `theory_verification!`, `stochastic_signature_verification!`) across raw source text. As a result, comment-only markers (such as `// theory_verification!` in `crates/oxidize_core/src/vfs.rs` and `math_explorer_gui/src/accessibility.rs`) marked files and crates as verified, any single `#[verified]` occurrence in a file marked every function in that file as verified, and conditional `#![cfg_attr(any(), verified(opt_out = "..."))]` attributes went undetected.
2. **Test Submodule Traversal & Macro Assertion Padding**: `crates/oxidize_core/src/traceability/module_tree.rs` traversed `#[cfg(test)]` outline submodules, counting `#[verified]` test functions alongside production code, while `crates/oxidize_core/src/ast_visitor.rs` padded `theory_verification!` and `stochastic_signature_verification!` macro counts via `.max(3)` even when `test = {}` was empty.
3. **Silent Debt Passes & Tracked Worktree Mutations**: `print_report` in `report.rs` printed unverified crate, file-length, and entropy debt items while still returning `passed = true`, and `save_report` unconditionally overwrote tracked `Verification_Certificate.json` and `README.md` during non-test runs without CI cleanliness enforcement.

## Decision

1. **Single `syn` AST Pipeline via `AstCache`**:
   - Replace regex and substring checks in `crates/unified_verification/src/report.rs` with `syn` AST visitors backed by `oxidize_core::traceability::AstCache` (`analyze_files_with_cache`, `collect_workspace_ast_metrics_with_cache`, `collect_ast_file_debt`, and `count_file_wasm_coverage_with_cache`).
   - Separate production (`prod_*`) and test (`test_*`) function and assertion counts while excluding `#![cfg(test)]` files, `#[cfg(test)]` outline modules, and dedicated test directories/files from `traceability` traversal and retaining `#[cfg(any(target_arch = "wasm32", test))]` items in `prod_*` counts.
   - Parse `module = "..."` and `test = { ... }` / `stochastic_test = { ... }` blocks in `theory_verification!` and `stochastic_signature_verification!` via `syn` token/block parsing and count actual assertion macro invocations with `AssertVisitor`, removing `.max(3)` padding.
   - Extract both direct `#[verified(opt_out = "...")]` and nested `cfg_attr(..., verified(opt_out = "..."))` attributes via `syn::Meta` while preserving inner `#![...]` vs item-scoped `#[...]` opt-out boundaries and rejecting dead `cfg_attr(any(), verified)` or test-only `cfg_attr(test, verified)` predicates.
2. **Ratcheted `[verification_debt]` Governance in `verification_whitelist.toml`**:
   - Add `[verification_debt]` (`exempt_crates`, `max_opt_outs_per_file`, `min_verified_density`, `min_total_density`, `min_unverified_density`, `opt_out_reasons`, `crate_opt_out_caps`, and `file_opt_out_caps`) alongside `[verify_suite]` in `crates/unified_verification/src/whitelist.rs` and `crates/oxidize_core/src/traceability/debt_floors.rs`.
   - Fail `verify-suite` (both default and `--coverage-only`), `check-certificate`, and `traceability` whenever unwhitelisted debt items, unauthorized crates, `opt_out` cap violations, or assertion density regressions occur.
3. **Read-Only Default `verify-suite` & Explicit CI Certificate Staging**:
   - Make `verify-suite` read-only (`--check`) by default, add `--output-cert <path>` to stage live `Verification_Certificate.json` artifacts (preserving all nine JSON keys validated by `deploy-pages`), and require `--write` / `--write-tracked` to update tracked `Verification_Certificate.json` and `README.md`.
   - Update `.github/workflows/ci.yml` to run `verify-suite --coverage-only --check --output-cert target/verification/Verification_Certificate.json`, enforce `git diff --exit-code` and clean `git status --porcelain`, and upload `target/verification/Verification_Certificate.json`.

## Consequences

- **Positive**: `verify-suite`, `check-certificate`, and `traceability` share one `syn` AST pipeline, comment/string markers can no longer spoof verification gates, and all `opt_out` attributes and assertion densities ratchet strictly via `[verify_suite]` and `[verification_debt]` in `verification_whitelist.toml`.
- **Positive**: Local `verify-suite` executions leave tracked repository files untouched by default while CI validates worktree cleanliness and stages live certificates for GitHub Pages deployment.
