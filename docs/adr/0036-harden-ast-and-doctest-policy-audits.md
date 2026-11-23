# ADR 0036: Harden AST and Doctest Policy Audits in `unified_verification`

## Status
Accepted (extends [ADR 0019](0019-purge-dead-code-bypasses.md), [ADR 0027](0027-enforce-workspace-clippy-lints.md), and [ADR 0034](0034-scoped-pre-commit-and-single-parse-ast-verification.md))

## Context
1. **Hardcoded Dependency Exemptions & Skipped Tables**: `test-policy-audit` in [`crates/unified_verification/src/policy_audit.rs`](../../crates/unified_verification/src/policy_audit.rs) previously read only `[dependencies]` and `[dev-dependencies]`, skipped `[build-dependencies]`, all 14 `[target.*]` dependency tables, and [`crates/markdown_tests/Cargo.toml`](../../crates/markdown_tests/Cargo.toml), and unconditionally stripped 14 crate names (`thiserror`, `serde`, `approx`, `wasm_bindgen`, `console_error_panic_hook`, `proc_macro2`, `quote`, `syn`, `serde_json`, `document_features`, `eframe`, `math_explorer`, `math_commons`, and `scientific_metadata`). As a result, 9 unused `[dev-dependencies]` (`approx` across 6 crates, `serde_json` in `math_explorer`, and `eframe`/`egui` in `crates/markdown_tests`) plus a redundant `wasm32` `getrandom` entry in [`math_explorer_gui/Cargo.toml`](../../math_explorer_gui/Cargo.toml) accumulated undetected.
2. **Attribute, Derive, and `rustdoc` Fence Blind Spots**: `DependencyVisitor` did not inspect attribute arguments or derive macro expansions (`#[derive(Serialize, Deserialize)]`, `#[derive(Error)]`, `#![cfg_attr(..., doc = document_features::document_features!())]`, `#[wasm_bindgen]`), nor did it parse compiled `rustdoc` code blocks inside `///` / `//!` doc comments, `#![doc = include_str!("../README.md")]` files, or workspace `.md` files compiled by `crates/markdown_tests`. In addition, staging a `.md` file during `lint-staged` did not re-run `policy_audit` on the owning crate or `crates/markdown_tests`.
3. **Clippy Group, `impl`/`trait`, Macro, and Stale `too_many_lines` Bypasses**: `ClippyBypassVisitor` in [`crates/unified_verification/src/policy_audit/lint_policy.rs`](../../crates/unified_verification/src/policy_audit/lint_policy.rs) checked only `clippy::all`, `too_many_lines`, and `cognitive_complexity` on inner and `mod` attributes. It missed the 9 Clippy lint groups (`pedantic`, `nursery`, `restriction`, `complexity`, `correctness`, `suspicious`, `style`, `perf`, `cargo`), `impl` and `trait` blocks, `macro_rules!` token streams (allowing [`crates/pure_math/src/theory_macro.rs`](../../crates/pure_math/src/theory_macro.rs) to inject `#[allow(unused_imports, clippy::all)]` and unconditional `use approx::assert_relative_eq;` into 30 test submodules), and 5 stale `too_many_lines` attributes on functions of 54 lines or fewer.

## Decision
1. **Full Cargo Table, Attribute/Derive, and `rustdoc` Fence Auditing**:
   - Extract [`crates/unified_verification/src/policy_audit/dependencies.rs`](../../crates/unified_verification/src/policy_audit/dependencies.rs) and [`crates/unified_verification/src/policy_audit/doctests.rs`](../../crates/unified_verification/src/policy_audit/doctests.rs) so every module remains well below 500 lines and every function within 60 lines while preserving the single-parse `AstCache` invariant on `.rs` files.
   - Collect dependencies across `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`, and all `[target.*]` tables across all 20 workspace members without hardcoded crate exemptions.
   - Extend `DependencyVisitor` to inspect attribute paths/tokens (while ignoring `cfg` predicates and compiler lint lists) and map `Serialize`/`Deserialize` to `serde` and `Error` to `thiserror`.
   - Parse compiled `rustdoc` code blocks from `///` / `//!` comments, `#[doc = include_str!(...)]` markdown files, and workspace `.md` files (enabling direct auditing of `crates/markdown_tests`), and update `select_touched_policy_members` in [`crates/unified_verification/src/batch.rs`](../../crates/unified_verification/src/batch.rs) so staged `.md` changes re-audit both the owning crate and `crates/markdown_tests`.
2. **Strict `[[unused_dependencies]]` Whitelist Schema**:
   - Add `UnusedDependencyEntry` (`#[serde(deny_unknown_fields)]`) to [`crates/unified_verification/src/whitelist.rs`](../../crates/unified_verification/src/whitelist.rs) requiring non-empty `crate`, `dependency`, and `reason` fields:
     ```toml
     [[unused_dependencies]]
     crate = "crates/oxidize_core"
     dependency = "getrandom"
     reason = "Activates the getrandom js backend feature on wasm32-unknown-unknown for rand without direct Rust imports"
     ```
3. **Expanded Clippy Bypass Enforcement**:
   - Extend `ClippyBypassVisitor` to reject all 9 Clippy groups (`pedantic`, `nursery`, `restriction`, `complexity`, `correctness`, `suspicious`, `style`, `perf`, `cargo`) alongside `all`, reject `too_many_lines` and `cognitive_complexity` on `impl` and `trait` blocks, scan `macro_rules!` token streams for Clippy suppressions, and flag `too_many_lines` on functions of 60 lines or fewer (per [`.clippy.toml`](../../.clippy.toml)).
4. **Workspace Dependency, Macro, and Attribute Cleanup**:
   - Remove `clippy::all` and unconditional `use approx::assert_relative_eq;` from [`crates/pure_math/src/theory_macro.rs`](../../crates/pure_math/src/theory_macro.rs) and import `approx::assert_relative_eq` explicitly at the 25 `domain_applied` and `domain_physics` `theory_verification!` call sites.
   - Remove the 9 unused `[dev-dependencies]`, the redundant `wasm32` `getrandom` entry in `math_explorer_gui/Cargo.toml`, obsolete `[package.metadata.cargo-machete]` sections (while retaining `bnjbvr/cargo-machete` in CI), and the 5 stale `too_many_lines` attributes.

## Consequences
- **Positive**: `test-policy-audit`, `batch-static`, and `lint-staged` enforce zero unused dependencies across all Cargo dependency tables and zero Clippy group, `impl`/`trait`, macro, or short-function bypasses across all first-party crates while preserving legitimate doctest/integration-test consumers (`thiserror` in `diagnostics` and `approx` in `verified_engine` and `math_explorer`).
- **Neutral**: Public mathematical API signatures in [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) remain unchanged.
