# ADR 0027: Enforce Workspace Clippy Lints and Eliminate Crate-Level Bypasses

## Status
Accepted

## Context
Although the root `Cargo.toml` defines `[workspace.lints.rust]` and `[workspace.lints.clippy]` (`too_many_lines = "deny"` at the 60-line `.clippy.toml` ceiling and `cognitive_complexity = "deny"`), three gaps weakened workspace-wide enforcement:
1. `crates/unified_verification/src/main.rs` suppressed Clippy across the verification engine itself with `#![allow(clippy::too_many_lines, clippy::cognitive_complexity)]` and `#![allow(clippy::all)]`.
2. `crates/scientific_metadata/Cargo.toml` omitted `[lints] workspace = true` (relying on a local `#![warn(missing_docs)]` attribute), and `crates/pure_math/src/pure_math/number_theory/ambs.rs` suppressed `clippy::too_many_lines` at both file and function scope (`AmbsDsp::tonelli_shanks`).
3. Neither `check-profiles` nor `test-policy-audit` verified that first-party `Cargo.toml` manifests inherit workspace lints or blocked crate/module-level Clippy bypass attributes (`clippy::all`, `clippy::too_many_lines`, `clippy::cognitive_complexity`).

## Decision
1. Removed `#![allow(clippy::too_many_lines, clippy::cognitive_complexity)]` and `#![allow(clippy::all)]` from `crates/unified_verification/src/main.rs`, refactored oversized functions in `main.rs`, `policy_audit.rs`, `profile.rs`, `report.rs`, and `workflow_linter/tests.rs` below 60 lines (reusing the shared `crate::whitelist`, `collect_rs_files_filtered`, and `test_utils` helpers), and resolved all `clippy::all` lints across `unified_verification`.
2. Added `[lints] workspace = true` to `crates/scientific_metadata/Cargo.toml` and removed redundant `#![warn(missing_docs)]` from `crates/scientific_metadata/src/lib.rs`.
3. Removed `#![allow(clippy::too_many_lines)]` and `#[allow(clippy::too_many_lines)]` from `crates/pure_math/src/pure_math/number_theory/ambs.rs`, decomposing `AmbsDsp::tonelli_shanks` (with explicit `p = 2` short-circuiting prior to odd-prime factoring/refinement) and `AmbsDsp::search` into private `#[verified_engine::verified]` helpers under 60 lines with `debug_assert!` preconditions while preserving `AmbsDsp::tonelli_shanks`'s exact public signature.
4. Added `crates/unified_verification/src/policy_audit/lint_policy.rs` and wired it into both `policy_audit` (`test-policy-audit`) and `profile` (`check-profiles`):
   - First-party member `Cargo.toml` manifests must declare:
     ```toml
     [lints]
     workspace = true
     ```
   - Vendored `egui_plot/Cargo.toml` and `egui_plot/src/lib.rs` remain exempt only when `egui_plot/Cargo.toml` defines explicit `[lints.rust]` and `[lints.clippy]` tables.
   - `ClippyBypassVisitor` (kept separate from `DeadCodeVisitor` per ADR 0019) rejects inner `#![allow]` / `#![expect]` attributes and `mod` attributes (including `cfg_attr`) for `clippy::all`, `clippy::too_many_lines`, or `clippy::cognitive_complexity`, as well as item-level `clippy::all`, while preserving targeted function-level `#[allow(clippy::too_many_lines)]` where needed.

## Consequences
- Every first-party crate uniformly inherits `[workspace.lints]`, including `unified_verification` and `scientific_metadata`.
- Crate-wide and module-wide Clippy suppressions are blocked statically by both `check-profiles` and `test-policy-audit` in pre-commit and CI gates.
- Public mathematical APIs and baseline signatures (`crates/unified_verification/api_baseline.json`) remain unchanged.
