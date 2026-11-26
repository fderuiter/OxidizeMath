# ADR 20261125: Metadata-Driven Path Abstraction and Five-Tier Workspace Migration

## Status

Accepted

## Context

Previously, the OxidizeMath Cargo workspace placed only `crates/02_engines/engine_linalg` in a tiered directory while leaving 18 crates in flat `crates/<name>` paths and 3 crates at the repository root (`math_explorer`, `math_explorer_gui`, and `egui_plot`). Relocating crates into architectural tiers or adding new `#![no_std]` engines was blocked by five coupled constraints:

1. Hardcoded directory path arrays in `profile.rs`, `oxidize_core/src/traceability/runner.rs`, `staged_traceability.rs`, `api_drift`, and `doc_audit`.
2. Fixed relative parent depths (`../..` and cross-crate `#[path = "..."]` includes) across `markdown_tests/build.rs`, `oxidize_core/src/traceability.rs`, `apps/xtask/src/gates.rs`, `policy_audit/doctests.rs`, and `facade_hygiene.rs`.
3. Path-keyed entries (`crates/<name>/src/...`) in `api_baseline.json` and `verification_whitelist.toml` that caused hundreds of lines of churn whenever a crate directory moved.
4. Verification blind spots for Tier 2 `#![no_std]` engines (`engine_linalg`, `engine_diffeq`, and `engine_pde`) in facade signature/error gates, `markdown_tests` library coverage, and `math_explorer_gui` transitive snapshot `cargo:rerun-if-changed` directives.
5. Conflation of the `default = []` library facade with 32 domain `[[example]]` targets inside `math_explorer`.

## Decision

1. **Declarative `[package.metadata.oxidize]` Contract**: Every first-party workspace member declares `tier` (`00_foundation`, `01_core`, `02_engines`, `03_domains`, `04_bindings`, `apps`, or `vendored`), `release_role`, `facade_role`, `traceability`, and optional `facade_module`, `traceability_subdir`, and `math_contract` fields parsed by `oxidize_core::workspace` and `unified_verification::doc_audit::manifest_model`.
2. **Metadata Queries & Ancestor Root Discovery**: Verification, traceability, doctest, and `xtask` gates resolve crates by package name and metadata role while locating the workspace root by walking ancestors for `[workspace]`. Shared `oxidize_core` and `traceability.toml` artifacts are staged into `OUT_DIR` via `build.rs` so `apps/xtask` remains dependency-free without cross-crate `#[path]` attributes.
3. **Package-Keyed Baselines & Whitelists**: `api_baseline.json` and crate-scoped entries in `verification_whitelist.toml` key files as `<package_name>:<crate_rel_path>`, eliminating baseline and whitelist churn across directory moves.
4. **Tier 2 Engine Coverage & Transitive GUI Snapshot Graph**: `oxidizemath` re-exports `engine_linalg`, `engine_diffeq`, and `engine_pde` alongside `LinalgError`, `DiffeqError`, and `PdeError` in `error.rs`; `pure_math` consumes `engine_pde` directly without manual shims; `policy_audit` enforces `markdown_tests` dev-dependency coverage across all non-vendored library crates; and `compute_gui_snapshot_code_dirs` walks Cargo's transitive feature graph so enabling `ai`, `applied`, `biology`, or `physics` watches `engine_diffeq` and `engine_pde`.
5. **Acyclic Five-Tier Layout & Facade/CLI Split**: `audit_tier_dependencies` enforces `tier_rank(dep) <= tier_rank(pkg)` across all first-party manifests. All library and application crates now reside under `crates/00_foundation` through `crates/04_bindings` and `apps/`, with `crates/04_bindings/oxidizemath` serving as the `default = []` primary facade, `crates/04_bindings/math_explorer` retained as a thin compatibility re-export crate, and `apps/oxidize-cli` owning all 32 domain CLI examples.

## Consequences

- **Positive**: Maintainers can relocate crates or introduce new `#![no_std]` engines and domain crates by updating `Cargo.toml` metadata without touching gate source lists, `api_baseline.json`, or `verification_whitelist.toml`.
- **Positive**: Downstream library consumers depend on `oxidizemath` (`default = []`) without compiling CLI targets, while existing callers importing `math_explorer` retain full source and feature compatibility.
- **Trade-off**: Every new first-party crate in `[workspace].members` must include a valid `[package.metadata.oxidize]` table and satisfy the acyclic tier rank invariant enforced in `check-profiles`.
