# ADR 20261123: Unify Rust 2024 Edition, Declarative `ExplorerTab` Contracts, and Dynamic Partition Invariants

Date: 2026-11-23

## Status
Accepted (partially supersedes [ADR 0027](0027-build-rerun-directives-and-cargo-cache.md); complements [ADR 20261123](20261123-inventory-tabs-and-scoped-build-scripts.md), [ADR 0040](0040-staged-index-snapshot-and-graph-scoped-gates.md), and [ADR 0044](0044-unify-build-time-traceability-engine.md))

## Context
1. **Split Workspace Rust Editions**: While 18 workspace crates targeted `edition = "2024"` (including [`crates/markdown_tests/Cargo.toml`](../../crates/markdown_tests/Cargo.toml) in [ADR 20261123](20261123-inventory-tabs-and-scoped-build-scripts.md)), [`math_explorer_gui/Cargo.toml`](../../math_explorer_gui/Cargo.toml) and [`crates/federated_registry/Cargo.toml`](../../crates/federated_registry/Cargo.toml) remained on `edition = "2021"`, forcing [`apps/xtask/src/staged.rs`](../../apps/xtask/src/staged.rs) to bucket staged `rustfmt` invocations by crate edition.
2. **Centralized Tab Match Tables, Adapter Boilerplate, and Implicit URL Slug Fallback**: Even after [ADR 20261123](20261123-inventory-tabs-and-scoped-build-scripts.md) introduced static [`TabDescriptor`](../../math_explorer_gui/src/tabs/mod.rs) inventory registration, twenty-four `SimulationFramework` domain tabs in `math_explorer_gui/src/tabs/*/mod.rs` duplicated identical `Default`, `ExplorerTab`, and `inventory::submit!` adapter blocks while [`math_explorer_gui/src/tabs/mod.rs`](../../math_explorer_gui/src/tabs/mod.rs) centralized hardcoded `academic_domain()` and `url_slug()` match tables with `"Applied Math"` and `"simulation"` fallbacks.
3. **Brittle Global Count Literals in Contract Suites**: Nine GUI test suites asserted hardcoded global counts for tabs, registered tools, starter presets, export capability buckets, DragValue widgets, and inverse challenges, causing unrelated domain PRs to collide on test literal bumps.

## Decision
1. **Unify All 20 Workspace Crates on `edition = "2024"`**:
   - Upgrade [`math_explorer_gui/Cargo.toml`](../../math_explorer_gui/Cargo.toml) and [`crates/federated_registry/Cargo.toml`](../../crates/federated_registry/Cargo.toml) (alongside [`crates/markdown_tests/Cargo.toml`](../../crates/markdown_tests/Cargo.toml)) to `edition = "2024"`, resolve all Rust 2024 edition and Clippy diagnostics on host and `wasm32-unknown-unknown`, and simplify [`apps/xtask/src/staged.rs`](../../apps/xtask/src/staged.rs) to run a single `rustfmt --edition 2024` pass across staged Rust files.
2. **Declarative `declare_simulation_tab!` Macro and Required `ExplorerTab` Metadata**:
   - Make `ExplorerTab::academic_domain(&self)` and `ExplorerTab::url_slug(&self)` required trait methods in [`math_explorer_gui/src/tabs/mod.rs`](../../math_explorer_gui/src/tabs/mod.rs), removing the centralized `match self.name()` lookup tables and `"Applied Math"` / `"simulation"` fallbacks.
   - Add `#[macro_export] macro_rules! declare_simulation_tab` in [`math_explorer_gui/src/tabs/mod.rs`](../../math_explorer_gui/src/tabs/mod.rs) generating the `SimulationFramework`-backed struct, `Default`, `ExplorerTab`, and static `inventory::submit!(TabDescriptor { id, order, build })` registration in one declaration, and migrate all 24 `SimulationFramework` domain tabs onto it while implementing `academic_domain` and `url_slug` directly on the 4 custom tabs (`FavoritismTab`, `MorphogenesisTab`, `MriTab`, `TraceabilityTab`).
3. **Dynamic Contract Partition Invariants and `mtime` Regression Coverage**:
   - Replace hardcoded global count literals across `starter_gallery/{tests,contract_tests}.rs`, `url_state/contract_tests.rs`, `framework/{deferred_export_audit_tests,drag_bounds_and_rollback_tests}.rs`, `inverse_challenges_tests.rs`, and `app/{tests_tab_cache_perf,tests_shortcuts/sweep_and_undo,tests_domain_bar,tests_secondary_menus}.rs` with dynamic non-emptiness, uniqueness, and exhaustive partition invariants over `instantiate_tabs()` and `inventory::iter`, and verify `write_if_changed` `mtime` preservation in `math_explorer_gui/src/tabs/tests.rs`.

## Consequences
- **Positive**: All 20 workspace crates compile and format uniformly under Rust 2024 on both desktop and `wasm32-unknown-unknown`.
- **Positive**: Adding a new `ExplorerTab`, `InteractiveTool`, or `StarterPresetMetadata` requires zero central match-table edits in `src/tabs/mod.rs` and zero hardcoded global count bumps across unrelated contract test suites, while preserving `rustfmt` visibility and feature-scoped build scripts from [ADR 20261123](20261123-inventory-tabs-and-scoped-build-scripts.md).
- **Neutral**: Single-binary layout (`math_explorer_gui/src/main.rs`), `traceability_manifest.json` schema parity, and [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json) remain unchanged.
