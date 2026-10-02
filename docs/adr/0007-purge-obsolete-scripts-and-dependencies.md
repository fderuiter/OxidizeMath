# 7. Purge Obsolete Root Scripts, Migrator Crate, and Unused Workspace Dependencies

Date: 2026-09-30

## Status

Accepted

## Context

Over time, temporary maintenance scripts (`fix_machete.py`, `extract_deps.py`, `test_diff.rs`) and an unused `migrator` tool crate accumulated at the repository root. Furthermore, unused dependency declarations and redundant ignore blocks remained in several workspace `Cargo.toml` files, introducing noise into dependency analysis tools such as `cargo-machete`.

## Decision

1. Remove obsolete root scripts (`fix_machete.py`, `extract_deps.py`, `test_diff.rs`).
2. Remove the unused `migrator` crate directory and update `Cargo.lock`.
3. Purge unused dependency declarations (`serde`, `thiserror`, `diagnostics`, `federated_registry`) from domain and helper crates (`crates/diagnostics`, `crates/markdown_tests`, `math_explorer`, etc.).
4. Add explicit `[package.metadata.cargo-machete]` ignore entries for `scientific_metadata` in domain crates where it is required for `#[derive(Theory)]` macro expansion.

## Consequences

- Workspace build files and root structure remain clean and free of dead maintenance scripts.
- `cargo-machete` passes workspace checks without false positives on macro-required dependencies.
- Build dependencies are strictly aligned with actual usage.
