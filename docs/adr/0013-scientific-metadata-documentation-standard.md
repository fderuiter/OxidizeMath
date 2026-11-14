# 13. Scientific Metadata Documentation and Hygiene Standard

Date: 2026-11-14

## Status

Accepted

## Context

The `scientific_metadata` crate previously lacked a localized `README.md` guide and permitted undocumented public APIs via `#![allow(missing_docs)]` and localized `#[allow(missing_docs)]` attributes across `theory.rs` and `citation_registry.rs`. As a result, its architecture, `TheoryDescribable` contract, and `CitationRegistry` public methods were omitted from crate-level rustdocs and automated workspace documentation checks.

## Decision

1. **Native Crate Documentation via `include_str!` and `README.md`**: Created `crates/scientific_metadata/README.md` covering crate architecture, module contracts, and developer instructions with executable examples. Integrated this file into `crates/scientific_metadata/src/lib.rs` via `#![doc = include_str!("../README.md")]`.
2. **Documentation Hygiene Enforcement**: Replaced crate-level `#![allow(missing_docs)]` with `#![warn(missing_docs)]` in `crates/scientific_metadata/src/lib.rs`.
3. **Trait and Struct Documentation**: Removed local `allow(missing_docs)` attributes from `crates/scientific_metadata/src/theory.rs` and added doc comments with verified doc tests for `ParameterConstraint` and all `TheoryDescribable` methods.
4. **Registry Interface Documentation**: Removed local `allow(missing_docs)` attributes from `crates/scientific_metadata/src/citation_registry.rs` and added complete doc comments and doc tests for `CitationRegistry`, `global()`, `register()`, and `get()`.
5. **Workspace Doc Test Integration**: Added `scientific_metadata` as a dev-dependency in `crates/markdown_tests/Cargo.toml` to ensure automated verification of markdown documentation code examples across the workspace.

## Consequences

- **Positive:** Complete rustdoc coverage and documentation hygiene across the `scientific_metadata` crate.
- **Positive:** Validated, executable doc tests for all public APIs and markdown guides.
- **Positive:** Standardized consistency with the Active Documentation Standard (ADR 0005) across all workspace crates.
