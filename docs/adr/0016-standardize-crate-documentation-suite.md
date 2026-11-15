# 16. Standardize Crate Documentation Suite Across Workspace

Date: 2026-11-14

## Status

Accepted

## Context

Fourteen workspace crates previously contained only an obsolete placeholder string in their top-level `README.md` files. Because each crate's `src/lib.rs` (or `src/main.rs`) includes its top-level manual via `#![doc = include_str!("../README.md")]` according to the Active Documentation Standard (ADR 0005), this placeholder generated superficial documentation stubs in compiler references, missing module hierarchies, trait contracts, and runnable workflow examples. Furthermore, `crates/markdown_tests/Cargo.toml` lacked registration for `federated_registry`, preventing automated verification of markdown code blocks referencing federated telemetry.

## Decision

1. **Standardized Crate Manuals**: Replaced placeholder stubs across all fourteen crates (`federated_registry`, `verified_engine`, `verified_engine_macros`, `oxidize_core`, `math_commons`, `pure_math`, `domain_ai`, `domain_physics`, `domain_biology`, `domain_applied`, `domain_climate`, `domain_epidemiology`, `markdown_tests`, and `unified_verification`) with comprehensive architecture guides modeled after `crates/scientific_metadata/README.md`. Each manual provides:
   - System overview, architectural layer, and design rationale.
   - Detailed module structure and trait contracts.
   - Verified, compilable Rust code examples with assertions demonstrating common workflows.
   - Ecosystem integration and interaction with `math_explorer` and `math_explorer_gui`.
2. **Elimination of Placeholder Strings**: Removed all occurrences of superficial placeholder strings across all workspace documentation and build files (`oxidize_core/build.rs`, `oxidize_core/src/bin/traceability_cli.rs`, `markdown_tests/build.rs`, `math_explorer_gui/build.rs`, and `math_commons/src/lib.rs`).
3. **Workspace Doc-Test Dependency Registration**: Registered `federated_registry` as a dev-dependency in `crates/markdown_tests/Cargo.toml` to support workspace-wide doctest execution.

## Consequences

- **Positive:** Complete rustdoc coverage and architecture documentation across all 14 core mathematical crates.
- **Positive:** Validated, executable doc tests for all newly authored markdown code samples via `cargo test --doc --workspace`.
- **Positive:** Full compliance with ADR 0005 and ADR 0013 documentation hygiene standards.
- **Positive:** Zero compiler warnings during `cargo doc --workspace --no-deps`.
