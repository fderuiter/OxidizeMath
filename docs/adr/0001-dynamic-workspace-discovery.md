# Architecture Decision Record: Dynamic Workspace Discovery for Traceability

## Status

Accepted

## Context
Previously, the `traceability_cli` tool relied on a hardcoded list of directories to scan for mathematical implementations and requirements (e.g., `math_explorer/src`, `crates/domain_ai/src`, etc.). As the project scales and new crates are added, maintaining this hardcoded list has become error-prone and tedious.

## Decision
We implemented a dynamic workspace discovery mechanism within `crates/oxidize_core/src/bin/traceability_cli.rs`. This system dynamically parses `Cargo.toml` (or `../../Cargo.toml` if executed within a crate subdirectory) to extract workspace members. It then iterates over the `src` directory of each member to identify Rust source files. Additionally, `crates/oxidize_core/src/traceability.rs` was enhanced with a fallback mechanism to recursively list directories and collect `.rs` files when traditional entrypoints (`lib.rs` or `main.rs`) are absent.

## Consequences
- **Positive:** Adding a new crate to the workspace automatically includes it in the traceability checks without needing to update the `traceability_cli` configuration.
- **Positive:** Reduced maintenance overhead and risk of out-of-sync configurations.
- **Negative:** Slightly increased complexity during initialization of the `traceability_cli` as it now relies on file system traversal and `toml` parsing instead of a static list.
