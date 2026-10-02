# Architecture Decision Record: Consolidate Workspace Discovery into oxidize_core

## Context
Previously, `get_workspace_members` was implemented separately in `crates/oxidize_core/src/bin/traceability_cli.rs` and `crates/unified_verification/src/main.rs`. These duplicate implementations created redundant maintenance and inconsistent directory resolution when CLI tools were executed from crate subdirectories.

## Decision
We centralized workspace member discovery into a shared module `oxidize_core::workspace` (`crates/oxidize_core/src/workspace.rs`). The module provides `get_workspace_members` with relative fallback handling (`Cargo.toml` and `../../Cargo.toml`). `traceability_cli` and `unified_verification` now use this consolidated function.

## Consequences
- **Positive:** Centralized workspace member discovery eliminates duplicate code and ensures consistent workspace parsing across all CLI tools.
- **Positive:** CLI tools running from subdirectories correctly fall back to the workspace root manifest.
- **Positive:** Added unit tests in `oxidize_core::workspace` to guarantee parsing and fallback accuracy.
- **Negative:** `unified_verification` now depends directly on `oxidize_core`.
