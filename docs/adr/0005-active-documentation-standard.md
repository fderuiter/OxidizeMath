# 5. Active Documentation Standard

Date: 2026-07-22

## Status

Accepted

## Context

Isolated helper scripts, unlinked design logs, and disconnected crate documentation previously cluttered the workspace, resulting in maintenance overhead and contributor confusion. The workspace contained many crates with inner doc-comments (`//!`) inside `src/lib.rs` and `src/main.rs`, but these descriptions were not easily discoverable or consistently validated without specialized scripts.

## Decision

We implement the **Active Documentation Standard** across all workspace crates.

1.  **Native Crate-Level Documentation via `include_str!`**: All existing `//!` inner doc-comments from the primary entry points of workspace crates have been extracted into localized `README.md` files. These files are integrated back into the source code using the compiler-validated attribute:
    ```text
    #![doc = include_str!("../README.md")]
    ```
    This ensures any documentation updates in markdown files are automatically reflected in Rustdocs upon compilation, eliminating duplication efforts.
2.  **Consolidated Safety Guidelines**: The NASA "Power of 10" safety guidelines and checklists from the standalone `AGENTS.md` file have been migrated directly into a new Safety Guidelines section in `CONTRIBUTING.md`. The legacy `AGENTS.md` file has been deleted.
3.  **Indexed Architecture Decision Records (ADRs)**: `CONTRIBUTING.md` was updated to include a dedicated Architecture Decision Records (ADRs) index, linking to historical decisions directly within our primary contributor manual.
4.  **Deprecated Script Cleanup**: Legacy Python maintenance scripts (`fix_docs.py`, `fix_inner_doc.py`) used for doc formatting are obsolete and have been deleted.

## Consequences

*   **Positive:** Single source of truth for documentation, automatically verified by `cargo doc` and `cargo test --doc`.
*   **Positive:** Centralized contributor guidelines that reduce onboarding friction.
*   **Positive:** Deprecation of custom Python scripts reduces toolchain complexity.
*   **Negative:** Developers must ensure the `README.md` is valid markdown that can be parsed correctly by `rustdoc`.
