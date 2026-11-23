# ADR 0013: Workspace Doc Lints and Intra-Doc Path Scoping

## Status

Accepted

## Context
Automated documentation generation across workspace crates encountered broken intra-doc links and warnings under strict flags (`RUSTDOCFLAGS="-D warnings"`). Furthermore, the markdown test harness compiles workspace documentation, exposing it to markdown formatting lints such as bare URLs and broken intra-doc links.

## Decision
1. Add `rustdoc::broken_intra_doc_links = "warn"` to workspace lints in `Cargo.toml`.
2. Decouple `markdown_tests` from doc generation lints by applying `#![allow(rustdoc::broken_intra_doc_links)]` and `#![allow(rustdoc::bare_urls)]` to its crate root.
3. Validate doc generation in CI pre-flight via `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features`.

## Consequences
- Documentation links are strictly validated across production workspace crates.
- Test harnesses do not fail documentation generation due to non-rustdoc markdown files.
- The CI pre-flight pipeline catches any broken intra-doc links early.
