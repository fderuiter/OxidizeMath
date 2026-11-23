# Markdown Tests

Workspace-wide markdown test harness validating code blocks across all documentation manuals.

## 1. Overview & Architecture

The `markdown_tests` crate provides automated doc-test discovery and verification for all Markdown documentation in the OxidizeMath repository. It operationalizes the Active Documentation Standard (ADR 0005) by ensuring that code snippets published in manuals, guides, and ADR records compile and run cleanly during standard testing cycles.

Rather than relying on unmaintained third-party tools or custom scripts, `markdown_tests` leverages native `rustdoc` capabilities by synthesizing module wrappers that embed Markdown files as compiler-validated documentation comments.

### Module Structure

- **`build.rs`**: Build-time file discovery script (Rust 2024 edition):
  - Recursively scans the workspace directory tree in deterministic sorted order for `*.md` files, skipping library crate `README.md` files that are already verified via `#![doc = include_str!("../README.md")]` under `doc_audit`.
  - Generates path-derived Rust test modules of the form `#[doc = include_str!(...)] pub mod docs_architecture_cera_framework_md {}` into `$OUT_DIR/generated_tests.rs`, writing only when bytes change.
  - Configures deterministic `cargo:rerun-if-changed` triggers for discovered Markdown paths and directories.
- **`lib.rs`**: The crate root module that includes the synthesized doctest files and enables rustdoc evaluation.
- **`Cargo.toml`**: Centralized test dependency catalog specifying workspace crate paths to satisfy imports across all documentation code samples.

---

## 2. Developer Instructions

### Writing Verifiable Documentation Code Blocks

Authors write standard Markdown with tagged Rust code blocks (```` ```rust ````). The `markdown_tests` harness ensures that any crate referenced in documentation examples is registered in its `[dev-dependencies]`:

```rust
// Verify that markdown_tests executes within the workspace directory tree
let manifest_dir = env!("CARGO_MANIFEST_DIR");
assert!(!manifest_dir.is_empty());
assert!(manifest_dir.contains("markdown_tests"));
```

To execute doctests across all workspace Markdown files in isolation:

```bash
cargo test --doc -p markdown_tests
```

---

## 3. Ecosystem Integration

The `markdown_tests` harness is executed as part of the workspace quality gate (`cargo test --doc --workspace`), ensuring that documentation changes are verified simultaneously with code modifications in continuous integration pipelines.
