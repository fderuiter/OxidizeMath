# AGENTS.md — Developer & Agent Workflow Guide

This guide documents local build, verification, and feature-matrix workflows for automated coding agents and human contributors working in **OxidizeMath**. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Documentation Index](docs/README.md) for full architectural and contribution standards.

## Quick Start & Quality Gates

Configure version-controlled Git hooks and run initial workspace checks:

```bash
cargo run -p xtask -- setup
```

Run tests and compile checks across all workspace members:

```bash
# Run unit and integration tests across all workspace crates
cargo test --workspace

# Check compilation across all workspace crates
cargo check --workspace
```

Run the complete workspace lint and verification suite before committing:

```bash
cargo run -p xtask -- lint-all
```

## Feature Matrix Testing (`xtask test-features`)

Both [math_explorer/Cargo.toml](math_explorer/Cargo.toml) and [math_explorer_gui/Cargo.toml](math_explorer_gui/Cargo.toml) gate optional scientific domains behind Cargo features (`pure_math`, `applied`, `ai`, `biology`, `climate`, `epidemiology`, `physics`). In `math_explorer_gui`, [math_explorer_gui/build.rs](math_explorer_gui/build.rs) conditionally registers domain tabs via `@explorer_feature` annotations, leaving only `TraceabilityTab` active under `--no-default-features`.

The `feature-matrix` job in [.github/workflows/ci.yml](.github/workflows/ci.yml) invokes [apps/xtask/src/features.rs](apps/xtask/src/features.rs), which runs `cargo test` for **both** `-p math_explorer` and `-p math_explorer_gui`.

### Run the Full Feature Matrix Locally

```bash
# Tests core-only (--no-default-features), each of the 7 domain features, and --all-features
# across both math_explorer and math_explorer_gui
cargo run -p xtask -- test-features
```

### Run Individual CI Matrix Configurations

```bash
# Core-only (--no-default-features)
cargo run -p xtask -- test-features core-only

# Individual domain features (matching .github/workflows/ci.yml)
cargo run -p xtask -- test-features pure_math
cargo run -p xtask -- test-features applied
cargo run -p xtask -- test-features ai
cargo run -p xtask -- test-features biology
cargo run -p xtask -- test-features climate
cargo run -p xtask -- test-features epidemiology
cargo run -p xtask -- test-features physics
```

## Repository Guardrails & Conventions

- **File & Function Size Limits**: Every `.rs` source file must remain at or below **500 lines** (`cargo run -p xtask -- check-file-lengths`), and functions must respect the **60-line** threshold configured in [.clippy.toml](.clippy.toml).
- **GUI Feature Guards**: When adding or modifying tests in `math_explorer_gui` that import domain-specific tabs/tools or assert full-catalog counts, guard imports and test functions with the corresponding `#[cfg(feature = "...")]` attribute so `core-only` and single-feature matrix builds compile without errors or unused-import/dead-code warnings.
- **Zero Compiler Warnings**: All targets must pass Clippy and `rustdoc` with `-D warnings`.
- **Markdown Code Fences & Links**: Workspace Markdown files are compiled by `crates/markdown_tests` during `cargo test --doc --workspace --all-features` and audited by `cargo run -p xtask -- check-markdown-links`. Always tag shell or non-Rust snippets with `bash` or `text` fences and ensure relative links resolve to existing repository paths.
- **Architectural Decision Records (ADRs)**: Structural modifications under `math_explorer/` or `crates/` (outside `unified_verification`) require an accompanying ADR update under `docs/adr/` verified by `cargo run -p xtask -- verify-records`.
