# Agent Guide for OxidizeMath

This document summarizes the repository conventions, quality guardrails, and local verification commands required to maintain parity with `.githooks/pre-commit` and `.github/workflows/ci.yml`.

## Architecture & Standards

- Review [Contributing Guidelines](CONTRIBUTING.md) and the [Documentation Index](docs/README.md) before modifying core domain crates or UI tools.
- Keep every Rust source file at or below **500 lines** (`check-file-lengths`) and keep functions within the **60-line** Clippy ceiling configured in `.clippy.toml`.
- Do not introduce unwhitelisted `dead_code`, `unused`, or `warnings` lint suppressions (`allow`, `expect`, or `cfg_attr`). Zero `dead_code` bypass entries are permitted per [ADR 0019](docs/adr/0019-purge-dead-code-bypasses.md); gate test-only helpers with `#[cfg(test)]` or delete unreferenced items instead.
- Modifying core mathematical or runtime crates (`crates/` excluding `crates/unified_verification`, or `math_explorer/`) requires creating or updating an Architectural Decision Record under `docs/adr/`.
- Use `bash` or `text` tags on non-Rust fenced code blocks in Markdown files because `crates/markdown_tests` compiles untagged fences as Rust doctests.

## Local Verification & CI Parity

Run the full pre-commit verification suite via `xtask`:

```bash
cargo run -p xtask -- lint-all
```

Run individual pre-flight and policy verification gates locally:

```bash
# Audit AST attributes for unauthorized dead_code/unused/warnings bypasses and unused dependencies
cargo run -p xtask -- test-policy-audit
cargo run -p unified_verification --release -- test-policy-audit

# Format, Clippy, file-length, workflow, documentation, link, traceability, and entropy gates
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo run -p unified_verification --release -- check-file-lengths
cargo run -p unified_verification --release -- lint-workflows
cargo run -p unified_verification --release -- verify-records
cargo run -p unified_verification --release -- traceability
cargo run -p unified_verification --release -- check-entropy
cargo run -p unified_verification --release -- check-docs
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo test --doc --workspace --all-features
cargo run -p xtask -- check-markdown-links
```

Run native unit/integration tests and WebAssembly target checks:

```bash
cargo test --workspace --lib --tests
cargo check --workspace --target wasm32-unknown-unknown --all-features
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features
```
