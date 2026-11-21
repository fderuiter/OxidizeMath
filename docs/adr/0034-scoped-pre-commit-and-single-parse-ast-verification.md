# ADR 0034: Scoped Pre-Commit Lint Gates and Single-Parse In-Process AST Verification

- **Status:** Accepted
- **Date:** 2026-11-21

## Context

Previously, `.githooks/pre-commit` executed `cargo run -p xtask -- lint-all` on every local commit. That workflow created four recurring developer-experience and build-latency bottlenecks:

1. **Workspace-wide checks on incremental commits:** Editing a single domain file triggered full-workspace host and `wasm32-unknown-unknown` Clippy passes, full `cargo doc` and doctest builds, live OSV network queries (`check-vulnerabilities`), public API baseline checks (`check-api-drift`), and architectural decision record checks (`verify-records --staged`), blocking local checkpoint commits.
2. **Release-profile fat LTO overhead for verification tooling:** `apps/xtask/src/gates.rs` built `unified_verification` with `--release`, inheriting `lto = true` and `codegen-units = 1` from the workspace root `Cargo.toml`.
3. **Redundant `syn::parse_file` passes and subprocesses:** `crates/oxidize_core/src/traceability.rs` parsed each Rust file twice (once for item extraction and submodule discovery, and a second time inside `traceability/analysis.rs` for citation validation), and `unified_verification` spawned `cargo run -p oxidize_core --bin traceability_cli` as a separate Cargo subprocess while re-parsing the same `.rs` files across `ast_visitor`, `api_drift`, `entropy_guard`, and `policy_audit`.
4. **Repository state mutation during tests:** `xtask`'s `test_git_hooks_path_configuration` invoked `git config core.hooksPath .githooks` directly against the live workspace repository, and `xtask setup` did not ensure the `wasm32-unknown-unknown` target was installed.

## Decision

1. **Scoped Pre-Commit Hook (`xtask lint-staged`):**
   - Update `.githooks/pre-commit` to run `cargo run -p xtask -- lint-staged`.
   - Scope `lint-staged` strictly to paths reported by `git diff --cached`:
     - Run `rustfmt --edition <edition> --check` only on staged `.rs` files.
     - Run `unified_verification batch-staged` in-process for staged file-length checks, staged secret/entropy scanning, touched-member policy auditing, staged duplicate heuristics, and `.github/`-gated workflow linting.
     - Run `check-markdown-links` only when staged `.md` files are present.
     - Run targeted host Clippy (`-p <pkg>`) only on member crates with staged `.rs` or `Cargo.toml` edits, and run targeted `wasm32-unknown-unknown` Clippy only when web crates (`oxidize_core`, `egui_plot`, `math_explorer_gui`) have staged edits.
   - Defer PR-level gates (`check-vulnerabilities`, `check-api-drift`, and `verify-records`) to `cargo run -p xtask -- lint-all` and `.github/workflows/ci.yml`.

2. **Development Profile for Local `unified_verification` Execution:**
   - Update `apps/xtask/src/gates.rs` to build and invoke `target/debug/unified_verification` (`cargo build -p unified_verification -q`) so local verification skips release fat LTO while keeping workspace `[profile.release]` invariants untouched.

3. **Shared `AstCache` and In-Process Traceability in `oxidize_core` and `unified_verification`:**
   - Add `CachedRustFile` and `AstCache` in `crates/oxidize_core/src/traceability/cache.rs`, refactor `TraceabilityEngine` (`crates/oxidize_core/src/traceability.rs` and `analysis.rs`) to reuse already-parsed `syn::File` ASTs in a single pass per file, and expose `run_workspace_traceability_with_cache` (`crates/oxidize_core/src/traceability/runner.rs`) as a direct library entry point.
   - Add `batch-ast`, `batch-static`, and `batch-staged` subcommands in `crates/unified_verification` that share one `AstCache` across clone detection, public API drift, traceability, entropy scanning, and policy auditing so each unique `.rs` file is parsed at most once per run.

4. **Toolchain Setup and Isolated Hook Tests:**
   - Update `xtask setup` to run `rustup target add wasm32-unknown-unknown`.
   - Parameterize `configure_git_hooks_in_dir(repo_dir)` so `test_git_hooks_path_configuration` validates hook configuration and Unix `0755` permissions inside a temporary Git repository without mutating workspace `core.hooksPath`.

## Verification Commands

```bash
# Scoped pre-commit verification on staged files
cargo run -p xtask -- lint-staged

# Single-pass AST batch across workspace Rust sources
cargo run -p unified_verification -- batch-ast

# Full PR-readiness verification suite
cargo run -p xtask -- lint-all
```

## Consequences

- **Positive:** Local commits complete offline in seconds, touching only modified crates and staged files without requiring premature ADRs or `api_baseline.json` regeneration during work-in-progress commits.
- **Positive:** Full static verification (`xtask lint-all` and `verify-suite`) avoids release fat LTO compilation of `unified_verification`, eliminates subprocess Cargo overhead for traceability, and parses each workspace `.rs` file once across all five `syn` visitors.
- **Neutral:** Contributors and coding agents must continue running `cargo run -p xtask -- lint-all` before opening or updating pull requests to catch deferred `check-api-drift`, `verify-records`, workspace-wide doc/Clippy, and `check-vulnerabilities` checks enforced in CI.
