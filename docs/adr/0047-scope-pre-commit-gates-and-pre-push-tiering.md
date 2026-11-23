# ADR 0047: Scope Pre-Commit Gates, Add Pre-Push Tiering, and Eliminate Build-Script `CARGO_MANIFEST_DIR` Drift

## Status

Accepted

## Context

Prior to this decision, `.githooks/pre-commit` (`cargo run -p xtask -- lint-staged`) materialized the staged Git index into `target/lint-staged-snapshot` (`ADR 0040`), yet still incurred repository-wide latency and Cargo cache invalidation on single-file edits:

1. **Unscoped Static Passes in `batch-staged`**: Even when `escalate` was `false`, `unified_verification batch-staged` ran `CloneDetectorVisitor` across all ~804 first-party `.rs` files, audited `Cargo.toml` profiles and crate documentation across all 20 member crates, parsed Rust fences across all ~155 Markdown files, and executed `run_traceability_with_cache` across all ~804 `.rs` files.
2. **Transitive Pre-Commit Clippy Fan-Out**: `apps/xtask/src/staged.rs` expanded every touched crate to all reverse dependents plus all 17 web-compiled crates reachable from `math_explorer_gui`, triggering heavy host and `wasm32-unknown-unknown` GUI Clippy runs during commits to leaf domain crates.
3. **`CARGO_MANIFEST_DIR` Fingerprint Drift**: `math_explorer_gui/build.rs` and `crates/markdown_tests/build.rs` embedded `env!("CARGO_MANIFEST_DIR")` inside generated `#[path = ...]` and `include_str!(...)` directives. Because `target/lint-staged-snapshot` and the working tree have distinct `CARGO_MANIFEST_DIR` values while sharing `CARGO_TARGET_DIR`, alternating between `cargo check` in the working tree and `cargo run -p xtask -- lint-staged` forced `rustc` recompilation of `math_explorer_gui` and `markdown_tests`.

## Decision

We split local Git guardrails into two 100% offline version-controlled tiers under `.githooks/` and make `build.rs` artifacts workspace-location independent:

1. **Scoped Pre-Commit Static Gates (`unified_verification batch-staged`)**:
   - When `escalate` is `false`, `CloneDetectorVisitor` (`ast_visitor.rs`) inspects only staged `.rs` files; `check_profiles_with_cache` (`profile.rs`) and `doc_audit` (`doc_audit.rs`, `facade_imports.rs`, `staged_guides.rs`) check only touched member crates and staged Markdown/guide files; Markdown fence policy auditing (`policy_audit/doctests.rs`) scans only staged `.md` files; and staged citation verification (`staged_traceability.rs`) validates `@cite` keys in staged `.rs` files while deferring full 804-file `run_traceability_with_cache` unless `traceability.toml` or `papers/` changes.
   - Staging root manifests (`Cargo.toml`, `Cargo.lock`), toolchain/lint configuration (`rust-toolchain*`, `.cargo/*`, `.clippy.toml`, `.rustfmt.toml`, `verification_whitelist.toml`), or shared policy tooling (`crates/unified_verification`, `crates/oxidize_core`) escalates `lint-staged` to workspace checks.
2. **Scoped Pre-Commit Clippy & `.githooks/pre-push` (`xtask lint-push`)**:
   - `xtask lint-staged` runs host Clippy only on `direct_crates` (directly modified workspace members) and runs `--target wasm32-unknown-unknown` Clippy only when `math_explorer_gui` or `wasm32`-specific sources change (unless escalated).
   - A new version-controlled `.githooks/pre-push` hook invokes `cargo run -p xtask -- lint-push`, resolving pushed commits against remote tracking refs (falling back to `origin/main` or `main`), running reverse-dependent host Clippy, conditional reverse-dependent `wasm32-unknown-unknown` Clippy across web-reachable crates, and full workspace `unified_verification traceability` offline.
   - `cargo run -p xtask -- setup` configures `core.hooksPath` to `.githooks` and ensures executable `0755` permissions on both `.githooks/pre-commit` and `.githooks/pre-push`.
3. **Workspace-Relative `OUT_DIR` Staging in Build Scripts**:
   - `math_explorer_gui/build.rs` (`build_stage.rs`) stages `src/tabs/` and compile-time embedded assets (`sw.js`, `Verification_Certificate.json`, `references.bib`) into `OUT_DIR` using content-guarded writes (`write_if_changed`) and emits generated modules referencing `env!("OUT_DIR")` instead of `env!("CARGO_MANIFEST_DIR")`.
   - `crates/markdown_tests/build.rs` stages workspace `.md` files into `OUT_DIR/staged_md/` with `write_if_changed` and generates `#[doc = include_str!(concat!(env!("OUT_DIR"), ...))]` doctest modules without referencing `CARGO_MANIFEST_DIR`.

```text
Hook Tier Summary:
- .githooks/pre-commit -> cargo run -p xtask -- lint-staged (scoped staged files + direct_crates Clippy, offline)
- .githooks/pre-push   -> cargo run -p xtask -- lint-push   (reverse-dependent host/wasm32 Clippy + full traceability, offline)
- Pre-PR / CI parity   -> cargo run -p xtask -- lint-all    (full workspace verification + OSV advisory check)
```

## Consequences

- Warm single-file Markdown and leaf-crate `batch-staged` runs complete in sub-second latency without parsing 804 Rust files or auditing untouched crates.
- Committing inside a leaf domain crate runs Clippy only on that crate at commit time while still enforcing reverse-dependent host/`wasm32` Clippy and full 804-file traceability before `git push`.
- Alternating `cargo check` in the working tree and `cargo run -p xtask -- lint-staged` in `target/lint-staged-snapshot` preserves Cargo fingerprints for `math_explorer_gui` and `markdown_tests`.
