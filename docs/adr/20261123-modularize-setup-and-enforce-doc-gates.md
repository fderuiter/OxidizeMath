# ADR 20261123: Modularize Setup and Enforce Static Documentation Gates

## Status
Accepted (extends [ADR 0008](0008-xtask-lint-all-pre-commit.md), [ADR 0036](0036-harden-ast-and-doctest-policy-audits.md), and [ADR 0043](0043-repair-guides-and-enforce-static-doc-gates.md))

## Context
1. **Local Hook Activation Blocked by Browser Prerequisites**: `cargo run -p xtask -- setup` ran `rustup`, enforced `wasm-bindgen`, `trunk`, `brotli`, and Playwright Chromium presence via `exit(1)`, and ran full `cargo build` and `cargo test` before configuring `core.hooksPath` to `.githooks`. Contributors without the WebAssembly browser stack could not activate `.githooks/pre-commit`.
2. **Walkthrough & Crate Topology Drift**: [`docs/architecture/interactive_tool_walkthrough.md`](../../docs/architecture/interactive_tool_walkthrough.md) wrapped its simulation state and `TheoryDescribable` implementations in `rust,ignore` fences instead of compiled `markdown_tests` doctests and taught `show_ui`/`draw` rather than `show_inside`, [`math_explorer_gui/src/tabs/mod.rs`](../../math_explorer_gui/src/tabs/mod.rs) imported from binary-only `math_explorer_gui::`, [`README.md`](../../README.md) described the GUI as MRI-only, [`math_explorer_gui/README.md`](../../math_explorer_gui/README.md) warned against `rand` despite its workspace dependency, and architecture indexes stated `18` crates instead of `20`.
3. **Static Documentation Audit Gaps**: `check-docs` in [`crates/unified_verification`](../../crates/unified_verification/src/doc_audit.rs) did not block unapproved `rust,ignore` bypasses, invalid `math_explorer_gui::` binary-crate imports, or drifted workspace crate counts.

## Decision
1. **Configure `.githooks` First and Modularize `xtask setup`**:
   - Reorder `xtask setup` in [`apps/xtask/src/setup.rs`](../../apps/xtask/src/setup.rs) so `configure_git_hooks_in_dir` executes first.
   - Support `--hooks-only` (configure `.githooks` and exit `0`), `--check-only` (skip `cargo check --workspace`), and `--web` (enforce WebAssembly and Playwright browser prerequisites with `exit(1)`), warning on missing browser tools by default and running `cargo check --workspace` instead of `cargo build` and `cargo test`.
2. **Compile Walkthrough Doctests and Sync Workspace Documentation**:
   - Add `serde` and `serde_json` to `[dev-dependencies]` in [`crates/markdown_tests/Cargo.toml`](../../crates/markdown_tests/Cargo.toml) (keeping `math_explorer_gui` binary-only so `api_baseline.json` stays unchanged), convert the `HarmonicOscillatorTool` state and `TheoryDescribable` blocks in [`docs/architecture/interactive_tool_walkthrough.md`](../../docs/architecture/interactive_tool_walkthrough.md) to compiled `rust` doctests, rewrite the `InteractiveTool` example around `show_inside` and `crate::framework`, and verify the full `HarmonicOscillatorTool` contract in [`math_explorer_gui/src/walkthrough_tool_tests.rs`](../../math_explorer_gui/src/walkthrough_tool_tests.rs).
   - Sync the `20`-crate count and `Developer_Tooling` topology across [`README.md`](../../README.md), [`CONTRIBUTING.md`](../../CONTRIBUTING.md), [`docs/README.md`](../README.md), [`docs/architecture/README.md`](../architecture/README.md), and [`docs/architecture/system_overview.md`](../architecture/system_overview.md), and update [`math_explorer_gui/README.md`](../../math_explorer_gui/README.md) and [`math_explorer_gui/src/tabs/mod.rs`](../../math_explorer_gui/src/tabs/mod.rs) for multi-domain `SimulationFramework` tabs, `crate::` imports, and seeded `rand` usage.
3. **Enforce Static Documentation Gates in `unified_verification`**:
   - Extend `doc_audit` (`check-docs`, `batch-static`, and `batch-staged`, excluding `docs/adr/` historical records) to reject unapproved `rust,ignore` fences (ratcheted via `[doc_verification.approved_ignore_fences]` in [`verification_whitelist.toml`](../../verification_whitelist.toml)), retired `Tectonic` instructions, invalid `math_explorer_gui::` imports across Markdown and Rust sources, and drifted workspace crate counts or `system_overview.md` catalog mismatches.

## Consequences
- **Positive**: Contributors on native Rust toolchains can activate `.githooks/pre-commit` immediately via `xtask setup` or `xtask setup --hooks-only`, while CI and web contributors retain strict `--web` validation.
- **Positive**: `markdown_tests`, `math_explorer_gui --bins`, and offline `check-docs` continuously guard tutorial snippets, `crate::` GUI imports, paper tooling docs, and workspace crate counts against future drift.
