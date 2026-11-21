# Automation Runner (xtask)

This directory contains the workspace automation runner, which is responsible for executing various quality gates and legacy utility tasks.

## Commands

### Active Quality Gates
These commands run automated checks in git hooks or CI.

| Command | Operational Role | Status |
|---|---|---|
| `check-api-drift` | Validates public API surface against `api_baseline.json`. | Active |
| `check-cargo-docs` | Verifies workspace documentation builds without warnings. | Active |
| `check-docs` | Runs crate documentation audit. | Active |
| `check-duplicates` | Runs AST-level code clone detector across first-party crates. | Active |
| `check-entropy` | Scans codebase for high entropy strings and secrets. | Active |
| `check-file-lengths` | Verifies file-length constraints. | Active |
| `check-markdown-links` | Check all internal relative markdown documentation links | Active |
| `check-profiles` | Verifies Cargo `[profile.release]` `overflow-checks = true` and rejects member overrides. | Active |
| `check-staged-duplicates` | Checks for duplicated patterns in git staging (non-blocking legacy helper). | Active |
| `check-vulnerabilities` | Scans workspace dependencies for known vulnerabilities. | Active |
| `lint-all` | Runs the full PR verification suite (formatting, single-pass batched static & AST checks across profiles, ADRs, file lengths, workflows, crate docs, AST clones, API drift, traceability, entropy, and policy audit, plus host & WASM clippy, rustdoc, doctests, markdown links, and OSV vulnerabilities). | Active |
| `lint-staged` | Runs fast diff-scoped pre-commit checks on staged files (`.githooks/pre-commit`): staged `rustfmt`, `batch-staged`, conditional `.github/` workflow linting, conditional `.md` link checks, and targeted host/`wasm32` Clippy. | Active |
| `lint-workflows` | Validates CI workflow definitions. | Active |
| `test-features` | Runs tests for various cargo features. | Active |
| `test-policy-audit` | Audits workspace crates for unauthorized dead-code bypasses and unused dependencies. | Active |
| `test-wasm` | Executes `wasm32-unknown-unknown` browser unit tests, builds `math_explorer_gui/dist` (unless `--skip-build` is passed), verifies the 3 MB bundle budget, and runs headless Chromium CSP & fault-injection smoke checks (`--screenshot-dir <dir>` supported). | Active |
| `traceability` | Generates and checks the traceability report. | Active |
| `verify-records` | Verifies architectural decision records (`--staged` supported). | Active |
| `verify-suite` | Runs the high-integrity verification suite. | Active |

### WebAssembly & Browser Smoke Prerequisites (`test-wasm`)

Running `cargo run -p xtask -- test-wasm` locally requires the `wasm32-unknown-unknown` Rust target, `trunk`, `wasm-bindgen-cli` (matching the workspace `wasm-bindgen` version), `brotli`, and Python `playwright` with headless Chromium installed:

```bash
rustup target add wasm32-unknown-unknown
sudo apt-get install -y brotli
cargo binstall --no-confirm trunk wasm-bindgen-cli@0.2.130
python3 -m pip install --user playwright
python3 -m playwright install --with-deps chromium
cargo run -p xtask -- test-wasm
```

### Legacy Utility Tasks
These commands are used for manual repository maintenance or legacy workflows.

| Command | Operational Role | Status |
|---|---|---|
| `compile-papers` | Compiles academic papers using Tectonic. | Legacy |
| `regenerate-baseline` | Regenerates the public API baseline. | Legacy |
| `setup` | Installs `wasm32-unknown-unknown`, builds and tests the workspace, and configures version-controlled git hooks. | Legacy |
