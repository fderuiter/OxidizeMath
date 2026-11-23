# Automation Runner (xtask)

Workspace automation runner orchestrating pre-commit hooks, batched verification gates, feature-matrix testing, WebAssembly smoke suites, and paper builds.

## 1. Overview & Architecture

The `xtask` binary (`apps/xtask`) implements the cargo-xtask automation pattern for the OxidizeMath monorepo. It provides a single cross-platform developer CLI that coordinates formatting, `unified_verification` static/AST checks, host and `wasm32` Clippy gates, markdown link checking, and Playwright browser smoke tests.

### Active Quality Gates

| Command | Operational Role | Status |
|---|---|---|
| `check-api-drift` | Validates public API surface against `api_baseline.json`. | Active |
| `check-cargo-docs` | Verifies workspace documentation builds without warnings. | Active |
| `check-docs` | Runs crate documentation audit across all workspace crates. | Active |
| `check-duplicates` | Runs AST-level code clone detector across first-party crates. | Active |
| `check-entropy` | Scans codebase for high entropy strings and unseeded RNG usage. | Active |
| `check-file-lengths` | Verifies the 500-line source file limit across first-party crates. | Active |
| `check-markdown-links` | Checks all internal relative Markdown documentation links. | Active |
| `check-profiles` | Verifies Cargo `[profile.release]` `overflow-checks = true` and rejects member overrides. | Active |
| `check-staged-duplicates` | Checks for duplicated patterns in git staging (non-blocking helper). | Active |
| `check-vulnerabilities` | Scans workspace dependencies for known OSV vulnerabilities. | Active |
| `lint-all` | Runs the full PR verification suite (formatting, single-pass batched static & AST checks across profiles, ADRs, file lengths, workflows, crate docs, AST clones, API drift, traceability, entropy, and policy audit, plus host & WASM clippy, rustdoc, doctests, markdown links, and OSV vulnerabilities). | Active |
| `lint-staged` | Syncs the staged Git index into `<target_dir>/lint-staged-snapshot` and runs offline pre-commit gates (`.githooks/pre-commit`): staged `rustfmt`, `batch-staged` (staged ADR verification, profiles, file lengths, docs, AST clones, traceability, entropy, policy audit), single `.github/` workflow linting, `.md` or deletion/rename link scans, and reverse-dependent + 17-web-crate host/`wasm32` Clippy with root config escalation. | Active |
| `lint-workflows` | Validates GitHub Actions workflow definitions and permissions. | Active |
| `test-features` | Exercises isolated Cargo domain feature flags (`pure_math`, `physics`, `core-only`, etc.). | Active |
| `test-policy-audit` | Audits workspace crates for unauthorized lint/dead-code/`missing_docs` bypasses and unused dependencies. | Active |
| `test-wasm` | Discovers and executes `wasm32-unknown-unknown` unit and browser tests across all compatible workspace crates (anchored to the workspace root), builds `math_explorer_gui/dist` (unless `--skip-build` is passed), verifies the 3 MB bundle budget, and runs headless Chromium CSP & fault-injection smoke checks (`--screenshot-dir <dir>` supported). | Active |
| `traceability` | Verifies academic paper citation bindings and `#[verified]` coverage. | Active |
| `verify-records` | Verifies architectural decision records (`--staged` supported), enforcing `NNNN-slug.md` naming, `# ` and `## Status` headings, unique 4-digit IDs, and ACMR updates on core changes. | Active |
| `verify-suite` | Runs the full coverage-instrumented high-integrity verification suite. | Active |

### Maintenance & Onboarding Tasks

| Command | Operational Role | Status |
|---|---|---|
| `compile-papers` | Compiles academic `.tex` manuscripts in `papers/` using Tectonic or TeXLive. | Maintenance |
| `regenerate-baseline` | Regenerates `crates/unified_verification/api_baseline.json`. | Maintenance |
| `setup` | Installs `wasm32-unknown-unknown`, checks prerequisites, builds and tests the workspace, and configures `.githooks/pre-commit`. | Maintenance |

---

## 2. Developer Instructions

### Common Developer Workflows

Run `xtask` commands from the workspace root using `cargo run -p xtask -- <command>`:

```bash
# Bootstrap local git hooks and wasm32 target
cargo run -p xtask -- setup

# Run fast scoped checks on staged changes
cargo run -p xtask -- lint-staged

# Run the complete pre-PR verification suite
cargo run -p xtask -- lint-all
```

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

---

## 3. Ecosystem Integration

`xtask` delegates AST and repository policy analysis to `unified_verification` while coordinating Cargo subcommands, `.githooks/pre-commit`, and `.github/workflows/ci.yml` jobs across Linux, macOS, Windows, and WebAssembly targets.
