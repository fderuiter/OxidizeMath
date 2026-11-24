# Automation Runner (xtask)

Workspace automation runner orchestrating pre-commit hooks, batched verification gates, feature-matrix testing, WebAssembly smoke suites, and paper builds.

## 1. Overview & Architecture

The `xtask` binary (`apps/xtask`) implements the cargo-xtask automation pattern for the OxidizeMath monorepo. It provides a single cross-platform developer CLI that coordinates formatting, `unified_verification` static/AST checks, host and `wasm32` Clippy gates, markdown link checking, and Playwright browser smoke tests.

### Active Quality Gates

| Command | Operational Role | Status |
|---|---|---|
| `check-api-drift` | Validates public API surface against `api_baseline.json`. | Active |
| `check-benches` | Builds and runs all workspace benchmark targets in test mode (`cargo test --workspace --benches --all-features`). | Active |
| `check-cargo-docs` | Verifies workspace documentation builds with `--all-features` and zero warnings. | Active |
| `check-certificate` | Verifies `Verification_Certificate.json` and `README.md` parity against workspace AST metrics and `[verify_suite]` floors. | Active |
| `check-docs` | Runs crate documentation audit across all workspace crates. | Active |
| `check-duplicates` | Runs AST-level code clone detector across first-party crates. | Active |
| `check-entropy` | Scans codebase for high entropy strings and unseeded RNG usage. | Active |
| `check-file-lengths` | Verifies the 500-line source file limit across first-party crates. | Active |
| `check-markdown-links` | Checks all internal relative Markdown documentation links. | Active |
| `check-maturity` | Verifies workspace compile health plus `Experimental`, `Validated`, and `Certified` qualification gates against `traceability.toml`. | Active |
| `check-profiles` | Verifies Cargo `[profile.release]` `overflow-checks = true`, `[workspace.package].version` inheritance/bounds, absence of `[package.metadata.packager].version`, `README.md` parity, and member `[lints]` inheritance. | Active |
| `check-public-debug` | Enforces `std::fmt::Debug` across all public structs and enums in mathematical crates. | Active |
| `check-release-tag` | Verifies `refs/tags/v*` / `RELEASE_TAG` parity against `[workspace.package].version` and emits WiX pre-release version mappings. | Active |
| `check-staged-duplicates` | Checks for duplicated patterns in git staging (non-blocking helper). | Active |
| `check-vulnerabilities` | Scans workspace dependencies for known OSV vulnerabilities. | Active |
| `compile-papers` | Compiles all 31 academic papers with multi-pass TeXLive `pdflatex` and `bibtex`, emits archive and GitHub Pages HTML catalogs (`index.html` and `pages-index.html`), and packages `papers-bundle.tar.gz` with `SHA256SUMS`. | Active |
| `lint-all` | Runs the full PR verification suite (formatting, single-pass batched static & AST checks across profiles, ADRs, file lengths, workflows, crate docs, AST clones, API drift, traceability, entropy, and policy audit, plus host & WASM clippy, rustdoc, doctests, markdown links, and OSV vulnerabilities). | Active |
| `lint-push` | Runs offline pre-push gates (`.githooks/pre-push`) across pushed commits: reverse-dependent host Clippy, web-reachable `wasm32-unknown-unknown` Clippy, and full workspace `traceability` (escalating root config edits to `--workspace`). | Active |
| `lint-staged` | Syncs the staged Git index into `<target_dir>/lint-staged-snapshot` and runs offline pre-commit gates (`.githooks/pre-commit`) scoped to staged files and directly touched crates: staged `rustfmt`, scoped `batch-staged` (staged ADRs, touched-crate profiles/docs, staged `.rs` clones/citations/entropy, staged `.md` fences, touched-crate policy audit), single `.github/` workflow linting, `.md` or deletion/rename link scans, and `direct_crates` host Clippy (plus `wasm32` Clippy when `math_explorer_gui` or `wasm32` sources change) with root config escalation. | Active |
| `lint-workflows` | Validates GitHub Actions workflow definitions and permissions. | Active |
| `smoke-examples` | Discovers every `[[example]]` target in `math_explorer/Cargo.toml` via `unified_verification list-examples`, builds all examples once, and runs each binary in an isolated temporary directory with `MATH_EXPLORER_SMOKE=1` and bounded timeout enforcement. | Active |
| `test-features` | Exercises isolated Cargo domain feature flags (`pure_math`, `physics`, `core-only`, etc.), running `math_explorer` unit/integration tests, doctests, and `--benches` plus `math_explorer_gui` `--bins --tests` type-checks and `-D warnings` Clippy per feature leg. | Active |
| `test-policy-audit` | Audits workspace crates for unauthorized lint/dead-code/`missing_docs` bypasses and unused dependencies. | Active |
| `test-wasm` | Discovers and executes `wasm32-unknown-unknown` unit and browser tests across all compatible workspace crates (anchored to the workspace root), builds `math_explorer_gui/dist` (unless `--skip-build` is passed), verifies the 4 MB bundle budget, and runs headless Chromium CSP & fault-injection smoke checks (`--screenshot-dir <dir>` supported). | Active |
| `traceability` | Verifies academic paper citation bindings and `#[verified]` coverage. | Active |
| `verify-records` | Verifies architectural decision records (`--staged` supported), enforcing `YYYYMMDD-slug.md` naming with valid dates and unique slugs for new ADRs, freezing legacy `NNNN-slug.md` prefixes, requiring `# ` and `## Status` headings, and checking ACMR updates on core changes. | Active |
| `verify-suite` | Runs the full coverage-instrumented high-integrity verification suite. | Active |

### Maintenance & Onboarding Tasks

| Command | Operational Role | Status |
|---|---|---|
| `regenerate-baseline` | Regenerates `crates/unified_verification/api_baseline.json`. | Maintenance |
| `setup` | Configures `.githooks/pre-commit` and `.githooks/pre-push` (`0755` executable permissions) first, then (unless `--hooks-only`) adds `wasm32-unknown-unknown`, checks `wasm-bindgen` `0.2.130`, `trunk`, `brotli`, and Playwright Chromium prerequisites (warning by default or enforcing with `--web`), and runs `cargo check --workspace` (unless `--check-only`). | Maintenance |

---

## 2. Developer Instructions

### Common Developer Workflows

Run `xtask` commands from the workspace root using `cargo run -p xtask -- <command>`:

```bash
# Configure local .githooks (.githooks/pre-commit and .githooks/pre-push) first, probe optional browser tools, and run cargo check --workspace
cargo run -p xtask -- setup

# Configure .githooks/pre-commit and .githooks/pre-push only (instant, skips rustup/browser probes and cargo check)
cargo run -p xtask -- setup --hooks-only

# Configure .githooks and probe toolchains without running cargo check --workspace
cargo run -p xtask -- setup --check-only

# Enforce WebAssembly & Playwright browser prerequisites (exits 1 with install fixes if missing)
cargo run -p xtask -- setup --web

# Run fast scoped pre-commit checks on staged changes
cargo run -p xtask -- lint-staged

# Run offline pre-push reverse-dependent Clippy and full traceability checks
cargo run -p xtask -- lint-push

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

`xtask` delegates AST and repository policy analysis to `unified_verification` while coordinating Cargo subcommands, `.githooks/pre-commit`, `.githooks/pre-push`, and `.github/workflows/ci.yml` jobs across Linux, macOS, Windows, and WebAssembly targets.
