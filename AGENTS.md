# OxidizeMath Agent & Developer Verification Guide

This guide documents repository conventions, local Cargo build and test commands, domain feature matrix checks, WebAssembly bundle verification, workflow linting, release packaging checks, and academic paper compilation checks for contributors and automated coding agents working in the OxidizeMath workspace. See [README.md](README.md) for an architectural overview, [CONTRIBUTING.md](CONTRIBUTING.md) for engineering and NASA Power of 10 safety standards (consolidated in `CONTRIBUTING.md` per [ADR 0005](docs/adr/0005-active-documentation-standard.md)), and [docs/README.md](docs/README.md) for the documentation index.

## 1. Architecture, Policy & Documentation Rules

All Markdown files in the repository (outside excluded build directories such as `target`, `dist`, and `artifacts`) are compiled by `crates/markdown_tests/build.rs` via `rustdoc`.

- Always tag non-Rust code blocks with `bash`, `yaml`, or `text` fences so `cargo test --doc` does not attempt to compile shell commands, workflow YAML, or plain text snippets as Rust doctests.
- Keep core engineering safety rules (such as the NASA Power of 10 standards) in [CONTRIBUTING.md](CONTRIBUTING.md) rather than duplicating them here.
- Ensure all relative Markdown links resolve to existing files in the repository.
- Keep every Rust source file at or below **500 lines** (`check-file-lengths`), keep functions within the **60-line** Clippy ceiling configured in `.clippy.toml`, and declare `[lints] workspace = true` in every first-party `Cargo.toml` manifest (`check-profiles` / `test-policy-audit`).
- Do not introduce unwhitelisted `dead_code`, `unused`, or `warnings` lint suppressions (`allow`, `expect`, or `cfg_attr`), Clippy lint-group suppressions (`clippy::all`, `pedantic`, `nursery`, `restriction`, `complexity`, `correctness`, `suspicious`, `style`, `perf`, or `cargo`), crate/module/`impl`/`trait` complexity bypasses (`too_many_lines` or `cognitive_complexity`), `macro_rules!` Clippy suppressions, or stale `too_many_lines` attributes on functions of **60 lines or fewer** per [ADR 0019](docs/adr/0019-purge-dead-code-bypasses.md), [ADR 0027](docs/adr/0027-enforce-workspace-clippy-lints.md), and [ADR 0036](docs/adr/0036-harden-ast-and-doctest-policy-audits.md). Every dependency across `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`, and `[target.*]` tables must be referenced in Rust AST/attributes, derive macros, or compiled `rustdoc` code blocks, or carry an explicit `[[unused_dependencies]]` entry (`crate`, `dependency`, `reason`) in [verification_whitelist.toml](verification_whitelist.toml). Gate test-only helpers with `#[cfg(test)]` or decompose large functions instead.
- Modifying core mathematical or runtime crates (`crates/` excluding `crates/unified_verification`, or `math_explorer/`) requires creating or updating an Architectural Decision Record under `docs/adr/`.

```text
Markdown verification gates:
- cargo run -p unified_verification --release -- check-docs
- RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
- cargo test --doc --workspace --all-features
- cargo run -p xtask -- check-markdown-links
```

## 2. Local Cargo Builds & CI Quality Gates

Rust jobs in [.github/workflows/ci.yml](.github/workflows/ci.yml) restore Cargo registry sources and profile-scoped `target/` caches via [.github/actions/cargo-cache/action.yml](.github/actions/cargo-cache/action.yml) (referencing `./.github/actions/cargo-cache` without caching mutable CLI binaries).

Run workspace builds, fast diff-scoped pre-commit checks (`lint-staged`), and the full PR verification suite (`lint-all`) aligned with [CONTRIBUTING.md](CONTRIBUTING.md) and [crates/unified_verification/README.md](crates/unified_verification/README.md) via `xtask` (`xtask` builds `unified_verification` with Cargo's default `dev` profile locally to skip release fat LTO):

```bash
# Type-check and build all workspace members
cargo check --workspace
cargo build --workspace --all-features

# Run fast diff-scoped pre-commit checks on staged paths (executed automatically by .githooks/pre-commit;
# checks staged formatting, file lengths, entropy, policy audit, staged duplicates, .github/-gated workflows,
# .md-gated links, and targeted host/wasm32 Clippy without OSV network calls, API drift, or ADR checks)
cargo run -p xtask -- lint-staged

# Run the full PR verification suite before opening or updating a pull request (formatting, batched static
# & single-parse AST checks across profiles, ADRs, file lengths, workflows, crate docs, AST clones, API drift,
# in-process traceability, entropy, and policy audit, plus host & WASM clippy, rustdoc, doctests, links, and OSV)
cargo run -p xtask -- lint-all
```

Before opening or updating a pull request, individual or batched quality gates defined in [.github/workflows/ci.yml](.github/workflows/ci.yml) can also be run directly:

```bash
# Run all five syn AST passes (duplicates, API drift, traceability, entropy, policy audit) in one parse per .rs file
cargo run -p unified_verification -- batch-ast

# Format and lint host and WebAssembly targets with Clippy (warnings denied)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --lib --bins --tests --target wasm32-unknown-unknown --all-features -- -D warnings

# Check Cargo release profile invariants (overflow-checks = true, zero member [profile.release] overrides, and [lints] workspace = true inheritance)
cargo run -p unified_verification --release -- check-profiles
cargo run -p xtask -- check-profiles

# Run AST-level code clone detection across first-party crates
cargo run -p unified_verification --release -- check-duplicates
cargo run -p xtask -- check-duplicates

# Validate reachable public API signatures against crates/unified_verification/api_baseline.json
cargo run -p unified_verification --release -- check-api-drift
cargo run -p xtask -- check-api-drift

# When intentionally changing public APIs, regenerate the baseline and record an ADR under docs/adr/
cargo run -p unified_verification --release -- regenerate-baseline
cargo run -p xtask -- regenerate-baseline

# Lint GitHub Actions workflows (permissions, concurrency, timeouts, 40-char SHA or ./ local action pins, and safe run expressions)
cargo run -p unified_verification --release -- lint-workflows

# Run workflow linter unit tests directly
cargo test -p unified_verification workflow_linter

# Audit member [lints] inheritance, AST/macro Clippy & dead_code bypasses, and Cargo/doctest dependency usage
cargo run -p xtask -- test-policy-audit
cargo run -p unified_verification --release -- test-policy-audit

# Verify crate documentation and workspace rustdoc builds
cargo run -p unified_verification --release -- check-docs
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo test --doc --workspace --all-features

# Validate relative Markdown links, file lengths (500-line ceiling), entropy/secrets, and traceability
cargo run -p xtask -- check-markdown-links
cargo run -p unified_verification --release -- check-file-lengths
cargo run -p unified_verification --release -- check-entropy
cargo run -p unified_verification --release -- traceability

# Verify architectural decision records across staged index edits or commit ranges when core crates change
cargo run -p unified_verification --release -- verify-records --staged
BASE_SHA="$(git merge-base HEAD main)" HEAD_SHA="HEAD" cargo run -p unified_verification --release -- verify-records

# Run the full high-integrity verified suite (including cargo llvm-cov coverage and Verification_Certificate.json)
cargo run -p unified_verification --release -- verify-suite
cargo run -p xtask -- verify-suite

# Native unit/integration tests and WebAssembly target checks
cargo test --workspace --lib --tests --all-features
cargo check --workspace --target wasm32-unknown-unknown --all-features
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features
```

### Workflow `run` Script Expression & Reusable Caller Policy

`lint-workflows` audits `.github/workflows` (including full permission, concurrency, timeout, pinning, and `run` expression checks on local reusable workflows) and `.github/actions` manifests. It rejects `${{ github.* }}`, `${{ inputs.* }}`, and non-static or unmapped `${{ env.* }}` expressions interpolated directly inside workflow `jobs.<id>.steps[*].run` and composite action `runs.steps[*].run` shell scripts (such as `${{ github.ref_name }}`, `${{ github.head_ref }}`, `${{ github.event.* }}`, or `${{ inputs.target }}`) to prevent script and argument injection. Reusable workflow caller jobs (`jobs.<id>.uses`) are exempt from `timeout-minutes` and must omit `timeout-minutes` at the caller level (configure `timeout-minutes` on concrete jobs inside the called workflow instead). Always bind `github` and `inputs` context values through a step `env:` block and validate untrusted inputs in `bash` before passing them to CLI tools:

```text
env:
  RELEASE_TAG: ${{ github.ref_name }}
  INPUT_TARGET: ${{ inputs.target }}
run: |
  # Reference "$RELEASE_TAG" and "$INPUT_TARGET" safely as shell variables after validation
```

## 3. Isolated Domain Feature Matrix (`test-features`)

Both `deploy-pages` and `publish-release` in [.github/workflows/ci.yml](.github/workflows/ci.yml) wait for `feature-matrix`, because [math_explorer_gui/Cargo.toml](math_explorer_gui/Cargo.toml) enables all seven scientific domains by default. Verify isolated domain feature compilation and tests locally via `xtask`:

```bash
# Run core-only (--no-default-features), all seven isolated domains, and --all-features sequentially
cargo run -p xtask -- test-features

# Run a single isolated domain suite (pure_math, applied, ai, biology, climate, epidemiology, or physics)
cargo run -p xtask -- test-features physics

# Run core-only (--no-default-features) without optional domain crates
cargo run -p xtask -- test-features core-only
```

## 4. Local WebAssembly (`Web/WASM`) & Pages Bundle Checks

In [.github/workflows/ci.yml](.github/workflows/ci.yml), the `Web/WASM` leg of `build-and-test` compiles the web application once with `trunk build --release` (which runs [scripts/finalize_web_dist.sh](scripts/finalize_web_dist.sh) during `post_build` to stage precomputed traceability JSON metadata and pre-compress assets), enforces the 3 MB compressed bundle budget, executes browser `wasm32` unit tests and Playwright headless Chromium CSP/fault-injection smoke checks via `cargo run -p xtask -- test-wasm --skip-build`, generates `SHA256SUMS`, and uploads `wasm_web_dist` on `refs/heads/main` before `deploy-pages` stages and verifies all 28 `compiled_papers` PDFs, refreshes `SHA256SUMS`, and publishes the GitHub Pages artifact.

```bash
# 1. Install the WebAssembly Rust target, Trunk, wasm-bindgen-cli, compression utilities, and Playwright Chromium (Ubuntu / Debian)
rustup target add wasm32-unknown-unknown
sudo apt-get update
sudo apt-get install -y brotli
cargo binstall --no-confirm trunk wasm-bindgen-cli@0.2.130
python3 -m pip install --user playwright
python3 -m playwright install --with-deps chromium

# 2. Type-check and lint workspace crates and tests for WebAssembly
cargo check --workspace --target wasm32-unknown-unknown --all-features --verbose
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features --verbose
cargo clippy --workspace --lib --bins --tests --target wasm32-unknown-unknown --all-features -- -D warnings

# 3. Run wasm32 browser unit tests, build math_explorer_gui/dist, verify the 3 MB budget, and run Playwright Chromium CSP & fault-injection smoke checks
cargo run -p xtask -- test-wasm

# 4. Generate and verify the Pages SHA256SUMS manifest
(
  cd math_explorer_gui/dist
  shopt -s nullglob
  shopt -u dotglob
  files=()
  for f in *; do
    if [ -f "$f" ] && [ "$f" != "SHA256SUMS" ]; then
      files+=("$f")
    fi
  done
  test "${#files[@]}" -gt 0
  sha256sum "${files[@]}" > SHA256SUMS
  sha256sum -c SHA256SUMS
)
```

## 5. Local Release Packaging Checks

The `package` job in [.github/workflows/ci.yml](.github/workflows/ci.yml) defines a static three-OS matrix (`ubuntu-latest`/`deb`, `macos-latest`/`dmg`, and `windows-latest`/`wix`) so every workflow event posts deterministic `Package Installers` check runs across all three operating systems:

- **Linux (`deb`)**: Runs on every `pull_request`, `merge_group`, `push`, and `workflow_dispatch` event.
- **macOS (`dmg`) & Windows (`wix`)**: Run natively on `merge_group`, `main` pushes, `v*` tag pushes, `workflow_dispatch`, and any `pull_request` where `pre-flight` sets `installer_changed=true` (diffs touching `math_explorer_gui/**`, [Cargo.toml](Cargo.toml), `Cargo.lock`, or [.github/workflows/ci.yml](.github/workflows/ci.yml), or when git diff falls back). On unrelated pull requests (`installer_changed=false`), the `macos-latest` and `windows-latest` legs route to `ubuntu-latest` and skip build, manifest, and upload steps via `if:`.

### Prerequisites (Ubuntu / Debian)

Install a C toolchain (if not already present) along with `cargo-packager` `0.11.8` using `cargo-binstall` without falling back to source compilation (`rfd` uses `xdg-portal` in [Cargo.toml](Cargo.toml), so `libgtk-3-dev` and `libxdo-dev` are not required):

```bash
sudo apt-get update
sudo apt-get install -y build-essential

# Install pinned cargo-packager prebuilt binary (fails fast if unavailable)
cargo binstall --no-confirm --disable-strategies compile cargo-packager@0.11.8
```

### Build Linux `.deb` Installer & SHA-256 Manifest

[math_explorer_gui/Cargo.toml](math_explorer_gui/Cargo.toml) configures `before-packaging-command = "cargo build --release -p math_explorer_gui"` and `out-dir = "dist"` under `[package.metadata.packager]`, so `cargo packager` compiles the `--release` binary automatically and writes installers to `math_explorer_gui/dist` alongside a hidden `.cargo-packager` metadata directory. Because `product-name = "Math Explorer"` introduces a space in macOS `.dmg` filenames, normalize spaces to underscores on `*.dmg` files before generating `SHA256SUMS` so `gh release create` does not rewrite spaces to dots and invalidate `SHA256SUMS-dmg`:

```bash
# 1. Build release GUI binary (via before-packaging-command) and package Debian installer
cargo packager --release -p math_explorer_gui -f deb

# 2. Inspect .deb control metadata and simulate installation
dpkg-deb -I math_explorer_gui/dist/*.deb
sudo dpkg --simulate -i math_explorer_gui/dist/*.deb

# 3. Normalize space-containing .dmg filenames and verify SHA256SUMS (skipping .cargo-packager)
(
  cd math_explorer_gui/dist
  shopt -s nullglob
  shopt -u dotglob
  for dmg in *.dmg; do
    if [ -f "$dmg" ] && [[ "$dmg" == *" "* ]]; then
      normalized="${dmg// /_}"
      test ! -e "$normalized"
      mv -- "$dmg" "$normalized"
    fi
  done
  files=()
  for f in *.deb *.dmg *.msi *.exe *.wix; do
    if [ -f "$f" ] && [ "$f" != "SHA256SUMS" ]; then
      files+=("$f")
    fi
  done
  test "${#files[@]}" -gt 0
  sha256sum "${files[@]}" > SHA256SUMS
  sha256sum -c SHA256SUMS
)
```

Clean up local `math_explorer_gui/dist` build outputs before committing so untracked web or installer binaries are not left in the workspace:

```bash
rm -rf math_explorer_gui/dist
```

## 6. Local Academic Paper Compilation Checks

Because `traceability` only checks `.tex` filename presence in `papers/` and `.githooks/pre-commit` omits LaTeX builds so contributors do not need TeXLive for unrelated Rust commits, [.github/workflows/ci.yml](.github/workflows/ci.yml) enforces full paper compilation in the `compile-papers` (`Compile Academic Papers`) job whenever `pre-flight` sets `papers_changed=true` on `pull_request` (diffs touching `papers/**`, `apps/xtask/src/papers*`, [traceability.toml](traceability.toml), or [.github/workflows/ci.yml](.github/workflows/ci.yml), or on git diff fallback) and unconditionally on `merge_group`, `push`, and `workflow_dispatch`.

### Prerequisites (Ubuntu / Debian)

Install the TeXLive packages used by `compile-papers` in [.github/workflows/ci.yml](.github/workflows/ci.yml):

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra texlive-science texlive-binaries
```

### Compile All 28 Academic Papers via `xtask`

Run the paper compiler defined in [apps/xtask/src/papers.rs](apps/xtask/src/papers.rs) to execute multi-pass `pdflatex` (`-no-shell-escape -interaction=nonstopmode -halt-on-error`) and `bibtex`, verify `.blg` and `.log` outputs for unresolved citations or undefined references across all 28 `.tex` manuscripts, generate `papers/output/index.html`, and package `papers/output/papers-bundle.tar.gz` alongside `papers/output/SHA256SUMS`:

```bash
cargo run -p xtask -- compile-papers
```

Clean up local `papers/output` build artifacts after verification so generated PDFs, auxiliary TeX logs, HTML index files, and release tarballs are not left in the workspace:

```bash
rm -rf papers/output
```
