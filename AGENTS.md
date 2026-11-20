# OxidizeMath Agent & Developer Verification Guide

This guide documents repository conventions, local Cargo build and test commands, domain feature matrix checks, WebAssembly bundle verification, workflow linting, and release packaging checks for contributors and automated coding agents working in the OxidizeMath workspace. See [README.md](README.md) for an architectural overview, [CONTRIBUTING.md](CONTRIBUTING.md) for engineering and NASA Power of 10 safety standards (consolidated in `CONTRIBUTING.md` per [ADR 0005](docs/adr/0005-active-documentation-standard.md)), and [docs/README.md](docs/README.md) for the documentation index.

## 1. Architecture, Policy & Documentation Rules

All Markdown files in the repository (outside excluded build directories such as `target`, `dist`, and `artifacts`) are compiled by `crates/markdown_tests/build.rs` via `rustdoc`.

- Always tag non-Rust code blocks with `bash`, `yaml`, or `text` fences so `cargo test --doc` does not attempt to compile shell commands, workflow YAML, or plain text snippets as Rust doctests.
- Keep core engineering safety rules (such as the NASA Power of 10 standards) in [CONTRIBUTING.md](CONTRIBUTING.md) rather than duplicating them here.
- Ensure all relative Markdown links resolve to existing files in the repository.
- Keep every Rust source file at or below **500 lines** (`check-file-lengths`) and keep functions within the **60-line** Clippy ceiling configured in `.clippy.toml`.
- Do not introduce unwhitelisted `dead_code`, `unused`, or `warnings` lint suppressions (`allow`, `expect`, or `cfg_attr`). Zero `dead_code` bypass entries are permitted per [ADR 0019](docs/adr/0019-purge-dead-code-bypasses.md); gate test-only helpers with `#[cfg(test)]` or delete unreferenced items instead.
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

Run workspace builds and the full pre-commit verification suite aligned with [CONTRIBUTING.md](CONTRIBUTING.md) and [crates/unified_verification/README.md](crates/unified_verification/README.md) via `xtask`:

```bash
# Type-check and build all workspace members
cargo check --workspace
cargo build --workspace --all-features

# Run the full verification suite (formatting, Cargo profiles, AST code duplicates, public API drift,
# staged ADR verification, file lengths, host & WASM clippy, rustdoc, doctests, markdown links,
# workflow lints, crate docs, traceability, entropy, vulnerabilities, and policy audit)
cargo run -p xtask -- lint-all
```

Before opening or updating a pull request, individual quality gates defined in [.github/workflows/ci.yml](.github/workflows/ci.yml) can also be run directly:

```bash
# Format and lint host and WebAssembly targets with Clippy (warnings denied)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --lib --bins --tests --target wasm32-unknown-unknown --all-features -- -D warnings

# Check Cargo release profile invariants (overflow-checks = true and zero member [profile.release] overrides)
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

# Audit AST attributes for unauthorized dead_code/unused/warnings bypasses and unused dependencies
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
cargo test --workspace --lib --tests
cargo check --workspace --target wasm32-unknown-unknown --all-features
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features
```

### Workflow `run` Script Expression Policy

`lint-workflows` rejects `${{ github.* }}` expressions interpolated directly inside `jobs.<id>.steps[*].run` shell scripts (such as `${{ github.ref_name }}`, `${{ github.head_ref }}`, or `${{ github.event.* }}`) to prevent script and argument injection. Always bind `github` context values through a step `env:` block and validate untrusted inputs in `bash` before passing them to CLI tools:

```text
env:
  RELEASE_TAG: ${{ github.ref_name }}
run: |
  # Reference "$RELEASE_TAG" safely as a shell variable after validation
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

In [.github/workflows/ci.yml](.github/workflows/ci.yml), the `Web/WASM` leg of `build-and-test` compiles the web application once with `trunk build --release`, pre-compresses assets, enforces the 3 MB compressed bundle budget, generates `SHA256SUMS`, and uploads the GitHub Pages artifact on `refs/heads/main` before `deploy-pages` publishes it.

```bash
# 1. Install the WebAssembly Rust target, Trunk, and compression utilities (Ubuntu / Debian)
rustup target add wasm32-unknown-unknown
sudo apt-get update
sudo apt-get install -y binaryen brotli
cargo binstall --no-confirm trunk

# 2. Type-check and lint workspace crates and tests for WebAssembly
cargo check --workspace --target wasm32-unknown-unknown --all-features --verbose
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features --verbose
cargo clippy --workspace --lib --bins --tests --target wasm32-unknown-unknown --all-features -- -D warnings

# 3. Build release WebAssembly distribution, compress assets, and verify the 3 MB size budget
(cd math_explorer_gui && trunk build --release)
bash scripts/compress_dist.sh math_explorer_gui/dist
bash scripts/verify_wasm_budget.sh math_explorer_gui/dist

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

Pull requests automatically execute the Linux Debian (`deb`) installer build in the `package` job of [.github/workflows/ci.yml](.github/workflows/ci.yml), while `main` pushes, `v*` tag pushes, and `workflow_dispatch` runs build all three desktop installer targets (`deb`, `dmg`, and `wix`).

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
