# OxidizeMath Agent & Developer Verification Guide

This guide documents local Cargo build, test, CI verification, and release packaging checks for contributors and automated coding agents working in the OxidizeMath workspace. See [README.md](README.md) for an architectural overview and [CONTRIBUTING.md](CONTRIBUTING.md) for engineering and NASA Power of 10 safety standards (kept consolidated in `CONTRIBUTING.md` per [docs/adr/0005-active-documentation-standard.md](docs/adr/0005-active-documentation-standard.md)).

## 1. Documentation & Markdown Rules

All Markdown files in the repository (outside excluded build directories such as `target`, `dist`, and `artifacts`) are compiled by `crates/markdown_tests/build.rs` via `rustdoc`.

- Always tag non-Rust code blocks with `bash`, `yaml`, or `text` fences so `cargo test --doc` does not attempt to compile shell commands, workflow YAML, or plain text snippets as Rust doctests.
- Ensure all relative Markdown links resolve to existing files in the repository.

```text
Markdown verification gates:
- cargo run -p unified_verification --release -- check-docs
- RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
- cargo test --doc --workspace --all-features
- cargo run -p xtask -- check-markdown-links
```

## 2. Local Cargo Builds & Workspace Tests

Run workspace compilation, clippy, unit/integration tests, feature matrix checks, and WebAssembly target checks locally before submitting changes:

```bash
# Type-check and build all workspace members
cargo check --workspace
cargo build --workspace --all-features

# Run formatting and strict Clippy lints across all targets and features
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run workspace library/integration tests and feature-isolated domain tests
cargo test --workspace --lib --tests
cargo run -p xtask -- test-features

# Type-check WebAssembly target builds (matching CI and GitHub Pages builds)
cargo check --workspace --target wasm32-unknown-unknown --all-features
```

## 3. CI Pre-Flight, Workflow Lints & Quality Gates

CI jobs in [.github/workflows/ci.yml](.github/workflows/ci.yml) and [.github/workflows/pages.yml](.github/workflows/pages.yml) restore Cargo registry and profile-scoped `target/` caches via [.github/actions/cargo-cache/action.yml](.github/actions/cargo-cache/action.yml) (referencing `./.github/actions/cargo-cache` without caching mutable CLI binaries).

Before opening or updating a pull request, run the unified lint suite from [CONTRIBUTING.md](CONTRIBUTING.md) or invoke individual verification gates directly:

```bash
# Run the complete workspace quality gate suite
cargo run -p xtask -- lint-all

# Lint GitHub Actions workflows (permissions, concurrency, timeouts, 40-char SHA or ./ local action pins)
cargo run -p unified_verification --release -- lint-workflows

# Verify crate documentation and workspace rustdoc builds
cargo run -p unified_verification --release -- check-docs
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo test --doc --workspace --all-features

# Validate relative Markdown links, file lengths (500-line ceiling), entropy/secrets, and traceability
cargo run -p xtask -- check-markdown-links
cargo run -p unified_verification --release -- check-file-lengths
cargo run -p unified_verification --release -- check-entropy
cargo run -p unified_verification --release -- traceability

# Verify architectural decision records when core crates change
BASE_SHA="$(git merge-base HEAD main)" HEAD_SHA="HEAD" cargo run -p unified_verification --release -- verify-records
```

## 4. Local Release Packaging Checks

Pull requests automatically execute the Linux Debian (`deb`) installer build in the `package` job of [.github/workflows/ci.yml](.github/workflows/ci.yml), while `main` pushes, `v*` tag pushes, and `workflow_dispatch` runs build all three desktop installer targets (`deb`, `dmg`, and `wix`).

### Prerequisites (Ubuntu / Debian)

Install native GUI and packaging libraries along with `cargo-packager` `0.11.8` using `cargo-binstall` without falling back to source compilation:

```bash
sudo apt-get update
sudo apt-get install -y libgtk-3-dev libxdo-dev build-essential

# Install pinned cargo-packager prebuilt binary (fails fast if unavailable)
cargo binstall --no-confirm --disable-strategies compile cargo-packager@0.11.8
```

### Build Linux `.deb` Installer & SHA-256 Manifest

`math_explorer_gui/Cargo.toml` configures `out-dir = "dist"` under `[package.metadata.packager]`, so `cargo packager` writes installers to `math_explorer_gui/dist` alongside a hidden `.cargo-packager` metadata directory.

```bash
# 1. Build release GUI binary and package Debian installer
cargo build --release -p math_explorer_gui
cargo packager --release -p math_explorer_gui -f deb

# 2. Generate and verify SHA256SUMS for installer files only (skipping .cargo-packager)
(
  cd math_explorer_gui/dist
  shopt -s nullglob
  shopt -u dotglob
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

Clean up local `math_explorer_gui/dist` build outputs before committing so untracked installer binaries are not left in the workspace:

```bash
rm -rf math_explorer_gui/dist
```
