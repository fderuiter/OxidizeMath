# OxidizeMath Agent & Developer Verification Guide

This guide documents repository conventions, local build and verification gates, and release packaging checks for contributors and automated coding agents working in the OxidizeMath workspace. See [README.md](README.md) for an architectural overview, [CONTRIBUTING.md](CONTRIBUTING.md) for engineering standards, and [docs/README.md](docs/README.md) for the documentation index.

## 1. Architecture, Policy & Documentation Rules

All Markdown files in the repository (outside excluded build directories such as `target`, `dist`, and `artifacts`) are compiled by `crates/markdown_tests/build.rs` via `rustdoc`.

- Always tag non-Rust code blocks with `bash` or `text` fences so `cargo test --doc` does not attempt to compile shell commands or plain text snippets as Rust doctests.
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

## 2. CI Pre-Flight & Quality Gates

Before opening or updating a pull request, run the full workspace verification suite aligned with [CONTRIBUTING.md](CONTRIBUTING.md) and [.github/workflows/ci.yml](.github/workflows/ci.yml) (all NASA Power of 10 coding and safety standards remain authoritative in [CONTRIBUTING.md](CONTRIBUTING.md)):

```bash
# Run all workspace quality gates in one command (format, clippy, duplicates, file-lengths, docs, doctests, markdown links, workflow lints, doc audit, traceability, entropy, vulnerabilities, policy audit)
cargo run -p xtask -- lint-all
```

You can also invoke individual CI pre-flight checks directly while iterating:

```bash
# Format and lint host and WebAssembly targets with Clippy (warnings denied, 60-line function limit in .clippy.toml)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --lib --bins --tests --target wasm32-unknown-unknown --all-features -- -D warnings

# Lint GitHub Actions workflows (.github/workflows/ci.yml and .github/workflows/pages.yml):
# enforces global permissions: read-all, scoped job write exceptions in verification_whitelist.toml,
# global concurrency, positive timeout-minutes, and 40-char hexadecimal commit SHA action pinning
cargo run -p unified_verification --release -- lint-workflows

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

# Verify architectural decision records when core crates change
BASE_SHA="$(git merge-base HEAD main)" HEAD_SHA="HEAD" cargo run -p unified_verification --release -- verify-records

# Native unit/integration tests and WebAssembly target checks
cargo test --workspace --lib --tests
cargo check --workspace --target wasm32-unknown-unknown --all-features
cargo check --tests --workspace --target wasm32-unknown-unknown --all-features
```

## 3. Local Release Packaging Checks

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

## 4. Local Paper Bundling & Split Checksum Verification

The `compile-papers` job in [.github/workflows/ci.yml](.github/workflows/ci.yml) downloads the fresh `Verification_Certificate.json` artifact produced by `build-and-test`, compiles all 28 LaTeX papers via [apps/xtask/src/papers.rs](apps/xtask/src/papers.rs), and emits split SHA-256 manifests so both downloaded release archives and extracted paper bundles verify cleanly.

### Prerequisites (Ubuntu / Debian)

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra texlive-science texlive-binaries
```

### Compile Papers & Verify Split Manifests

```bash
# Optional: regenerate a fresh Verification_Certificate.json locally (requires cargo-llvm-cov)
# cargo run -p unified_verification --release -- verify-suite

# Compile all 28 PDFs, index.html, papers-bundle.tar.gz, SHA256SUMS, and SHA256SUMS-papers
cargo run -p xtask -- compile-papers

# 1. Verify outer release manifest (hashes only papers-bundle.tar.gz)
(
  cd papers/output
  sha256sum -c SHA256SUMS-papers
)

# 2. Extract papers-bundle.tar.gz into a clean directory and verify internal SHA256SUMS
#    (hashes all 28 .pdf files, index.html, and Verification_Certificate.json)
(
  verify_dir="$(mktemp -d)"
  trap 'rm -rf "$verify_dir"' EXIT
  tar -xzf papers/output/papers-bundle.tar.gz -C "$verify_dir"
  cd "$verify_dir"
  sha256sum -c SHA256SUMS
)
```

### Consolidated Release Manifest & SLSA Provenance Rules

```text
Release asset & attestation invariants (.github/workflows/ci.yml & .github/workflows/pages.yml):
- release-assets/SHA256SUMS hashes only deliverable binaries (.deb, .dmg, .msi/.exe),
  papers-bundle.tar.gz, and fresh Verification_Certificate.json (omitting all SHA256SUMS* files).
- Prepare release assets fails with exit code 1 if any expected platform installer,
  papers-bundle.tar.gz, Verification_Certificate.json, or SHA256SUMS-papers is missing.
- publish-release (ci.yml) and deploy (pages.yml) generate GitHub SLSA v1.0 provenance
  attestations via actions/attest-build-provenance; build jobs remain strictly read-only.
```

