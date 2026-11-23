# Unified Verification

Unified verification CLI and quality gate enforcing architecture records, file lengths, and code traceability.

## 1. Overview & Architecture

The `unified_verification` crate serves as the primary verification gatekeeper and quality engine for the OxidizeMath ecosystem. It enforces strict architectural constraints, documentation standards, and code hygiene policies across all workspace crates.

By consolidating compliance checks into a unified command-line tool, developer workflows remain streamlined while safeguarding repository invariants such as the NASA Power of 10 safety rules, architecture decision tracking, and file length boundaries.

### Module Structure

- **`batch-ast` / `batch-static` / `batch-staged`**: In-process batch verification modes sharing a single `oxidize_core::traceability::AstCache` so each unique `.rs` file is parsed at most once with `syn::parse_file` across clone detection, API drift (`batch-ast`/`batch-static`), traceability, entropy scanning, and policy auditing, alongside release profile, file length, staged ADR (`batch-staged`), and crate documentation checks in `batch-static` and `batch-staged`.
- **`verify-records`**: Audits git commit ranges (resolving `git merge-base` against `origin/main` or `main` when `BASE_SHA` is unset or zero and failing closed on invalid explicit `BASE_SHA` values) or staged blobs (`--staged` via `git show :<path>`), requiring every ADR in `docs/adr/` to match `NNNN-slug.md`, include `# ` and `## Status` headings outside code fences, use unique 4-digit IDs (allowing legacy collisions listed under `[adr_verification]` in `verification_whitelist.toml`), and accompany non-comment code modifications in `crates/` or `math_explorer/` via `--diff-filter=ACMR`.
- **`traceability`**: Invokes `oxidize_core::traceability::TraceabilityEngine` directly in-process as a library to verify literature citation bindings (`papers/`) and module verification attributes without spawning a Cargo subprocess.
- **`check-file-lengths`**: Enforces the 500-line ceiling per source file across workspace members to maintain modularity.
- **`test-policy-audit`**: Enforces `[lints] workspace = true` inheritance in first-party `Cargo.toml` manifests, detects unauthorized `#[allow(dead_code)]`, `#[allow(unused)]`, or `#[allow(warnings)]` bypasses, blocks Clippy group suppressions (`all`, `pedantic`, `nursery`, `restriction`, `complexity`, `correctness`, `suspicious`, `style`, `perf`, `cargo`), inner/`mod`/`impl`/`trait` `too_many_lines` and `cognitive_complexity` bypasses, `macro_rules!` Clippy suppressions, and stale `too_many_lines` attributes on functions of 60 lines or fewer, and audits `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`, and `[target.*]` tables against AST, attribute/derive, and compiled `rustdoc` code blocks.
- **`check-entropy`**: Audits Rust AST and macro token streams, configuration files, `.githooks`, dotfiles (such as `.env`), and extensionless UTF-8 files (such as `Dockerfile`) against leaked secrets and the No-Entropy policy, ensuring that all random number generation relies on explicit seeds via `OxidizeRng`.
- **`check-vulnerabilities`**: Audits workspace dependencies against known vulnerabilities via OSV database.
- **`check-duplicates` / `check-staged-duplicates`**: AST-level code clone detector identifying duplicated logic across crates.
- **`check-api-drift` / `regenerate-baseline`**: Validates public API surface against baseline definitions to prevent accidental breaking changes.
- **`check-profiles`**: Verifies workspace `[profile.release]` `overflow-checks = true` enforcement, rejects member-level `[profile.release]` overrides, validates `[workspace.package].version` SemVer/WiX bounds (rejecting `+build` metadata), enforces `version.workspace = true` inheritance across all 11 release crates (preserving `0.34.1` on vendored `egui_plot` and `0.1.0` on internal foundation/tooling crates), forbids `[package.metadata.packager].version` overrides, validates `README.md` version parity, and enforces `[lints] workspace = true` inheritance across member manifests (allowing vendored `egui_plot` only with explicit `[lints.rust]` and `[lints.clippy]` tables).
- **`check-release-tag`**: Validates `refs/tags/v*` and `RELEASE_TAG` parity against `[workspace.package].version`, rejects `+build` metadata, maps `alpha.N` (`1000 + N`), `beta.N` (`10000 + N`), and `rc.N` (`20000 + N`) pre-release suffixes within WiX `<= 65535` revision bounds, and emits `is_prerelease` and `wix_version` outputs for CI packaging and release publishing.
- **`check-docs`**: Audits crate READMEs, facade import hygiene, Markdown Cargo CLI target/feature flags, consumer `[dependencies]` feature blocks, `math_explorer/examples` 3-way catalog parity, TeXLive paper & `--all-features` rustdoc gate alignment, and canonical `TheoryDescribable` / `InteractiveTool` / `ExplorerTab` trait contracts.
- **`lint-workflows`**: Validates GitHub Actions workflow configurations (including full checks on local reusable workflows) and `.github/actions` composite/Docker manifests, enforcing root and job-level read-only permissions (with scoped deployment exceptions in `verification_whitelist.toml`), event-scoped root and job-level concurrency controls across `on.push.branches`, job timeouts (exempting `jobs.<id>.uses` reusable workflow callers and rejecting schema-invalid `timeout-minutes` on `uses` jobs), commit-pinned actions, and rejecting direct `${{ github.* }}`, `${{ inputs.* }}`, and non-static `${{ env.* }}` expressions across workflow and composite `run` scripts.

---

## 2. Developer Instructions

### Manifest Parsing and Member Inspection

The verification suite inspects workspace members using structured manifest parsers:

```rust
use oxidize_core::workspace::parse_workspace_members;

let manifest_toml = r#"
[workspace]
members = [
    "crates/oxidize_core",
    "crates/unified_verification",
]
"#;

let members = parse_workspace_members(manifest_toml).expect("Valid workspace manifest");
assert_eq!(members.len(), 2);
assert_eq!(members[0], "crates/oxidize_core");
assert_eq!(members[1], "crates/unified_verification");
```

### Running Verification Gates

Execute specific verification commands from the repository root:

```bash
# Lint GitHub Actions workflows for permissions, pinning, and safe run expressions
cargo run -p unified_verification --release -- lint-workflows

# Verify architecture records for recent changes
cargo run -p unified_verification --release -- verify-records

# Audit source file length limits
cargo run -p unified_verification --release -- check-file-lengths
```

---

## 3. Ecosystem Integration

`unified_verification` is executed in `batch-staged` mode by `.githooks/pre-commit` (`cargo run -p xtask -- lint-staged`), in `batch-static` mode by `cargo run -p xtask -- lint-all`, and within `.github/workflows/ci.yml`, serving as the definitive verification authority for all proposed changes.
