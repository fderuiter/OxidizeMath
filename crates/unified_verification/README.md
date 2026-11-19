# Unified Verification

Unified verification CLI and quality gate enforcing architecture records, file lengths, and code traceability.

## 1. Overview & Architecture

The `unified_verification` crate serves as the primary verification gatekeeper and quality engine for the OxidizeMath ecosystem. It enforces strict architectural constraints, documentation standards, and code hygiene policies across all workspace crates.

By consolidating compliance checks into a unified command-line tool, developer workflows remain streamlined while safeguarding repository invariants such as the NASA Power of 10 safety rules, architecture decision tracking, and file length boundaries.

### Module Structure

- **`verify-records`**: Audits git commit diffs to enforce that modifying core logic (`crates/` or `math_explorer/`) is accompanied by creating or updating an Architectural Decision Record in `docs/adr/`.
- **`traceability`**: Performs static AST scans across Rust source files to verify literature citation bindings (`papers/`) and module verification attributes.
- **`check-file-lengths`**: Enforces the 500-line ceiling per source file across workspace members to maintain modularity.
- **`test-policy-audit`**: Detects unauthorized bypasses such as unapproved `#[allow(dead_code)]` or `#[allow(unused)]` attributes.
- **`check-entropy`**: Audits sources against the No-Entropy policy, ensuring that all random number generation relies on explicit seeds via `OxidizeRng`.
- **`check-vulnerabilities`**: Audits workspace dependencies against known vulnerabilities via OSV database.
- **`check-duplicates` / `check-staged-duplicates`**: AST-level code clone detector identifying duplicated logic across crates.
- **`check-api-drift` / `regenerate-baseline`**: Validates public API surface against baseline definitions to prevent accidental breaking changes.
- **`lint-workflows`**: Validates GitHub Actions workflow configurations, enforcing root and job-level read-only permissions (with scoped deployment exceptions in `verification_whitelist.toml`), concurrency controls, job timeouts, and commit-pinned actions.

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
# Verify architecture records for recent changes
cargo run -p unified_verification --release -- verify-records

# Audit source file length limits
cargo run -p unified_verification --release -- check-file-lengths
```

---

## 3. Ecosystem Integration

`unified_verification` is executed by `.githooks/pre-commit`, by `cargo run -p xtask -- lint-all`, and within `.github/workflows/ci.yml`, serving as the definitive verification authority for all proposed changes.
