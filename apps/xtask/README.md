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
| `lint-all` | Runs formatting, profile, AST duplicate, API drift, staged ADR, file-length, clippy, doc, doctest, markdown-link, workflow, traceability, entropy, vulnerability, and policy audit checks. | Active |
| `lint-workflows` | Validates CI workflow definitions. | Active |
| `test-features` | Runs tests for various cargo features. | Active |
| `test-policy-audit` | Audits workspace crates for unauthorized dead-code bypasses and unused dependencies. | Active |
| `traceability` | Generates and checks the traceability report. | Active |
| `verify-records` | Verifies architectural decision records (`--staged` supported). | Active |
| `verify-suite` | Runs the high-integrity verification suite. | Active |

### Legacy Utility Tasks
These commands are used for manual repository maintenance or legacy workflows.

| Command | Operational Role | Status |
|---|---|---|
| `compile-papers` | Compiles academic papers using Tectonic. | Legacy |
| `regenerate-baseline` | Regenerates the public API baseline. | Legacy |
| `setup` | Configures version-controlled git hooks and runs initial checks. | Legacy |
