# 18. CI Coverage Gate Hardening and Architectural Verification

## Status
Accepted

## Context
The continuous integration and verification pipeline required automated in-job test coverage collection and strict threshold gating to prevent untested regressions from reaching main releases. Previously, coverage collection was prone to silent masking when `cargo llvm-cov` encountered errors, synthetic averaging obscured native line coverage against WASM path coverage, and matrix jobs across non-Linux targets incurred unnecessary coverage tooling overhead. Furthermore, core workspace crates required coverage enrichment and AST visitor handling to maintain high-integrity verification.

## Decision
1. **Automated In-Job Coverage Collection**:
   - Gated coverage collection, suite verification, and certificate upload to Linux native runners (`runner.os == 'Linux' && matrix.target != 'wasm32-unknown-unknown'`) in `.github/workflows/ci.yml`.
   - Installed `llvm-tools-preview` in the Rust toolchain and pinned `cargo-llvm-cov` installation using immutable action commit hashes (`taiki-e/install-action@4a76afe23f9e08d2d9550fb56e556444e8611d21`).
2. **Unified Verification Failure Handling**:
   - Updated `crates/unified_verification/src/main.rs` to replace silent error masking with stderr logging and immediate process exit (`std::process::exit(1)`) upon `cargo llvm-cov` failure.
3. **Strict Coverage Threshold Gate**:
   - Enforced an independent 50.0% minimum threshold for native line coverage in `crates/unified_verification/src/report.rs`, eliminating synthetic averaging with WASM path coverage.
   - Updated README badge generation and verification certificate metadata to reflect native coverage.
4. **Coverage Enrichment and AST Visitor Verification**:
   - Added comprehensive tests across `crates/pure_math/tests/test_math_types.rs`, `crates/oxidize_core/tests/test_traceability_engine.rs`, `crates/oxidize_core/src/ast_visitor.rs`, `egui_plot` item constructors, and GUI interactive tools.
5. **Architectural Verification Consistency**:
   - Propagated pull request base and head commit SHAs (`BASE_SHA` and `HEAD_SHA`) to `verify-suite` in `.github/workflows/ci.yml` to ensure consistent ADR and API drift verification across CI jobs.

## Consequences
- Independent native code coverage of at least 50.0% is strictly validated in CI.
- Coverage collection failures immediately fail the CI verification gate with actionable diagnostics.
- Architectural decision records and verification checks pass cleanly across all environments.
