# 13. Retire Deprecated Battery Degradation Primitives and Modernize Integration Tests

## Status
Accepted

## Context
The applied battery degradation module previously maintained deprecated legacy calculation functions (`n70`, `capacity`, and `cycles_to_capacity`) alongside the modern, strongly-typed `PowerLawModel`. These legacy functions relied on unvalidated primitive float arguments, bypassed the domain type safety invariants provided by `DepthOfDischarge` and `Capacity`, and required obsolete `#[allow(deprecated)]` suppressions across integration test suites.

## Decision
1. Removed deprecated legacy calculation functions (`n70`, `capacity`, `cycles_to_capacity`) from `crates/domain_applied/src/applied/battery_degradation/mod.rs`.
2. Migrated integration tests in `crates/domain_applied/tests/battery_degradation.rs` to interface directly with `PowerLawModel::standard()` using domain types (`DepthOfDischarge`, `Capacity`).
3. Removed `#[allow(deprecated)]` lint suppressions and legacy wrapper tests from `crates/domain_applied/tests/test_battery_degradation.rs`.
4. Updated the public verification API baseline in `crates/unified_verification/api_baseline.json` to reflect the retired primitives.

## Consequences
- **Positive:** Enforces strict domain type invariants and eliminates deprecated legacy calculation paths across `domain_applied`.
- **Positive:** Cleaner integration test suites with zero `#[allow(deprecated)]` annotations.
- **Negative:** Downstream code relying on the deprecated primitive float functions must migrate to `PowerLawModel::standard()`.
