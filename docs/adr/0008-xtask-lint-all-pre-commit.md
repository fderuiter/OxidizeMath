# ADR 0008: `xtask lint-all` Command and Pre-commit Hook

## Context
To improve code quality enforcement and streamline local developer workflows, an `xtask lint-all` command was added alongside workspace formatting. Standardizing code formatting updated whitespace and line endings in core crates (`crates/`).

## Decision
We introduced a dedicated `lint-all` task in `apps/xtask` and updated repository setup to configure git pre-commit hooks to invoke `cargo run -p xtask -- lint-all`. Code formatting adjustments across crates were applied to pass `cargo fmt --check`.

## Consequences
- Clean workspace state passes `cargo fmt --check` and `cargo clippy`.
- The `verify-records` CI check requirement for core logic changes is satisfied by this ADR.
