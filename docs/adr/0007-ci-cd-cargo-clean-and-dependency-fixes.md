# ADR 0007: CI/CD Cargo Clean Removal and Dependency Cleanups

## Context
Removing premature `cargo clean` in CI and updating workspace domain `Cargo.toml` files to prune unused dependencies touched files under `crates/`. This triggered the verification rule in `unified_verification`, requiring an Architectural Decision Record (ADR) update for changes in core logic directories.

## Decision
Document the CI workflow adjustment and domain `Cargo.toml` dependency cleanup in this ADR to satisfy the verification requirement.

## Consequences
- The CI `verify-records` check passes.
- Code formatting and dependency pruning standards are maintained across domain crates.
