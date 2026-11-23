# ADR 0004: CI/CD Fixes for Dependencies

## Status

Accepted

## Context
Recent changes to `math_explorer` and `crates/markdown_tests` updated `Cargo.toml` to restore necessary dev-dependencies for testing. This triggered the core-modification check in the unified verification suite, which expects an Architectural Decision Record (ADR) update for any changes in these directories.

## Decision
We've added this ADR to document that adding test/dev dependencies does not change the core architecture, but satisfies the strict CI requirements of the unified verification suite.

## Consequences
- The CI pipeline will now pass the `verify-records` check.
- Future dependency-only modifications may also require ADR documentation or an update to the CI logic to exempt `Cargo.toml`.
