# Architecture Decision Record: Portable Build Paths

## Status

Accepted

## Context
Previously, build scripts (`crates/markdown_tests/build.rs`, `math_explorer_gui/build.rs`) and procedural macros (`crates/verified_engine_macros/src/embed_theory.rs`) generated Rust code using hardcoded absolute filesystem paths derived via `fs::canonicalize`. This caused issues with isolated sandboxed CI/CD builds where the filesystem layout differs at test execution time, and prevented remote caching systems from sharing artifacts because generated file contents were machine-specific.

## Decision
We switched to using relative paths resolved dynamically at compile time via the `CARGO_MANIFEST_DIR` environment variable. Build scripts now emit paths relative to the crate's manifest or workspace root, and generated code includes files using `concat!(env!("CARGO_MANIFEST_DIR"), "/...", ...)`. This forces the rust compiler to resolve the path dynamically based on where the crate is currently being compiled rather than where the code generation script ran.

## Consequences
- **Positive:** Enables deterministic builds and successful test execution within arbitrary isolated CI containers.
- **Positive:** Fixes remote build cache misses, accelerating team-wide build times.
- **Negative:** Slightly more verbose `include_str!` macros in generated code.
