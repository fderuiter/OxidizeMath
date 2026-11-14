# 13. Trait-Level Executable Doc-Test Suites Across Foundational Crates

## Status
Accepted

## Context
Foundational architectural traits and contracts (`SimulationModel`, `ModelState`, `ModelConfig` in `oxidize_core`, `TheoryDescribable` and `ParameterConstraint` in `scientific_metadata`, telemetry dispatches in `federated_registry`, attribute macros in `verified_engine`, and utility math kernels in `math_commons`) lacked executable trait-level documentation tests. Without runnable examples embedded directly into the public trait definitions, developers had to inspect integration tests or external crates to understand usage patterns, and interface contract regressions were not verified during standard `cargo test --doc`.

## Decision
1. Implemented runnable doc-tests on core simulation traits (`ModelConfig`, `ModelState`, `SimulationModel`) in `crates/oxidize_core/src/lib.rs` verifying deterministic simulation initialization (`OxidizeRng`), state retrieval, and stepping.
2. Implemented runnable doc-tests on scientific metadata traits (`ParameterConstraint`, `TheoryDescribable`) in `crates/scientific_metadata/src/theory.rs` verifying citation metadata, phonetic screen-reader strings, parameter constraints, and dynamic getters/setters.
3. Added runnable doc-test for the re-exported `verified` attribute macro in `crates/verified_engine/src/lib.rs` demonstrating runtime invariant enforcement and execution metrics tracking.
4. Added runnable doc-tests for `FederatedRegistry`, `register_source`, and `emit` in `crates/federated_registry/src/lib.rs` demonstrating telemetry registration, subscription, and event streaming.
5. Added calculation contract doc-tests in `crates/math_commons` for `flatten_2d_index`, `heatmap_color`, `mapped_gradient`, and `ClebschGordan`.

## Consequences
- All public trait contracts are accompanied by runnable, compiler-verified doc-tests executed during CI pre-flight checks (`cargo test --doc --workspace --all-features`).
- Developer onboarding is streamlined by providing working implementation examples directly in generated API documentation.
- Architectural contract regressions are immediately caught during trait documentation builds and tests.
