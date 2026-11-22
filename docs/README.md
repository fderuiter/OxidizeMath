# OxidizeMath Documentation

Welcome to the OxidizeMath documentation directory.

## Architecture Guides

The [Architecture Guide Suite](architecture/README.md) provides centralized blueprints, diagrams, and developer walkthroughs:

- **[System Overview & Crate Topology](architecture/system_overview.md)**: Architectural layers, crate dependency graphs, and catalog across all 18 crates.
- **[CERA Framework Guide](architecture/cera_framework.md)**: Climate-invariant Encoding through Representation Alignment and climate modeling pipelines.
- **[Zero-Allocation Runtime Guide](architecture/zero_allocation_runtime.md)**: NASA Power of 10 Rule 3 enforcement, `VerifiedAllocator`, and verified engine execution rules.
- **[Interactive Tool Walkthrough](architecture/interactive_tool_walkthrough.md)**: Step-by-step developer guide to implement interactive simulations with `InteractiveTool`.

## Architectural Decision Records (ADRs)

The [`docs/adr/`](adr/0001-dynamic-workspace-discovery.md) directory documents accepted design decisions for OxidizeMath:

- [ADR 0001: Dynamic Workspace Discovery](adr/0001-dynamic-workspace-discovery.md)
- [ADR 0002: Active Documentation Standard](adr/0002-active-documentation-standard.md)
- [ADR 0002: Portable Build Paths](adr/0002-portable-build-paths.md)
- [ADR 0003: Traceability Refactoring](adr/0003-traceability-refactoring.md)
- [ADR 0005: Active Documentation Standard (Extended)](adr/0005-active-documentation-standard.md)
- [ADR 0008: Pre-Commit Lint Suite](adr/0008-xtask-lint-all-pre-commit.md)
- [ADR 0012: Centralized Architecture Guide Suite](adr/0012-centralized-architecture-suite.md)
- [ADR 0013: Retire Deprecated Battery Primitives](adr/0013-retire-deprecated-battery-primitives.md)
- [ADR 0013: Retire Legacy BCS Gap Solver Function](adr/0013-retire-legacy-bcs-gap-solver.md)
- [ADR 0013: Scientific Metadata Documentation Standard](adr/0013-scientific-metadata-documentation-standard.md)
- [ADR 0013: Trait-Level Executable Doc-Test Suites](adr/0013-trait-level-runnable-doctests.md)
- [ADR 0016: Standardize Crate Documentation Suite](adr/0016-standardize-crate-documentation-suite.md)
- [ADR 0034: Curate `math_explorer` Facade Exports and Doctests](adr/0034-curate-math-explorer-facade-exports-and-doctests.md)
- [ADR 0035: Zero-Allocation Age-Structured Cannibalism PDE Solvers and Interactive Explorer Tab](adr/0035-age-structured-cannibalism-pde-solvers-and-explorer-tab.md)

## Contributing

For guidelines on coding style, safety standards, and repository workflows, see the [Contributing Guide](../CONTRIBUTING.md).
