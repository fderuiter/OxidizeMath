# OxidizeMath Documentation

Welcome to the OxidizeMath documentation directory.

## Architecture Guides

The [Architecture Guide Suite](architecture/README.md) provides centralized blueprints, diagrams, and developer walkthroughs:

- **[System Overview & Crate Topology](architecture/system_overview.md)**: Architectural layers, crate dependency graphs, and catalog across all 21 crates.
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
- [ADR 0035: Spatial Fermi Games and Ising Droplet Brushes](adr/0035-fermi-spatial-games-and-ising-droplet-brushes.md)
- [ADR 0037: Frozen-Period Glicko-2 Tournament Engine and Interactive Simulator](adr/0037-glicko2-frozen-period-tournament-simulator.md)
- [ADR 0038: Prime-Exponent FRACTRAN Virtual Machine and Minsky Counter Automaton Explorer](adr/0038-prime-exponent-fractran-vm-and-minsky-gui.md)
- [ADR 0039: Network Epidemic Containment Primitives, Active Degree Moments, and Zero-Allocation Stepping](adr/0039-network-epidemic-containment-primitives.md)
- [ADR 0040: Galactic Dynamics Kernels and Auto-Discovered Astrophysics Tab](adr/0040-galactic-dynamics-and-astrophysics-tab.md)
- [ADR 0041: Unified 3D Volumetric Field Studio and Zero-Allocation Surface/Splat Kernels](adr/0041-volumetric-field-studio-zero-alloc-3d-kernels.md)
- [ADR 0042: Zero-Allocation 2D Chladni Plate and Wave Tank Solver](adr/0042-2d-chladni-plate-and-wave-tank-solver.md)
- [ADR 0043: Auto-Discovered Paradox Lab Tab for Stochastic Ratchets and Network Congestion](adr/0043-paradox-lab-interactive-tab.md)
- [ADR 0043: Repair Guides, Enable Full-Feature Rustdoc, and Enforce Static Documentation Gates](adr/0043-repair-guides-and-enforce-static-doc-gates.md)
- [ADR 0044: Stack-Allocated Geodesic Engine and General Relativity Tab](adr/0044-geodesic-engine-and-relativity-tab.md)
- [ADR 0044: Ratchet `missing_docs` Exemptions and Document Core API Entrypoints](adr/0044-ratchet-missing-docs-and-core-api-documentation.md)
- [ADR 0044: Unify Build-Time and Runtime Traceability Scanning via `TraceabilityEngine`](adr/0044-unify-build-time-traceability-engine.md)
- [ADR 0045: SpatialBrushKit and Multi-Domain Field Sculpting](adr/0045-spatial-brush-kit-and-field-sculpting.md)
- [ADR 0045: Short-Weierstrass Elliptic Curves and Modular GUI Tab](adr/0045-weierstrass-elliptic-curves-and-modular-gui-tab.md)
- [ADR 0046: Symplectic Hamiltonian Integrators and Poincaré Section Lab](adr/0046-symplectic-hamiltonian-solvers-and-poincare-lab.md)
- [ADR 20261123: Zero-Allocation 2D Abelian Sandpile Engine, Recurrent Group Algebra, and Solid-State Studio](adr/20261123-abelian-sandpile-engine-and-studio.md)
- [ADR 20261123: Acoustic Luthiery Sonification, RLE Soundboard Sculpting, and Phononic Metamaterials](adr/20261123-acoustic-luthiery-sonification-and-phononic-metamaterials.md)
- [ADR 20261123: Zero-Allocation Age-Structured Cannibalism PDE Solvers and Interactive Explorer Tab](adr/20261123-age-structured-cannibalism-pde-solvers-and-explorer-tab.md)
- [ADR 20261123: Bidirectional Zero-Allocation Odds Conversions, Format-Aware Kelly Errors, and Bet Size Calculator Synchronization](adr/20261123-bidirectional-odds-conversions-and-bet-calculator-sync.md)
- [ADR 20261123: Verified Zero-Allocation Botanical Morphogenesis Kernels and Multi-Scale Biology Studios](adr/20261123-botanical-kernels-and-biology-studios.md)
- [ADR 20261123: Calibrated 2D Score Grids and Venue-Adjusted Sports Elo Engine](adr/20261123-calibrated-2d-score-grids-and-sports-elo.md)
- [ADR 20261123: Right-Censored Win-Ratio U-Statistics, Bivariate Weibull Power Sweeps, and Interactive Stress Tester](adr/20261123-censored-win-ratio-u-statistics-and-weibull-power-sweeps.md)
- [ADR 20261123: Checked Companion APIs, Typed Domain Errors, and Human-Readable Diagnostics](adr/20261123-checked-apis-and-typed-domain-errors.md)
- [ADR 20261123: Local Win Ratio and Clinical Cohort Workbench](adr/20261123-clinical-cohort-and-win-ratio-workbench.md)
- [ADR 20261123: Zero-Allocation Discretized Q-Learning Helpers and Unified Closed-Loop Control Workbench](adr/20261123-closed-loop-ekf-grpo-control-workbench.md)
- [ADR 20261123: DEFLATE-Compressed Workspace Share URLs and Browser History Bridge](adr/20261123-compressed-workspace-urls-and-history-bridge.md)
- [ADR 20261123: Counterfactual Shadow Twin Branching](adr/20261123-counterfactual-shadow-twin-branching.md)
- [ADR 20261123: Coupled Electrochemical Battery Aging, Zero-Allocation CSV Calibration, and Multi-Segment Lifetime Exports](adr/20261123-coupled-battery-aging-and-lifetime-exports.md)
- [ADR 20261123: Zero-Allocation CSR Spatial Grid and Cortical Thickness Workspace](adr/20261123-csr-spatial-grid-thickness-workspace.md)
- [ADR 20261123: Elliptic Curve Discriminant Atlas, Singular Cubic Rendering, and Hasse-Weil Challenges](adr/20261123-elliptic-curve-atlas-singular-loci-and-challenges.md)
- [ADR 20261123: Feature-Gate Core AST Tools and Unify Workspace Crates](adr/20261123-feature-gate-core-ast-tools-and-unify-crates.md)
- [ADR 20261123: Zero-Allocation 2D FFT Lenia Engine and Promoted Gray-Scott Kinetics](adr/20261123-fft-lenia-engine-and-gray-scott-kinetics.md)
- [ADR 20261123: Finite-Temperature BCS Superconductivity, CODATA SI Unit Bridges, and Solid-State Workbenches](adr/20261123-finite-temperature-bcs-si-unit-bridges-and-solid-state-workbenches.md)
- [ADR 20261123: Zero-Allocation Prime-Exponent FRACTRAN VM, Kolmogorov Bounds, and Auto-Discovered `AlgorithmicInfoTab`](adr/20261123-fractran-vm-kolmogorov-and-algorithmic-info-tab.md)
- [ADR 20261123: Framework-Hosted 2D Parameter Atlases and Zero-Allocation Cartographers](adr/20261123-framework-hosted-2d-parameter-atlases.md)
- [ADR 20261123: Zero-Allocation FreeSurfer Cortical Surface Physics, Potts MRF Segmentation, and 3D Cortical Morphometry Studio](adr/20261123-freesurfer-deformable-mrf-and-cortical-morphometry.md)
- [ADR 20261123: Deterministic Genz Gaussian Copula and Parlay Engine](adr/20261123-genz-gaussian-copula-and-parlay-engine.md)
- [ADR 20261123: Correlated Risk & OU Workbenches, Bet-Sizing Growth Curves, and Kelly Net-Odds Semantics](adr/20261123-glicko2-copula-ou-workbenches-and-kelly-net-odds.md)
- [ADR 20261123: Exact Elimination Treewidth, Zero-Allocation Network Flow & MST Solvers, and Unified Graph Interchange](adr/20261123-graph-flows-treewidth-and-dot-interchange.md)
- [ADR 20261123: Hierarchical Umbrella Error Enums, Full Submodule Error Re-Exports, and DiagnosticBus Emission Helpers](adr/20261123-hierarchical-umbrella-errors-and-diagnostic-bus-helpers.md)
- [ADR 20261123: Hybrid Analytical and Zero-Allocation Monte Carlo Edge Validation and Bankroll Solvency](adr/20261123-hybrid-analytical-and-monte-carlo-edge-and-ruin.md)
- [ADR 20261123: Poincaré Disk Hyperbolic Geometry Kernel and Geodesic Billiard Laboratory](adr/20261123-hyperbolic-geometry-and-geodesic-billiards.md)
- [ADR 20261123: Implicit CAD Volumetric Studio, Boundary-Capped Marching Cubes, and Millimeter STL Export](adr/20261123-implicit-cad-capped-isosurface-and-mm-stl.md)
- [ADR 20261123: Static Inventory Tab Registration and Scoped Build Scripts](adr/20261123-inventory-tabs-and-scoped-build-scripts.md)
- [ADR 20261123: Downsampled 2D FFT Lenia Soliton Viability Atlas, Potential Histograms, and AST Rule Sandboxes](adr/20261123-lenia-viability-atlas-and-sandboxes.md)
- [ADR 20261123: Modularize Setup and Enforce Static Documentation Gates](adr/20261123-modularize-setup-and-enforce-doc-gates.md)
- [ADR 20261123: Multi-State, Portfolio, and Drawdown-Constrained Kelly Convex Solvers](adr/20261123-multi-state-kelly-convex-solvers.md)
- [ADR 20261123: Myerson Mechanism-Design Auction Lab and Analytical Ising / Fermi Spatial Parameter Atlas](adr/20261123-myerson-auction-lab-and-ising-parameter-atlas.md)
- [ADR 20261123: Root-Owned Pane Dataflow Bus and Cycle-Checked Cross-Domain Pipelines](adr/20261123-pane-dataflow-bus-and-cross-domain-pipelines.md)
- [ADR 20261123: Variable-Dose Pharmacokinetic Regimen Superposition and Interactive Planner](adr/20261123-pharmacokinetic-regimen-superposition-and-analytics.md)
- [ADR 20261123: Planar ODE AST Evaluation, Poincaré Stability Maps, AST Caching, and Recovery-Based URL Sharing](adr/20261123-planar-ode-ast-poincare-and-url-recovery.md)
- [ADR 20261123: Precompiled Verification Patterns, AST Source Prefetch, and Zero-Allocation Scans](adr/20261123-precompiled-verification-patterns-and-ast-prefetch.md)
- [ADR 20261123: `core`-Slice Thomas Solver, `CrankNicolson1D` Workspace, and Closed-Form $2\times 2$ Quantum Helpers](adr/20261123-pure-math-thomas-solver-and-closed-form-quantum-workspaces.md)
- [ADR 20261123: Live 4D Radar-Gated IMRT Contouring Workbench](adr/20261123-radar-gated-4d-imrt-contouring-workbench.md)
- [ADR 20261123: Zero-Allocation Fixed-Size Christoffel Symbols, Relativistic Thin-Lens Splat Jacobians, and Paczyński-Wiita Galactic Orbits](adr/20261123-relativistic-lensed-splats-and-orbits.md)
- [ADR 20261123: Reusable Flat CSR Spatial Grid for Cortical Thickness](adr/20261123-reusable-flat-spatial-grid-cortical-thickness.md)
- [ADR 20261123: Scope Pre-Commit Gates, Add Pre-Push Tiering, and Eliminate Build-Script `CARGO_MANIFEST_DIR` Drift](adr/20261123-scope-pre-commit-gates-and-pre-push-tiering.md)
- [ADR 20261123: Workspace-Scoped `SharedSimulationBus`, Copy-on-Write CSR Topologies, and Zero-Allocation Graph Kernels](adr/20261123-shared-simulation-topology-bus.md)
- [ADR 20261123: Zero-Allocation Laplace-Beltrami Surface Morphogenesis and Mean-Curvature Buckling](adr/20261123-surface-morphogenesis-laplace-beltrami-buckling.md)
- [ADR 20261123: Tridiagonal Crank-Nicolson 1D Quantum Propagation and Persistent Unified GIF Stepping](adr/20261123-tridiagonal-crank-nicolson-quantum-propagation.md)
- [ADR 20261123: Unified RL Maze Sculpting, Exact Bellman Solvers, and TD Surprise Sonification](adr/20261123-unified-rl-maze-sculpting-bellman-and-sonification.md)
- [ADR 20261123: Volumetric 3D Implicit Formulas, Boundary-Padded Solid Closure, and Unified 3D Exports](adr/20261123-volumetric-implicit-formulas-and-solid-exports.md)
- [ADR 20261123: Quantum Tunneling and 2D Chladni Resonance Parameter Atlases](adr/20261123-wave-and-chladni-resonance-atlases.md)
- [ADR 20261123: Workspace Release Version Governance, Tag Parity Guards, and WiX Pre-Release Mapping](adr/20261123-workspace-release-version-guards-and-wix-mapping.md)
- [ADR 20261123: Add xtask Smoke Gates and Zero-Allocation Benchmark Checks](adr/20261123-xtask-smoke-gates-and-zero-allocation-bench-checks.md)
- [ADR 20261123: Zero-Allocation Brent Root-Finder, Market Pricing Devigging, and Execution Calculators](adr/20261123-market-pricing-brent-and-execution-calculators.md)
- [ADR 20261123: `#![no_std]` Linear Algebra Engine (`engine_linalg`), Solver Unification, and Cached Cohort GLM QR](adr/20261123-no-std-engine-linalg-and-solver-unification.md)
- [ADR 20261123: Parameter-Substituted LaTeX and Multi-Target Code Artifact Export](adr/20261123-parameter-substituted-latex-and-code-artifact-export.md)

## Contributing

For guidelines on coding style, safety standards, and repository workflows, see the [Contributing Guide](../CONTRIBUTING.md).
