# ADR 20261123: Zero-Allocation Structure-of-Arrays Physarum Engine and Live Steiner Benchmarks

## Status

Accepted (2026-11-23)

## Context

Previously, biological spatial solvers in [`crates/domain_biology/src/biology/diffusion/fd2d/mod.rs`](../../crates/domain_biology/src/biology/diffusion/fd2d/mod.rs) and [`crates/domain_biology/src/biology/lenia/engine.rs`](../../crates/domain_biology/src/biology/lenia/engine.rs) evolved fixed Eulerian concentration grids without mobile chemotactic agents, while graph algorithms in [`crates/pure_math/src/pure_math/graph_theory/dijkstra.rs`](../../crates/pure_math/src/pure_math/graph_theory/dijkstra.rs) and [`crates/pure_math/src/pure_math/graph_theory/parameters/network_metrics.rs`](../../crates/pure_math/src/pure_math/graph_theory/parameters/network_metrics.rs) operated only on static topologies. Simulating emergent *Physarum polycephalum* transport highways, shortest-path maze solving, and $120^\circ$ Fermat-Torricelli Steiner junctions in WebAssembly requires coupling up to `16,384` mobile Lagrangian agents to diffused, obstacle-masked Eulerian trail fields under `VerifiedAllocator` without dynamic heap allocations in the hot simulation loop.

## Decision

1. **Zero-Allocation SoA `PhysarumEngine` (`domain_biology::biology::physarum`)**:
   - Added [`PhysarumParams`](../../crates/domain_biology/src/biology/physarum/params.rs) (`#[derive(Theory)]`), [`PhysarumTerminal`](../../crates/domain_biology/src/biology/physarum/params.rs), [`PhysarumTelemetry`](../../crates/domain_biology/src/biology/physarum/params.rs), [`PhysarumPresetKind`](../../crates/domain_biology/src/biology/physarum/presets.rs) (`SteinerTriangle`, `AsymmetricMaze`, `TokyoRail`), and [`PhysarumEngine`](../../crates/domain_biology/src/biology/physarum/engine.rs).
   - Pre-allocated Structure-of-Arrays `(agent_x, agent_y, agent_heading, agent_energy, agent_origin)` vectors up to `MAX_PHYSARUM_AGENTS = 16,384`, row-major `trail`, `trail_scratch`, and `obstacles` slices indexed via `flatten_2d_index` on `64..=256` grids, fixed-size `[PhysarumTerminal; 16]` nutrient beacons, and seeded `OxidizeRng` streams so `PhysarumEngine::step` executes with **zero dynamic heap allocations** and bit-identical native/`wasm32` trajectories.
   - Coupled Jones (2010) 3-sensor forward-cone chemotaxis with obstacle-aware $3\times 3$ zero-flux diffusion and Tero et al. (2010) sub-threshold Hill filament pruning so detour tubes collapse while reinforced corridors stabilize.

2. **Morphological `Graph` Skeleton Extraction & Steiner/Dijkstra Benchmarks (`physarum::network`)**:
   - Implemented adaptive trail thresholding, interior eyelet hole filling, terminal-pinned Zhang-Suen 8-connected medial-axis thinning, dead-end whisker pruning, junction clustering, and iterative Ramer-Douglas-Peucker simplification into `pure_math::pure_math::graph_theory::graph::Graph<[f64; 2], f64>`.
   - Benchmarked extracted `wire_cost`, `length_reduction_pct`, pairwise `dijkstra` `mean_route_stretch`, and single-edge `fault_survival` / `pairwise_fault_survival` against the exact Euclidean Minimum Spanning Tree (`mst_cost`) over active terminals.

3. **Interactive `PhysarumTool` ("Physarum Network Lab") (`math_explorer_gui::tabs::morphogenesis::physarum`)**:
   - Registered `PhysarumTool` inside `MorphogenesisTab` alongside `MorphogenesisTool` and `LeniaTool` for split-viewport comparison, backed by three `StarterPresetMetadata` cards (`physarum_steiner`, `physarum_maze`, `physarum_tokyo_rail`).
   - Added `SemanticPalette` & high-contrast trail/skeleton/MST rendering, `AnimatedSlider` genome controls, mouse and keyboard brushes (`PlaceCity`, `PaintWall`, `EraseWall`, `SeverTube`), `FiniteStateGuard` recovery, `8 ms` cooperative frame stepping, `.png`/`.gif`/`.csv`/`.json` deferred exports, and `< 4 KiB` URL state sharing (`URL_JSON_BOUNDS`).

## Consequences

- **Positive**: `PhysarumEngine::step` runs with `0` heap allocations under `VerifiedAllocator` up to `256x256` grids and `16,384` agents with zero new crate dependencies.
- **Positive**: Quantitative regression tests verify `>= 90%` short-corridor trail routing on the two-branch asymmetric maze within `450` steps and `>= 12%` wire-length reduction below the static two-edge spanning tree on three-terminal Fermat-Torricelli Steiner networks, plus severed-artery self-healing on the 11-hub Tokyo-rail mesh.
- **Verification**: Validated via `domain_biology` zero-allocation and quantitative suites, `math_explorer_gui` contract/accessibility/URL/export tests, and regenerated `crates/unified_verification/api_baseline.json`.
