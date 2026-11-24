# ADR 20261123: Zero-Allocation SoA Physarum Engine, Steiner Network Benchmarks, and Interactive Lab

## Status

Accepted

## Context

Biological transport networks formed by *Physarum polycephalum* (Nakagaki et al., 2000; Jones, 2010; Tero et al., 2010) bridge continuous chemotactic reaction-diffusion fields and combinatorial network optimization (minimum spanning trees, Fermat-Torricelli Steiner junctions, shortest-path maze routing, and fault-tolerant urban rail meshes). Supporting interactive simulation of up to $16{,}384$ chemotactic agents alongside live topological extraction and graph-theoretic benchmarking within OxidizeMath requires:

1. **Zero-Allocation Hot Loop:** Every per-frame chemotactic sense/turn/move/deposit/diffuse/prune step and runtime spatial sculpting operation must execute with zero dynamic heap allocations under `verified_engine::allocator::VerifiedAllocator` on native and `wasm32-unknown-unknown` targets (NASA Power of 10 Rule 3).
2. **Cache-Coherent Structure-of-Arrays Layout:** Array-of-Structs (`Vec<Agent>`) layouts waste cache bandwidth during sensory lookahead and propulsion updates across thousands of agents.
3. **Quantitative Graph Benchmarking:** Extracting weighted undirected planar skeletons into `pure_math::pure_math::graph_theory::graph::Graph<[f64; 2], f64>` and comparing extracted wire length, pairwise `dijkstra` shortest-path route stretch, and single-edge fault survival against the static Euclidean Minimum Spanning Tree (MST).
4. **Unified Framework Sculpting, Telemetry & Maturity Qualification:** Seamless integration with `RasterBrushCanvas` (`#1733`: aspect-locked coordinate mapping, continuous Bresenham pointer-drag interpolation, and pre-allocation-validated RLE `base64url` obstacle masks), `PaneDataflowBus` (`#1765`: named scalar `observables()`), and three-tier maturity qualification (`#1794`: `MaturityTier::Validated`).

## Decision

1. **Structure-of-Arrays `PhysarumEngine` (`crates/domain_biology/src/biology/physarum`)**:
   - Pre-allocate contiguous `f64`/`u8` agent buffers (`agent_x`, `agent_y`, `agent_heading`, `agent_energy`, `agent_origin`) up to `MAX_PHYSARUM_AGENTS = 16,384`, row-major Eulerian `trail`, `trail_scratch`, and `obstacles` slices (`64..=256`, indexed via `math_commons::math_kernel::types::flatten_2d_index`), fixed-size `[PhysarumTerminal; 16]` nutrient beacons, and deterministic `OxidizeRng` streams.
   - Implement 3-sensor forward-cone chemotaxis (Jones, 2010), obstacle-aware $3\times 3$ zero-flux diffusion, and nonlinear Hill sub-threshold filament pruning (Tero et al., 2010) with zero heap allocations during `step`, `set_agent_count`, `paint_obstacle_disc`, `apply_obstacle_mask`, and `sever_trail_disc` (verified in `crates/domain_biology/tests/test_physarum_zero_alloc.rs`).
   - Ship canonical `PhysarumPresetKind::{SteinerTriangle, AsymmetricMaze, TokyoRail}` topologies and classify `PhysarumParams` / `PhysarumTheory` at `MaturityTier::Validated` (`#1794`).

2. **Morphological Thinning & Graph Benchmark Pipeline (`skeleton.rs`, `network.rs`)**:
   - Extract weighted undirected `Graph<[f64; 2], f64>` topologies via adaptive foreground thresholding, interior eyelet filling, terminal-pinned Zhang-Suen two-subiteration thinning, short-spur pruning, junction clustering, and iterative Ramer-Douglas-Peucker polyline simplification.
   - Compute exact terminal Euclidean MST `mst_cost`, extracted `wire_cost`, `cost_ratio`, `length_reduction_pct`, pairwise `dijkstra` `mean_route_stretch`, and single-edge `fault_survival` / `pairwise_fault_survival` in `PhysarumNetworkBenchmark`.

3. **Interactive `PhysarumTool` Studio (`math_explorer_gui/src/tabs/morphogenesis/physarum`)**:
   - Register `PhysarumTool` ("Physarum Network Lab") inside `MorphogenesisTab` with three starter presets (`physarum_steiner`, `physarum_maze`, `physarum_tokyo_rail`), `8 ms` cooperative frame stepping, `FiniteStateGuard` recovery, paired `egui_plot` telemetry charts, and `.png`/`.gif`/`.csv`/`.json` deferred exports.
   - Use `#1733`'s `compute_aspect_locked_rect`, `map_pointer_to_cell`, continuous `bresenham_line` drag interpolation, and pre-allocation-validated `encode_u8_grid` / `decode_u8_grid` (`obstacle_mask_rle`) so complex multi-stroke obstacle topologies round-trip losslessly under `< 4 KiB` URL bounds (`URL_JSON_BOUNDS`), alongside AccessKit keyboard navigation, `announce_live_status` screen-reader announcements, and `#1765` scalar `observables()`.

## Consequences

- Achieves bit-identical, zero-allocation chemotactic morphogenesis stepping across desktop and WebAssembly targets, quantitative Steiner/Dijkstra/MST benchmarking, lossless compact URL sharing of arbitrary sculpted obstacle masks, and WCAG 2.2 AA accessible keyboard/pointer interaction and `[VAL]` maturity badges.
- Morphological skeletonization and all-pairs terminal shortest-path evaluation run every `BENCHMARK_REFRESH_STRIDE = 8` steps (and immediately on brush edits) so interactive frame budgets remain well within `8 ms`.
