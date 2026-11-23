# ADR 20261123: Exact Elimination Treewidth, Zero-Allocation Network Flow & MST Solvers, and Unified Graph Interchange

## Status
Accepted (extends [ADR 0009](0009-reusable-graph-layout-algorithms.md), [ADR 0022](0022-unify-preset-state-serialization.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`crates/pure_math/README.md`](../../crates/pure_math/README.md) positions `pure_math::graph_theory` as the canonical graph and network reference module feeding `math_explorer_gui`. Four gaps fragmented graph exploration across the library and GUI:

1. **Placeholder `treewidth` and Unwired Structural Parameters**: [`crates/pure_math/src/pure_math/graph_theory/parameters/treewidth.rs`](../../crates/pure_math/src/pure_math/graph_theory/parameters/treewidth.rs) returned `0` unconditionally, and [`math_explorer_gui/src/tabs/graph_theory/network_metrics.rs`](../../math_explorer_gui/src/tabs/graph_theory/network_metrics.rs) left `degeneracy`, `vertex_cover`, and `treewidth` unwired in the UI.
2. **Missing Classical Combinatorial Optimization Solvers and Graph Interchange Codecs**: `pure_math::graph_theory` lacked `s-t` Max-Flow / Min-Cut, Minimum Spanning Tree (MST), and deterministic `.dot`, `.graphml`, and edge-list `.csv` interchange codecs, preventing users from importing or exporting external network topologies.
3. **False Shortest-Path Tree Highlighting in `AlgorithmVisualizerTool`**: [`math_explorer_gui/src/tabs/graph_theory/algorithm_visualizer/canvas.rs`](../../math_explorer_gui/src/tabs/graph_theory/algorithm_visualizer/canvas.rs) discarded `DijkstraResult::predecessors` and highlighted every edge `(u, v)` whose endpoints were both visited (`24/24` edges on the default `4x4` grid instead of the `15`-edge shortest-path spanning tree).
4. **Siloed Topologies and Deferred Figure Exports Across `GraphTheoryTab`**: [`math_explorer_gui/src/tabs/graph_theory/mod.rs`](../../math_explorer_gui/src/tabs/graph_theory/mod.rs) isolated `GraphEditorTool`, `AlgorithmVisualizerTool`, and `NetworkMetricsTool` without cross-tool topology handoff, while all three tools returned `ExportCapabilities::none()`.

## Decision
1. **Exact Branch-and-Bound Elimination `treewidth` with Min-Fill Fallback**:
   - Implement an iterative fixed-stack `[u32; 24]` bitmask elimination branch-and-bound solver in [`crates/pure_math/src/pure_math/graph_theory/parameters/treewidth.rs`](../../crates/pure_math/src/pure_math/graph_theory/parameters/treewidth.rs) (`|V| <= 24`) with simplicial and almost-simplicial (`deg(v) <= lb`) safe reductions, maximum-minimum-degree (`MMD`) degeneracy lower bounds, min-fill upper bounds, and a 4096-entry stack transposition cache, executing with zero heap allocations.
   - Provide preallocated `TreewidthScratch` and `treewidth_in_place` min-fill upper-bound elimination for larger graphs (`|V| > 24`) with zero post-init heap allocations.
2. **Zero-Allocation `max_flow_min_cut` and `minimum_spanning_tree` Engines plus Typed `GraphTheoryError`**:
   - Add [`crates/pure_math/src/pure_math/graph_theory/error.rs`](../../crates/pure_math/src/pure_math/graph_theory/error.rs) (`GraphTheoryError`), wired through [`crates/pure_math/src/error/umbrella.rs`](../../crates/pure_math/src/error/umbrella.rs) (`PureMathError::GraphTheory`) and [`math_explorer/src/error.rs`](../../math_explorer/src/error.rs) (`MathExplorerError::GraphTheory`).
   - Add [`crates/pure_math/src/pure_math/graph_theory/max_flow_min_cut.rs`](../../crates/pure_math/src/pure_math/graph_theory/max_flow_min_cut.rs) (`EdmondsKarpScratch`, `solve_max_flow_min_cut_in_place`, `max_flow_min_cut`) computing residual augmenting-path steps, net edge flows, and reachable `S`/`T` `s-t` minimum-cut vertex partitions with zero steady-state allocations.
   - Add [`crates/pure_math/src/pure_math/graph_theory/minimum_spanning_tree.rs`](../../crates/pure_math/src/pure_math/graph_theory/minimum_spanning_tree.rs) (`KruskalScratch`, `solve_minimum_spanning_tree_in_place`, `minimum_spanning_tree`) using in-place iterative heapsort and union-find with path compression and union-by-rank.
3. **Deterministic `.dot`, `.graphml`, and `.csv` Interchange Codecs**:
   - Add [`crates/pure_math/src/pure_math/graph_theory/interchange/mod.rs`](../../crates/pure_math/src/pure_math/graph_theory/interchange/mod.rs) (`InterchangeGraph`, `parse_dot`, `to_dot`, `to_styled_dot`, `parse_graphml`, `to_graphml`, `parse_edge_csv`, `to_edge_csv`), enforcing `<= 256` vertices, `<= 2,048` edges, and finite coordinates/weights while supporting styled Graphviz `color`, `fillcolor`, `penwidth`, and `xlabel` attributes.
4. **Unify `GraphTheoryTab` Topologies, Reconstruct Predecessor Trees, and Enable `ExportCapabilities::plot()`**:
   - Introduce [`math_explorer_gui/src/tabs/graph_theory/shared_topology.rs`](../../math_explorer_gui/src/tabs/graph_theory/shared_topology.rs) (`SharedGraphTopology`, `CompactTopologyState`), one-click cross-tool handoff (`Send to Visualizer`, `Send to Metrics`, `Send to Editor`), and `.dot`/`.graphml`/`.csv`/`.json` file dialogs via [`math_explorer_gui/src/web_io.rs`](../../math_explorer_gui/src/web_io.rs).
   - Reconstruct exact `DijkstraResult::predecessors` and BFS/DFS discovery tree edges in [`math_explorer_gui/src/tabs/graph_theory/algorithm_visualizer/solvers.rs`](../../math_explorer_gui/src/tabs/graph_theory/algorithm_visualizer/solvers.rs) (`15` tree edges and `0` false non-tree highlights on the `4x4` grid), and add stepped `MaxFlowMinCut` and `MinimumSpanningTree` modes with source/sink selectors.
   - Wire `degeneracy`, `vertex_cover` modulator rings, and exact `treewidth` into `NetworkMetricsTool`, and upgrade all three `GraphTheoryTab` tools to `ExportCapabilities::plot()` via [`math_explorer_gui/src/tabs/graph_theory/plot_export.rs`](../../math_explorer_gui/src/tabs/graph_theory/plot_export.rs).

```text
Updated GUI ExportCapabilities inventory:
- InteractiveTools: 87 (69 plot + 8 raster + 10 non-figure)
```

## Consequences
- **Positive**: Users can sketch or import `.dot`, `.graphml`, `.csv`, and `.json` networks, step through exact Dijkstra predecessor trees, Edmonds-Karp `s-t` min-cuts, and Kruskal MSTs, inspect exact `treewidth`, `degeneracy`, and `vertex_cover` invariants, and export PNG, SVG, CSV, JSON, `.dot`, and `.graphml` figures across desktop and WebAssembly.
- **Negative / Trade-offs**: Exact elimination treewidth has exponential worst-case time in `|V|`, bounded safely at `|V| <= 24` with safe simplicial reductions before transitioning to polynomial-time min-fill upper bounds for `|V| > 24`. Expanding `pure_math::graph_theory` requires regenerating `api_baseline.json`.
