# ADR 20261124: Unified `GraphTopologyDocument` Codecs, TikZ/SVG Publication Diagrams, and Cross-Domain GUI Topology Exchange

## Status
Accepted (extends [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), [ADR 20261123: Graph Flows, Treewidth, and DOT Interchange](20261123-graph-flows-treewidth-and-dot-interchange.md), and [ADR 20261123: Shared Simulation Topology Bus](20261123-shared-simulation-topology-bus.md))

## Context
 While [ADR 20261123](20261123-graph-flows-treewidth-and-dot-interchange.md) introduced integer-indexed `InterchangeGraph` `.dot`/`.graphml`/`.csv` codecs for `GraphTheoryTab` and [ADR 20261123](20261123-shared-simulation-topology-bus.md) connected live workspaces via `SharedSimulationBus`, three cross-domain gaps remained:

1. **String-Identifier Automata & Publication Topologies**: Cross-domain network tools—including FRACTRAN Minsky counter automata (`q0`, `q_p`), Braess Wheatstone routing graphs (`S`, `A`, `B`, `T`), Hodgkin-Huxley / Kuramoto neural rings (`N0`..`N4`, `O0`..`O4`), and budgeted epidemic contact graphs—require string vertex IDs, node shapes (`Circle`, `DoubleCircle`, `Box`), edge stroke styles (`Solid`, `Dashed`, `Thick`), and standalone compilable LaTeX `TikZ` (`.tex`) and vector `.svg` topology diagrams alongside `.dot`, `.graphml`, and `.csv`/`.tsv` tables.
2. **Framework Action-Bar Integration**: `SimulationFramework` lacked a unified `ExportCapabilities::topology` profile (`with_topology()`, `topology_only()`), deferred `ExportFormat::Topology*` file/clipboard dispatch, and `.dot`/`.gv`/`.graphml` browser MIME mappings in [`egui_plot/src/file_io.rs`](../../egui_plot/src/file_io.rs).
3. **External Contact Network Import in `NetworkPropagationTool` & Parser Hardening**: `NetworkPropagationTool` supported seeded geometric graphs, `SharedSimulationBus`, and `PaneDataflowBus`, but could not import standalone `.dot`/`.graphml`/`.csv`/`.tsv` contact topologies from disk; meanwhile, interchange parsers benefited from a uniform `2 MiB` byte cap, UTF-8 BOM stripping, `.tsv` delimiter detection, and explicit XML `<!DOCTYPE` / `<!ENTITY` (XXE) rejection.

## Decision
1. **Extend `pure_math::graph_theory::interchange` with `GraphTopologyDocument` and Hardened Codecs**:
   - Add [`crates/pure_math/src/pure_math/graph_theory/interchange/document.rs`](../../crates/pure_math/src/pure_math/graph_theory/interchange/document.rs) (`GraphTopologyDocument`, `TopologyNode`, `TopologyEdge`, `TopologyFormat`, `TopologyNodeShape`, `TopologyEdgeStyle`, `from_interchange`, `to_interchange`, `to_undirected_petgraph`, `ensure_positions`, `serialize`, `parse_with_filename`) sharing escaping, XML entity decoding, and bounds validation with `InterchangeGraph`.
   - Add [`crates/pure_math/src/pure_math/graph_theory/interchange/tikz.rs`](../../crates/pure_math/src/pure_math/graph_theory/interchange/tikz.rs) (`to_tikz_topology` standalone `\documentclass[tikz,border=10pt]{standalone}` figure generator) and [`crates/pure_math/src/pure_math/graph_theory/interchange/svg.rs`](../../crates/pure_math/src/pure_math/graph_theory/interchange/svg.rs) (`to_svg_topology` standalone vector network diagram generator).
   - Add `GraphTheoryError::PayloadTooLarge` and `GraphTheoryError::ForbiddenXmlDeclaration` in [`crates/pure_math/src/pure_math/graph_theory/error.rs`](../../crates/pure_math/src/pure_math/graph_theory/error.rs), enforcing `MAX_TOPOLOGY_IMPORT_BYTES` (`2 MiB`), UTF-8 BOM stripping, `.tsv` tab support, and DTD/XXE rejection across both `InterchangeGraph` and `GraphTopologyDocument` parsers.
2. **Framework-Level Topology Export Menu, Deferred Dispatch, and MIME Inference**:
   - Map `.dot`/`.gv` to `text/vnd.graphviz` and `.graphml` to `application/graphml+xml` in [`egui_plot/src/file_io.rs`](../../egui_plot/src/file_io.rs).
   - Extend `ExportCapabilities` (`topology`, `topology_disabled_reason`, `with_topology()`, `topology_only()`), `InteractiveTool::export_topology_document`, `ExportFormat` (`TopologyDot`, `TopologyGraphMl`, `TopologyTikz`, `TopologySvg`, `TopologyCsv`), and [`math_explorer_gui/src/framework/export_topology.rs`](../../math_explorer_gui/src/framework/export_topology.rs) (`🕸 Topology ▾` download and clipboard actions).
3. **Wire Seven Cross-Domain Network & Automata Tools**:
   - Enable `ExportCapabilities::plot().with_topology()` on `GraphEditorTool`, `AlgorithmVisualizerTool`, `NetworkMetricsTool`, `FractranMachineTool`, and `BraessNetworkParadoxTool`.
   - Enable `ExportCapabilities::topology_only()` on `NeuralNetworkVizTool` (exporting both Hodgkin-Huxley synaptic weights/voltages and Kuramoto phase/order-parameter graphs) and `NetworkPropagationTool` (adding non-blocking `.dot`/`.graphml`/`.csv`/`.tsv` contact-network import in [`math_explorer_gui/src/tabs/epidemiology/network_propagation/topology_io.rs`](../../math_explorer_gui/src/tabs/epidemiology/network_propagation/topology_io.rs)).

```text
Updated GUI ExportCapabilities topology coverage:
- Topology-enabled InteractiveTools: 7 (5 plot+topology + 2 topology_only)
```

## Consequences
- **Positive**: Any of the 7 discrete network and automata tools can export or copy `.dot`, `.graphml`, standalone `TikZ` (`.tex`), `.svg`, and `.csv` topologies through a single framework action bar while preserving full compatibility with `InterchangeGraph`, `SharedSimulationBus`, and `PaneDataflowBus`.
- **Negative / Trade-offs**: Expanding `pure_math::graph_theory::interchange` and `GraphTheoryError` adds public types and enum variants, requiring `api_baseline.json` regeneration via `cargo run -p xtask -- regenerate-baseline`.
