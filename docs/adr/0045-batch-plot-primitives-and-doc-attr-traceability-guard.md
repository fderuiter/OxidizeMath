# ADR 0045: Batched Plot Primitives, Interactive Grid Caching, and Doc-Attribute Traceability Guard

## Status
Accepted (extends [ADR 0011](0011-ast-citation-binding-and-cli-check-enforcement.md), [ADR 0012](0012-background-simulation-worker-offload.md), and [ADR 0044](0044-unify-build-time-traceability-engine.md))

## Context
1. **Per-Frame Plot Item Churn and Dataset Cloning**: Interactive 2D and 3D tools (`CurvatureHeatmap`, `SurfaceViewer`, `PotentialFlowTool`, `ComplexMappingTool`, and `LossLandscapeTool`) previously emitted hundreds to thousands of individual `egui_plot` line/point items and eagerly cloned 2D/3D export datasets on idle render frames, causing multi-second frame stalls at high grid densities.
2. **Unbounded Interactive Slider Ranges and Missing Per-Element Deletion**: Interactive grid sliders allowed densities up to solver registry limits (`1000`), stalling synchronous UI frames, while `PotentialFlowTool` lacked accessible individual superposition element removal.
3. **Rustdoc Example False Positives in `AstVisitor`**: Following [ADR 0044](0044-unify-build-time-traceability-engine.md) and [ADR 0044 (Missing Docs Ratchet)](0044-ratchet-missing-docs-and-core-api-documentation.md), `AstVisitor::process_fn_attrs` in [`crates/oxidize_core/src/ast_visitor.rs`](../../crates/oxidize_core/src/ast_visitor.rs) stringified `#[doc = "..."]` attributes when matching `#[verified]` and `#[embed_theory(...)]`, misclassifying Rustdoc code examples in [`crates/verified_engine_macros/src/lib.rs`](../../crates/verified_engine_macros/src/lib.rs) as unlinked paper citations.

## Decision
1. **Batched `egui_plot` Primitives, Shared-Axis Tabular Sidecar, and Deferred Datasets**:
   - Add `ColoredSegment`, `Line::from_colored_segments`, `PlotImage::with_backing_image`, `TabularSeries`, and `Arc`-backed deferred `Plot3DSeriesSidecar` / `LAST_PLOT_DATASETS*` handles in [`egui_plot/src/lib.rs`](../../egui_plot/src/lib.rs) and [`egui_plot/src/export_primitives/mod.rs`](../../egui_plot/src/export_primitives/mod.rs) so idle frames perform zero dataset vector clones while preserving PNG, SVG, CSV, JSON, and `Data Table` parity.
2. **Grid Caching, Interactive Slider Caps, and Individual Flow Element Deletion**:
   - Consolidate 3D wireframes in [`math_explorer_gui/src/camera/scene_batch.rs`](../../math_explorer_gui/src/camera/scene_batch.rs), cache manifold/field/conformal/landscape evaluations across `CurvatureHeatmap`, `SurfaceViewer`, `PotentialFlowTool`, `ComplexMappingTool`, and `LossLandscapeTool`, enforce `MAX_INTERACTIVE_GRID_SIZE = 120`, `MAX_INTERACTIVE_SURFACE_RESOLUTION = 120`, and `MAX_INTERACTIVE_CONFORMAL_GRID_SIZE = 150` in [`math_explorer_gui/src/presets.rs`](../../math_explorer_gui/src/presets.rs) (preserving `MAX_GRID_SIZE = 1000` in [`crates/math_commons/src/registry.rs`](../../crates/math_commons/src/registry.rs)), and add AccessKit-labeled `"✕ Remove"` controls in [`math_explorer_gui/src/tabs/fluid_dynamics/potential_flow/ui.rs`](../../math_explorer_gui/src/tabs/fluid_dynamics/potential_flow/ui.rs).
3. **Exclude `#[doc]` Attributes from Macro Attribute Matching in `AstVisitor`**:
   - Guard `#[verified]` and `#[embed_theory]` attribute inspection in [`crates/oxidize_core/src/ast_visitor.rs`](../../crates/oxidize_core/src/ast_visitor.rs) with `!attr.path().is_ident("doc")` while retaining `[cite:...]` docstring extraction.

## Consequences
- **Positive**: Sustains `< 16.67 ms` (`60 FPS`) headless `1920x1080` frame budgets across all slider positions, eliminates idle dataset allocations, and restores 100% build-time and runtime `TraceabilityReport` parity across all 28 papers.
- **Neutral**: Headless solver crates retain `MAX_GRID_SIZE = 1000`, while interactive UI sliders and untrusted preset/URL payloads clamp cleanly to interactive caps.
