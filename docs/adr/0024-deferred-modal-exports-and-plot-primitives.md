# ADR 0024: Deferred Modal Exports, Plot Primitive Extraction, and Raster Tool Hooks

## Status
Accepted

## Context
Interactive tools in `math_explorer_gui` expose CSV, JSON, clipboard CSV, PNG, SVG, and GIF exports through `SimulationFramework`'s action bar and export modal. Three architectural defects undermined export fidelity and user feedback:

1. **Frame-Ordering Race in Modal Figure Exports**: `math_explorer_gui` clears `egui_plot`'s thread-local plot buffers at the start of every frame, after which `SimulationFramework::show` rendered the top bar and export modal *before* `tool.show(ctx)` populated the frame's `Plot` series. Clicking `Save PNG`, `Save SVG`, `Save GIF`, or `Copy to Clipboard` inside the modal therefore ran against empty thread-local plot state whenever a tool did not override `export_datasets`, producing blank background-only figures.
2. **Incomplete `egui_plot` Primitive Extraction**: Only `Line`, `Points`, and `Polygon` implemented `PlotItem::extract_points`, and offscreen PNG/SVG renderers treated every dataset as a connected polyline. Tools built on `BarChart` (`PartitionsWidget`, `ClebschGordanTool`, `ReplicatorDynamicsTool`, `SpikeAnalysisTool`), `Arrows` (`PotentialFlowTool`), `BoxPlot`, or raster textures (`DoseCalculationTool`, `LatticeBoltzmannTool`, `FractalViewer`, `IsingModelTool`, `PrimeSpiralWidget`, `AttentionMapsTool`) lost bar/vector rows in CSV/JSON or produced distorted polylines / blank backgrounds in PNG/SVG/GIF.
3. **Silent Empty Exports and Uniform Status Styling**: Tools without tabular datasets or figures left export buttons enabled, writing 12-byte header-only CSV (`Dataset,X,Y\n`) and 4-byte JSON (`[\n\n]`) files while coloring failure banners `Color32::GREEN` without dispatching screen-reader live-region announcements.

## Decision
1. **Unified End-of-Frame Deferred Export Queue (`File` + `Clipboard`)**: Extend `ExportFormat` (`Csv`, `Json`, `Png`, `Svg`, `Gif`) and `ExportRequest { format, target }` (`ExportTarget::{File, Clipboard}`) in `math_explorer_gui/src/framework/export_queue.rs` so both action-bar actions (`Export CSV`, `Copy CSV`, `Export JSON`) and modal figure actions (`Save PNG`, `Save SVG`, `Save GIF`, `Copy to Clipboard`) execute inside `dispatch_pending_exports` at end-of-frame *after* `tool.show(ctx)` has populated all `Plot` series and raster snapshots.
2. **Structured Points and Log-Aware `ExportPrimitive` Pipeline in `egui_plot`**:
   - Implement `PlotItem::extract_points` on `BarChart`, `Arrows`, and `BoxPlot` alongside `Line`, `Points`, and `Polygon`, and filter zero-point items (such as `HLine`/`VLine` guides) when recording `LAST_PLOT_DATASETS` so mixed-series tools preserve every tabular series.
   - Introduce `egui_plot::ExportPrimitive` (`Polyline`, `Points`, `Bar`, `Arrow`, `BoxElem`, `RasterImage`) captured across all `Plot` widgets in a frame via `LAST_PLOT_PRIMITIVES`, backed by shared log-10-aware coordinate projection (`PrimitiveCanvasLayout` honoring `ExportOptions::{log_x, log_y}`) across offscreen `ColorImage` (`export_primitives/raster.rs`) and vector SVG (`export_primitives/svg.rs`) renderers.
3. **Capability-Gated UI and Raster Hooks on `InteractiveTool`**:
   - Add `ExportCapabilities`, `export_plot_primitives`, and `export_raster_image` hooks to `InteractiveTool`.
   - Implement `export_raster_image` (`ExportCapabilities::raster_only()`) on texture-backed tools (`DoseCalculationTool`, `LatticeBoltzmannTool`, `FractalViewer`, `IsingModelTool`, `PrimeSpiralWidget`, `AttentionMapsTool`) and declare `ExportCapabilities::none()` on non-plot calculator/inspector tools so unsupported CSV, JSON, or figure buttons render disabled with explanatory hover tooltips.
4. **Payload Validation, Severity Coloring, and `TwinSlotLiveRegion` Announcements**:
   - Validate non-empty finite datasets and drawable primitives before writing files in `dispatch_pending_exports` and `PlotUi` exporters.
   - Track `ExportStatusSeverity` (`Success` -> `GREEN`/`polite`, `Warning` -> `YELLOW`/`assertive`, `Error` -> `RED`/`assertive`) and wire `announce_status_with_priority` through `TwinSlotLiveRegion` on both native and `wasm32` targets.

## Consequences
- **Positive**: Live modal PNG, SVG, and GIF exports capture populated frame data across all `Plot` and raster-backed tools; `BarChart`, `Arrows`, and `BoxPlot` items export accurately to CSV, JSON, PNG, and SVG; unsupported exports are prevented in the UI and rejected before touching disk; status banners and ARIA live-region announcements distinguish successes from warnings and errors.
- **Negative / Trade-offs**: Introducing `ExportCapabilities` and `ExportPrimitive` slightly expands the `InteractiveTool` and `egui_plot` public surfaces, requiring baseline API regeneration and modular submodules (`export_primitives/`, `export_types.rs`, `export_figure.rs`) to stay strictly within the 500-line file limit. All new allocations remain confined to cold UI export paths with zero impact on hot simulation loops.
