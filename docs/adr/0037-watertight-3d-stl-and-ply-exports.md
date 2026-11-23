# ADR 0037: Unified Watertight 3D STL, Colored PLY, and 3DGS PLY Exports

## Status
Accepted

## Context
Prior to this decision, 3D simulations in `math_explorer_gui` suffered from fragmented or non-manifold export paths:

1. **Projected 2D Flattening and Disabled 3D Exports**: Five of the six 3D tools (`CurvatureHeatmap`, `AttractorPlotter`, `CrystalViewer`, `SpinVisualizer`, and `SurfaceViewer` in the top action bar) flattened 3D geometry onto 2D plot polylines or disabled exports altogether (`GaussianSplattingTool` returned `ExportCapabilities::none()`).
2. **Incomplete Core Mesh Serialization and Open/Singular Topology**: `oxidize_core::mesh` provided only `export_mesh_to_obj_string` without non-finite vertex guards or fallback facet normals, and `geometry_topology::export_utils::surface_to_mesh` swept `Sphere` over `[0, 2*PI]`, producing `NaN` normals at the poles and leaving periodic boundary edges open, while `KleinBottle` immersions lacked physical wall thickness for 3D slicing.
3. **Ad-Hoc Browser/Desktop File Saving**: `SurfaceViewer` bypassed `SimulationFramework`'s deferred export queue (ADR 0024) by invoking `rfd::FileDialog` and `WasmVfs::write_to_file` inline during `show_inside`, skipping `TwinSlotLiveRegion` screen-reader announcements and MIME-typed `Blob` downloads.

## Decision
1. **Zero-Dependency STL, Colored PLY, Hardened OBJ, and 3DGS PLY Serializers in `oxidize_core::mesh`**:
   - Keep `Mesh` unchanged for backward compatibility and introduce `VertexColor` and `ColoredMesh` in `crates/oxidize_core/src/mesh/types.rs`.
   - Implement little-endian binary (`84 + 50 * F` bytes) and ASCII `.stl` writers/parsers (`export_stl.rs`), binary and ASCII uncolored/colored `.ply` writers (`export_ply.rs`), and an INRIA/`SuperSplat`-standard 3D Gaussian Splatting binary `.ply` writer (`export_splat_ply.rs`) encoding degree-0 spherical harmonic DC terms (`f_dc_0..2`), logit opacity, log-scales (`scale_0..2`), and unit quaternions (`rot_0..3`, scalar-first `(w, x, y, z)`).
   - Harden `export_mesh_to_obj_string` (`export.rs`) to reject non-finite coordinates or out-of-bounds indices and replace degenerate vertex normals with finite right-hand facet normals.
   - Add rotation-minimizing frame (double-reflection RMF) tube extrusion (`extrude_polyline_tube`, clamped to `<= 32` radial segments with welded end-caps or holonomy-distributed closed loops) alongside closed 2-manifold `build_uv_sphere`, `build_cylinder`, `build_cone`, and `build_arrow` builders in `primitives.rs`.
2. **Manifold Seam Welding and Normal-Offset Shell Extrusion**:
   - Update `math_explorer_gui/src/tabs/geometry_topology/export_utils.rs` so polar surfaces (`Sphere`) sweep `v` over `[0, PI]` with welded pole fans and exact `(0, 0, ±1)` normals, doubly periodic surfaces (`Torus` and periodic `CustomParametric` immersions) weld `u` and `v` seams, `KleinBottle` immersions extrude a normal-offset wall shell (`0.05..=0.50`) stitching seams into an orientable watertight solid, and open rectangular patches (`CustomHeightfield` and non-periodic `CustomParametric` formulas) extrude a rim-stitched solid slab (`0` open boundary edges, `0` `NaN` normals).
3. **Unified Deferred 3D Export Dispatch (`Stl`, `Ply`, `Obj`)**:
   - Extend `ExportCapabilities` (`stl`, `ply`, `obj`, `with_3d_solid()`, `splat_ply_only()`) and `ExportFormat` (`Stl`, `Ply`, `Obj`), routing cold-path `export_mesh_3d`, `export_colored_mesh_3d`, and `export_splat_ply_3d` hooks through `SimulationFramework::dispatch_3d_export` (`export_3d.rs`) and `complete_file_export`.
   - Map `.stl` (`model/stl`), `.ply` (`model/ply`), and `.obj` (`model/obj`) in `egui_plot::file_io::infer_mime_type` so WebAssembly downloads strictly use CSP-compliant local `Blob` URLs, and retire the inline `Export to OBJ` `WasmVfs`/`rfd` block in `SurfaceViewer`.
   - Wire all six 3D tools (`SurfaceViewer`, `CurvatureHeatmap`, `AttractorPlotter`, `CrystalViewer`, `SpinVisualizer`, and `GaussianSplattingTool`) on cold export paths so hot `step` loops allocate `0` mesh bytes.

## Consequences
- **Positive**: All five geometric 3D tools export watertight, outward-wound `.stl`, `.ply`, and `.obj` solids ready for PrusaSlicer, Cura, MeshLab, and Blender, while `GaussianSplattingTool` exports standard 3DGS `.ply` scenes for `SuperSplat` — with zero external crate dependencies, zero hot-loop allocations, full `TwinSlotLiveRegion` accessibility announcements, and compressed `.wasm` size well within the 3 MiB budget.
- **Negative / Trade-offs**: Expanding `oxidize_core::mesh`, `domain_ai::gaussian_splatting::Gaussian3D`, `egui_plot::file_io`, and `SimulationFramework` requires regenerating `api_baseline.json` and partitioning tool export logic into dedicated submodules (`export_3d.rs`, `curvature_export.rs`, `attractors/export.rs`, `spin_export.rs`, `gaussian_splatting_export.rs`) to stay strictly within the 500-line file and 60-line function ceilings.
