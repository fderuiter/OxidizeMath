# Implicit CAD Volumetric Studio, Boundary-Capped Marching Cubes, and Millimeter STL Export

## Status

Accepted (2026-11-23)

## Context

`VolumetricFieldStudioTool` previously evolved 3D hydrogenic orbital superpositions and 3D Schnakenberg reaction-diffusion fields using uncapped `MarchingCubes` extraction and a bespoke Wavefront `.obj`-only save dialog (`trigger_obj_export_dialog`). Periodic surfaces (such as Gyroid and Schwarz-P TPMS lattices) and cutting-plane cross sections left open boundary loops where the level set met the bounding box or cutting plane, preventing direct 3D slicer ingestion as watertight `.stl` or `.ply` solids. Additionally, desktop 3D slicers interpret raw `.stl` coordinates in millimeters (`mm`) and look for `UNITS=mm` in the 80-byte binary header.

## Decision

1. **Boundary-Capped Zero-Allocation Isosurface Extraction (`domain_applied::isosurface::capped`)**:
   - Introduce `extract_capped_isosurface_into` and `extract_capped_isosurface` along with `CutPlaneCap` and `CappedBoxBounds`.
   - Embed the $N_x \times N_y \times N_z$ scalar field inside a pre-allocated $(N_x + 2) \times (N_y + 2) \times (N_z + 2)$ 1-voxel outer apron (and sub-voxel cutting-plane apron) initialized strictly outside the solid so complement-disambiguated `MarchingCubes` seals every boundary and cutting-plane intersection into a closed, outward-wound 2-manifold (`(0, 0)` open/non-manifold edges, `signed_volume > 0`, and finite unit normals).
2. **Millimeter Scaling and `UNITS=mm` Binary `.stl` Header (`oxidize_core::mesh::export_stl`)**:
   - Update `STL_HEADER_TEXT` to `b"OxidizeMath binary STL export; UNITS=mm"`.
   - Add `MIN_EXPORT_SPAN_MM` (`5.0`), `MAX_EXPORT_SPAN_MM` (`250.0`), `scale_mesh_to_millimeters`, and `export_mesh_to_stl_binary_scaled` to center and uniformly scale exported meshes to user-selected physical spans in millimeters.
3. **Browser-Native Implicit CAD Mode and Unified Solid 3D Export (`VolumetricFieldStudioTool`)**:
   - Add `FieldSourceMode::CustomImplicit` and `ImplicitClosureMode` (`HalfSpaceSolid` for $f(x, y, z) \le c$ and `ThickenedWall` for $|f(x, y, z) - c| \le w/2$) backed by `CompiledExpr::eval_3d` (supporting trigonometric, exponential, and CSG `min`/`max`/`clamp` functions), auto-synthesized parameter sliders (`MathExpressionInputBar`), and ordered X/Y/Z `[min, max]` domain bounds, executing with zero heap allocations on `refresh_zero_alloc` and `step_zero_alloc`.
   - Wire `VolumetricFieldStudioTool` into `SimulationFramework`'s deferred 3D solid export pipeline (`ExportCapabilities::plot().with_3d_solid()` and `export_mesh_3d()`), retire `trigger_obj_export_dialog`, and register bookmarkable starter presets for Gyroid TPMS, Schwarz-P, and CSG Boolean solids.

## Consequences

- `VolumetricFieldStudioTool` joins the audited 3D solid tool suite (`solid_3d_count == 6`), producing closed, outward-wound `.stl`, `.ply`, and `.obj` solids scaled in millimeters (`5.0..=250.0 mm`) across implicit CAD formulas, hydrogenic orbitals, and cut-plane 3D Turing labyrinths.
- Hot-path parameter drags (`refresh_zero_alloc`) and playback ticks (`step_zero_alloc`) remain strictly zero-allocation under `VerifiedAllocator` while preserving the single-binary `< 3 MB` WebAssembly budget.
