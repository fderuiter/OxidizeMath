# ADR 20261123: Volumetric 3D Implicit Formulas, Boundary-Padded Solid Closure, and Unified 3D Exports

## Status

Accepted — 2026-11-23

## Context

`VolumetricFieldStudioTool` (`math_explorer_gui/src/tabs/quantum/volumetric_studio/`) previously supported two volumetric generators (`QuantumSuperposition` and `TuringMorphogenesis`), extracted open-boundary `MarchingCubes` level sets, and exposed a standalone inline `.obj` file dialog while reporting `ExportCapabilities::plot()` (`stl = false`, `ply = false`, `obj = false`) to the framework toolbar export modal.

Three workflow and topological gaps resulted:

1. **No user-authored 3D implicit / TPMS / CSG formula mode**: While `math_commons::ast::CompiledExpr::eval_3d` already supported stack-bytecode evaluation of `f(x, y, z)` expressions (including `min`, `max`, `clamp`, and `atan2`), users could not author custom 3D implicit level sets, triply-periodic minimal surfaces (Gyroid, Schwarz P, Schwarz D), or signed-distance CSG solids inside `VolumetricFieldStudioTool`.
2. **Open-boundary `MarchingCubes` meshes at box faces and cutting planes**: Standard `MarchingCubes::extract` leaves open boundary loops wherever an interior solid intersects the 6 bounding-box faces or an active half-space cutting plane, failing closed 2-manifold (`0` open edges, `0` non-manifold edges) and positive signed-volume requirements for 3D printing (`.stl`).
3. **Fragmented 3D export UX**: `VolumetricFieldStudioTool` bypassed the shared `SimulationFramework` 3D export pipeline (`export_mesh_3d`, `export_colored_mesh_3d`, `export_splat_ply_3d`) and could not export binary `.stl`, vertex-colored `.ply`, or INRIA-standard 3D Gaussian Splatting `.ply` scenes.

## Decision

### 1. Virtual 1-Voxel Collar Watertight Closure in `domain_applied::applied::isosurface`

We introduce `SolidPolarity` (`Sublevel` and `Superlevel`), `CuttingPlane`, `SolidClosureConfig`, `MarchingCubes::extract_solid`, `MarchingCubes::extract_solid_into`, `extract_solid_isosurface`, and `extract_solid_isosurface_into` in [solid_closure.rs](../../crates/domain_applied/src/applied/isosurface/solid_closure.rs) while keeping existing open-boundary `extract_isosurface` and `MarchingCubes::extract` unchanged:

- Instead of duplicating `48^3` voxel buffers on the heap, `MarchingCubes::extract_solid_into` evaluates a virtual `(W + 2) x (H + 2) x (D + 2)` scalar collar (`<= 50^3` for `48^3` voxels) on the fly, clamping exterior collar samples to `tau + eps_collar` beyond all 6 bounding-box faces and beyond any active `CuttingPlane`.
- Sublevel (`v < threshold`) and superlevel (`v > threshold`) regions are normalized onto a single signed level-set convention with deterministic exact-tie nudging (`2e-6`), outward-wound triangle indices, and exact planar unit normals on box/cutting-plane cap faces.
- `VolumetricEngine` pre-allocates `IsosurfaceWorkspace::with_locked_capacity(50, 50)` up front so per-frame solid extraction executes with `0` heap bytes under `VerifiedAllocator`, while cold-path 3D file exports allocate a fresh `IsosurfaceWorkspace::new_lazy` to guarantee zero triangle truncation at any grid resolution.

### 2. Custom 3D Implicit & TPMS Shell Mode (`FieldSourceMode::CustomImplicitFormula`)

We extend `VolumetricStudioConfig` and add [formula.rs](../../math_explorer_gui/src/tabs/quantum/volumetric_studio/formula.rs) to [mod.rs](../../math_explorer_gui/src/tabs/quantum/volumetric_studio/mod.rs):

- `MathExpressionInputBar` is configured with `.with_axes(&["x", "y", "z"])` and up to `MAX_SYNTHESIZED_PARAMS = 6` synthesized parameter sliders, plus built-in templates for Gyroid TPMS, Schwarz P, Schwarz D, Torus SDF, CSG boolean union/subtraction (`min`/`max`), and twisted helical tori (`atan2`/`clamp`), along with a new `StarterPresetMetadata` card (`volumetric_3d_gyroid_tpms`).
- Optional sheet-thickening (`shell_enabled` + `wall_thickness`) evaluates `|f(x, y, z) - c| - 0.5 * wall_thickness <= 0` so zero-thickness minimal surfaces become closed printable shells.
- Transient syntax errors during interactive typing keep `VolumetricStudioConfig::implicit_expr` and `VolumetricEngine`'s cached `CompiledExpr` pinned to the last valid bytecode so `FiniteStateGuard` never triggers false divergence rollbacks, whereas `try_load_state` strictly rejects malformed expressions or unknown JSON fields (`#[serde(deny_unknown_fields)]`).

### 3. Mode-Aware Unified 3D Solid & 3DGS `.ply` Exports

We remove the standalone inline `.obj` save dialog from `volumetric_studio/ui.rs` and route all 3D exports through `SimulationFramework`:

- In mesh-bearing render modes (`MarchingCubesMesh` and `HybridShellAndCloud`), `export_capabilities()` returns `ExportCapabilities::plot().with_3d_solid()` and exposes `export_mesh_3d` and `export_colored_mesh_3d` for watertight binary `.stl`, vertex-colored `.ply`, and `.obj`.
- In `VolumetricRenderMode::GaussianRadianceCloud`, `export_capabilities()` returns `ExportCapabilities::plot().with_3d_splat_ply()`, exporting INRIA-compatible 3DGS binary `.ply` scenes via `Gaussian3D::to_splat_ply` while disabling `.stl` and `.obj` with explanatory hover text and emitting `"assertive"` ARIA warnings via `TwinSlotLiveRegion` if requested.

```text
CustomImplicitFormula: CompiledExpr::eval_3d -> VoxelGrid (<= 48^3)
                                                    |
               +------------------------------------+------------------------------------+
               | MarchingCubesMesh / HybridShellAndCloud                                 | GaussianRadianceCloud
               v                                                                         v
MarchingCubes::extract_solid_into (virtual <= 50^3 collar + CuttingPlane)      Gaussian3D::to_splat_ply
               |                                                                         |
               v                                                                         v
Watertight 2-manifold ColoredMesh -> Binary .stl / Colored .ply / .obj         INRIA 3DGS Binary .ply
```

## Consequences

- **Positive**: Clipped spheres, `32^3` Gyroid TPMS shells, hydrogenic orbitals, and sliced 3D Turing gyroid labyrinths export as closed, outward-oriented 2-manifolds (`(0, 0)` open/non-manifold edges, `signed_volume > 0.0`) ready for 3D slicers, while `GaussianRadianceCloud` scenes load directly in INRIA-compatible 3DGS viewers.
- **Positive**: Hot-path 3D bytecode evaluation, collar-padded `MarchingCubes` extraction, and painter facet projection allocate `0` heap bytes per frame under `VerifiedAllocator`.
- **Trade-off**: Interactive cubic voxel resolution remains capped at `16..=48` (`50^3` collar cubes) to preserve 60 FPS interactive responsiveness on native CPU and WebAssembly targets without octree dual contouring.

## Verification

```bash
cargo test -p domain_applied --test test_isosurface --test test_isosurface_zero_alloc
cargo test -p math_explorer_gui --all-features volumetric_studio
cargo test -p math_explorer_gui --all-features deferred_export_audit_tests
cargo run -p unified_verification -- verify-records
```
