# ADR 20261123: Volumetric 3D Implicit Formulas, Boundary-Padded Solid Closure, and Unified 3D Exports

## Status

Accepted — 2026-11-23

## Context

`VolumetricFieldStudioTool` (`math_explorer_gui/src/tabs/quantum/volumetric_studio/`) supports volumetric quantum eigenstates, 3D Turing reaction-diffusion fields, and analytical 3D implicit CAD equations (`FieldSourceMode::CustomImplicit`, see [ADR 20261123: Implicit CAD Volumetric Studio, Boundary-Capped Marching Cubes, and Millimeter STL Export](20261123-implicit-cad-capped-isosurface-and-mm-stl.md)).

Three complementary kernel and export capabilities complete the studio's 3D solid and radiance-cloud pipeline:

1. **Zero-buffer-duplication virtual-collar solid closure in `domain_applied`**: Callers operating on arbitrary `VoxelGrid` instances without an auxiliary `(W + 2) x (H + 2) x (D + 2)` apron buffer need a zero-allocation `MarchingCubes::extract_solid[_into]` entrypoint supporting both `Sublevel` (`v < threshold`) and `Superlevel` (`v > threshold`) polarities and axis-aligned `CuttingPlane` caps.
2. **Extended SDF / helical implicit templates and implicit splat sampling**: `ImplicitFormulaUi` benefits from Torus SDF and twisted helical torus (`atan2`/`clamp`) presets with 6-parameter slider specifications (`k`, `a`, `b`, `c`, `r`, `R`) plus gradient-aligned `Gaussian3D` splat sampling inside implicit shells.
3. **Vertex-colored millimeter `.ply` and mode-aware 3DGS `.ply` exports**: In mesh-bearing render modes, `.ply` exports should carry height-shaded RGB `VertexColor` attributes (`export_colored_mesh_3d`) scaled in millimeters alongside `.stl` (`UNITS=mm`) and `.obj`, whereas `VolumetricRenderMode::GaussianRadianceCloud` should advertise `ExportCapabilities::plot().with_3d_splat_ply()` to export INRIA-standard 3DGS binary `.ply` scenes while disabling `.stl`/`.obj` with `"assertive"` ARIA warnings.

## Decision

### 1. Virtual 1-Voxel Collar Watertight Closure in `domain_applied::applied::isosurface`

We introduce `SolidPolarity` (`Sublevel` and `Superlevel`), `CuttingPlane`, `SolidClosureConfig`, `MarchingCubes::extract_solid`, `MarchingCubes::extract_solid_into`, `extract_solid_isosurface`, and `extract_solid_isosurface_into` in [solid_closure.rs](../../crates/domain_applied/src/applied/isosurface/solid_closure.rs) alongside [capped.rs](../../crates/domain_applied/src/applied/isosurface/capped.rs) and open-boundary `MarchingCubes::extract`:

- Without allocating a padded voxel copy, `MarchingCubes::extract_solid_into` evaluates a virtual `(W + 2) x (H + 2) x (D + 2)` scalar collar on the fly, clamping exterior collar samples beyond all 6 bounding-box faces and beyond any active `CuttingPlane`.
- Sublevel and superlevel regions are normalized onto a single signed level-set convention with deterministic exact-tie nudging (`2e-6`), outward-wound triangle indices, and exact planar unit normals on box/cutting-plane cap faces, executing with `0` heap bytes per frame under `VerifiedAllocator`.

### 2. Unified Implicit CAD Templates & Implicit `Gaussian3D` Splat Sampling

We enrich [implicit.rs](../../math_explorer_gui/src/tabs/quantum/volumetric_studio/implicit.rs) and [export_3d.rs](../../math_explorer_gui/src/tabs/quantum/volumetric_studio/export_3d.rs):

- `ImplicitFormulaUi` adds `Torus SDF` and `Twisted Helical Torus` (`atan2`/`clamp`) templates and 6 named parameter specs (`k`, `a`, `b`, `c`, `r`, `R`), keeping transient syntax errors isolated in `MathExpressionInputBar` so `FiniteStateGuard` never triggers false divergence rollbacks while `try_load_state` strictly rejects malformed formulas or unknown JSON fields (`#[serde(deny_unknown_fields)]`).
- `extract_implicit_splats_into` samples gradient-aligned `Gaussian3D` splats across occupied implicit solid/wall voxels with `0` heap allocations on the hot path.

### 3. Mode-Aware Colored Millimeter `.ply` & INRIA 3DGS `.ply` Exports

We route mode-aware 3D exports through [mod.rs](../../math_explorer_gui/src/tabs/quantum/volumetric_studio/mod.rs) and [export_3d.rs](../../math_explorer_gui/src/tabs/quantum/volumetric_studio/export_3d.rs):

- In `MarchingCubesMesh` and `HybridShellAndCloud`, `export_capabilities()` returns `ExportCapabilities::plot().with_3d_solid()`, exposing millimeter-scaled `export_mesh_3d` (`.stl` with `UNITS=mm`, `.obj`) and `export_colored_mesh_3d` (vertex-colored `.ply`).
- In `GaussianRadianceCloud`, `export_capabilities()` returns `ExportCapabilities::plot().with_3d_splat_ply()`, exporting INRIA-compatible 3DGS binary `.ply` scenes via `Gaussian3D::to_splat_ply` while disabling `.stl` and `.obj` with explanatory hover text and `"assertive"` `TwinSlotLiveRegion` ARIA announcements.

```text
CustomImplicit / Quantum / Turing -> VoxelGrid (<= 48^3)
                                         |
            +----------------------------+----------------------------+
            | MarchingCubesMesh / HybridShellAndCloud                 | GaussianRadianceCloud
            v                                                         v
Watertight Capped Mesh + scale_mesh_to_millimeters          Gaussian3D::to_splat_ply
            |                                                         |
            v                                                         v
Binary .stl (UNITS=mm) / Colored .ply / .obj                INRIA 3DGS Binary .ply
```

## Consequences

- **Positive**: Clipped spheres, Gyroid/Schwarz-P TPMS shells, CSG solids, hydrogenic orbitals, and sliced 3D Turing gyroid labyrinths export as closed, outward-oriented millimeter-scaled 2-manifolds (`(0, 0)` open/non-manifold edges, `signed_volume > 0.0`) with vertex-colored `.ply` support, while `GaussianRadianceCloud` scenes load directly in INRIA-compatible 3DGS viewers.
- **Positive**: Hot-path 3D bytecode evaluation, isosurface extraction, and splat projection allocate `0` heap bytes per frame under `VerifiedAllocator`.
- **Trade-off**: Interactive cubic voxel resolution remains capped at `16..=48` to sustain 60 FPS on native CPU and WebAssembly targets within the 4 MB compressed `.wasm` budget in [verify_wasm_budget.sh](../../scripts/verify_wasm_budget.sh) across the full multi-tab workspace (`28` tabs, `87` tools, `148` presets).

## Verification

```bash
cargo test -p domain_applied --test test_isosurface --test test_isosurface_zero_alloc
cargo test -p math_explorer_gui --all-features volumetric_studio
cargo test -p math_explorer_gui --all-features deferred_export_audit_tests
cargo run -p unified_verification -- verify-records
```
