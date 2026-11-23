# ADR 0041: Unified 3D Volumetric Field Studio and Zero-Allocation Surface/Splat Kernels

## Status
Accepted (extends [ADR 0025](0025-rolling-slice-edge-cache-marching-cubes.md))

## Context
Prior to this change, `math_explorer_gui` had dimensional and cross-domain silos across quantum mechanics, morphogenesis, computational geometry, and 3D neural rendering:
- `MarchingCubes`, `VoxelGrid`, and `legendre_p_associated` had zero production callers in `math_explorer_gui`.
- `WaveUnified` (`QuantumTab`) operated strictly in 1D, and `MorphogenesisRunner` (`MorphogenesisTab`) operated strictly in 2D via `FiniteDifference2D`.
- `GaussianSplattingTool` (`AiTab`) rendered only a hardcoded 8-splat `CANONICAL_SCENE`.
- `MarchingCubes::extract` and `project_scene` allocated fresh output `Vec` buffers per call, which prevented running per-frame 3D PDE stepping, surface extraction, and EWA splat projection inside a `VerifiedAllocator`-locked hot path (`VerifiedEngine::run_verified`).

## Decision
1. **3D Special Functions & Quantum Eigenstate Superpositions (`pure_math`, `domain_physics`)**:
   - Add `associated_laguerre(n, alpha, x)` in `pure_math::special_functions::polynomials` and extend `legendre_p_associated(l, m, x)` to negative magnetic quantum numbers (`-l <= m < 0`) alongside orthonormal `real_spherical_harmonic(l, m, cos_theta, phi)`.
   - Introduce `HydrogenicSuperposition3D` in `domain_physics::quantum`, pre-computing spatial basis grids up to `48^3` voxels, evaluating `write_field_into` via zipped slice iterators, and advancing two-state complex amplitudes in-place via `construct_two_state_eigen_hamiltonian`, `time_evolution_operator`, and `evolve_state_in_place`.
2. **7-Point Neumann 3D Reaction-Diffusion (`pure_math`, `domain_biology`)**:
   - Extend `StencilOperators` and `FusedStencilStepper` with `step_3d_coupled_neumann` and implement `FiniteDifference3D` + `TuringSystem::new_3d` in `domain_biology` for zero-allocation 3D Schnakenberg morphogenesis.
3. **Zero-Allocation Rolling-Slice Isosurface & Splat Projection (`oxidize_core`, `domain_applied`, `domain_ai`)**:
   - Add `Mesh::with_capacity` / `Mesh::clear` in `oxidize_core`, `VoxelGrid::reconfigure_in_place` and `IsosurfaceWorkspace::with_locked_capacity` + `MarchingCubes::extract_into` in `domain_applied` (reusing the rolling two-slice `SliceEdgeCache` from [ADR 0025](0025-rolling-slice-edge-cache-marching-cubes.md)), and `project_scene_into` in `domain_ai` using zero-allocation `sort_unstable_by`.
4. **Unified `VolumetricFieldStudioTool` & Shared Scene Builders (`math_explorer_gui`)**:
   - Register `VolumetricFieldStudioTool` inside `QuantumTab` on `Plot3DViewport` and `Camera3D`, clamping interactive cubic resolution to `16..=48` voxels for responsive cooperative WebAssembly execution.
   - Support toggling between `MarchingCubes` nodal meshes (with Wavefront `.obj` export via `export_mesh_to_obj_string` and `spawn_save_as_file_dialog`), gradient-aligned `Gaussian3D` EWA radiance clouds (with `determine_density_action` telemetry), and hybrid shell-in-cloud overlays, plus a 3D half-space cutting plane and click perturbation/slice probe.
   - Share `volumetric_scene` dynamic `Scene` builders across `QuantumTab`, `AiTab` (`GaussianSplattingTool` presets beyond `CANONICAL_SCENE`), and `MorphogenesisTab` (`2D -> 3D` slice splat preview) under `Cargo.toml` feature guards.

```text
HydrogenicSuperposition3D / TuringSystem<FiniteDifference3D>
                      │ (zero heap alloc)
                      ▼
             VoxelGrid (16^3 ..= 48^3)
           ┌──────────┴──────────┐
           ▼                     ▼
MarchingCubes::extract_into   extract_voxel_splats_into
 + Cutting Plane + .obj        + project_scene_into (EWA)
           └──────────┬──────────┘
                      ▼
          Plot3DViewport + Camera3D
```

## Consequences
- **Positive**: Activates `MarchingCubes`, `VoxelGrid`, `legendre_p_associated`, `associated_laguerre`, `FiniteDifference3D`, `export_mesh_to_obj_string`, and dynamic `Gaussian3D` projection in production with `VerifiedAllocator`-verified zero-allocation hot-path execution.
- **Positive**: Enables interactive exploration of dipole selection rules (`1s + 2pz`), concentric `2s`/`3s` radial nodal shells, `3dz2` nodal cones, and 3D Turing gyroid labyrinths on both native desktop and `wasm32` targets.
- **Negative / Trade-offs**: Pre-allocating workspace buffers up to `48^3` voxels reserves ~4–6 MB of reusable heap memory per active `VolumetricFieldStudioTool` instance, and interactive grid dimensions are capped at `48^3` until future WebGPU compute shader offloads are introduced.
