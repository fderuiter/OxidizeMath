# Oxidize Core

Foundational simulation traits, deterministic random generation, virtual filesystems, and indexed 3D meshes.

## 1. Overview & Architecture

The `oxidize_core` crate defines the primary architectural contracts and runtime utilities for simulations in the OxidizeMath ecosystem. It provides the standardized lifecycle models for configuration, state representation, and step execution.

By decoupling simulation state contracts from concrete numerical algorithms and graphical renderers, simulation models can be executed deterministically in native headless tests, multi-threaded worker pools, and WebAssembly browser targets without architecture divergence.

### Module Structure

- **Core Lifecycle Traits**:
  - `ModelConfig`: Serializable configuration contract for simulation parameters.
  - `ModelState`: Cloneable state representation supporting checkpoints and history buffers.
  - `SimulationModel`: Lifecycle interface defining `initialize`, `step`, and `get_state`.
- **`rng`**: Deterministic pseudo-random number generator (`OxidizeRng`) based on ChaCha8, enforcing workspace reproducible simulation policies.
- **`mesh`**: Indexed 3D geometry primitives (`Point3D`, `Mesh`, `VertexColor`, `ColoredMesh`), watertight 2-manifold solid builders (`extrude_polyline_tube`, `build_uv_sphere`, `build_cylinder`, `build_cone`, `build_arrow`), and zero-dependency binary/ASCII STL (`export_mesh_to_stl_binary`, `export_mesh_to_stl_ascii`), colored PLY (`export_colored_mesh_to_ply_binary`), INRIA 3DGS PLY (`export_splats_to_ply_binary`), and hardened Wavefront OBJ (`export_mesh_to_obj_string`) serializers.
- **`boundary`**: Coordinate boundary condition traits and clamping resolvers (`BoundaryCondition`, `NeumannBoundary`).
- **`iteration`**: Interior and boundary coordinate traversal helpers (`IterationPattern`).
- **`vfs`**: Virtual file system layer abstracting asset access across native and web environments.
- **`workspace`**: Cargo workspace inspection, path normalization, and manifest parsing utilities.

---

## 2. Developer Instructions

### Working with Watertight 3D Meshes, STL/PLY Exports, and Deterministic RNG

Initialize reproducible simulation seeds with `OxidizeRng`, build closed 2-manifold primitives or swept tubes, and serialize geometry to binary `.stl`, per-vertex colored `.ply`, and Wavefront `.obj`:

```rust
use oxidize_core::mesh::{
    ColoredMesh, Point3D, VertexColor, build_uv_sphere, export_colored_mesh_to_ply_binary,
    export_mesh_to_obj_string, export_mesh_to_stl_binary, extrude_polyline_tube,
};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

// Initialize a deterministic random provider
let mut rng = OxidizeRng::new(42);
let seed_val = rng.next_u64();
assert!(seed_val > 0);

// Build a closed 2-manifold UV sphere and serialize to binary STL (84 + 50 * F bytes)
let sphere = build_uv_sphere(Point3D::new(0.0, 0.0, 0.0), 1.0, 16, 12).unwrap();
let stl_bytes = export_mesh_to_stl_binary(&sphere).unwrap();
assert_eq!(stl_bytes.len(), 84 + 50 * (sphere.indices.len() / 3));

// Extrude a 3D polyline into a capped rotation-minimizing tube and export to OBJ
let curve = [
    Point3D::new(0.0, 0.0, 0.0),
    Point3D::new(1.0, 0.5, 0.5),
    Point3D::new(2.0, 0.0, 1.0),
];
let tube = extrude_polyline_tube(&curve, 0.15, 12, false).unwrap();
let obj = export_mesh_to_obj_string(&tube).unwrap();
assert!(obj.contains("f "));

// Attach per-vertex RGB colors and serialize to little-endian binary PLY
let colors = vec![VertexColor::new(40, 180, 240); sphere.vertices.len()];
let colored = ColoredMesh::new(sphere, colors);
let ply_bytes = export_colored_mesh_to_ply_binary(&colored).unwrap();
assert!(ply_bytes.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
```

---

## 3. Ecosystem Integration

All domain simulation engines (`domain_ai`, `domain_physics`, `domain_biology`, `domain_applied`, `domain_climate`, `domain_epidemiology`) implement the lifecycle traits from `oxidize_core`, allowing the unified runner in `math_explorer` to schedule simulation steps seamlessly.
