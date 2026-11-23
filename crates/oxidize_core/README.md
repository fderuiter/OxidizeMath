# Oxidize Core

Deterministic random generation, indexed 3D meshes, spatial boundary and iteration utilities, virtual filesystems, and workspace traceability infrastructure.

## 1. Overview & Architecture

The `oxidize_core` crate provides shared runtime primitives and static verification utilities across the OxidizeMath ecosystem. It supplies platform-independent foundations for deterministic stochastic seeding, indexed 3D geometry export, grid traversal, virtual asset I/O, Cargo workspace inspection, and academic citation traceability across native targets and `wasm32-unknown-unknown`.

### Module Structure

- **`ast_visitor`**: Syntax tree visitor (`AstVisitor`) counting functions, assertions, and `#[verified]` coverage across Rust source files.
- **`boundary`**: Coordinate boundary condition traits and clamping resolvers (`BoundaryCondition`, `NeumannBoundary`) for spatial grids.
- **`iteration`**: Interior and boundary 2D coordinate traversal helpers (`IterationPattern`) separating hot interior loops from edge handling.
- **`mesh`**: Indexed 3D geometry primitives (`Point3D`, `Mesh`, `VertexColor`, `ColoredMesh`), watertight 2-manifold solid builders (`extrude_polyline_tube`, `build_uv_sphere`, `build_cylinder`, `build_cone`, `build_arrow`), and zero-dependency binary/ASCII STL (`export_mesh_to_stl_binary`, `export_mesh_to_stl_ascii`), colored PLY (`export_colored_mesh_to_ply_binary`), INRIA 3DGS PLY (`export_splats_to_ply_binary`), and hardened Wavefront OBJ (`export_mesh_to_obj_string`) serializers.
- **`path_utils`**: Cross-platform forward-slash path normalization helpers (`normalize_path`, `strip_and_normalize`, `join_and_normalize`).
- **`rng`**: Deterministic pseudo-random number generator (`OxidizeRng`) backed by ChaCha8, enforcing workspace reproducible simulation policies.
- **`traceability`**: Literature citation scanner, shared `syn` AST cache (`AstCache`), and `TraceabilityEngine` / `TraceabilityReport` auditing `papers/` bindings.
- **`vfs`**: Asynchronous virtual file system layer (`VirtualFileSystem`, `NativeVfs`, `WasmVfs`, `DefaultVfs`) abstracting asset access across native and browser environments.
- **`workspace`**: Cargo workspace member discovery and manifest parsing utilities (`get_workspace_members`, `parse_workspace_members`).

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

Domain crates (`pure_math`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, `domain_physics`) and the `math_explorer` facade consume `rng`, `mesh`, `boundary`, and `iteration` primitives from `oxidize_core`, while numerical stepping contracts (`TimeStepper`, `EvolutionEngine`) live in `pure_math` and interactive orchestration (`SimulationFramework`) lives in `math_explorer_gui`. Developer gates in `unified_verification` and traceability views in `math_explorer_gui` share `ast_visitor`, `path_utils`, `traceability`, `vfs`, and `workspace`.
