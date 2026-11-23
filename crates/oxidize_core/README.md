# Oxidize Core

Deterministic random generation, indexed 3D meshes, spatial boundary and iteration utilities, virtual filesystems, and workspace traceability infrastructure.

## 1. Overview & Architecture

The `oxidize_core` crate provides shared runtime primitives and static verification utilities across the OxidizeMath ecosystem. It supplies platform-independent foundations for deterministic stochastic seeding, indexed 3D geometry export, grid traversal, virtual asset I/O, Cargo workspace inspection, and academic citation traceability across native targets and `wasm32-unknown-unknown`.

### Module Structure

- **`ast_visitor`**: Syntax tree visitor (`AstVisitor`) counting functions, assertions, and `#[verified]` coverage across Rust source files.
- **`boundary`**: Coordinate boundary condition traits and clamping resolvers (`BoundaryCondition`, `NeumannBoundary`) for spatial grids.
- **`iteration`**: Interior and boundary 2D coordinate traversal helpers (`IterationPattern`) separating hot interior loops from edge handling.
- **`mesh`**: Indexed 3D geometry primitives (`Point3D`, `Mesh`), manifold diagnostics (`MeshDiagnostics`, `analyze_mesh`), and binary STL / Wavefront OBJ serialization (`export_mesh_to_stl_bytes`, `export_mesh_to_obj_string`).
- **`path_utils`**: Cross-platform forward-slash path normalization helpers (`normalize_path`, `strip_and_normalize`, `join_and_normalize`).
- **`rng`**: Deterministic pseudo-random number generator (`OxidizeRng`) backed by ChaCha8, enforcing workspace reproducible simulation policies.
- **`traceability`**: Literature citation scanner, shared `syn` AST cache (`AstCache`), and `TraceabilityEngine` / `TraceabilityReport` auditing `papers/` bindings.
- **`vfs`**: Asynchronous virtual file system layer (`VirtualFileSystem`, `NativeVfs`, `WasmVfs`, `DefaultVfs`) abstracting asset access across native and browser environments.
- **`workspace`**: Cargo workspace member discovery and manifest parsing utilities (`get_workspace_members`, `parse_workspace_members`).

---

## 2. Developer Instructions

### Working with Indexed Meshes and Deterministic RNG

Initialize reproducible simulation seeds with `OxidizeRng` and construct indexed 3D surface geometry using `Mesh` and `Point3D`:

```rust
use oxidize_core::mesh::{Mesh, Point3D, export_mesh_to_obj_string};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

// Initialize a deterministic random provider
let mut rng = OxidizeRng::new(42);
let seed_val = rng.next_u64();
assert!(seed_val > 0);

// Construct an indexed 3D triangle mesh
let mesh = Mesh {
    vertices: vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(1.0, 0.0, 0.0),
        Point3D::new(0.0, 1.0, 0.0),
    ],
    normals: vec![Point3D::new(0.0, 0.0, 1.0); 3],
    indices: vec![0, 1, 2],
};

// Export the indexed mesh buffer to Wavefront OBJ format
let obj = export_mesh_to_obj_string(&mesh).unwrap();
assert!(obj.contains("f 1//1 2//2 3//3"));
```

---

## 3. Ecosystem Integration

Domain crates (`pure_math`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, `domain_physics`) and the `math_explorer` facade consume `rng`, `mesh`, `boundary`, and `iteration` primitives from `oxidize_core`, while numerical stepping contracts (`TimeStepper`, `EvolutionEngine`) live in `pure_math` and interactive orchestration (`SimulationFramework`) lives in `math_explorer_gui`. Developer gates in `unified_verification` and traceability views in `math_explorer_gui` share `ast_visitor`, `path_utils`, `traceability`, `vfs`, and `workspace`.
