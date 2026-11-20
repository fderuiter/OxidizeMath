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
- **`mesh`**: Indexed 3D geometry primitives (`Point3D`, `Mesh`) and Wavefront OBJ serialization (`export_mesh_to_obj_string`).
- **`boundary`**: Coordinate boundary condition traits and clamping resolvers (`BoundaryCondition`, `NeumannBoundary`).
- **`iteration`**: Interior and boundary coordinate traversal helpers (`IterationPattern`).
- **`vfs`**: Virtual file system layer abstracting asset access across native and web environments.
- **`workspace`**: Cargo workspace inspection, path normalization, and manifest parsing utilities.

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

All domain simulation engines (`domain_ai`, `domain_physics`, `domain_biology`, `domain_applied`, `domain_climate`, `domain_epidemiology`) implement the lifecycle traits from `oxidize_core`, allowing the unified runner in `math_explorer` to schedule simulation steps seamlessly.
