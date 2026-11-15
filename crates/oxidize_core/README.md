# Oxidize Core

Foundational simulation traits, deterministic random generation, virtual filesystems, and spatial grids.

## 1. Overview & Architecture

The `oxidize_core` crate defines the primary architectural contracts and runtime utilities for simulations in the OxidizeMath ecosystem. It provides the standardized lifecycle models for configuration, state representation, and step execution.

By decoupling simulation state contracts from concrete numerical algorithms and graphical renderers, simulation models can be executed deterministically in native headless tests, multi-threaded worker pools, and WebAssembly browser targets without architecture divergence.

### Module Structure

- **Core Lifecycle Traits**:
  - `ModelConfig`: Serializable configuration contract for simulation parameters.
  - `ModelState`: Cloneable state representation supporting checkpoints and history buffers.
  - `SimulationModel`: Lifecycle interface defining `initialize`, `step`, and `get_state`.
- **`grid`**: Generic 2D spatial grid container (`Grid2D`) supporting coordinate flattening, row/column slicing, and element mutation.
- **`rng`**: Deterministic pseudo-random number generator (`OxidizeRng`) based on ChaCha8, enforcing workspace reproducible simulation policies.
- **`double_buffer`**: Ping-pong buffer abstractions (`DoubleBuffer`, `DoubleBufferedState`) to prevent race conditions during state stepping.
- **`boundary`**: Invariant boundary condition representations (Periodic, Dirichlet, Neumann).
- **`vfs`**: Virtual file system layer abstracting asset access across native and web environments.
- **`workspace`**: Cargo workspace inspection and manifest parsing utilities.

---

## 2. Developer Instructions

### Working with Spatial Grids and Deterministic RNG

Initialize reproducible spatial simulation environments using `Grid2D` and `OxidizeRng`:

```rust
use oxidize_core::grid::Grid2D;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

// Initialize a deterministic random provider
let mut rng = OxidizeRng::new(42);
let seed_val = rng.next_u64();
assert!(seed_val > 0);

// Allocate a 2D spatial simulation grid
let mut grid = Grid2D::new(8, 8, 0.0);
assert_eq!(grid.width, 8);
assert_eq!(grid.height, 8);

// Access and update grid cells via (row, col) coordinates
grid[(2, 3)] = 100.0;
assert_eq!(grid[(2, 3)], 100.0);
assert_eq!(grid[(0, 0)], 0.0);
```

---

## 3. Ecosystem Integration

All domain simulation engines (`domain_ai`, `domain_physics`, `domain_biology`, `domain_applied`, `domain_climate`, `domain_epidemiology`) implement the lifecycle traits from `oxidize_core`, allowing the unified runner in `math_explorer` to schedule simulation steps seamlessly.
