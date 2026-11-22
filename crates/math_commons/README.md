# Math Commons

Standardized mathematical constants, spatial indexing, colormaps, and primitive geometry.

## 1. Overview & Architecture

The `math_commons` crate provides foundational mathematical primitives, physical constant registries, and numerical utilities across the OxidizeMath ecosystem.

By standardizing universal constants, color mapping formulas, and spatial coordinate indexing in a single shared crate, numerical kernels and graphical user interfaces maintain consistent calculation baselines and render outputs without duplicate implementations.

### Module Structure

- **`ast`**: Native mathematical expression parser, LaTeX formatter, and zero-allocation postfix stack bytecode evaluator (`CompiledExpr`) for real and complex formulas.
- **`constants`**: Centralized registry of physical, astronomical, and mathematical constants:
  - Speed of light (`C`), Gravitational constant (`G`), Solar mass (`SOLAR_MASS`).
  - Reduced Planck constant (`HBAR_C`), Proton mass (`PROTON_MASS`), Neutron mass (`NEUTRON_MASS`).
  - Liquid drop nuclear model constants and floating-point tolerances (`EPSILON`).
- **`contours`**: 2D marching-squares isoline extraction (`extract_contours`), undirected segment deduplication (`deduplicate_segments`), nearest-neighbor toolpath ordering (`order_paths_nearest_neighbor`), and RK4 streamline tracing (`trace_streamlines_rk4`).
- **`math_kernel`**: Standardized numerical routines:
  - `types`: 2D/3D index flattening (`flatten_2d_index`) and spatial offsets.
  - `colormap`: Heatmap color generation (`heatmap_color`) and gradient interpolation for scientific visualization.
- **`primitives`**: Basic geometric constructs and coordinate structures.
- **`registry`**: Constant lookup and numeric tolerance validation.
- **`wigner`**: Clebsch-Gordan coefficients and quantum angular momentum arithmetic.

---

## 2. Developer Instructions

### Coordinate Flattening and Colormap Evaluation

Utilize spatial index flattening and scientific color palette evaluation:

```rust
use math_commons::constants::C;
use math_commons::math_kernel::colormap::heatmap_color;
use math_commons::math_kernel::types::flatten_2d_index;

// Access standardized physical constants
assert_eq!(C, 299_792_458.0);

// Flatten 2D simulation coordinates into a linear buffer index
let index = flatten_2d_index(3, 2, 10);
assert_eq!(index, 23);

// Evaluate normalized scalar values to RGB heatmaps for visualizers
let min_color = heatmap_color(0.0);
assert_eq!(min_color, [0, 0, 255]); // Blue at lower bound

let max_color = heatmap_color(1.0);
assert_eq!(max_color, [255, 0, 0]); // Red at upper bound
```

---

## 3. Ecosystem Integration

`math_commons` is consumed by domain simulation engines for universal constants and coordinate operations, and by `math_explorer_gui` for palette generation and shader data pipelines.
