# Math Commons

Standardized mathematical constants, spatial indexing, colormaps, and primitive geometry.

## 1. Overview & Architecture

The `math_commons` crate provides foundational mathematical primitives, physical constant registries, and numerical utilities across the OxidizeMath ecosystem.

By standardizing universal constants, color mapping formulas, and spatial coordinate indexing in a single shared crate, numerical kernels and graphical user interfaces maintain consistent calculation baselines and render outputs without duplicate implementations.

### Module Structure

- **`ast`**: Native mathematical expression lexer, parser, LaTeX formatter, zero-allocation postfix stack bytecode evaluator (`CompiledExpr`), and iterative GLSL ES 3.00 transpiler (`ast::glsl`) for real and complex formulas:
  - `Expr::to_glsl_scalar` / `Expr::to_glsl_scalar_with_vars`: Iterative `float` emitter with IEEE-safe helpers (`GLSL_SCALAR_HELPERS`).
  - `Expr::to_glsl_complex` / `Expr::to_glsl_complex_with_vars`: Iterative `vec2` complex emitter with branch-cut `cln`/`cpow`/`csqrt` and `cdiv` guards (`GLSL_COMPLEX_HELPERS`).
- **`constants`**: Centralized registry of physical, astronomical, and mathematical constants:
  - Speed of light (`C`), Gravitational constant (`G`), Solar mass (`SOLAR_MASS`).
  - Reduced Planck constant (`HBAR_C`), Proton mass (`PROTON_MASS`), Neutron mass (`NEUTRON_MASS`).
  - Liquid drop nuclear model constants and floating-point tolerances (`EPSILON`).
- **`math_kernel`**: Standardized numerical routines:
  - `types`: 2D/3D index flattening (`flatten_2d_index`) and spatial offsets.
  - `colormap`: Heatmap color generation (`heatmap_color`), cyclic cosine palettes, and WCAG 2.2 dark-canvas contrast guards.
  - `glsl_colormap`: Standalone GLSL ES 3.00 emitters (`glsl_heatmap_color_fn`, `glsl_cyclic_cosine_palette_fn`, `glsl_ensure_dark_canvas_contrast_fn`, `glsl_colormap_library`).
- **`primitives`**: Basic geometric constructs and coordinate structures.
- **`registry`**: Constant lookup and numeric tolerance validation.
- **`series`**: Zero-copy [`PointSeries`](crate::series::PointSeries) views over `[f64; 2]`, `(f64, f64)`, and paired `(&[f64], &[f64])` coordinate slices.
- **`fit`**: Stack-allocated polynomial ordinary least squares (`1..=4`), damped Levenberg-Marquardt nonlinear regression, closed-form seed estimators, canonical [`FitModel`](crate::fit::FitModel) presets, and typed [`FitError`](crate::error::FitError) diagnostics.
- **`poi`**: Zero-allocation sub-segment Hermite threshold crossings, vertical crossings, 3-point Lagrange parabolic extrema, and monotone/general 2D polyline intersections.
- **`wigner`**: Clebsch-Gordan coefficients and quantum angular momentum arithmetic.

---

## 2. Developer Instructions

### Zero-Allocation Curve Fitting and Critical-Point Queries

Fit canonical polynomial or nonlinear curves and query sub-segment crossings directly over paired slices:

```rust
use math_commons::fit::{FitModel, fit_model};
use math_commons::poi::{closest_threshold_crossing, parabolic_vertex};

let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
let ys = [0.0, 3.0, 4.0, 3.0, 0.0];
let series = (&xs[..], &ys[..]);

let fit = fit_model(&series, FitModel::Quadratic, false).expect("quadratic fit");
assert!((fit.eval(2.0) - 4.0).abs() < 1e-9);

let root = closest_threshold_crossing(&series, 2.0, 0.5).expect("crossing");
assert!(root > 0.0 && root < 1.0);

let vertex = parabolic_vertex([1.0, 3.0], [2.0, 4.0], [3.0, 3.0]);
assert_eq!(vertex, [2.0, 4.0]);
```

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

### Iterative AST-to-GLSL Transpilation

Compile scalar (`float`) and complex (`vec2`) AST expressions into standalone GLSL ES 3.00 expressions:

```rust
use math_commons::ast::parse;
use math_commons::math_kernel::glsl_colormap::glsl_colormap_library;

let expr = parse("z^3 - 1/z").unwrap();
let glsl = expr.to_glsl_complex_with_vars(&["z"]).unwrap();
assert_eq!(glsl, "cpow(z, vec2(3.0, 0.0)) - cdiv(vec2(1.0, 0.0), z)");

let palette_glsl = glsl_colormap_library();
assert!(palette_glsl.contains("vec3 ensure_dark_canvas_contrast(vec3 srgb)"));
```

Example emitted GLSL helper signature (`text` block):

```text
vec2 cdiv(vec2 a, vec2 b);
vec2 cpow(vec2 a, vec2 b);
vec3 ensure_dark_canvas_contrast(vec3 srgb);
```

---

## 3. Ecosystem Integration

`math_commons` is consumed by domain simulation engines for universal constants and coordinate operations, and by `math_explorer_gui` for palette generation and shader data pipelines.
