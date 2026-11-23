# Math Commons

Standardized mathematical constants, spatial indexing, colormaps, and primitive geometry.

## 1. Overview & Architecture

The `math_commons` crate provides foundational mathematical primitives, physical constant registries, and numerical utilities across the OxidizeMath ecosystem.

By standardizing universal constants, color mapping formulas, and spatial coordinate indexing in a single shared crate, numerical kernels and graphical user interfaces maintain consistent calculation baselines and render outputs without duplicate implementations.

### Module Structure

- **`ast`**: Native mathematical expression lexer, bounded Pratt parser, iterative LaTeX formatter, zero-allocation postfix stack bytecode evaluator (`CompiledExpr`), real/complex dynamic evaluators, and iterative GPU shader transpilers:
  - `ast::glsl` (`Expr::to_glsl_scalar`, `Expr::to_glsl_scalar_with_vars`, `Expr::to_glsl_complex`, `Expr::to_glsl_complex_with_vars`): Iterative GLSL ES 3.00 `float` and `vec2` emitters with IEEE-safe helpers (`GLSL_SCALAR_HELPERS`, `GLSL_COMPLEX_HELPERS`).
  - `ast::shader` (`Expr::to_wgsl`, `Expr::to_glsl`, `Expr::to_wgsl_complex`, `Expr::to_shader`, `Expr::to_shader_complex`): Iterative WebGPU WGSL and GLSL ES 3.00 emitters with `shader_preludes` (`wgsl_real_prelude`, `glsl_real_prelude`, `wgsl_complex_prelude`, `glsl_complex_prelude`).
- **`constants`**: Centralized registry of physical, astronomical, and mathematical constants:
  - Speed of light (`C`), Gravitational constant (`G`), Solar mass (`SOLAR_MASS`).
  - Reduced Planck constant (`HBAR_C`), Proton mass (`PROTON_MASS`), Neutron mass (`NEUTRON_MASS`).
  - Liquid drop nuclear model constants and floating-point tolerances (`EPSILON`).
- **`math_kernel`**: Standardized numerical routines:
  - `types`: 2D/3D index flattening (`flatten_2d_index`) and spatial offsets.
  - `colormap`: Heatmap color generation (`heatmap_color`), cyclic cosine palettes, continuous complex domain coloring (`domain_color`, `ShaderPalette`), WCAG 2.2 `3.0:1` dark-canvas contrast guards, and WGSL/GLSL shader palette preludes.
  - `glsl_colormap`: Standalone GLSL ES 3.00 emitters (`glsl_heatmap_color_fn`, `glsl_cyclic_cosine_palette_fn`, `glsl_ensure_dark_canvas_contrast_fn`, `glsl_colormap_library`).
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
