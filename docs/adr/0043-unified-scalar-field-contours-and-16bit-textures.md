# ADR 0043: Unified 2D Scalar-Field Contours and 16-Bit Texture Exports

## Status

Accepted — 2026-11-22

- **Supersedes (partial):** [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md) (base64 raster `<image>` SVG fallback on 2D spatial tools)
- **Relates to:** [ADR 0016](0016-delete-unreferenced-boundary-conditions.md) (`PeriodicBoundary` removal from `oxidize_core`), [ADR 0035](0035-iterative-ast-glsl-transpiler-and-shader-exports.md) (`ExportCapabilities::with_shader` composition on `FractalViewer`), [ADR 0042](0042-fft-lenia-engine-and-gray-scott-kinetics.md) (`GrayScottKinetics` and `MorphogenesisKineticsMode`)

## Context

Under [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), all seven 2D spatial/matrix tools in `math_explorer_gui` (`MorphogenesisTool`, `FractalViewer`, `LatticeBoltzmannTool` / `LbmUnified`, `DoseCalculationTool`, `IsingModelTool`, `PrimeSpiralWidget`, and `AttentionMapsTool`) declared `ExportCapabilities::raster_only()` and discarded their underlying `f64` scalar fields or discrete lattices before figure export:

1. **Pixelated Base64 `<image>` SVG Wrapper:** `egui_plot::export_primitives::svg::export_raster_svg` wrapped 8-bit quantized `ColorImage` buffers inside `<image image-rendering="pixelated" href="data:image/png;base64,..."/>`, preventing scalable vector isoline or lattice plotting in Inkscape, LaTeX, or pen plotters.
2. **8-Bit Terracing and Border Padding in Texture Exports:** `egui_plot::export::color32_image_to_png` hardcoded `png::ColorType::Rgba` and `png::BitDepth::Eight`, while `scale_color_image` added border padding that broke seamless tileability and terraced 3D displacement and normal maps.
3. **Clamped Diffusion Edges, Missing Gray-Scott Kinetics, and Banded Fractals:** Following [ADR 0016](0016-delete-unreferenced-boundary-conditions.md)'s deletion of unreferenced `PeriodicBoundary` from `oxidize_core`, `FiniteDifference2D` in `domain_biology` only supported clamped `NeumannBoundary` edges via `FusedStencilStepper::step_2d_coupled_neumann`. Furthermore, `domain_biology::biology::morphogenesis::reaction` exposed only `SchnakenbergKinetics` despite `crates/domain_biology/README.md` advertising Gray-Scott reaction-diffusion, and `FractalViewer` capped raster exports at `480x320` integer escape counts.

## Decision

1. **Toroidal 2D PDE Boundary Conditions and `GrayScottKinetics` (`pure_math` & `domain_biology`):**
   - Introduce `PeriodicBoundary` and `BoundaryMode::{Neumann, Periodic}` directly in `crates/domain_biology/src/biology/diffusion/fd2d/boundary.rs` (where active 2D finite-difference callers reside, rather than re-bloating `oxidize_core`), paired with zero-allocation `FusedStencilStepper::step_2d_coupled_periodic` in `crates/pure_math/src/pure_math/analysis/pde/fused_stepper.rs`.
   - Consolidate with [ADR 0042](0042-fft-lenia-engine-and-gray-scott-kinetics.md)'s `GrayScottKinetics { f, k }` in `crates/domain_biology/src/biology/morphogenesis/reaction.rs` and expose boundary (`Neumann` / `Periodic`) selection alongside `Schnakenberg` / `GrayScott` kinetics in `MorphogenesisTool`.
2. **Continuous Escape-Time Renormalization (`domain_physics`):**
   - Add zero-allocation `smooth_escape_mandelbrot` and `smooth_escape_julia` (`n + 1 - log2(ln|z|)`) in `crates/domain_physics/src/physics/chaos/fractals.rs` and remove the `480x320` export cap in `FractalViewer` via resolution-aware `export_scalar_field_at`.
3. **Feature-Independent `ScalarField2D` and `MarchingSquares` (`math_commons`):**
   - Add `ScalarField2D` and `MarchingSquares` in `crates/math_commons/src/math_kernel/marching_squares.rs` with linear edge interpolation, asymptotic decider disambiguation for saddle cases `5` and `10`, toroidal seam wrapping when `periodic == true`, and segment chaining into closed loops and open polylines. Placing `MarchingSquares` in unconditional `math_commons` keeps `math_explorer_gui` cleanly compilable under every isolated domain feature flag (`biology`, `physics`, `pure_math`, `ai`).
4. **Vector Contour `<svg>` and Borderless 16-Bit PNG Encoders (`egui_plot`):**
   - Add `egui_plot/src/export_primitives/svg_contours.rs` (`export_contour_layers_svg`, `export_rect_grid_svg`, `copy_svg_to_clipboard`) supporting `1..=16` contour thresholds and standalone `<path>` / `<rect>` vector serialization.
   - Add `egui_plot/src/export_primitives/texture_maps.rs` reusing existing `png` `0.18` (zero new dependencies) to encode borderless power-of-two (`256`, `512`, `1024`, `2048`) 16-bit grayscale heightmaps (`png::ColorType::Grayscale`, `png::BitDepth::Sixteen`) and 16-bit RGB tangent-space normal maps (`encode_normal_map_png`) with periodic or clamped central differences and adjustable normal strength.
5. **Unified Spatial Export Hooks (`math_explorer_gui`):**
   - Extend `InteractiveTool` with `export_scalar_field`, `export_scalar_field_at`, and `export_vector_svg`, extend `ExportCapabilities` with `scalar_field()`, `discrete_lattice()`, and `vector_grid()`, and add `ExportFormat::{Heightmap16Png, NormalMapPng}` plus `Copy SVG` clipboard dispatch in `SimulationFramework`.
   - Wire continuous `ScalarField2D` contours and 16-bit textures on `MorphogenesisTool`, `FractalViewer` (composed with `.with_shader()`), `DoseCalculationTool`, and `LbmUnified`; discrete `<path>` geometry plus 16-bit textures on `IsingModelTool` and `PrimeSpiralWidget`; and crisp `<rect>` cell SVG plus scalar-field exports on `AttentionMapsTool`. This supersedes ADR 0024's base64 PNG-in-SVG fallback for all seven spatial tools while preserving the workspace audit invariant across all registered GUI tools (`45` plot tools, `7` spatial figure tools, and `16` non-export tools).

## Verification Commands

```bash
# Verify domain kernels, Marching Squares, and zero-allocation PDE/fractal guards
cargo test -p pure_math -p domain_biology -p domain_physics -p math_commons -p egui_plot

# Verify vector SVG and 16-bit PNG export audits across all 68 registered GUI tools
cargo test -p math_explorer_gui deferred_export

# Full static, isolated-feature, and WebAssembly budget gates
cargo run -p xtask -- lint-all
cargo run -p xtask -- test-features
cargo run -p xtask -- test-wasm
./scripts/verify_wasm_budget.sh
```

## Consequences

- **Positive:** All seven 2D spatial tools now export pure `<path>` or `<rect>` vector SVGs (zero `data:image/png` base64 wrappers) alongside borderless `256`–`2048` 16-bit grayscale heightmaps and seamless normal maps suitable for Blender, Godot, Three.js, LaTeX, and pen plotters.
- **Positive:** Hot-path PDE (`step_2d_coupled_periodic`, `PeriodicBoundary`, `GrayScottKinetics`) and smooth fractal kernels allocate `0` heap bytes under `VerifiedAllocator`, and reusing `png` `0.18` adds zero external crates while keeping compressed WASM output comfortably within the `3,145,728`-byte budget.
- **Trade-offs:** Expanding public kernel, `egui_plot`, and `InteractiveTool` signatures requires regenerating `crates/unified_verification/api_baseline.json` and splitting cold-path export modules (`export_spatial.rs`, `export_ui_modal.rs`, `morphogenesis/export.rs`, `ising/export.rs`, `fd2d/tests.rs`) so every `.rs` file outside `egui_plot/` remains strictly `<= 500` lines.
