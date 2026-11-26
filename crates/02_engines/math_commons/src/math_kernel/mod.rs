/// Scalar-to-RGB heatmap, diverging, and cyclic palette mappings.
#[cfg(feature = "std")]
pub mod colormap;
/// GLSL shader source snippets for GPU colormap evaluation.
#[cfg(feature = "std")]
pub mod glsl_colormap;
/// Strongly typed step-size, grid-index, and flattened index primitives.
pub mod types;
// [cite:essay]
