#![doc = include_str!("../README.md")]
// [cite:math_commons]

/// Symbolic mathematical expression AST parser, evaluator, and GLSL emitter.
pub mod ast;
pub mod constants;
/// Shared numerical, bounds, and curve-fitting diagnostic error types.
pub mod error;
/// Stack-allocated polynomial and damped Levenberg-Marquardt curve-fitting solvers.
pub mod fit;
/// Colormap gradients and strongly typed grid/step primitives.
pub mod math_kernel;
/// Zero-allocation point-of-interest detection (extrema, roots, inflections, intersections).
pub mod poi;
/// Validated bounded floating-point wrapper types (`UnitInterval`, `PositiveFloat`, `NonNegativeFloat`).
pub mod primitives;
pub mod registry;
/// Contiguous and strided 2D `(x, y)` series traits for zero-allocation numerical kernels.
pub mod series;
/// Quantum angular-momentum coupling and Wigner $3j$ / Clebsch-Gordan coefficients.
pub mod wigner;
