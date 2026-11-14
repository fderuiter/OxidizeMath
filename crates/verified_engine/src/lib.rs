#![doc = include_str!("../README.md")]
#[allow(missing_docs)]
pub mod allocator;
#[allow(missing_docs)]
pub mod engine;
#[allow(missing_docs)]
pub mod metrics;

/// Attribute macro to enforce high-integrity verification rules (NASA Power of 10)
/// and track runtime metrics (call counts and arithmetic operations).
///
/// Functions marked with `#[verified]` are checked for direct recursion and statement
/// limits, and instrumented for verification metrics collection.
///
/// # Example
/// ```
/// use verified_engine::verified;
///
/// #[verified]
/// fn kinetic_energy(mass: f64, velocity: f64) -> f64 {
///     0.5 * mass * velocity * velocity
/// }
///
/// let energy = kinetic_energy(2.0, 3.0);
/// assert_eq!(energy, 9.0);
/// ```
pub use verified_engine_macros::verified;

// Re-export the macros
pub use verified_engine_macros::{Theory, embed_theory};
