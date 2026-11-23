#![doc = include_str!("../README.md")]

/// Zero-allocation global allocator guard enforcing NASA Power of 10 Rule 3.
pub mod allocator;
/// Verified execution harness coordinating allocation locks and complexity metrics.
pub mod engine;
/// Thread-local runtime counters tracking calls, arithmetic operations, and memory loads.
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

// Re-export the macros and hidden runtime support crate for macro expansion
#[doc(hidden)]
pub use scientific_metadata;

/// Derive macro implementing `scientific_metadata::theory::TheoryDescribable` with compile-time
/// validation of parameter bounds and academic citations.
///
/// # Example
/// ```
/// use verified_engine::Theory;
/// use verified_engine::scientific_metadata::theory::TheoryDescribable;
///
/// #[derive(Theory)]
/// #[theory(description = "Hookean spring law", citation = "Hooke (1678)")]
/// pub struct SpringParams {
///     #[theory(min = 0.1, max = 500.0, step = 0.5)]
///     pub stiffness: f64,
/// }
///
/// let spring = SpringParams { stiffness: 12.5 };
/// assert_eq!(spring.theory_description(), "Hookean spring law");
/// assert_eq!(spring.theory_parameters()["stiffness"].min, 0.1);
/// ```
pub use verified_engine_macros::Theory;

/// Attribute macro embedding labeled LaTeX equations from `papers/*.tex` into item docstrings
/// while verifying AST operator semantic alignment at compile time.
pub use verified_engine_macros::embed_theory;
