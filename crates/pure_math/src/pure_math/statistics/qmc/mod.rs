//! Quasi-Monte Carlo (QMC) low-discrepancy point generators and randomized shifts.
//!
//! Provides preallocated prime-base [`HaltonSequence`] (up to 64 dimensions) and Joe-Kuo
//! Gray-code [`SobolSequence`] (up to 32 dimensions) with Cranley-Patterson random shifts
//! seeded explicitly from [`oxidize_core::rng::OxidizeRng`].
//!
//! ## Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::prelude::OxidizeRng;
//! use math_explorer::pure_math::statistics::qmc::SobolSequence;
//!
//! fn main() -> math_explorer::error::Result<()> {
//!     let mut sobol = SobolSequence::new(3)?;
//!     let mut rng = OxidizeRng::new(42);
//!     sobol.apply_cranley_patterson_shift(&mut rng);
//!
//!     let mut point = [0.0; 3];
//!     sobol.next_point(&mut point)?;
//!     assert!(point.iter().all(|&u| u > 0.0 && u < 1.0));
//!     Ok(())
//! }
//! ```

mod halton;
mod sobol;
mod sobol_data;

pub use halton::{FIRST_64_PRIMES, HaltonSequence, MAX_HALTON_DIM, radical_inverse};
pub use sobol::SobolSequence;
pub use sobol_data::MAX_SOBOL_DIM;

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
