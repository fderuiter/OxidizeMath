//! Quasi-Monte Carlo (QMC) low-discrepancy point generators and randomized shifts.
//!
//! Delegates to Tier 0 [`math_rng`] static 53-bit tables and zero-allocation views, providing
//! 64-dimensional scrambled [`HaltonSequence`], 64-dimensional Joe-Kuo [`SobolSequence`], and
//! 64-dimensional base-2 [`NiederreiterSequence`] alongside borrowed-slice [`HaltonView`],
//! [`SobolView`], and [`NiederreiterView`].
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
mod niederreiter;
mod sobol;

pub use halton::{FIRST_64_PRIMES, HaltonSequence, MAX_HALTON_DIM, radical_inverse};
pub use math_rng::{HaltonView, MAX_NIEDERREITER_DIM, MAX_SOBOL_DIM, NiederreiterView, SobolView};
pub use niederreiter::NiederreiterSequence;
pub use sobol::SobolSequence;

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
