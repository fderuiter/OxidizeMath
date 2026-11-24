//! Zero-allocation 64-dimensional Quasi-Monte Carlo views (`SobolView`, `HaltonView`,
//! `NiederreiterView`) over compact static read-only tables (`< 16 KiB` `.rodata`).
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

/// Error types for QMC views and owned sequences.
pub mod error;
/// Static 64-prime table and packed Faure prime-radix permutations.
pub mod faure_data;
/// Borrowed-slice 64D Halton generator with Faure + stateless hash Owen scrambling.
pub mod halton;
/// Borrowed-slice 64D base-2 Niederreiter `(t, s)`-sequence generator.
pub mod niederreiter;
/// Static 64D 53-bit base-2 Niederreiter `GF(2)` generator tables.
pub mod niederreiter_data;
/// Borrowed-slice 64D 53-bit Joe-Kuo Sobol generator.
pub mod sobol;
/// Static 64D 53-bit Joe-Kuo Sobol direction tables.
pub mod sobol_data;

/// Maximum dimensionality supported uniformly across all QMC views (`64`).
pub const MAX_QMC_DIM: usize = 64;

pub use error::QmcError;
pub use faure_data::{FIRST_64_PRIMES, MAX_HALTON_DIM, faure_permute, faure_static_bytes};
#[cfg(feature = "alloc")]
pub use halton::HaltonSequence;
pub use halton::{HaltonView, radical_inverse};
#[cfg(feature = "alloc")]
pub use niederreiter::NiederreiterSequence;
pub use niederreiter::NiederreiterView;
pub use niederreiter_data::{
    MAX_NIEDERREITER_DIM, NIEDERREITER_BITS, niederreiter_generator_53, niederreiter_static_bytes,
};
#[cfg(feature = "alloc")]
pub use sobol::SobolSequence;
pub use sobol::SobolView;
pub use sobol_data::{MAX_SOBOL_DIM, SOBOL_BITS, sobol_direction_53, sobol_static_bytes};

/// Returns the combined static `.rodata` footprint in bytes of all 64D Sobol, Niederreiter,
/// and Faure tables (`< 16 KiB`).
#[inline]
#[must_use]
pub fn total_qmc_static_table_bytes() -> usize {
    sobol_static_bytes() + niederreiter_static_bytes() + faure_static_bytes()
}
