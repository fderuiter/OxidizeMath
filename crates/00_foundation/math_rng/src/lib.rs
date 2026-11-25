//! Zero-allocation `#![no_std]` ChaCha8 random streams, generic IEEE-754 scalar samplers,
//! and borrowed-slice 64-dimensional Quasi-Monte Carlo (QMC) views over compact static tables.
// [cite:clinical_trials_statistics]
#![no_std]
#![doc = include_str!("../README.md")]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
extern crate std;

/// Quasi-Monte Carlo (`SobolView`, `HaltonView`, `NiederreiterView`) generators and tables.
pub mod qmc;
/// Deterministic `ChaCha8Rng` wrapper with orthogonal stream splitting and word seeking.
pub mod rng;
/// Generic `f32`/`f64` uniform, symmetric, Gaussian, and unbiased bounded-index samplers.
pub mod scalar;
/// Stateless 53-bit nested hash Owen scrambling and prime-radix Faure-Owen permutations.
pub mod scramble;

pub use qmc::{
    FIRST_64_PRIMES, HaltonView, MAX_HALTON_DIM, MAX_NIEDERREITER_DIM, MAX_QMC_DIM, MAX_SOBOL_DIM,
    NiederreiterView, QmcError, SobolView, radical_inverse,
};
#[cfg(feature = "alloc")]
pub use qmc::{HaltonSequence, NiederreiterSequence, SobolSequence};
pub use rand_core::{CryptoRng, RngCore, SeedableRng};
pub use rng::{BASE_SEED, OxidizeRng};
pub use scalar::{
    RngError, RngScalar, fill_cranley_patterson_shifts, sample_bounded_u64, sample_bounded_usize,
    sample_closed_open, sample_open, sample_standard_normal, sample_standard_normal_f64,
    sample_symmetric_open, try_sample_bounded_u64, try_sample_bounded_usize,
};
pub use scramble::{dyadic_53_to_shifted_open, owen_scramble_53, splitmix64, torus_shift_open};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_qmc;
