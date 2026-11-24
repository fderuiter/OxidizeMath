//! Zero-allocation borrowed-slice 64D 53-bit [`SobolView`] and optional [`SobolSequence`].
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use super::error::QmcError;
use super::sobol_data::{MAX_SOBOL_DIM, SOBOL_BITS, sobol_direction_53};
use crate::scalar::{RngScalar, fill_cranley_patterson_shifts};
use crate::scramble::dyadic_53_to_shifted_open;
use rand_core::RngCore;

/// Zero-allocation 64D Sobol generator borrowing caller-supplied `state` and `shifts` slices.
#[derive(Debug)]
pub struct SobolView<'a> {
    dimension: usize,
    index: u64,
    scramble_seed: Option<u64>,
    state: &'a mut [u64],
    shifts: &'a mut [f64],
}

impl<'a> SobolView<'a> {
    /// Binds a 64D 53-bit `SobolView` to caller-owned `state` and `shifts` slices and zeroes them.
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::InvalidDimension`] if `dimension == 0` or `dimension > 64`,
    /// [`QmcError::StateLengthMismatch`] if `state.len() != dimension`, or
    /// [`QmcError::BufferLengthMismatch`] if `shifts.len() != dimension`.
    pub fn new(
        dimension: usize,
        state: &'a mut [u64],
        shifts: &'a mut [f64],
    ) -> Result<Self, QmcError> {
        validate_sobol_slices(dimension, state.len(), shifts.len())?;
        state.fill(0);
        shifts.fill(0.0);
        Ok(Self {
            dimension,
            index: 0,
            scramble_seed: None,
            state,
            shifts,
        })
    }

    /// Configures stateless 53-bit nested hash Owen scrambling (`Some(seed)`) or disables it (`None`).
    #[inline]
    #[must_use]
    pub const fn with_scramble_seed(mut self, seed: Option<u64>) -> Self {
        self.scramble_seed = seed;
        self
    }

    /// Sets or clears the 53-bit nested hash Owen scramble seed in place.
    #[inline]
    pub fn set_scramble_seed(&mut self, seed: Option<u64>) {
        self.scramble_seed = seed;
    }

    /// Returns the configured dimension (`1..=64`).
    #[inline]
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of points emitted since creation or reset.
    #[inline]
    #[must_use]
    pub const fn current_index(&self) -> u64 {
        self.index
    }

    /// Resets the Gray-code state and index to `0` while preserving Cranley-Patterson shifts and scramble seed.
    #[inline]
    pub fn reset_index(&mut self) {
        self.index = 0;
        self.state.fill(0);
    }

    /// Repositions the Gray-code state cursor to `index` in $O(d)$ without mutating active shifts.
    pub fn seek(&mut self, index: u64) {
        self.index = index;
        let gray = index ^ (index >> 1);
        for (dim, slot) in self.state.iter_mut().enumerate() {
            let mut raw = 0_u64;
            let mut bits = gray;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                if bit < SOBOL_BITS {
                    raw ^= sobol_direction_53(dim, bit);
                }
                bits &= bits - 1;
            }
            *slot = raw;
        }
    }

    /// Resets index, Gray-code state, and Cranley-Patterson shifts to `0` (preserving `scramble_seed`).
    #[inline]
    pub fn reset(&mut self) {
        self.reset_index();
        self.shifts.fill(0.0);
    }

    /// Draws fresh Cranley-Patterson shifts in `[0, 1)` (and refreshes any active scramble seed) across all axes and resets the sequence index to `0`.
    #[inline]
    pub fn apply_cranley_patterson_shift<R: RngCore + ?Sized>(&mut self, rng: &mut R) {
        if self.scramble_seed.is_some() {
            self.scramble_seed = Some(rng.next_u64());
        }
        fill_cranley_patterson_shifts(self.shifts, rng);
        self.reset_index();
    }

    /// Generates the next Gray-code Sobol point into `out` (`f32` or `f64`) in $O(d)$ XOR operations.
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::BufferLengthMismatch`] if `out.len() != self.dimension()`.
    pub fn next_point<T: RngScalar>(&mut self, out: &mut [T]) -> Result<(), QmcError> {
        if out.len() != self.dimension {
            return Err(QmcError::BufferLengthMismatch {
                expected: self.dimension,
                actual: out.len(),
            });
        }
        if self.index > 0 {
            let step_bit = self.index.trailing_zeros() as usize;
            if step_bit < SOBOL_BITS {
                for (dim, slot) in self.state.iter_mut().enumerate() {
                    *slot ^= sobol_direction_53(dim, step_bit);
                }
            }
        }
        for (dim, slot) in out.iter_mut().enumerate() {
            *slot = dyadic_53_to_shifted_open(
                self.state[dim],
                dim,
                self.scramble_seed,
                self.shifts[dim],
            );
        }
        self.index = self.index.wrapping_add(1);
        Ok(())
    }

    /// Evaluates the Gray-code Sobol point at `index` directly into `out` without mutating state.
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::BufferLengthMismatch`] if `out.len() != self.dimension()`.
    pub fn point_at<T: RngScalar>(&self, index: u64, out: &mut [T]) -> Result<(), QmcError> {
        if out.len() != self.dimension {
            return Err(QmcError::BufferLengthMismatch {
                expected: self.dimension,
                actual: out.len(),
            });
        }
        let gray = index ^ (index >> 1);
        for (dim, slot) in out.iter_mut().enumerate() {
            let mut raw = 0_u64;
            let mut bits = gray;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                if bit < SOBOL_BITS {
                    raw ^= sobol_direction_53(dim, bit);
                }
                bits &= bits - 1;
            }
            *slot = dyadic_53_to_shifted_open(raw, dim, self.scramble_seed, self.shifts[dim]);
        }
        Ok(())
    }
}

const fn validate_sobol_slices(
    dimension: usize,
    state_len: usize,
    shifts_len: usize,
) -> Result<(), QmcError> {
    if dimension == 0 || dimension > MAX_SOBOL_DIM {
        return Err(QmcError::InvalidDimension {
            dimension,
            max_supported: MAX_SOBOL_DIM,
        });
    }
    if state_len != dimension {
        return Err(QmcError::StateLengthMismatch {
            expected: dimension,
            actual: state_len,
        });
    }
    if shifts_len != dimension {
        return Err(QmcError::BufferLengthMismatch {
            expected: dimension,
            actual: shifts_len,
        });
    }
    Ok(())
}

/// Owned 64D 53-bit Sobol sequence wrapper backed by [`SobolView`] (requires the `alloc` feature).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SobolSequence {
    dimension: usize,
    index: u64,
    scramble_seed: Option<u64>,
    state: alloc::vec::Vec<u64>,
    shifts: alloc::vec::Vec<f64>,
}

#[cfg(feature = "alloc")]
impl SobolSequence {
    /// Allocates an owned 64D 53-bit Sobol generator (`1 <= dimension <= 64`).
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::InvalidDimension`] if `dimension == 0` or `dimension > 64`.
    pub fn new(dimension: usize) -> Result<Self, QmcError> {
        if dimension == 0 || dimension > MAX_SOBOL_DIM {
            return Err(QmcError::InvalidDimension {
                dimension,
                max_supported: MAX_SOBOL_DIM,
            });
        }
        Ok(Self {
            dimension,
            index: 0,
            scramble_seed: None,
            state: alloc::vec![0_u64; dimension],
            shifts: alloc::vec![0.0_f64; dimension],
        })
    }

    /// Configures 53-bit nested hash Owen scrambling (`Some(seed)`) or disables it (`None`).
    #[inline]
    #[must_use]
    pub const fn with_scramble_seed(mut self, seed: Option<u64>) -> Self {
        self.scramble_seed = seed;
        self
    }

    /// Sets or clears the 53-bit nested hash Owen scramble seed in place.
    #[inline]
    pub fn set_scramble_seed(&mut self, seed: Option<u64>) {
        self.scramble_seed = seed;
    }

    /// Returns the configured dimension (`1..=64`).
    #[inline]
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the current sequence index.
    #[inline]
    #[must_use]
    pub const fn current_index(&self) -> u64 {
        self.index
    }

    /// Resets index and Gray-code state to `0` while keeping active shifts and scramble seed.
    #[inline]
    pub fn reset_index(&mut self) {
        self.index = 0;
        self.state.fill(0);
    }

    /// Resets index, Gray-code state, and Cranley-Patterson shifts to `0`.
    #[inline]
    pub fn reset(&mut self) {
        self.reset_index();
        self.shifts.fill(0.0);
    }

    /// Draws fresh Cranley-Patterson shifts in `[0, 1)` (and refreshes any active scramble seed) and resets the sequence index to `0`.
    #[inline]
    pub fn apply_cranley_patterson_shift<R: RngCore + ?Sized>(&mut self, rng: &mut R) {
        if self.scramble_seed.is_some() {
            self.scramble_seed = Some(rng.next_u64());
        }
        fill_cranley_patterson_shifts(&mut self.shifts, rng);
        self.reset_index();
    }

    /// Generates the next Sobol point into `out` without heap allocation.
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::BufferLengthMismatch`] if `out.len() != self.dimension()`.
    pub fn next_point<T: RngScalar>(&mut self, out: &mut [T]) -> Result<(), QmcError> {
        let mut view = SobolView {
            dimension: self.dimension,
            index: self.index,
            scramble_seed: self.scramble_seed,
            state: &mut self.state,
            shifts: &mut self.shifts,
        };
        view.next_point(out)?;
        self.index = view.index;
        Ok(())
    }

    /// Evaluates the Gray-code Sobol point at `index` into `out` without mutating state.
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::BufferLengthMismatch`] if `out.len() != self.dimension()`.
    pub fn point_at<T: RngScalar>(&self, index: u64, out: &mut [T]) -> Result<(), QmcError> {
        if out.len() != self.dimension {
            return Err(QmcError::BufferLengthMismatch {
                expected: self.dimension,
                actual: out.len(),
            });
        }
        let gray = index ^ (index >> 1);
        for (dim, slot) in out.iter_mut().enumerate() {
            let mut raw = 0_u64;
            let mut bits = gray;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                if bit < SOBOL_BITS {
                    raw ^= sobol_direction_53(dim, bit);
                }
                bits &= bits - 1;
            }
            *slot = dyadic_53_to_shifted_open(raw, dim, self.scramble_seed, self.shifts[dim]);
        }
        Ok(())
    }
}
