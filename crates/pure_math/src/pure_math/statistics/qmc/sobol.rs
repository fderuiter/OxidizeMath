//! Preallocated Joe-Kuo Gray-code Sobol low-discrepancy sequence with Cranley-Patterson shifts.

use super::halton::{OPEN_UNIT_EPS, next_unit_closed_open};
use super::sobol_data::{MAX_SOBOL_DIM, SOBOL_BITS, compute_direction_numbers};
use crate::pure_math::statistics::monte_carlo::MonteCarloError;
use oxidize_core::rng::OxidizeRng;

/// Reciprocal of $2^{32}$ mapping 32-bit Sobol integer states into `[0.0, 1.0)`.
const INV_TWO_POW_32: f64 = 1.0 / 4_294_967_296.0;

/// Preallocated Joe-Kuo Sobol low-discrepancy generator using Antonov-Saleev Gray-code stepping.
///
/// All direction-number tables, integer state registers, and Cranley-Patterson shifts are allocated
/// once in [`SobolSequence::new`] so [`SobolSequence::next_point`] and [`SobolSequence::point_at`]
/// perform zero heap allocations.
#[derive(Debug, Clone)]
pub struct SobolSequence {
    directions: Vec<[u32; SOBOL_BITS]>,
    state: Vec<u32>,
    shift: Vec<f64>,
    index: u32,
}

impl SobolSequence {
    /// Constructs an unshifted `dimension`-dimensional Joe-Kuo Sobol sequence starting at index `0`.
    #[verified_engine::verified]
    pub fn new(dimension: usize) -> Result<Self, MonteCarloError> {
        if dimension == 0 || dimension > MAX_SOBOL_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension,
                max_dimension: MAX_SOBOL_DIM,
            });
        }
        let mut directions = Vec::with_capacity(dimension);
        for dim_idx in 0..dimension {
            directions.push(compute_direction_numbers(dim_idx));
        }
        Ok(Self {
            directions,
            state: vec![0_u32; dimension],
            shift: vec![0.0; dimension],
            index: 0,
        })
    }

    /// Coordinate dimensionality of the sequence.
    #[inline]
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.directions.len()
    }

    /// Current Gray-code sequence index to be emitted on the next [`Self::next_point`] call.
    #[inline]
    #[verified_engine::verified]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Computes the raw 32-bit Gray-code Sobol state at `index` for dimension `dim_idx`.
    #[inline]
    #[verified_engine::verified]
    fn state_at(&self, dim_idx: usize, index: u32) -> u32 {
        let mut gray = index ^ (index >> 1);
        let dirs = &self.directions[dim_idx];
        let mut acc = 0_u32;
        let mut bit = 0_usize;
        while gray != 0 {
            if (gray & 1) != 0 {
                acc ^= dirs[bit];
            }
            gray >>= 1;
            bit += 1;
        }
        acc
    }

    /// Repositions the Gray-code state cursor to `start_index` while preserving the active shift.
    #[verified_engine::verified]
    pub fn reset_index(&mut self, start_index: u32) {
        self.index = start_index;
        for dim_idx in 0..self.directions.len() {
            self.state[dim_idx] = self.state_at(dim_idx, start_index);
        }
    }

    /// Resets the sequence cursor to index `0` and clears any Cranley-Patterson shift.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.shift.fill(0.0);
        self.reset_index(0);
    }

    /// Draws an independent Cranley-Patterson uniform shift in `[0, 1)^d` from `rng` and rewinds to index `0`.
    #[verified_engine::verified]
    pub fn apply_cranley_patterson_shift(&mut self, rng: &mut OxidizeRng) {
        for slot in &mut self.shift {
            *slot = next_unit_closed_open(rng);
        }
        self.reset_index(0);
    }

    /// Evaluates the Sobol point at `index` into `out` without mutating the sequence cursor.
    #[verified_engine::verified]
    pub fn point_at(&self, index: u32, out: &mut [f64]) -> Result<(), MonteCarloError> {
        let dim = self.directions.len();
        if out.len() != dim {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: dim,
                actual: out.len(),
            });
        }
        for (dim_idx, (slot, &shift)) in out.iter_mut().zip(&self.shift).enumerate() {
            let raw_u32 = self.state_at(dim_idx, index);
            let shifted = (((raw_u32 as f64) + 0.5) * INV_TWO_POW_32 + shift).fract();
            *slot = shifted.clamp(OPEN_UNIT_EPS, 1.0 - OPEN_UNIT_EPS);
        }
        Ok(())
    }

    /// Emits the next Sobol point in `(0, 1)^d` and advances in $O(d)$ via Antonov-Saleev XOR stepping.
    #[verified_engine::verified]
    pub fn next_point(&mut self, out: &mut [f64]) -> Result<(), MonteCarloError> {
        let dim = self.directions.len();
        if out.len() != dim {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: dim,
                actual: out.len(),
            });
        }
        let bit = (!self.index).trailing_zeros() as usize;
        let step_bit = bit.min(SOBOL_BITS - 1);
        for ((slot, state_val), (dirs, &shift)) in out
            .iter_mut()
            .zip(&mut self.state)
            .zip(self.directions.iter().zip(&self.shift))
        {
            let shifted = (((*state_val as f64) + 0.5) * INV_TWO_POW_32 + shift).fract();
            *slot = shifted.clamp(OPEN_UNIT_EPS, 1.0 - OPEN_UNIT_EPS);
            *state_val ^= dirs[step_bit];
        }
        self.index = self.index.wrapping_add(1);
        Ok(())
    }
}

// [cite:clinical_trials_statistics]
