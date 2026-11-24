//! Prime-base Halton low-discrepancy sequence with Cranley-Patterson random shifts.

use crate::pure_math::number_theory::primes::is_prime;
use crate::pure_math::statistics::monte_carlo::MonteCarloError;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Maximum coordinate dimension supported by [`HaltonSequence::new`] using built-in primes.
pub const MAX_HALTON_DIM: usize = 64;

/// Guard margin keeping open-unit-cube coordinates strictly inside `(0.0, 1.0)`.
pub(crate) const OPEN_UNIT_EPS: f64 = 1e-15;

/// Scale factor converting a 53-bit integer draw into `[0.0, 1.0)`.
const INV_TWO_POW_53: f64 = 1.0 / ((1_u64 << 53) as f64);

/// First 64 prime numbers used as pairwise-coprime radical-inverse bases.
pub const FIRST_64_PRIMES: [u64; MAX_HALTON_DIM] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311,
];

/// Converts an `OxidizeRng` 64-bit word into a uniform `[0.0, 1.0)` float without heap allocation.
#[inline]
#[verified_engine::verified]
pub(crate) fn next_unit_closed_open(rng: &mut OxidizeRng) -> f64 {
    ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53
}

/// Evaluates the radical-inverse function $\Phi_b(k)$ in base `base` for index `k`.
#[inline]
#[verified_engine::verified]
pub fn radical_inverse(mut index: u64, base: u64) -> f64 {
    let inv_base = 1.0 / (base as f64);
    let mut factor = inv_base;
    let mut acc = 0.0;
    while index > 0 {
        let digit = index % base;
        acc += (digit as f64) * factor;
        index /= base;
        factor *= inv_base;
    }
    acc
}

/// Preallocated prime-base Halton quasi-Monte Carlo sequence with optional Cranley-Patterson shift.
#[derive(Debug, Clone)]
pub struct HaltonSequence {
    bases: Vec<u64>,
    shift: Vec<f64>,
    index: u64,
}

impl HaltonSequence {
    /// Constructs an unshifted `dimension`-dimensional Halton sequence starting at index `1`.
    #[verified_engine::verified]
    pub fn new(dimension: usize) -> Result<Self, MonteCarloError> {
        if dimension == 0 || dimension > MAX_HALTON_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension,
                max_dimension: MAX_HALTON_DIM,
            });
        }
        Ok(Self {
            bases: FIRST_64_PRIMES[..dimension].to_vec(),
            shift: vec![0.0; dimension],
            index: 1,
        })
    }

    /// Constructs a Halton sequence from explicit prime bases.
    #[verified_engine::verified]
    pub fn with_bases(bases: &[u64]) -> Result<Self, MonteCarloError> {
        if bases.is_empty() || bases.len() > MAX_HALTON_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension: bases.len(),
                max_dimension: MAX_HALTON_DIM,
            });
        }
        for &base in bases {
            if base < 2 || !is_prime(base) {
                return Err(MonteCarloError::InvalidBase { base });
            }
        }
        Ok(Self {
            bases: bases.to_vec(),
            shift: vec![0.0; bases.len()],
            index: 1,
        })
    }

    /// Coordinate dimensionality of the sequence.
    #[inline]
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.bases.len()
    }

    /// Next 1-based sequence index to be emitted by [`Self::next_point`].
    #[inline]
    #[verified_engine::verified]
    pub fn index(&self) -> u64 {
        self.index
    }

    /// Resets the sequence index to `1` and clears any Cranley-Patterson shift.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.index = 1;
        self.shift.fill(0.0);
    }

    /// Repositions the sequence cursor to `start_index.max(1)` while preserving the active shift.
    #[verified_engine::verified]
    pub fn reset_index(&mut self, start_index: u64) {
        self.index = start_index.max(1);
    }

    /// Draws an independent Cranley-Patterson uniform shift in `[0, 1)^d` from `rng` and rewinds to index `1`.
    #[verified_engine::verified]
    pub fn apply_cranley_patterson_shift(&mut self, rng: &mut OxidizeRng) {
        for slot in &mut self.shift {
            *slot = next_unit_closed_open(rng);
        }
        self.index = 1;
    }

    /// Evaluates the point at `1`-based `index` into `out` without mutating the sequence cursor.
    #[verified_engine::verified]
    pub fn point_at(&self, index: u64, out: &mut [f64]) -> Result<(), MonteCarloError> {
        let dim = self.bases.len();
        if out.len() != dim {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: dim,
                actual: out.len(),
            });
        }
        let idx = index.max(1);
        for ((slot, &base), &shift) in out.iter_mut().zip(&self.bases).zip(&self.shift) {
            let raw = (radical_inverse(idx, base) + shift).fract();
            *slot = raw.clamp(OPEN_UNIT_EPS, 1.0 - OPEN_UNIT_EPS);
        }
        Ok(())
    }

    /// Writes the next low-discrepancy point in `(0, 1)^d` into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn next_point(&mut self, out: &mut [f64]) -> Result<(), MonteCarloError> {
        self.point_at(self.index, out)?;
        self.index = self.index.saturating_add(1);
        Ok(())
    }
}

// [cite:clinical_trials_statistics]
