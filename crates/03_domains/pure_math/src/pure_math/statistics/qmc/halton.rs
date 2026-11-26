//! Prime-base Halton low-discrepancy sequence delegating to [`math_rng::HaltonView`]
//! with static Faure permutations, stateless hash Owen scrambling, and Cranley-Patterson shifts.

use crate::pure_math::number_theory::primes::is_prime;
use crate::pure_math::statistics::monte_carlo::MonteCarloError;
use math_rng::{BASE_SEED, HaltonView, fill_cranley_patterson_shifts, torus_shift_open};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Maximum coordinate dimension supported by [`HaltonSequence::new`] (`64`).
pub const MAX_HALTON_DIM: usize = math_rng::MAX_HALTON_DIM;

/// First 64 prime numbers used as pairwise-coprime radical-inverse bases.
pub const FIRST_64_PRIMES: [u64; MAX_HALTON_DIM] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311,
];

/// Evaluates the radical-inverse function $\Phi_b(k)$ in `base >= 2` for index `k`.
#[inline]
#[verified_engine::verified]
pub fn radical_inverse(mut index: u64, base: u64) -> f64 {
    if let Ok(b32) = u32::try_from(base)
        && let Ok(val) = math_rng::radical_inverse(index, b32)
    {
        return val;
    }
    if base < 2 {
        return 0.0;
    }
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

/// Preallocated 64D prime-base Halton quasi-Monte Carlo sequence delegating canonical prime axes
/// to [`math_rng::HaltonView`] (static Faure permutations + stateless hash Owen scrambling).
#[derive(Debug, Clone)]
pub struct HaltonSequence {
    bases: Vec<u64>,
    shift: Vec<f64>,
    index: u64,
    scramble_seed: Option<u64>,
    canonical_primes: bool,
}

impl HaltonSequence {
    /// Constructs a scrambled `dimension`-dimensional Halton sequence starting at index `1`.
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
            scramble_seed: Some(BASE_SEED),
            canonical_primes: true,
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
        let canonical = bases == &FIRST_64_PRIMES[..bases.len()];
        Ok(Self {
            bases: bases.to_vec(),
            shift: vec![0.0; bases.len()],
            index: 1,
            scramble_seed: canonical.then_some(BASE_SEED),
            canonical_primes: canonical,
        })
    }

    /// Configures stateless hash Owen scrambling (`Some(seed)`) or disables it (`None`).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn with_scramble_seed(mut self, seed: Option<u64>) -> Self {
        self.scramble_seed = seed;
        self
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

    /// Draws an independent Cranley-Patterson shift in `[0, 1)^d` (and fresh Owen scramble seed) and rewinds to index `1`.
    #[verified_engine::verified]
    pub fn apply_cranley_patterson_shift(&mut self, rng: &mut OxidizeRng) {
        if self.canonical_primes && self.scramble_seed.is_some() {
            self.scramble_seed = Some(rng.next_u64());
        }
        fill_cranley_patterson_shifts(&mut self.shift, rng);
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
        if self.canonical_primes {
            let mut zero_shifts = [0.0_f64; MAX_HALTON_DIM];
            let view = HaltonView::new(dim, &mut zero_shifts[..dim])
                .expect("validated dim")
                .with_scramble_seed(self.scramble_seed);
            view.point_at(idx, out).expect("validated out len");
            for (slot, &shift) in out.iter_mut().zip(&self.shift) {
                if shift != 0.0 {
                    *slot = torus_shift_open(*slot, shift);
                }
            }
            return Ok(());
        }
        for ((slot, &base), &shift) in out.iter_mut().zip(&self.bases).zip(&self.shift) {
            *slot = torus_shift_open(radical_inverse(idx, base), shift);
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
