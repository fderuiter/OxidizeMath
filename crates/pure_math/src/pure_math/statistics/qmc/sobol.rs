//! 64-dimensional 53-bit Joe-Kuo Gray-code Sobol sequence delegating to [`math_rng::SobolView`].

use crate::pure_math::statistics::monte_carlo::MonteCarloError;
use math_rng::{MAX_SOBOL_DIM, SobolView, fill_cranley_patterson_shifts};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Preallocated 64D 53-bit Joe-Kuo Sobol low-discrepancy generator backed by [`math_rng::SobolView`].
///
/// Direction tables live in Tier 0 `.rodata`; only the `d`-element state and shift slices are
/// allocated in [`SobolSequence::new`], keeping [`Self::next_point`] and [`Self::point_at`]
/// strictly zero-allocation.
#[derive(Debug, Clone)]
pub struct SobolSequence {
    state: Vec<u64>,
    shift: Vec<f64>,
    scramble_seed: Option<u64>,
    index: u32,
}

impl SobolSequence {
    /// Constructs an unshifted `dimension`-dimensional 53-bit Joe-Kuo Sobol sequence (`1..=64`).
    #[verified_engine::verified]
    pub fn new(dimension: usize) -> Result<Self, MonteCarloError> {
        if dimension == 0 || dimension > MAX_SOBOL_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension,
                max_dimension: MAX_SOBOL_DIM,
            });
        }
        Ok(Self {
            state: vec![0_u64; dimension],
            shift: vec![0.0_f64; dimension],
            scramble_seed: None,
            index: 0,
        })
    }

    /// Configures stateless 53-bit nested hash Owen scrambling (`Some(seed)`) or disables it (`None`).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn with_scramble_seed(mut self, seed: Option<u64>) -> Self {
        self.scramble_seed = seed;
        self
    }

    /// Coordinate dimensionality of the sequence (`1..=64`).
    #[inline]
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.state.len()
    }

    /// Current Gray-code sequence index to be emitted on the next [`Self::next_point`] call.
    #[inline]
    #[verified_engine::verified]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Repositions the Gray-code state cursor to `start_index` while preserving the active shift.
    #[verified_engine::verified]
    pub fn reset_index(&mut self, start_index: u32) {
        self.index = start_index;
        let dim = self.state.len();
        let mut tmp_shifts = [0.0_f64; MAX_SOBOL_DIM];
        let mut view = SobolView::new(dim, &mut self.state, &mut tmp_shifts[..dim])
            .expect("validated Sobol dimension");
        view.seek(start_index as u64);
    }

    /// Resets the sequence cursor to index `0` and clears any Cranley-Patterson shift.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.shift.fill(0.0);
        self.state.fill(0);
        self.index = 0;
    }

    /// Draws an independent Cranley-Patterson shift in `[0, 1)^d` from `rng` and rewinds to index `0`.
    #[verified_engine::verified]
    pub fn apply_cranley_patterson_shift(&mut self, rng: &mut OxidizeRng) {
        if self.scramble_seed.is_some() {
            self.scramble_seed = Some(rng.next_u64());
        }
        fill_cranley_patterson_shifts(&mut self.shift, rng);
        self.state.fill(0);
        self.index = 0;
    }

    /// Evaluates the 53-bit Sobol point at `index` into `out` without mutating the sequence cursor.
    #[verified_engine::verified]
    pub fn point_at(&self, index: u32, out: &mut [f64]) -> Result<(), MonteCarloError> {
        let dim = self.state.len();
        if out.len() != dim {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: dim,
                actual: out.len(),
            });
        }
        let gray = (index as u64) ^ ((index as u64) >> 1);
        for (d, slot) in out.iter_mut().enumerate() {
            let mut raw = 0_u64;
            let mut bits = gray;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                if bit < math_rng::qmc::SOBOL_BITS {
                    raw ^= math_rng::qmc::sobol_direction_53(d, bit);
                }
                bits &= bits - 1;
            }
            *slot = math_rng::dyadic_53_to_shifted_open(raw, d, self.scramble_seed, self.shift[d]);
        }
        Ok(())
    }

    /// Emits the next 53-bit Sobol point in `(0, 1)^d` and advances in $O(d)$ via Gray-code XOR stepping.
    #[verified_engine::verified]
    pub fn next_point(&mut self, out: &mut [f64]) -> Result<(), MonteCarloError> {
        let dim = self.state.len();
        if out.len() != dim {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: dim,
                actual: out.len(),
            });
        }
        if self.index > 0 {
            let step_bit = self.index.trailing_zeros() as usize;
            if step_bit < math_rng::qmc::SOBOL_BITS {
                for (d, slot) in self.state.iter_mut().enumerate() {
                    *slot ^= math_rng::qmc::sobol_direction_53(d, step_bit);
                }
            }
        }
        for (d, slot) in out.iter_mut().enumerate() {
            *slot = math_rng::dyadic_53_to_shifted_open(
                self.state[d],
                d,
                self.scramble_seed,
                self.shift[d],
            );
        }
        self.index = self.index.wrapping_add(1);
        Ok(())
    }
}

// [cite:clinical_trials_statistics]
