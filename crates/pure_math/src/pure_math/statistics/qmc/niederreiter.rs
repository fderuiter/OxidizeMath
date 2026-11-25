//! 64-dimensional 53-bit base-2 Niederreiter `(t, s)`-sequence delegating to [`math_rng::NiederreiterView`].

use crate::pure_math::statistics::monte_carlo::MonteCarloError;
use math_rng::{BASE_SEED, MAX_NIEDERREITER_DIM, NiederreiterView, fill_cranley_patterson_shifts};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Preallocated 64D 53-bit base-2 Niederreiter generator with Gray-code stepping,
/// nested 53-bit hash Owen scrambling, and Cranley-Patterson shifts.
#[derive(Debug, Clone)]
pub struct NiederreiterSequence {
    state: Vec<u64>,
    shift: Vec<f64>,
    scramble_seed: Option<u64>,
    index: u64,
}

impl NiederreiterSequence {
    /// Constructs a 64D base-2 Niederreiter sequence (`1..=64`) with default hash Owen scrambling.
    #[verified_engine::verified]
    pub fn new(dimension: usize) -> Result<Self, MonteCarloError> {
        if dimension == 0 || dimension > MAX_NIEDERREITER_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension,
                max_dimension: MAX_NIEDERREITER_DIM,
            });
        }
        Ok(Self {
            state: vec![0_u64; dimension],
            shift: vec![0.0_f64; dimension],
            scramble_seed: Some(BASE_SEED),
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
    pub fn index(&self) -> u64 {
        self.index
    }

    /// Repositions the Gray-code state cursor to `start_index` while preserving active shifts.
    #[verified_engine::verified]
    pub fn reset_index(&mut self, start_index: u64) {
        self.index = start_index;
        let dim = self.state.len();
        let mut tmp_shifts = [0.0_f64; MAX_NIEDERREITER_DIM];
        let mut view = NiederreiterView::new(dim, &mut self.state, &mut tmp_shifts[..dim])
            .expect("validated Niederreiter dimension");
        view.seek(start_index);
    }

    /// Resets the sequence cursor to index `0` and clears any Cranley-Patterson shift.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.shift.fill(0.0);
        self.state.fill(0);
        self.index = 0;
    }

    /// Draws fresh Cranley-Patterson shifts (and refreshes any active Owen scramble seed) and rewinds to `0`.
    #[verified_engine::verified]
    pub fn apply_cranley_patterson_shift(&mut self, rng: &mut OxidizeRng) {
        if self.scramble_seed.is_some() {
            self.scramble_seed = Some(rng.next_u64());
        }
        fill_cranley_patterson_shifts(&mut self.shift, rng);
        self.state.fill(0);
        self.index = 0;
    }

    /// Evaluates the 53-bit base-2 Niederreiter point at `index` into `out` without mutating state.
    #[verified_engine::verified]
    pub fn point_at(&self, index: u64, out: &mut [f64]) -> Result<(), MonteCarloError> {
        let dim = self.state.len();
        if out.len() != dim {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: dim,
                actual: out.len(),
            });
        }
        let gray = index ^ (index >> 1);
        for (d, slot) in out.iter_mut().enumerate() {
            let mut raw = 0_u64;
            let mut bits = gray;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                if bit < math_rng::qmc::NIEDERREITER_BITS {
                    raw ^= math_rng::qmc::niederreiter_generator_53(d, bit);
                }
                bits &= bits - 1;
            }
            *slot = math_rng::dyadic_53_to_shifted_open(raw, d, self.scramble_seed, self.shift[d]);
        }
        Ok(())
    }

    /// Emits the next 53-bit base-2 Niederreiter point in `(0, 1)^d` in $O(d)$ Gray-code XOR operations.
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
            if step_bit < math_rng::qmc::NIEDERREITER_BITS {
                for (d, slot) in self.state.iter_mut().enumerate() {
                    *slot ^= math_rng::qmc::niederreiter_generator_53(d, step_bit);
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
