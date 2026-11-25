//! Zero-allocation borrowed-slice 64D [`HaltonView`] with static Faure permutations
//! and stateless hash Owen scrambling.
// [cite:clinical_trials_statistics]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]

use super::error::QmcError;
use super::faure_data::{FIRST_64_PRIMES, MAX_HALTON_DIM, faure_owen_permute};
use crate::rng::BASE_SEED;
use crate::scalar::{RngScalar, fill_cranley_patterson_shifts};
use crate::scramble::{splitmix64, torus_shift_open};
use rand_core::RngCore;

const MAX_HALTON_DIGITS: u32 = 54;
const MIN_FACTOR_F64: f64 = 1.0 / ((1_u64 << 53) as f64);

/// Evaluates the classical van der Corput radical inverse $\Phi_b(n) \in [0, 1)$ in `base >= 2`.
///
/// # Errors
///
/// Returns [`QmcError::InvalidBase`] if `base < 2`.
#[inline]
pub fn radical_inverse(mut index: u64, base: u32) -> Result<f64, QmcError> {
    if base < 2 {
        return Err(QmcError::InvalidBase { base });
    }
    let b_u64 = base as u64;
    let inv_base = 1.0 / (base as f64);
    let mut factor = inv_base;
    let mut result = 0.0_f64;
    while index > 0 {
        let digit = index % b_u64;
        result += (digit as f64) * factor;
        index /= b_u64;
        factor *= inv_base;
    }
    debug_assert!((0.0..1.0).contains(&result));
    Ok(result)
}

#[inline]
fn eval_halton_axis(dim: usize, mut n: u64, scramble_seed: Option<u64>) -> f64 {
    let prime = FIRST_64_PRIMES[dim];
    let p_u64 = prime as u64;
    let inv_p = 1.0 / (prime as f64);
    let mut factor = inv_p;
    let mut acc = 0.0_f64;
    let mut prefix_hash = 0_u64;
    let mut depth = 0_u32;

    while depth < MAX_HALTON_DIGITS {
        let keep_going = if scramble_seed.is_some() {
            factor >= MIN_FACTOR_F64
        } else {
            n > 0
        };
        if !keep_going {
            break;
        }
        let raw_digit = (n % p_u64) as u32;
        let permuted = faure_owen_permute(dim, prime, depth, prefix_hash, raw_digit, scramble_seed);
        acc += (permuted as f64) * factor;
        if scramble_seed.is_some() {
            prefix_hash = splitmix64(prefix_hash ^ ((raw_digit as u64).wrapping_add(1)));
        }
        n /= p_u64;
        factor *= inv_p;
        depth += 1;
    }
    acc
}

/// Zero-allocation 64D Halton generator borrowing a caller-supplied `shifts` slice.
///
/// By default, `HaltonView::new` enables deterministic Faure + stateless hash Owen scrambling
/// (`Some(BASE_SEED)`) to eliminate high-dimensional prime collinearity (`r = 1.0` on axes `62..=64`).
/// Call [`HaltonView::unscrambled`] for pure Faure-permuted deterministic points or
/// [`HaltonView::raw_unpermuted`] for textbook unpermuted radical-inverse points.
#[derive(Debug)]
pub struct HaltonView<'a> {
    dimension: usize,
    index: u64,
    scramble_seed: Option<u64>,
    use_faure: bool,
    shifts: &'a mut [f64],
}

impl<'a> HaltonView<'a> {
    /// Binds a 64D scrambled `HaltonView` (`index = 1`, `scramble_seed = Some(BASE_SEED)`) to `shifts`.
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::InvalidDimension`] if `dimension == 0` or `dimension > 64`, or
    /// [`QmcError::BufferLengthMismatch`] if `shifts.len() != dimension`.
    pub fn new(dimension: usize, shifts: &'a mut [f64]) -> Result<Self, QmcError> {
        if dimension == 0 || dimension > MAX_HALTON_DIM {
            return Err(QmcError::InvalidDimension {
                dimension,
                max_supported: MAX_HALTON_DIM,
            });
        }
        if shifts.len() != dimension {
            return Err(QmcError::BufferLengthMismatch {
                expected: dimension,
                actual: shifts.len(),
            });
        }
        shifts.fill(0.0);
        Ok(Self {
            dimension,
            index: 1,
            scramble_seed: Some(BASE_SEED),
            use_faure: true,
            shifts,
        })
    }

    /// Configures the stateless hash Owen scramble seed (`Some(seed)`) or pure Faure mode (`None`).
    #[inline]
    #[must_use]
    pub const fn with_scramble_seed(mut self, seed: Option<u64>) -> Self {
        self.scramble_seed = seed;
        self.use_faure = true;
        self
    }

    /// Disables hash Owen scrambling while retaining static Faure prime-radix permutations.
    #[inline]
    #[must_use]
    pub const fn unscrambled(mut self) -> Self {
        self.scramble_seed = None;
        self.use_faure = true;
        self
    }

    /// Disables both hash Owen scrambling and Faure permutations (raw van der Corput radical inverse).
    #[inline]
    #[must_use]
    pub const fn raw_unpermuted(mut self) -> Self {
        self.scramble_seed = None;
        self.use_faure = false;
        self
    }

    /// Sets or clears the stateless hash Owen scramble seed in place.
    #[inline]
    pub fn set_scramble_seed(&mut self, seed: Option<u64>) {
        self.scramble_seed = seed;
        if seed.is_some() {
            self.use_faure = true;
        }
    }

    /// Returns the configured dimension (`1..=64`).
    #[inline]
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the 1-based index of the next Halton point to be emitted.
    #[inline]
    #[must_use]
    pub const fn current_index(&self) -> u64 {
        self.index
    }

    /// Resets the sequence counter back to `1` while retaining active shifts and scramble settings.
    #[inline]
    pub fn reset_index(&mut self) {
        self.index = 1;
    }

    /// Resets the sequence counter to `1` and zeroes all Cranley-Patterson shifts.
    #[inline]
    pub fn reset(&mut self) {
        self.index = 1;
        self.shifts.fill(0.0);
    }

    /// Draws fresh `[0, 1)` Cranley-Patterson shifts and a fresh scramble seed (if scrambled), resetting `index` to `1`.
    #[inline]
    pub fn apply_cranley_patterson_shift<R: RngCore + ?Sized>(&mut self, rng: &mut R) {
        if self.scramble_seed.is_some() {
            self.scramble_seed = Some(rng.next_u64());
        }
        fill_cranley_patterson_shifts(self.shifts, rng);
        self.index = 1;
    }

    /// Generates the next 64D Halton point into `out` (`f32` or `f64`) without heap allocation.
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::BufferLengthMismatch`] if `out.len() != self.dimension()`.
    pub fn next_point<T: RngScalar>(&mut self, out: &mut [T]) -> Result<(), QmcError> {
        self.point_at(self.index, out)?;
        self.index = self.index.saturating_add(1);
        Ok(())
    }

    /// Evaluates the Halton point at 1-based `index` into `out` without mutating the sequence counter.
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
        for (dim, slot) in out.iter_mut().enumerate() {
            let unit = if self.use_faure {
                eval_halton_axis(dim, index, self.scramble_seed)
            } else {
                radical_inverse(index, FIRST_64_PRIMES[dim])?
            };
            *slot = torus_shift_open(unit, self.shifts[dim]);
        }
        Ok(())
    }
}

/// Owned 64D Halton sequence wrapper backed by [`HaltonView`] (requires the `alloc` feature).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct HaltonSequence {
    dimension: usize,
    index: u64,
    scramble_seed: Option<u64>,
    use_faure: bool,
    shifts: alloc::vec::Vec<f64>,
}

#[cfg(feature = "alloc")]
impl HaltonSequence {
    /// Allocates an owned 64D scrambled Halton generator (`1 <= dimension <= 64`).
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::InvalidDimension`] if `dimension == 0` or `dimension > 64`.
    pub fn new(dimension: usize) -> Result<Self, QmcError> {
        if dimension == 0 || dimension > MAX_HALTON_DIM {
            return Err(QmcError::InvalidDimension {
                dimension,
                max_supported: MAX_HALTON_DIM,
            });
        }
        Ok(Self {
            dimension,
            index: 1,
            scramble_seed: Some(BASE_SEED),
            use_faure: true,
            shifts: alloc::vec![0.0_f64; dimension],
        })
    }

    /// Configures the stateless hash Owen scramble seed (`Some(seed)`) or pure Faure (`None`).
    #[inline]
    #[must_use]
    pub const fn with_scramble_seed(mut self, seed: Option<u64>) -> Self {
        self.scramble_seed = seed;
        self.use_faure = true;
        self
    }

    /// Disables hash Owen scrambling and Faure permutations (raw van der Corput radical inverse).
    #[inline]
    #[must_use]
    pub const fn raw_unpermuted(mut self) -> Self {
        self.scramble_seed = None;
        self.use_faure = false;
        self
    }

    /// Sets or clears the stateless hash Owen scramble seed in place.
    #[inline]
    pub fn set_scramble_seed(&mut self, seed: Option<u64>) {
        self.scramble_seed = seed;
        if seed.is_some() {
            self.use_faure = true;
        }
    }

    /// Returns the configured dimension (`1..=64`).
    #[inline]
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the 1-based index of the next Halton point.
    #[inline]
    #[must_use]
    pub const fn current_index(&self) -> u64 {
        self.index
    }

    /// Resets the sequence counter to `1` while retaining active shifts and scramble settings.
    #[inline]
    pub fn reset_index(&mut self) {
        self.index = 1;
    }

    /// Resets the sequence counter to `1` and zeroes all Cranley-Patterson shifts.
    #[inline]
    pub fn reset(&mut self) {
        self.index = 1;
        self.shifts.fill(0.0);
    }

    /// Draws fresh `[0, 1)` Cranley-Patterson shifts (and refreshes any active scramble seed) and resets `index` to `1`.
    #[inline]
    pub fn apply_cranley_patterson_shift<R: RngCore + ?Sized>(&mut self, rng: &mut R) {
        if self.scramble_seed.is_some() {
            self.scramble_seed = Some(rng.next_u64());
        }
        fill_cranley_patterson_shifts(&mut self.shifts, rng);
        self.index = 1;
    }

    /// Generates the next Halton point into `out` without heap allocation.
    ///
    /// # Errors
    ///
    /// Returns [`QmcError::BufferLengthMismatch`] if `out.len() != self.dimension()`.
    pub fn next_point<T: RngScalar>(&mut self, out: &mut [T]) -> Result<(), QmcError> {
        let mut view = HaltonView {
            dimension: self.dimension,
            index: self.index,
            scramble_seed: self.scramble_seed,
            use_faure: self.use_faure,
            shifts: &mut self.shifts,
        };
        view.next_point(out)?;
        self.index = view.index;
        Ok(())
    }

    /// Evaluates the Halton point at 1-based `index` into `out` without mutating state.
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
        for (dim, slot) in out.iter_mut().enumerate() {
            let unit = if self.use_faure {
                eval_halton_axis(dim, index, self.scramble_seed)
            } else {
                radical_inverse(index, FIRST_64_PRIMES[dim])?
            };
            *slot = torus_shift_open(unit, self.shifts[dim]);
        }
        Ok(())
    }
}
