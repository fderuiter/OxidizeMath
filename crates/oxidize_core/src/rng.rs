use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[path = "rng/qmc.rs"]
mod qmc;
#[path = "rng/sobol.rs"]
mod sobol;

pub use qmc::{
    HaltonSequence, MAX_QMC_DIM, PRIMES_64, RichtmyerSequence, baker_tent, prime_at,
    radical_inverse, richtmyer_generator,
};
pub use sobol::{MAX_SOBOL_DIM, SobolSequence};

const INV_TWO_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Default deterministic seed used by [`OxidizeRng::default`] and XOR-offset rerun streams.
pub const BASE_SEED: u64 = 0x0001_D1CE_CAFE_BABE;

/// A standard deterministic random number generator provider for the `oxidize` framework.
/// This enforces the "No-Entropy" policy by relying exclusively on explicit seeds
/// and preventing direct OS-level random calls via getrandom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OxidizeRng {
    inner: ChaCha8Rng,
}

impl OxidizeRng {
    /// Creates a new deterministic RNG from a 64-bit seed.
    /// This ensures identical stochastic behavior across platforms (Native and Web).
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Creates a deterministic RNG from an explicit 64-bit seed (e.g., `BASE_SEED ^ seed_counter`).
    pub fn from_seed(seed: u64) -> Self {
        Self::new(seed)
    }

    /// Draws a raw deterministic 32-bit integer without requiring `rand::RngCore` in scope.
    #[inline]
    pub fn raw_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    /// Draws a raw deterministic 64-bit integer without requiring `rand::RngCore` in scope.
    #[inline]
    pub fn raw_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Draws a deterministic uniform `f64` in $[0, 1)$ using 53 bits of precision (`#![no_std]`).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.inner.next_u64() >> 11) as f64) * INV_TWO_POW_53
    }

    /// Draws a deterministic uniform `f64` strictly inside $(0, 1)$ (`#![no_std]`).
    #[inline]
    pub fn next_f64_open01(&mut self) -> f64 {
        (((self.inner.next_u64() >> 11) as f64) + 0.5) * INV_TWO_POW_53
    }

    /// Draws a deterministic bounded index in `0..upper_bound` via unbiased Lemire rejection (`#![no_std]`).
    #[inline]
    pub fn next_index(&mut self, upper_bound: usize) -> usize {
        assert!(upper_bound > 0, "upper_bound must be > 0");
        let bound = upper_bound as u64;
        if bound.is_power_of_two() {
            return (self.inner.next_u64() & (bound - 1)) as usize;
        }
        let threshold = bound.wrapping_neg() % bound;
        for _ in 0..64 {
            let x = self.inner.next_u64();
            let prod = u128::from(x) * u128::from(bound);
            let low = prod as u64;
            if low >= threshold {
                return (prod >> 64) as usize;
            }
        }
        (self.inner.next_u64() % bound) as usize
    }
}

impl Default for OxidizeRng {
    fn default() -> Self {
        Self::from_seed(BASE_SEED)
    }
}

impl RngCore for OxidizeRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.inner.try_fill_bytes(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_seed_deterministic_and_distinct_counters() {
        let mut r0_a = OxidizeRng::from_seed(BASE_SEED);
        let mut r0_b = OxidizeRng::default();
        assert_eq!(r0_a.next_u64(), r0_b.next_u64());

        let mut r1 = OxidizeRng::from_seed(BASE_SEED ^ 1);
        let mut r2 = OxidizeRng::from_seed(BASE_SEED ^ 2);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn test_qmc_sequences_deterministic_and_unit_hypercube() {
        let mut richt = RichtmyerSequence::with_baker_tent(4, true).unwrap();
        let mut halton = HaltonSequence::new(4).unwrap();
        let mut sobol = SobolSequence::new(32).unwrap();
        let mut buf_4 = [0.0; 4];
        let mut buf_32 = [0.0; 32];

        for idx in 0..64_u32 {
            richt.next_into(&mut buf_4);
            assert!(buf_4.iter().all(|&u| (0.0..=1.0).contains(&u)));

            halton.next_into(&mut buf_4);
            assert!(buf_4.iter().all(|&u| (0.0..1.0).contains(&u)));

            sobol.next_into(&mut buf_32);
            let mut direct = [0.0; 32];
            sobol.point_at(idx, None, &mut direct);
            assert_eq!(buf_32, direct);
            assert!(buf_32.iter().all(|&u| u > 0.0 && u < 1.0));
        }
    }
}
