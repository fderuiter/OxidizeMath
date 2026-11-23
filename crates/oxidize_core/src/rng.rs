use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Default deterministic seed used by [`OxidizeRng::default`] and XOR-offset rerun streams.
pub const BASE_SEED: u64 = 0x0001_D1CE_CAFE_BABE;

/// A standard deterministic random number generator provider for the `oxidize` framework.
/// This enforces the "No-Entropy" policy by relying exclusively on explicit seeds
/// and preventing direct OS-level random calls via getrandom.
#[derive(Clone, Debug)]
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
}
