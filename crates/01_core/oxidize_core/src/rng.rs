//! Compatibility and ergonomic re-export of Tier 0 [`math_rng`] deterministic ChaCha8 streams,
//! generic `f32`/`f64` scalar samplers, and 64D Quasi-Monte Carlo views.

pub use math_rng::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_seed_deterministic_and_orthogonal_streams() {
        let mut r0_a = OxidizeRng::from_seed(BASE_SEED);
        let mut r0_b = OxidizeRng::default();
        assert_eq!(r0_a.next_u64(), r0_b.next_u64());

        let base = OxidizeRng::new(BASE_SEED);
        let mut r1 = base.split_stream(1);
        let mut r2 = base.split_stream(2);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }
}
