// [cite:engine_diffeq]
pub use math_commons::math_kernel::types::ssa_sample_event;

#[cfg(test)]
mod tests {
    use super::ssa_sample_event;
    use math_rng::OxidizeRng;
    use rand_core::RngCore;

    struct EndpointWordRng(u64);

    impl RngCore for EndpointWordRng {
        #[verified_engine::verified]
        fn next_u32(&mut self) -> u32 {
            let w = self.0 as u32;
            debug_assert_eq!(u64::from(w), self.0 & 0xffff_ffff);
            w
        }

        #[verified_engine::verified]
        fn next_u64(&mut self) -> u64 {
            let w = self.0;
            debug_assert_eq!(w, self.0);
            w
        }

        #[verified_engine::verified]
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(0);
            debug_assert!(dest.first().is_none_or(|&b| b == 0));
        }

        #[verified_engine::verified]
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(dest);
            debug_assert!(dest.first().is_none_or(|&b| b == 0));
            Ok(())
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_ssa_sample_event_f32_waiting_times_strictly_positive() {
        let rates = [0.25_f32, 0.75_f32, 1.5_f32];
        let (tau_max_u, idx_max) =
            ssa_sample_event(&rates, &mut EndpointWordRng(u64::MAX)).expect("active propensities");
        assert!(tau_max_u > 0.0_f32 && !tau_max_u.is_sign_negative());
        assert_eq!(idx_max, 2);

        let (tau_min_u, idx_min) =
            ssa_sample_event(&rates, &mut EndpointWordRng(0)).expect("active propensities");
        assert!(tau_min_u > tau_max_u && tau_min_u.is_finite());
        assert_eq!(idx_min, 0);

        let mut rng = OxidizeRng::new(0x55A0_2026);
        for _ in 0..512 {
            let (tau, chosen) = ssa_sample_event(&rates, &mut rng).expect("active propensities");
            assert!(tau > 0.0_f32 && tau.is_finite() && !tau.is_sign_negative());
            assert!(chosen < rates.len());
        }
    }
}
