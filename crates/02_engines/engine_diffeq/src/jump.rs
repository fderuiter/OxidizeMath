// [cite:engine_diffeq]
use math_commons::math_kernel::types::{Scalar, sample_open_unit};
use rand_core::RngCore;

/// Samples the next reaction waiting time $\tau > 0$ and reaction index $j \in 0..\text{propensities.len()}$
/// via Gillespie's exact Stochastic Simulation Algorithm (direct method) without heap allocation.
///
/// Returns `None` if `propensities` is empty or the total propensity $a_0 = \sum_j a_j \le 0$.
#[verified_engine::verified]
pub fn ssa_sample_event<T: Scalar, R: RngCore + ?Sized>(
    propensities: &[T],
    rng: &mut R,
) -> Option<(T, usize)> {
    if propensities.is_empty() {
        return None;
    }
    let mut total = T::ZERO;
    for &rate in propensities {
        debug_assert!(rate >= T::ZERO, "propensity must be non-negative");
        if rate > T::ZERO {
            total += rate;
        }
    }
    if total <= T::ZERO {
        return None;
    }

    let u1 = sample_open_unit::<T, R>(rng);
    let u2 = sample_open_unit::<T, R>(rng);
    debug_assert!(u1 > T::ZERO && u1 < T::ONE);
    let tau = -(u1.ln()) / total;
    let threshold = u2 * total;

    let mut cumulative = T::ZERO;
    let mut chosen = propensities.len() - 1;
    for (idx, &rate) in propensities.iter().enumerate() {
        if rate > T::ZERO {
            cumulative += rate;
            if cumulative >= threshold {
                chosen = idx;
                break;
            }
        }
    }
    debug_assert!(tau > T::ZERO && chosen < propensities.len());
    Some((tau, chosen))
}

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
