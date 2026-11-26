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
    debug_assert!(tau >= T::ZERO && chosen < propensities.len());
    Some((tau, chosen))
}
