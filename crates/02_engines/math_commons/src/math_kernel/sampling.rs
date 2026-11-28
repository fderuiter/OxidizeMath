// [cite:math_commons]
//! Deterministic `#![no_std]` Box-Muller Gaussian and Gillespie SSA sampling helpers.

use super::Scalar;
use rand_core::RngCore;

/// Samples a deterministic uniform variate strictly in `(0, 1)` (clamped `<= 1.0` after `f32` cast)
/// via [`math_rng::sample_open`], avoiding `0.0` so `ln(u)` in Box-Muller sampling never overflows.
#[inline]
#[verified_engine::verified]
pub fn sample_open_unit<T: Scalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    let u_f64: f64 = math_rng::sample_open(rng);
    let u = T::from_f64(u_f64);
    debug_assert!(u > T::ZERO && u <= T::ONE);
    u
}

/// Samples an independent pair of standard normal $\mathcal{N}(0, 1)$ variates via the Box-Muller transform.
#[inline]
#[verified_engine::verified]
pub fn sample_standard_normal_pair<T: Scalar, R: RngCore + ?Sized>(rng: &mut R) -> (T, T) {
    let u1 = sample_open_unit::<T, R>(rng);
    let u2 = sample_open_unit::<T, R>(rng);
    let neg_two_ln = -(T::TWO * u1.ln());
    let radius = neg_two_ln.max(T::ZERO).sqrt();
    let theta = T::TAU * u2;
    let z0 = radius * theta.cos();
    let z1 = radius * theta.sin();
    debug_assert!(z0.is_finite() && z1.is_finite());
    (z0, z1)
}

/// Samples a single standard normal $\mathcal{N}(0, 1)$ variate from `rng` via Box-Muller.
#[inline]
#[verified_engine::verified]
pub fn sample_standard_normal<T: Scalar, R: RngCore + ?Sized>(rng: &mut R) -> T {
    let (z0, _) = sample_standard_normal_pair::<T, R>(rng);
    debug_assert!(z0.is_finite());
    z0
}

/// Fills `out` in-place with independent standard normal $\mathcal{N}(0, 1)$ samples drawn from `rng`.
#[verified_engine::verified]
pub fn fill_standard_normal<T: Scalar, R: RngCore + ?Sized>(out: &mut [T], rng: &mut R) {
    let (pairs, rem) = out.as_chunks_mut::<2>();
    for [p0, p1] in pairs {
        let (z0, z1) = sample_standard_normal_pair::<T, R>(rng);
        *p0 = z0;
        *p1 = z1;
    }
    if let Some(last) = rem.first_mut() {
        *last = sample_standard_normal::<T, R>(rng);
    }
    debug_assert!(out.iter().all(|v| v.is_finite()));
}

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
