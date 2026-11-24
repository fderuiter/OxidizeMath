//! Forward-mode multi-tangent [`DualVec`] Levenberg-Marquardt nonlinear least-squares solvers.
// [cite:math_commons]

use math_scalars::{DualVec, Scalar};

use super::lm::{MAX_FIT_PARAMS, finite_xy, lm_core};
use crate::error::FitError;
use crate::series::PointSeries;

/// Minimizes $\sum_i (y_i - \hat{f}(x_i; \theta))^2$ ($1 \le k \le 6$) evaluating both the model
/// prediction and exact parameter gradient row in a single `DualVec<f64, MAX_FIT_PARAMS>` pass
/// per observation with zero heap allocations.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn levenberg_marquardt_ad<P, F>(
    points: &P,
    initial_params: &[f64],
    eval_ad: F,
) -> Result<[f64; MAX_FIT_PARAMS], FitError>
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[DualVec<f64, MAX_FIT_PARAMS>]) -> Option<DualVec<f64, MAX_FIT_PARAMS>>,
{
    levenberg_marquardt_ad_filtered(points, initial_params, finite_xy, eval_ad)
}

/// Minimizes nonlinear least squares over observations satisfying `domain_ok(x, y)` using a
/// single `DualVec<f64, MAX_FIT_PARAMS>` forward pass per sample.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn levenberg_marquardt_ad_filtered<P, F>(
    points: &P,
    initial_params: &[f64],
    domain_ok: fn(f64, f64) -> bool,
    eval_ad: F,
) -> Result<[f64; MAX_FIT_PARAMS], FitError>
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[DualVec<f64, MAX_FIT_PARAMS>]) -> Option<DualVec<f64, MAX_FIT_PARAMS>>,
{
    let k = initial_params.len();
    let eval_val = |x: f64, theta: &[f64; MAX_FIT_PARAMS]| -> f64 {
        let mut const_p = [DualVec::<f64, MAX_FIT_PARAMS>::ZERO; MAX_FIT_PARAMS];
        for (j, slot) in const_p.iter_mut().enumerate().take(k) {
            *slot = DualVec::constant(theta[j]);
        }
        eval_ad(x, &const_p[..k]).map_or(f64::NAN, |v| v.val)
    };
    let eval_jac = |x: f64, theta: &[f64; MAX_FIT_PARAMS]| -> (f64, [f64; MAX_FIT_PARAMS]) {
        forward_dual_vec_jacobian(x, theta, k, &eval_ad)
    };
    lm_core::<MAX_FIT_PARAMS, P, _, _>(
        points,
        initial_params,
        domain_ok,
        true,
        &eval_val,
        &eval_jac,
    )
}

#[inline]
fn forward_dual_vec_jacobian<F>(
    x: f64,
    theta: &[f64; MAX_FIT_PARAMS],
    k: usize,
    eval_ad: &F,
) -> (f64, [f64; MAX_FIT_PARAMS])
where
    F: Fn(f64, &[DualVec<f64, MAX_FIT_PARAMS>]) -> Option<DualVec<f64, MAX_FIT_PARAMS>>,
{
    debug_assert!(k <= MAX_FIT_PARAMS);
    let mut seeded = [DualVec::<f64, MAX_FIT_PARAMS>::ZERO; MAX_FIT_PARAMS];
    for (j, slot) in seeded.iter_mut().enumerate().take(k) {
        *slot = DualVec::variable(theta[j], j);
    }
    let Some(out) = eval_ad(x, &seeded[..k]) else {
        return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
    };
    if !out.is_finite_val() {
        return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
    }
    let mut jac = [0.0_f64; MAX_FIT_PARAMS];
    jac[..k].copy_from_slice(&out.eps[..k]);
    (out.val, jac)
}
