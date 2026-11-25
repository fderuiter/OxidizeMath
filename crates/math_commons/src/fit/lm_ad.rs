//! Single-pass multi-tangent forward-AD Levenberg-Marquardt nonlinear curve-fitting solvers.
// [cite:math_commons]

use super::lm::{MAX_FIT_PARAMS, finite_xy, lm_core};
use crate::error::FitError;
use crate::series::PointSeries;
use math_scalars::{DualVec, Scalar};

/// Minimizes $\sum_i (y_i - \hat{f}(x_i; \theta))^2$ ($1 \le k \le 6$) evaluating exact Jacobian rows in a single forward multi-tangent [`DualVec`] pass per observation.
///
/// Reference: Levenberg (1944), DOI: 10.1090/qam/10666; Marquardt (1963), DOI: 10.1137/0111030;
/// [math_commons specification](../../../../docs/math_specs/math_commons.md).
///
/// # Stability
/// Eliminates the $2k + 1$ finite-difference callback overhead and step-size truncation/cancellation error via exact stack-allocated `DualVec<f64, MAX_FIT_PARAMS>` tangents regularized with Marquardt damping $\lambda > 0$.
///
/// # Domain of Validity
/// Requires `1 <= initial_params.len() <= 6`, at least `initial_params.len()` finite `(x, y)` pairs, and finite `eval_ad` values and partial derivatives near $\theta$.
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

/// Minimizes $\sum_{i:\,\mathrm{domain\_ok}(x_i,y_i)} (y_i - \hat{f}(x_i; \theta))^2$ ($1 \le k \le 6$) using single-pass forward-AD [`DualVec`] Jacobians.
///
/// Reference: Marquardt (1963), DOI: 10.1137/0111030; [math_commons specification](../../../../docs/math_specs/math_commons.md).
///
/// # Stability
/// Computes both primal residual $\hat{f}(x_i; \theta)$ and exact row gradient $\nabla_\theta \hat{f}(x_i; \theta)$ in one pass per filtered point without heap allocation.
///
/// # Domain of Validity
/// Requires `1 <= initial_params.len() <= 6` and at least `initial_params.len()` observations satisfying `domain_ok(x, y)`.
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
        let mut const_theta = [DualVec::<f64, MAX_FIT_PARAMS>::ZERO; MAX_FIT_PARAMS];
        let limit = k.min(MAX_FIT_PARAMS);
        for j in 0..limit {
            const_theta[j] = DualVec::constant(theta[j]);
        }
        eval_ad(x, &const_theta[..limit]).map_or(f64::NAN, |d| d.val)
    };
    let eval_jac = |x: f64, theta: &[f64; MAX_FIT_PARAMS]| -> (f64, [f64; MAX_FIT_PARAMS]) {
        let mut dual_theta = [DualVec::<f64, MAX_FIT_PARAMS>::ZERO; MAX_FIT_PARAMS];
        let limit = k.min(MAX_FIT_PARAMS);
        for j in 0..limit {
            dual_theta[j] = DualVec::variable(theta[j], j);
        }
        let Some(jet) = eval_ad(x, &dual_theta[..limit]) else {
            return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
        };
        if !jet.val.is_finite() {
            return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
        }
        (jet.val, jet.eps)
    };
    lm_core::<MAX_FIT_PARAMS, P, _, _>(
        points,
        initial_params,
        domain_ok,
        false,
        &eval_val,
        &eval_jac,
    )
}
