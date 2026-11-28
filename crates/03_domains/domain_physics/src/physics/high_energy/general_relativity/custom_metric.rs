//! Compiled AST radial-lapse $f(r)$ and wormhole throat $b(l)$ metrics and lensing evaluators.

use std::f64::consts::PI;

use math_commons::ast::CompiledExpr;
use nalgebra::{SMatrix, SVector};
use pure_math::pure_math::tensor::{StaticRiemannianMetric, TensorError};

use super::kerr_radial::{LensingRegime, RadialAtlasSummary};
use super::schwarzschild::GeodesicKind;

/// Minimum regular radial lapse threshold guarding against horizon division by zero.
pub const MIN_REGULAR_LAPSE: f64 = 0.02;

/// Minimum regular circumferential throat radius guarding against pinch-off singularities.
pub const MIN_REGULAR_THROAT: f64 = 0.10;

/// Estimates the outermost radial horizon where $f(r; R_s) \le 0.02$ on $[0.15, 12.0]$.
#[must_use]
#[verified_engine::verified]
pub fn estimate_compiled_lapse_horizon(
    compiled: Option<&CompiledExpr>,
    params: &[f64],
    rs: f64,
) -> f64 {
    let mut horizon = 0.0_f64;
    for idx in 1..=96 {
        let r = 0.15 + 11.85 * (idx as f64) / 96.0;
        let f = compiled.map_or(1.0 - rs / r.max(0.1), |c| c.eval_2d(r, rs, params));
        if !f.is_finite() || f <= MIN_REGULAR_LAPSE {
            horizon = r;
        }
    }
    debug_assert!(horizon >= 0.0 && horizon.is_finite());
    horizon
}

/// Wraps a compiled $f(r; R_s, \text{params})$ into a stack-allocated 4D [`StaticRiemannianMetric`].
///
/// Returns `Err(TensorError::SingularMetric)` whenever $f(r) \le 0.02$, $r \le 0.05$,
/// $|\sin\theta| \le 10^{-4}$, or any coordinate/bytecode output is non-finite.
#[verified_engine::verified]
pub fn build_lapse_static_metric<'a>(
    compiled: &'a CompiledExpr,
    rs: f64,
    params: &'a [f64],
) -> StaticRiemannianMetric<
    impl Fn(&SVector<f64, 4>) -> Result<SMatrix<f64, 4, 4>, TensorError> + 'a,
    4,
> {
    StaticRiemannianMetric::new(move |pt: &SVector<f64, 4>| {
        if !pt.iter().all(|v| v.is_finite()) || !rs.is_finite() {
            return Err(TensorError::SingularMetric);
        }
        let r = pt[1];
        let sin_t = pt[2].sin();
        if r <= 0.05 || sin_t.abs() <= 1e-4 {
            return Err(TensorError::SingularMetric);
        }
        let f = compiled.eval_2d(r, rs, params);
        if !f.is_finite() || f <= MIN_REGULAR_LAPSE {
            return Err(TensorError::SingularMetric);
        }
        let mut g = SMatrix::<f64, 4, 4>::zeros();
        g[(0, 0)] = -f;
        g[(1, 1)] = 1.0 / f;
        g[(2, 2)] = r * r;
        g[(3, 3)] = r * r * sin_t * sin_t;
        Ok(g)
    })
}

/// Wraps a compiled $b(l; b_0, \text{params})$ into a stack-allocated 3D [`StaticRiemannianMetric`].
#[verified_engine::verified]
pub fn build_wormhole_static_metric<'a>(
    compiled: &'a CompiledExpr,
    b0: f64,
    params: &'a [f64],
) -> StaticRiemannianMetric<
    impl Fn(&SVector<f64, 3>) -> Result<SMatrix<f64, 3, 3>, TensorError> + 'a,
    3,
> {
    StaticRiemannianMetric::new(move |pt: &SVector<f64, 3>| {
        if !pt.iter().all(|v| v.is_finite()) || !b0.is_finite() {
            return Err(TensorError::SingularMetric);
        }
        let b_val = compiled.eval_2d(pt[1], b0, params);
        if !b_val.is_finite() || b_val <= MIN_REGULAR_THROAT {
            return Err(TensorError::SingularMetric);
        }
        let r2 = b_val * b_val;
        if !r2.is_finite() {
            return Err(TensorError::SingularMetric);
        }
        Ok(SMatrix::<f64, 3, 3>::new(
            -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, r2,
        ))
    })
}

/// Evaluates one 2D regime cell for a custom radial-lapse metric $g_{00} = -f(r; R_s)$.
///
/// # Examples
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_commons::ast::{parse, CompiledExpr};
/// use math_explorer::physics::high_energy::{eval_custom_lapse_lensing_cell, GeodesicKind};
///
/// let ast = parse("1 - Rs / r").unwrap();
/// let expr = CompiledExpr::compile_with_vars(&ast, &["r", "Rs"]).unwrap();
/// let cell = eval_custom_lapse_lensing_cell(GeodesicKind::Null, -0.82, 2.0, &expr, &[], 10.0, 7.0);
/// assert!(cell.primary.is_finite());
/// ```
#[must_use]
#[verified_engine::verified]
pub fn eval_custom_lapse_lensing_cell(
    kind: GeodesicKind,
    radial_bias: f64,
    rs: f64,
    compiled: &CompiledExpr,
    params: &[f64],
    r0_in: f64,
    b_val: f64,
) -> RadialAtlasSummary {
    let r0 = r0_in.max(0.35);
    let f0 = compiled.eval_2d(r0, rs, params);
    if !f0.is_finite() || f0 <= MIN_REGULAR_LAPSE {
        return RadialAtlasSummary {
            regime: LensingRegime::HorizonCaptureOrTransit,
            primary: 1.0,
            secondary: r0,
        };
    }
    let kappa = kind.kappa();
    let v0 = 0.5 * f0 * (kappa + (b_val * b_val) / (r0 * r0));
    let e_half_sq = match kind {
        GeodesicKind::Null => 0.5,
        GeodesicKind::Timelike => 0.5 * (radial_bias * 0.45).powi(2) + v0,
    };
    if v0 > e_half_sq + 1e-6 {
        return RadialAtlasSummary {
            regime: LensingRegime::CentrifugalBarrier,
            primary: 0.0,
            secondary: r0,
        };
    }
    let sign = if radial_bias > 0.0 { 1.0 } else { -1.0 };
    let vr0 = sign * (2.0 * (e_half_sq - v0).max(0.0)).sqrt();
    integrate_custom_lapse_radial(rs, compiled, params, kappa, b_val, r0, vr0)
}

#[verified_engine::verified]
fn integrate_custom_lapse_radial(
    rs: f64,
    compiled: &CompiledExpr,
    params: &[f64],
    kappa: f64,
    b_val: f64,
    r0: f64,
    vr0: f64,
) -> RadialAtlasSummary {
    let (mut r, mut vr, mut phi, mut r_min) = (r0, vr0, 0.0_f64, r0);
    for _ in 0..64 {
        let h = 1e-4 * r.max(1.0);
        let fp = compiled.eval_2d(r + h, rs, params);
        let fm = compiled.eval_2d((r - h).max(0.06), rs, params);
        if !fp.is_finite() || !fm.is_finite() || fm <= MIN_REGULAR_LAPSE {
            return capture_summary(phi, r.max(0.0));
        }
        let vp = 0.5 * fp * (kappa + (b_val * b_val) / ((r + h) * (r + h)));
        let vm = 0.5 * fm * (kappa + (b_val * b_val) / (r - h).max(0.06).powi(2));
        vr -= ((vp - vm) / (2.0 * h)) * 0.24;
        r += vr * 0.24;
        phi += (b_val / (r * r).max(0.04)) * 0.24;
        r_min = r_min.min(r);
        if r <= 0.2 || compiled.eval_2d(r, rs, params) <= MIN_REGULAR_LAPSE {
            return capture_summary(phi, r.max(0.0));
        }
        if r > 26.0 && vr > 0.0 {
            break;
        }
    }
    let regime = if phi.abs() >= 1.35 * PI {
        LensingRegime::PhotonRingWinding
    } else {
        LensingRegime::WeakDeflectionEscape
    };
    RadialAtlasSummary {
        regime,
        primary: (phi.abs() / (2.5 * PI)).clamp(0.0, 1.0),
        secondary: r_min,
    }
}

#[inline]
fn capture_summary(phi: f64, secondary: f64) -> RadialAtlasSummary {
    RadialAtlasSummary {
        regime: LensingRegime::HorizonCaptureOrTransit,
        primary: (phi.abs() / (2.0 * PI)).clamp(0.0, 1.0),
        secondary,
    }
}

/// Evaluates one 2D regime cell for a Morris-Thorne or custom $b(l)$ wormhole throat.
#[must_use]
#[verified_engine::verified]
pub fn eval_wormhole_lensing_cell(
    kind: GeodesicKind,
    radial_bias: f64,
    b0: f64,
    custom: Option<(&CompiledExpr, &[f64])>,
    l0: f64,
    b_val: f64,
) -> RadialAtlasSummary {
    let eval_r2 = |l: f64| match custom {
        Some((c, p)) => {
            let bv = c.eval_2d(l, b0, p);
            if bv.is_finite() && bv > MIN_REGULAR_THROAT {
                bv * bv
            } else {
                0.01
            }
        }
        None => b0 * b0 + l * l,
    };
    let r0_sq = eval_r2(l0);
    let throat_sq = eval_r2(0.0);
    let kinetic_0 = match kind {
        GeodesicKind::Null => 1.0 - (b_val * b_val) / r0_sq,
        GeodesicKind::Timelike => radial_bias * radial_bias,
    };
    if kinetic_0 < 0.0 {
        return RadialAtlasSummary {
            regime: LensingRegime::CentrifugalBarrier,
            primary: 0.0,
            secondary: l0.abs(),
        };
    }
    let e_kin = kinetic_0 + (b_val * b_val) / r0_sq;
    let transits = e_kin > (b_val * b_val) / throat_sq && l0 * radial_bias <= 0.0;
    let ratio = (b_val.abs() / throat_sq.sqrt().max(0.1)).clamp(0.0, 2.0);
    let regime = if transits {
        LensingRegime::HorizonCaptureOrTransit
    } else if (ratio - 1.0).abs() < 0.12 {
        LensingRegime::PhotonRingWinding
    } else {
        LensingRegime::WeakDeflectionEscape
    };
    let primary = (1.0 - 0.5 * ratio).clamp(0.0, 1.0);
    debug_assert!((0.0..=1.0).contains(&primary));
    RadialAtlasSummary {
        regime,
        primary,
        secondary: throat_sq.sqrt(),
    }
}

// [cite:general_relativity_geodesics]

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_custom_lapse_and_wormhole_lensing_cells_zero_alloc() {
        let lapse_ast = math_commons::ast::parse("1 - Rs / r").expect("parse lapse");
        let lapse = CompiledExpr::compile_with_vars(&lapse_ast, &["r", "Rs"])
            .expect("valid lapse expression");
        let wh_ast = math_commons::ast::parse("sqrt(b0^2 + l^2)").expect("parse wormhole");
        let wh = CompiledExpr::compile_with_vars(&wh_ast, &["l", "b0"])
            .expect("valid wormhole expression");
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((c_lapse, c_wh), _) = VerifiedEngine::run_verified(|| {
            let a = eval_custom_lapse_lensing_cell(
                GeodesicKind::Null,
                -0.82,
                2.0,
                &lapse,
                &[],
                8.0,
                6.0,
            );
            let b = eval_wormhole_lensing_cell(
                GeodesicKind::Null,
                -0.82,
                2.0,
                Some((&wh, &[])),
                6.0,
                0.5,
            );
            (a, b)
        });
        assert_eq!(c_lapse.regime, LensingRegime::WeakDeflectionEscape);
        assert_eq!(c_wh.regime, LensingRegime::HorizonCaptureOrTransit);
    }
}
