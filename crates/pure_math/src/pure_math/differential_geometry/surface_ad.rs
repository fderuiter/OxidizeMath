// [cite:modular_polynomials_review]
//! Exact zero-allocation `DualVec<f64, 2>` and `HyperDual<f64>` surface jet evaluators.

use math_commons::ast::CompiledExpr;
use math_scalars::{DualVec, HyperDual, Scalar};
use nalgebra::{Point3, Vector3};

use super::surface::{KleinBottle, Sphere, Torus};

/// Exact first- and second-order parametric surface derivatives at $(u, v)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceJet2 {
    /// Surface point $r(u, v)$.
    pub pos: Point3<f64>,
    /// First partial $\partial r / \partial u$.
    pub ru: Vector3<f64>,
    /// First partial $\partial r / \partial v$.
    pub rv: Vector3<f64>,
    /// Second partial $\partial^2 r / \partial u^2$.
    pub ruu: Vector3<f64>,
    /// Mixed second partial $\partial^2 r / \partial u \partial v$.
    pub ruv: Vector3<f64>,
    /// Second partial $\partial^2 r / \partial v^2$.
    pub rvv: Vector3<f64>,
}

#[inline]
pub(super) fn sanitize_scalar<S: Scalar>(val: S) -> S {
    let r = val.real_part();
    if !val.is_finite_val() || !r.is_finite() {
        S::ZERO
    } else if r < -50.0 {
        S::from_f64(-50.0)
    } else if r > 50.0 {
        S::from_f64(50.0)
    } else {
        val
    }
}

#[inline]
pub(super) fn eval_partials_ad<F>(u: f64, v: f64, f: F) -> (Vector3<f64>, Vector3<f64>)
where
    F: Fn(DualVec<f64, 2>, DualVec<f64, 2>) -> [DualVec<f64, 2>; 3],
{
    let out = f(DualVec::variable(u, 0), DualVec::variable(v, 1));
    let ru = Vector3::new(out[0].eps[0], out[1].eps[0], out[2].eps[0]);
    let rv = Vector3::new(out[0].eps[1], out[1].eps[1], out[2].eps[1]);
    (ru, rv)
}

#[inline]
pub(super) fn eval_jet2_ad<F>(u: f64, v: f64, f: F) -> SurfaceJet2
where
    F: Fn(HyperDual<f64>, HyperDual<f64>) -> [HyperDual<f64>; 3],
{
    let uu = f(HyperDual::second_order(u), HyperDual::constant(v));
    let vv = f(HyperDual::constant(u), HyperDual::second_order(v));
    let uv = f(HyperDual::var1(u), HyperDual::var2(v));
    SurfaceJet2 {
        pos: Point3::new(uu[0].val, uu[1].val, uu[2].val),
        ru: Vector3::new(uu[0].eps1, uu[1].eps1, uu[2].eps1),
        rv: Vector3::new(vv[0].eps1, vv[1].eps1, vv[2].eps1),
        ruu: Vector3::new(uu[0].eps12, uu[1].eps12, uu[2].eps12),
        ruv: Vector3::new(uv[0].eps12, uv[1].eps12, uv[2].eps12),
        rvv: Vector3::new(vv[0].eps12, vv[1].eps12, vv[2].eps12),
    }
}

#[inline]
pub(super) fn sphere_scalar<S: Scalar>(s: &Sphere, u: S, v: S) -> [S; 3] {
    let r = S::from_f64(s.radius);
    let (su, cu) = u.sin_cos();
    let (sv, cv) = v.sin_cos();
    [r * sv * cu, r * sv * su, r * cv]
}

#[inline]
pub(super) fn torus_scalar<S: Scalar>(t: &Torus, u: S, v: S) -> [S; 3] {
    let r_maj = S::from_f64(t.major_radius);
    let r_min = S::from_f64(t.minor_radius);
    let (su, cu) = u.sin_cos();
    let (sv, cv) = v.sin_cos();
    let ring = r_maj + r_min * cv;
    [ring * cu, ring * su, r_min * sv]
}

#[inline]
pub(super) fn klein_scalar<S: Scalar>(k: &KleinBottle, u: S, v: S) -> [S; 3] {
    let r = S::from_f64(k.radius);
    let half_r = r * S::HALF;
    let (su, cu) = u.sin_cos();
    let (su2, cu2) = (u * S::HALF).sin_cos();
    let (sv, cv) = v.sin_cos();
    let profile = r + half_r * cv * cu2 - half_r * sv * su2;
    [
        profile * cu,
        profile * su,
        half_r * sv * cu2 + half_r * cv * su2,
    ]
}

#[inline]
pub(super) fn heightfield_scalar<S: Scalar>(
    compiled: &CompiledExpr,
    params: &[f64],
    u: S,
    v: S,
) -> [S; 3] {
    let z = sanitize_scalar(compiled.eval_2d_scalar(u, v, params));
    [u, v, z]
}

#[inline]
pub(super) fn parametric_triple_scalar<S: Scalar>(
    x_expr: &CompiledExpr,
    y_expr: &CompiledExpr,
    z_expr: &CompiledExpr,
    params: &[f64],
    u: S,
    v: S,
) -> [S; 3] {
    [
        sanitize_scalar(x_expr.eval_2d_scalar(u, v, params)),
        sanitize_scalar(y_expr.eval_2d_scalar(u, v, params)),
        sanitize_scalar(z_expr.eval_2d_scalar(u, v, params)),
    ]
}
