// [cite:modular_polynomials_review]
use engine_symbolic::Jet2;
use math_commons::ast::{CompiledExpr, Expr};
use nalgebra::Vector3;

use super::surface::ParametricSurface;

/// Differentiable 2D scalar field $f(u, v)$ on a [`ParametricSurface`] domain.
///
/// Implemented for:
/// - [`Jet2Field`] wrapping `Fn(Jet2<2>, Jet2<2>) -> Jet2<2>` (**1** `Jet2<2>` pass),
/// - [`CompiledExpr`] / [`Expr`] (**1** `Jet2<2>` pass),
/// - Legacy closures `Fn(f64, f64) -> f64` (backward-compatible Taylor lift).
pub trait ScalarField2D {
    /// Evaluates the second-order Taylor jet of $f(u, v)$ at `(u, v)`.
    fn eval_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> Jet2<2>;
}

/// Zero-allocation wrapper lifting a `Jet2<2>` closure into [`ScalarField2D`] so
/// [`laplace_beltrami`] executes in **1** `Jet2<2>` pass with zero finite differences.
#[derive(Debug, Clone, Copy)]
pub struct Jet2Field<G>(pub G);

impl<G> ScalarField2D for Jet2Field<G>
where
    G: Fn(Jet2<2>, Jet2<2>) -> Jet2<2>,
{
    #[inline]
    fn eval_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> Jet2<2> {
        (self.0)(u, v)
    }
}

impl ScalarField2D for CompiledExpr {
    #[inline]
    #[verified_engine::verified]
    fn eval_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> Jet2<2> {
        self.eval_2d_jet2(u, v, &[])
    }
}

impl ScalarField2D for Expr {
    #[verified_engine::verified]
    fn eval_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> Jet2<2> {
        let free = self.free_variables();
        let uses_xy =
            free.iter().any(|s| s == "x" || s == "y") && !free.iter().any(|s| s == "u" || s == "v");
        let axes = if uses_xy { ["x", "y"] } else { ["u", "v"] };
        CompiledExpr::compile_with_vars(self, &axes)
            .map_or(Jet2::NAN, |c| c.eval_2d_jet2(u, v, &[]))
    }
}

impl<F> ScalarField2D for F
where
    F: Fn(f64, f64) -> f64,
{
    #[verified_engine::verified]
    fn eval_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> Jet2<2> {
        let (u0, v0) = (u.val, v.val);
        let h = 1e-5;
        let f0 = self(u0, v0);
        let f_up = self(u0 + h, v0);
        let f_um = self(u0 - h, v0);
        let f_vp = self(u0, v0 + h);
        let f_vm = self(u0, v0 - h);
        let f_pp = self(u0 + h, v0 + h);
        let f_pm = self(u0 + h, v0 - h);
        let f_mp = self(u0 - h, v0 + h);
        let f_mm = self(u0 - h, v0 - h);
        let fu = (f_up - f_um) / (2.0 * h);
        let fv = (f_vp - f_vm) / (2.0 * h);
        let fuu = (f_up - 2.0 * f0 + f_um) / (h * h);
        let fvv = (f_vp - 2.0 * f0 + f_vm) / (h * h);
        let fuv = (f_pp - f_mp - f_pm + f_mm) / (4.0 * h * h);
        let du = u - u0;
        let dv = v - v0;
        Jet2::constant(f0)
            + du * fu
            + dv * fv
            + du * du * (0.5 * fuu)
            + du * dv * fuv
            + dv * dv * (0.5 * fvv)
    }
}

#[verified_engine::verified]
fn eval_laplace_beltrami_from_jets<S: ParametricSurface + ?Sized>(
    surface: &S,
    u: f64,
    v: f64,
    fj: Jet2<2>,
) -> f64 {
    let rj = surface.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
    let ru = Vector3::new(rj[0].du(), rj[1].du(), rj[2].du());
    let rv = Vector3::new(rj[0].dv(), rj[1].dv(), rj[2].dv());
    let ruu = Vector3::new(rj[0].duu(), rj[1].duu(), rj[2].duu());
    let ruv = Vector3::new(rj[0].duv(), rj[1].duv(), rj[2].duv());
    let rvv = Vector3::new(rj[0].dvv(), rj[1].dvv(), rj[2].dvv());

    let e = ru.dot(&ru);
    let f = ru.dot(&rv);
    let g = rv.dot(&rv);
    let det_g = e * g - f * f;
    if det_g <= 0.0 || !det_g.is_finite() {
        return f64::NAN;
    }

    let e_u = 2.0 * ruu.dot(&ru);
    let e_v = 2.0 * ruv.dot(&ru);
    let f_u = ruu.dot(&rv) + ru.dot(&ruv);
    let f_v = ruv.dot(&rv) + ru.dot(&rvv);
    let g_u = 2.0 * ruv.dot(&rv);
    let g_v = 2.0 * rvv.dot(&rv);

    let det_u = e_u * g + e * g_u - 2.0 * f * f_u;
    let det_v = e_v * g + e * g_v - 2.0 * f * f_v;

    let (fu, fv, fuu, fuv, fvv) = (fj.du(), fj.dv(), fj.duu(), fj.duv(), fj.dvv());
    let n1 = g * fu - f * fv;
    let n2 = e * fv - f * fu;
    let dn1_du = g_u * fu + g * fuu - f_u * fv - f * fuv;
    let dn2_dv = e_v * fv + e * fvv - f_v * fu - f * fuv;

    (dn1_du + dn2_dv) / det_g - (n1 * det_u + n2 * det_v) / (2.0 * det_g * det_g)
}

/// Computes the Laplace-Beltrami operator $\Delta_S f$ at $(u, v)$ in a single `Jet2<2>` pass.
#[verified_engine::verified]
pub fn laplace_beltrami<S, F>(surface: &S, u: f64, v: f64, func: &F) -> f64
where
    S: ParametricSurface + ?Sized,
    F: ScalarField2D + ?Sized,
{
    let fj = func.eval_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
    eval_laplace_beltrami_from_jets(surface, u, v, fj)
}

/// Computes $\Delta_S f(u, v)$ directly from a `Jet2<2>` closure in **1** pass without stencils.
#[verified_engine::verified]
pub fn laplace_beltrami_jet2<S, G>(surface: &S, u: f64, v: f64, func: G) -> f64
where
    S: ParametricSurface + ?Sized,
    G: FnOnce(Jet2<2>, Jet2<2>) -> Jet2<2>,
{
    let fj = func(Jet2::var(u, 0), Jet2::var(v, 1));
    eval_laplace_beltrami_from_jets(surface, u, v, fj)
}
