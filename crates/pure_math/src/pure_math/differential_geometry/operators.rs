// [cite:modular_polynomials_review]
//! Intrinsic differential operators on [`ParametricSurface`] manifolds (`laplace_beltrami` and `laplace_beltrami_ad`).

use math_scalars::HyperDual;

use super::surface::{ParametricSurface, SurfaceAnalysis, SurfaceJet2};

#[inline]
fn eval_lb_from_jet_and_hessian(
    jet: &SurfaceJet2,
    fu: f64,
    fv: f64,
    fuu: f64,
    fuv: f64,
    fvv: f64,
) -> f64 {
    let e = jet.ru.dot(&jet.ru);
    let f = jet.ru.dot(&jet.rv);
    let g = jet.rv.dot(&jet.rv);
    let det_g = e * g - f * f;
    if det_g <= 1e-12 || !det_g.is_finite() {
        return 0.0;
    }

    let e_u = 2.0 * jet.ru.dot(&jet.ruu);
    let f_u = jet.ruu.dot(&jet.rv) + jet.ru.dot(&jet.ruv);
    let g_u = 2.0 * jet.rv.dot(&jet.ruv);

    let e_v = 2.0 * jet.ru.dot(&jet.ruv);
    let f_v = jet.ruv.dot(&jet.rv) + jet.ru.dot(&jet.rvv);
    let g_v = 2.0 * jet.rv.dot(&jet.rvv);

    let det_u = e_u * g + e * g_u - 2.0 * f * f_u;
    let det_v = e_v * g + e * g_v - 2.0 * f * f_v;

    let num1 = g * fu - f * fv;
    let dnum1_du = g_u * fu + g * fuu - f_u * fv - f * fuv;
    let term_u = (dnum1_du - 0.5 * num1 * det_u / det_g) / det_g;

    let num2 = e * fv - f * fu;
    let dnum2_dv = e_v * fv + e * fvv - f_v * fu - f * fuv;
    let term_v = (dnum2_dv - 0.5 * num2 * det_v / det_g) / det_g;

    term_u + term_v
}

/// Evaluates the exact intrinsic Laplace-Beltrami operator $\Delta_S f(u, v)$ via second-order
/// [`HyperDual`] automatic differentiation with zero subtractive cancellation.
#[verified_engine::verified]
pub fn laplace_beltrami_ad<S, F>(surface: &S, u: f64, v: f64, func: F) -> f64
where
    S: ParametricSurface + ?Sized,
    F: Fn(HyperDual<f64>, HyperDual<f64>) -> HyperDual<f64>,
{
    debug_assert!(u.is_finite() && v.is_finite());
    if let Some(jet) = surface.eval_jet2(u, v) {
        let huu = func(HyperDual::second_order(u), HyperDual::constant(v));
        let hvv = func(HyperDual::constant(u), HyperDual::second_order(v));
        let huv = func(HyperDual::var1(u), HyperDual::var2(v));
        return eval_lb_from_jet_and_hessian(
            &jet, huu.eps1, hvv.eps1, huu.eps12, huv.eps12, hvv.eps12,
        );
    }
    laplace_beltrami_fallback(surface, u, v, &|uu, vv| {
        func(HyperDual::constant(uu), HyperDual::constant(vv)).val
    })
}

#[inline]
fn stencil_hessian_2d<F: Fn(f64, f64) -> f64>(u: f64, v: f64, h: f64, func: &F) -> [f64; 5] {
    let f0 = func(u, v);
    let fu = (func(u + 3.0 * h, v) - 9.0 * func(u + 2.0 * h, v) + 45.0 * func(u + h, v)
        - 45.0 * func(u - h, v)
        + 9.0 * func(u - 2.0 * h, v)
        - func(u - 3.0 * h, v))
        / (60.0 * h);
    let fv = (func(u, v + 3.0 * h) - 9.0 * func(u, v + 2.0 * h) + 45.0 * func(u, v + h)
        - 45.0 * func(u, v - h)
        + 9.0 * func(u, v - 2.0 * h)
        - func(u, v - 3.0 * h))
        / (60.0 * h);
    let fuu = (2.0 * func(u + 3.0 * h, v) - 27.0 * func(u + 2.0 * h, v) + 270.0 * func(u + h, v)
        - 490.0 * f0
        + 270.0 * func(u - h, v)
        - 27.0 * func(u - 2.0 * h, v)
        + 2.0 * func(u - 3.0 * h, v))
        / (180.0 * h * h);
    let fvv = (2.0 * func(u, v + 3.0 * h) - 27.0 * func(u, v + 2.0 * h) + 270.0 * func(u, v + h)
        - 490.0 * f0
        + 270.0 * func(u, v - h)
        - 27.0 * func(u, v - 2.0 * h)
        + 2.0 * func(u, v - 3.0 * h))
        / (180.0 * h * h);
    let d1 = func(u + h, v + h) - func(u + h, v - h) - func(u - h, v + h) + func(u - h, v - h);
    let d2 = func(u + 2.0 * h, v + 2.0 * h)
        - func(u + 2.0 * h, v - 2.0 * h)
        - func(u - 2.0 * h, v + 2.0 * h)
        + func(u - 2.0 * h, v - 2.0 * h);
    let fuv = (16.0 * d1 - d2) / (48.0 * h * h);
    [fu, fv, fuu, fuv, fvv]
}

/// Computes the Laplace-Beltrami operator $\Delta_S f$ at point $(u, v)$.
///
/// Uses exact forward-AD metric partials when `surface.eval_jet2(u, v)` is available, paired
/// with high-order central stencils on `func`.
#[verified_engine::verified]
pub fn laplace_beltrami<S, F>(surface: &S, u: f64, v: f64, func: &F) -> f64
where
    S: ParametricSurface,
    F: Fn(f64, f64) -> f64,
{
    debug_assert!(u.is_finite() && v.is_finite());
    if let Some(jet) = surface.eval_jet2(u, v) {
        let e = jet.ru.dot(&jet.ru);
        let f = jet.ru.dot(&jet.rv);
        let g = jet.rv.dot(&jet.rv);
        if e * g - f * f <= 1e-12 {
            return 0.0;
        }
        let [fu, fv, fuu, fuv, fvv] = stencil_hessian_2d(u, v, 0.01, func);
        return eval_lb_from_jet_and_hessian(&jet, fu, fv, fuu, fuv, fvv);
    }
    laplace_beltrami_fallback(surface, u, v, func)
}

fn laplace_beltrami_fallback<S, F>(surface: &S, u: f64, v: f64, func: &F) -> f64
where
    S: ParametricSurface + ?Sized,
    F: Fn(f64, f64) -> f64,
{
    let (e0, f0, g0) = surface.first_fundamental_form(u, v);
    let det_g0 = e0 * g0 - f0 * f0;
    if det_g0.abs() < 1e-12 {
        return 0.0;
    }
    let h = 1e-5;
    let inner = |u_curr: f64, v_curr: f64| -> (f64, f64) {
        let (e, f, g) = surface.first_fundamental_form(u_curr, v_curr);
        let det_g = e * g - f * f;
        if det_g.abs() < 1e-12 {
            return (0.0, 0.0);
        }
        let det_sqrt = det_g.sqrt();
        let f_u = (func(u_curr + h, v_curr) - func(u_curr - h, v_curr)) / (2.0 * h);
        let f_v = (func(u_curr, v_curr + h) - func(u_curr, v_curr - h)) / (2.0 * h);
        (
            (g * f_u - f * f_v) / det_sqrt,
            (e * f_v - f * f_u) / det_sqrt,
        )
    };
    let d_du_term1 = (inner(u + h, v).0 - inner(u - h, v).0) / (2.0 * h);
    let d_dv_term2 = (inner(u, v + h).1 - inner(u, v - h).1) / (2.0 * h);
    let det_sqrt = surface.area_element(u, v).max(1e-12);
    (d_du_term1 + d_dv_term2) / det_sqrt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::differential_geometry::surface::{Sphere, Torus};
    use math_scalars::Scalar;
    use std::f64::consts::PI;

    #[test]
    fn test_laplace_beltrami_pole_guard_and_spherical_harmonic() {
        let sphere = Sphere { radius: 1.0 };
        let y10 = |_u: f64, v: f64| v.cos();

        let north = laplace_beltrami(&sphere, 0.5, 0.0, &y10);
        let south = laplace_beltrami(&sphere, 1.2, PI, &y10);
        assert!(north.is_finite() && north.abs() < 1e-12);
        assert!(south.is_finite() && south.abs() < 1e-12);

        let v_mid = 0.9;
        let approx = laplace_beltrami(&sphere, 0.7, v_mid, &y10);
        let exact = -2.0 * v_mid.cos();
        assert!(
            (approx - exact).abs() < 1e-11,
            "got {approx}, expected {exact}"
        );

        let exact_ad = laplace_beltrami_ad(&sphere, 0.7, v_mid, |_u, v| v.cos());
        assert!(
            (exact_ad - exact).abs() < 1e-14,
            "AD LB error: {}",
            (exact_ad - exact).abs()
        );
    }

    #[test]
    fn test_laplace_beltrami_ad_torus_matches_exact_within_1e_14() {
        let (r_maj, r_min) = (2.3_f64, 0.7_f64);
        let torus = Torus {
            major_radius: r_maj,
            minor_radius: r_min,
        };
        let (u, v) = (0.65_f64, 1.15_f64);
        // f(u, v) = cos(u) * sin(v) on Torus (E = (R + r cos v)^2, F = 0, G = r^2):
        // \Delta f = f_uu / E + (1 / (r^2 (R + r cos v))) * d/dv ((R + r cos v) f_v)
        let lb = laplace_beltrami_ad(&torus, u, v, |hu, hv| hu.cos() * hv.sin());
        let ring = r_maj + r_min * v.cos();
        let exact = -u.cos() * v.sin() / (ring * ring)
            + u.cos() * (-r_min * v.sin() * v.cos() - ring * v.sin()) / (r_min * r_min * ring);
        assert!((lb - exact).abs() < 1e-14, "got {lb}, expected {exact}");
    }
}
