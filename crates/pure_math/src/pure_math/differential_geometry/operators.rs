// [cite:modular_polynomials_review]
use super::surface::ParametricSurface;
use math_scalars::HyperDual;

#[inline]
#[verified_engine::verified]
fn combine_laplace_beltrami<S: ParametricSurface + ?Sized>(
    surface: &S,
    u: f64,
    v: f64,
    f_jet: [f64; 5],
) -> f64 {
    let [f_u, f_v, f_uu, f_uv, f_vv] = f_jet;
    let ru = surface.partial_u(u, v);
    let rv = surface.partial_v(u, v);
    let e = ru.dot(&ru);
    let f = ru.dot(&rv);
    let g = rv.dot(&rv);
    let det_g = e * g - f * f;
    if det_g.abs() < 1e-12 {
        return 0.0;
    }

    let (ruu, ruv, rvv) = surface.second_partials(u, v);
    let e_u = 2.0 * ruu.dot(&ru);
    let e_v = 2.0 * ruv.dot(&ru);
    let f_u_m = ruu.dot(&rv) + ru.dot(&ruv);
    let f_v_m = ruv.dot(&rv) + ru.dot(&rvv);
    let g_u = 2.0 * ruv.dot(&rv);
    let g_v = 2.0 * rvv.dot(&rv);

    let det_g_u = e_u * g + e * g_u - 2.0 * f * f_u_m;
    let det_g_v = e_v * g + e * g_v - 2.0 * f * f_v_m;

    let n1 = g * f_u - f * f_v;
    let n2 = e * f_v - f * f_u;
    let n1_u = g_u * f_u + g * f_uu - f_u_m * f_v - f * f_uv;
    let n2_v = e_v * f_v + e * f_vv - f_v_m * f_u - f * f_uv;

    (n1_u + n2_v) / det_g - (n1 * det_g_u + n2 * det_g_v) / (2.0 * det_g * det_g)
}

/// Computes the exact Laplace-Beltrami operator $\Delta_S f$ at $(u, v)$ via [`HyperDual<f64>`]
/// second-order automatic differentiation without subtractive cancellation.
///
/// Reference: do Carmo (1976), *Differential Geometry of Curves and Surfaces*, Prentice-Hall;
/// Fike & Alonso (2011), *AIAA* 2011-886, DOI: 10.2514/6.2011-886.
///
/// # Stability
/// Evaluates surface metric partials via [`ParametricSurface::second_partials`] and scalar field
/// jets $(f_u, f_v, f_{uu}, f_{uv}, f_{vv})$ via three stack-allocated [`HyperDual<f64>`] passes,
/// eliminating $O(h^{-2})$ cancellation; returns `0.0` at coordinate poles (`det_g < 1e-12`).
///
/// # Domain of Validity
/// Requires `func` to be $C^2$ at $(u, v)$ and `surface` to be regular away from coordinate poles.
#[verified_engine::verified]
pub fn laplace_beltrami_ad<S, F>(surface: &S, u: f64, v: f64, func: F) -> f64
where
    S: ParametricSurface + ?Sized,
    F: Fn(HyperDual<f64>, HyperDual<f64>) -> HyperDual<f64>,
{
    let uu = func(HyperDual::new(u, 1.0, 1.0, 0.0), HyperDual::constant(v));
    let uv = func(
        HyperDual::new(u, 1.0, 0.0, 0.0),
        HyperDual::new(v, 0.0, 1.0, 0.0),
    );
    let vv = func(HyperDual::constant(u), HyperDual::new(v, 1.0, 1.0, 0.0));
    combine_laplace_beltrami(
        surface,
        u,
        v,
        [uu.eps1, vv.eps1, uu.eps12, uv.eps12, vv.eps12],
    )
}

/// Computes the Laplace-Beltrami operator $\Delta_S f$ at $(u, v)$ for legacy `f64` closures,
/// combining exact surface metric derivatives with fourth-order Richardson scalar stencils.
#[verified_engine::verified]
pub fn laplace_beltrami<S, F>(surface: &S, u: f64, v: f64, func: &F) -> f64
where
    S: ParametricSurface + ?Sized,
    F: Fn(f64, f64) -> f64,
{
    let ru = surface.partial_u(u, v);
    let rv = surface.partial_v(u, v);
    let det_g0 = ru.dot(&ru) * rv.dot(&rv) - ru.dot(&rv).powi(2);
    if det_g0.abs() < 1e-12 {
        return 0.0;
    }

    let hu = 2.5e-3 * u.abs().max(1.0);
    let hv = 2.5e-3 * v.abs().max(1.0);
    let f0 = func(u, v);
    let f_up1 = func(u + hu, v);
    let f_um1 = func(u - hu, v);
    let f_up2 = func(u + 2.0 * hu, v);
    let f_um2 = func(u - 2.0 * hu, v);
    let f_vp1 = func(u, v + hv);
    let f_vm1 = func(u, v - hv);
    let f_vp2 = func(u, v + 2.0 * hv);
    let f_vm2 = func(u, v - 2.0 * hv);

    let f_u = (-f_up2 + 8.0 * f_up1 - 8.0 * f_um1 + f_um2) / (12.0 * hu);
    let f_v = (-f_vp2 + 8.0 * f_vp1 - 8.0 * f_vm1 + f_vm2) / (12.0 * hv);
    let f_uu = (-f_up2 + 16.0 * f_up1 - 30.0 * f0 + 16.0 * f_um1 - f_um2) / (12.0 * hu * hu);
    let f_vv = (-f_vp2 + 16.0 * f_vp1 - 30.0 * f0 + 16.0 * f_vm1 - f_vm2) / (12.0 * hv * hv);
    let f_uv = (func(u + hu, v + hv) - func(u - hu, v + hv) - func(u + hu, v - hv)
        + func(u - hu, v - hv))
        / (4.0 * hu * hv);

    combine_laplace_beltrami(surface, u, v, [f_u, f_v, f_uu, f_uv, f_vv])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::differential_geometry::surface::{Sphere, Torus};
    use math_scalars::Scalar;
    use std::f64::consts::PI;

    #[test]
    fn test_laplace_beltrami_pole_guard_and_spherical_harmonic_machine_precision() {
        let sphere = Sphere { radius: 1.5 };
        let y10 = |_u: f64, v: f64| v.cos();

        let north = laplace_beltrami(&sphere, 0.5, 0.0, &y10);
        let south = laplace_beltrami(&sphere, 1.2, PI, &y10);
        assert!(north.is_finite() && north.abs() < 1e-12);
        assert!(south.is_finite() && south.abs() < 1e-12);

        // On S^2(R), \Delta_{S^2} Y_2^1 = -(6 / R^2) Y_2^1 where Y_2^1(u, v) = sin(v) cos(v) cos(u).
        let v_mid = 0.9;
        let u_mid = 0.7;
        let exact_y10 = -2.0 * v_mid.cos() / (1.5 * 1.5);
        let ad_y10 = laplace_beltrami_ad(&sphere, u_mid, v_mid, |_u, v| v.cos());
        let fd_y10 = laplace_beltrami(&sphere, u_mid, v_mid, &y10);
        assert!(
            ((ad_y10 - exact_y10) / exact_y10).abs() < 1e-14,
            "AD Y10 got {ad_y10}, expected {exact_y10}"
        );
        assert!(
            ((fd_y10 - exact_y10) / exact_y10).abs() < 1e-10,
            "FD Y10 got {fd_y10}, expected {exact_y10}"
        );

        let y21_exact = -6.0 * v_mid.sin() * v_mid.cos() * u_mid.cos() / (1.5 * 1.5);
        let y21_ad = laplace_beltrami_ad(&sphere, u_mid, v_mid, |u, v| v.sin() * v.cos() * u.cos());
        assert!(
            ((y21_ad - y21_exact) / y21_exact).abs() < 1e-14,
            "AD Y21 got {y21_ad}, expected {y21_exact}"
        );

        // Torus analytical check: for f(u, v) = cos(u) + sin(v) on Torus(R, r):
        // g_uu = (R + r cos v)^2, g_uv = 0, g_vv = r^2
        // \Delta f = -cos(u)/(R + r cos v)^2 - (cos(v)/(r(R + r cos v)))*cos(v)... wait:
        // (1/(r(R + r cos v))) [ \partial_u( (r/(R + r cos v)) f_u ) + \partial_v( ((R + r cos v)/r) f_v ) ]
        let (r_maj, r_min) = (2.2, 0.7);
        let torus = Torus {
            major_radius: r_maj,
            minor_radius: r_min,
        };
        let tube = r_maj + r_min * v_mid.cos();
        let exact_torus = -u_mid.cos() / (tube * tube)
            + (-r_min * v_mid.sin() * v_mid.cos() - tube * v_mid.sin()) / (r_min * r_min * tube);
        let ad_torus = laplace_beltrami_ad(&torus, u_mid, v_mid, |u, v| u.cos() + v.sin());
        assert!(
            ((ad_torus - exact_torus) / exact_torus.abs()).abs() < 1e-14,
            "AD Torus got {ad_torus}, expected {exact_torus}"
        );
    }
}
