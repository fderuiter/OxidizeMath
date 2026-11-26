use super::surface::{ParametricSurface, SurfaceAnalysis};

/// Computes the Laplace-Beltrami operator $\Delta_S f$ at point $(u, v)$.
/// $\Delta_S f = \frac{1}{\sqrt{g}} [ \partial_u (\frac{G f_u - F f_v}{\sqrt{g}}) + \partial_v (\frac{E f_v - F f_u}{\sqrt{g}}) ]$
#[verified_engine::verified]
pub fn laplace_beltrami<S, F>(surface: &S, u: f64, v: f64, func: &F) -> f64
where
    S: ParametricSurface,
    F: Fn(f64, f64) -> f64,
{
    let (e0, f0, g0) = surface.first_fundamental_form(u, v);
    let det_g0 = e0 * g0 - f0 * f0;
    if det_g0.abs() < 1e-12 {
        return 0.0;
    }

    let h = 1e-5;

    // Inner term helper with pole guard for singular coordinate boundaries.
    let inner = |u_curr: f64, v_curr: f64| -> (f64, f64) {
        let (e, f, g) = surface.first_fundamental_form(u_curr, v_curr);
        let det_g = e * g - f * f;
        if det_g.abs() < 1e-12 {
            return (0.0, 0.0);
        }
        let det_sqrt = det_g.sqrt();

        let f_u = (func(u_curr + h, v_curr) - func(u_curr - h, v_curr)) / (2.0 * h);
        let f_v = (func(u_curr, v_curr + h) - func(u_curr, v_curr - h)) / (2.0 * h);

        let term1 = (g * f_u - f * f_v) / det_sqrt;
        let term2 = (e * f_v - f * f_u) / det_sqrt;

        (term1, term2)
    };

    let (term1_plus_u, _) = inner(u + h, v);
    let (term1_minus_u, _) = inner(u - h, v);
    let d_du_term1 = (term1_plus_u - term1_minus_u) / (2.0 * h);

    let (_, term2_plus_v) = inner(u, v + h);
    let (_, term2_minus_v) = inner(u, v - h);
    let d_dv_term2 = (term2_plus_v - term2_minus_v) / (2.0 * h);

    let det_sqrt = surface.area_element(u, v).max(1e-12);
    (1.0 / det_sqrt) * (d_du_term1 + d_dv_term2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::differential_geometry::surface::Sphere;
    use std::f64::consts::PI;

    #[test]
    fn test_laplace_beltrami_pole_guard_and_spherical_harmonic() {
        let sphere = Sphere { radius: 1.0 };
        let y10 = |_u: f64, v: f64| v.cos();

        // Exact poles v = 0 and v = PI have det_g = 0 and must return finite 0.0 (no NaN/Inf).
        let north = laplace_beltrami(&sphere, 0.5, 0.0, &y10);
        let south = laplace_beltrami(&sphere, 1.2, PI, &y10);
        assert!(north.is_finite() && north.abs() < 1e-12);
        assert!(south.is_finite() && south.abs() < 1e-12);

        // Interior point: \Delta_{S^2} cos(v) = -2 cos(v).
        let v_mid = 0.9;
        let approx = laplace_beltrami(&sphere, 0.7, v_mid, &y10);
        let exact = -2.0 * v_mid.cos();
        assert!(
            (approx - exact).abs() < 1e-5,
            "got {approx}, expected {exact}"
        );
    }
}
