use super::surface::{ParametricSurface, SurfaceAnalysis};

/// Computes the Laplace-Beltrami operator $\Delta_S f$ at point $(u, v)$.
/// $\Delta_S f = \frac{1}{\sqrt{g}} [ \partial_u (\frac{G f_u - F f_v}{\sqrt{g}}) + \partial_v (\frac{E f_v - F f_u}{\sqrt{g}}) ]$
#[verified_engine::verified]
pub fn laplace_beltrami<S, F>(surface: &S, u: f64, v: f64, func: &F) -> f64
where
    S: ParametricSurface,
    F: Fn(f64, f64) -> f64,
{
    assert!(u.is_finite() && v.is_finite());
    let det_sqrt_center = surface.area_element(u, v);
    if det_sqrt_center <= 1e-12 || !det_sqrt_center.is_finite() {
        return 0.0;
    }
    let h = 1e-5;

    // Inner term helper
    let inner = |u_curr: f64, v_curr: f64| -> (f64, f64) {
        let (e, f, g) = surface.first_fundamental_form(u_curr, v_curr);
        let det = e * g - f * f;
        if det <= 1e-12 || !det.is_finite() {
            return (0.0, 0.0);
        }
        let det_sqrt = det.sqrt();

        // Derivatives of the function
        let f_u = (func(u_curr + h, v_curr) - func(u_curr - h, v_curr)) / (2.0 * h);
        let f_v = (func(u_curr, v_curr + h) - func(u_curr, v_curr - h)) / (2.0 * h);

        let term1 = (g * f_u - f * f_v) / det_sqrt;
        let term2 = (e * f_v - f * f_u) / det_sqrt;

        (term1, term2)
    };

    // Outer derivatives
    let (term1_plus_u, _) = inner(u + h, v);
    let (term1_minus_u, _) = inner(u - h, v);
    let d_du_term1 = (term1_plus_u - term1_minus_u) / (2.0 * h);

    let (_, term2_plus_v) = inner(u, v + h);
    let (_, term2_minus_v) = inner(u, v - h);
    let d_dv_term2 = (term2_plus_v - term2_minus_v) / (2.0 * h);

    (1.0 / det_sqrt_center) * (d_du_term1 + d_dv_term2)
}
