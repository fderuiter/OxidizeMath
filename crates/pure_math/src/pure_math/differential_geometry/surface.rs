// [cite:modular_polynomials_review]
use math_commons::ast::CompiledExpr;
use nalgebra::{Matrix2, Point3, Vector3};

/// Maximum inline parameter slots stored on compiled [`ParametricSurface`] evaluators.
pub const MAX_SURFACE_PARAMS: usize = 6;

/// Represents a parametric surface $r(u, v)$ in $\mathbb{R}^3$.
pub trait ParametricSurface {
    /// Returns the 3D point at parameters $(u, v)$.
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64>;

    /// Returns the partial derivative $\frac{\partial r}{\partial u}$.
    /// Default implementation uses finite differences.
    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let h = math_commons::registry::TOLERANCE_FAST;
        (self.position(u + h, v) - self.position(u - h, v)) / (2.0 * h)
    }

    /// Returns the partial derivative $\frac{\partial r}{\partial v}$.
    /// Default implementation uses finite differences.
    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let h = math_commons::registry::TOLERANCE_FAST;
        (self.position(u, v + h) - self.position(u, v - h)) / (2.0 * h)
    }

    /// Returns the natural parameter interval $[u_{\min}, u_{\max}]$ for single-cover meshing.
    #[verified_engine::verified]
    fn u_domain(&self) -> (f64, f64) {
        (0.0, std::f64::consts::TAU)
    }

    /// Returns the natural parameter interval $[v_{\min}, v_{\max}]$ for single-cover meshing.
    #[verified_engine::verified]
    fn v_domain(&self) -> (f64, f64) {
        (0.0, std::f64::consts::TAU)
    }

    /// Returns `true` when $r(u_{\min}, v) = r(u_{\max}, v)$ so seam vertices weld along $u$.
    #[verified_engine::verified]
    fn is_u_periodic(&self) -> bool {
        true
    }

    /// Returns `true` when $r(u, v_{\min}) = r(u, v_{\max})$ so seam vertices weld along $v$.
    #[verified_engine::verified]
    fn is_v_periodic(&self) -> bool {
        true
    }

    /// Returns `true` when $v = v_{\min}$ and $v = v_{\max}$ collapse to single polar vertices.
    #[verified_engine::verified]
    fn has_v_poles(&self) -> bool {
        false
    }

    /// Finite unit normal fallback used at singular points where $|r_u \times r_v| = 0$.
    #[verified_engine::verified]
    fn singular_normal(&self, _u: f64, _v: f64) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, 1.0)
    }

    /// Outward-pointing finite unit normal for 3D mesh export without pole `NaN` singularities.
    #[verified_engine::verified]
    fn outward_normal(&self, u: f64, v: f64) -> Vector3<f64>
    where
        Self: Sized,
    {
        self.unit_normal(u, v)
    }

    /// Object-safe outward unit normal query for `dyn ParametricSurface` mesh builders.
    #[verified_engine::verified]
    fn mesh_normal(&self, u: f64, v: f64) -> Vector3<f64> {
        let ru = self.partial_u(u, v);
        let rv = self.partial_v(u, v);
        let cross = ru.cross(&rv);
        let norm = cross.norm();
        if norm > 1.0e-12 && norm.is_finite() {
            cross / norm
        } else {
            self.singular_normal(u, v)
        }
    }
}

/// Extension trait providing geometric analysis methods for any `ParametricSurface`.
/// Separating these satisfies the Interface Segregation Principle.
pub trait SurfaceAnalysis {
    /// Returns the unit normal vector $\mathbf{n} = \frac{r_u \times r_v}{|r_u \times r_v|}$.
    #[verified_engine::verified]
    fn unit_normal(&self, u: f64, v: f64) -> Vector3<f64>;

    /// Computes the First Fundamental Form coefficients $(E, F, G)$.
    /// $E = r_u \cdot r_u$, $F = r_u \cdot r_v$, $G = r_v \cdot r_v$
    #[verified_engine::verified]
    fn first_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64);

    /// Computes the Second Fundamental Form coefficients $(L, M, N)$.
    /// $L = r_{uu} \cdot n$, $M = r_{uv} \cdot n$, $N = r_{vv} \cdot n$
    #[verified_engine::verified]
    fn second_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64);

    /// Computes the Gaussian Curvature $K = \frac{LN - M^2}{EG - F^2}$.
    #[verified_engine::verified]
    fn gaussian_curvature(&self, u: f64, v: f64) -> f64;

    /// Computes the Mean Curvature $H = \frac{EN - 2FM + GL}{2(EG - F^2)}$.
    #[verified_engine::verified]
    fn mean_curvature(&self, u: f64, v: f64) -> f64;

    /// Computes the metric tensor $g_{ij}$ and its inverse $g^{ij}$.
    /// Returns (g_det, g_inv)
    #[verified_engine::verified]
    fn metric_tensor_inverse(&self, u: f64, v: f64) -> (f64, Matrix2<f64>);

    /// Differential area element $\sqrt{EG - F^2}$.
    #[verified_engine::verified]
    fn area_element(&self, u: f64, v: f64) -> f64;
}

impl<T: ParametricSurface> SurfaceAnalysis for T {
    #[verified_engine::verified]
    fn unit_normal(&self, u: f64, v: f64) -> Vector3<f64> {
        let ru = self.partial_u(u, v);
        let rv = self.partial_v(u, v);
        let cross = ru.cross(&rv);
        let norm = cross.norm();
        if norm > 1.0e-12 && norm.is_finite() {
            cross / norm
        } else {
            self.singular_normal(u, v)
        }
    }

    #[verified_engine::verified]
    fn first_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let ru = self.partial_u(u, v);
        let rv = self.partial_v(u, v);
        (ru.dot(&ru), ru.dot(&rv), rv.dot(&rv))
    }

    #[verified_engine::verified]
    fn second_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let h = 1e-5;
        let n = self.unit_normal(u, v);

        // Second derivatives via finite differences
        let p = self.position(u, v);
        let pu_plus = self.position(u + h, v);
        let pu_minus = self.position(u - h, v);
        let ruu = (pu_plus.coords - p.coords * 2.0 + pu_minus.coords) / (h * h);

        let pv_plus = self.position(u, v + h);
        let pv_minus = self.position(u, v - h);
        let rvv = (pv_plus.coords - p.coords * 2.0 + pv_minus.coords) / (h * h);

        let puv_pp = self.position(u + h, v + h);
        let puv_mm = self.position(u - h, v - h);
        let puv_pm = self.position(u + h, v - h);
        let puv_mp = self.position(u - h, v + h);
        let ruv = (puv_pp.coords - puv_mp.coords - puv_pm.coords + puv_mm.coords) / (4.0 * h * h); // Central difference mixed

        (ruu.dot(&n), ruv.dot(&n), rvv.dot(&n))
    }

    #[verified_engine::verified]
    fn gaussian_curvature(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let (l, m, n) = self.second_fundamental_form(u, v);

        let det_g = e * g - f * f;
        let det_ii = l * n - m * m;

        det_ii / det_g
    }

    #[verified_engine::verified]
    fn mean_curvature(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let (l, m, n) = self.second_fundamental_form(u, v);

        let det_g = e * g - f * f;

        (e * n - 2.0 * f * m + g * l) / (2.0 * det_g)
    }

    #[verified_engine::verified]
    fn metric_tensor_inverse(&self, u: f64, v: f64) -> (f64, Matrix2<f64>) {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let det = e * g - f * f;
        let inv = Matrix2::new(g, -f, -f, e) / det;
        (det, inv)
    }

    #[verified_engine::verified]
    fn area_element(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        (e * g - f * f).sqrt()
    }
}

/// A Sphere of radius R.
pub struct Sphere {
    #[allow(missing_docs)]
    pub radius: f64,
}

impl ParametricSurface for Sphere {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        // u = theta (azimuthal in [0, 2*pi)), v = phi (polar in [0, pi])
        let x = self.radius * v.sin() * u.cos();
        let y = self.radius * v.sin() * u.sin();
        let z = self.radius * v.cos();
        Point3::new(x, y, z)
    }

    #[verified_engine::verified]
    fn v_domain(&self) -> (f64, f64) {
        (0.0, std::f64::consts::PI)
    }

    #[verified_engine::verified]
    fn is_v_periodic(&self) -> bool {
        false
    }

    #[verified_engine::verified]
    fn has_v_poles(&self) -> bool {
        true
    }

    #[verified_engine::verified]
    fn singular_normal(&self, _u: f64, v: f64) -> Vector3<f64> {
        let sign = if v.cos() >= 0.0 { -1.0 } else { 1.0 };
        Vector3::new(0.0, 0.0, sign)
    }

    #[verified_engine::verified]
    fn outward_normal(&self, u: f64, v: f64) -> Vector3<f64> {
        Vector3::new(v.sin() * u.cos(), v.sin() * u.sin(), v.cos())
    }

    #[verified_engine::verified]
    fn mesh_normal(&self, u: f64, v: f64) -> Vector3<f64> {
        Vector3::new(v.sin() * u.cos(), v.sin() * u.sin(), v.cos())
    }
}

/// A Torus with major radius R and minor radius r.
pub struct Torus {
    #[allow(missing_docs)]
    pub major_radius: f64,
    #[allow(missing_docs)]
    pub minor_radius: f64,
}

impl ParametricSurface for Torus {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        // u \in [0, 2pi), v \in [0, 2pi)
        let x = (self.major_radius + self.minor_radius * v.cos()) * u.cos();
        let y = (self.major_radius + self.minor_radius * v.cos()) * u.sin();
        let z = self.minor_radius * v.sin();
        Point3::new(x, y, z)
    }
}

/// A Klein Bottle immersion (Figure-8 parametrization) in R^3.
/// This self-intersects since a true Klein bottle cannot be embedded in R^3 without intersection.
pub struct KleinBottle {
    #[allow(missing_docs)]
    pub radius: f64,
}

impl ParametricSurface for KleinBottle {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        // Standard "figure-8" immersion:
        // u \in [0, 2pi), v \in [0, 2pi)
        let r = self.radius;
        let x = (r + r / 2.0 * v.cos() * (u / 2.0).cos() - r / 2.0 * v.sin() * (u / 2.0).sin())
            * u.cos();
        let y = (r + r / 2.0 * v.cos() * (u / 2.0).cos() - r / 2.0 * v.sin() * (u / 2.0).sin())
            * u.sin();
        let z = r / 2.0 * v.sin() * (u / 2.0).cos() + r / 2.0 * v.cos() * (u / 2.0).sin();
        Point3::new(x, y, z)
    }

    #[verified_engine::verified]
    fn is_u_periodic(&self) -> bool {
        false
    }
}

#[inline]
fn sanitize_surface_coord(val: f64) -> f64 {
    if val.is_finite() {
        val.clamp(-50.0, 50.0)
    } else {
        0.0
    }
}

fn pack_surface_params(params: &[f64]) -> ([f64; MAX_SURFACE_PARAMS], usize) {
    let mut inline = [0.0; MAX_SURFACE_PARAMS];
    let len = params.len().min(MAX_SURFACE_PARAMS);
    inline[..len].copy_from_slice(&params[..len]);
    (inline, len)
}

/// Heightfield surface $r(x, y) = (x, y, f(x, y))$ backed by zero-allocation [`CompiledExpr`] bytecode.
#[derive(Debug, Clone)]
pub struct CompiledHeightfieldSurface {
    compiled: CompiledExpr,
    params: [f64; MAX_SURFACE_PARAMS],
    param_len: usize,
}

impl CompiledHeightfieldSurface {
    /// Constructs a heightfield surface from a [`CompiledExpr`] compiled over axes `["x", "y"]`.
    #[must_use]
    pub fn new(compiled: CompiledExpr, params: &[f64]) -> Self {
        let (packed, param_len) = pack_surface_params(params);
        Self {
            compiled,
            params: packed,
            param_len,
        }
    }

    /// Returns a reference to the underlying compiled height expression.
    #[must_use]
    pub const fn compiled(&self) -> &CompiledExpr {
        &self.compiled
    }
}

impl ParametricSurface for CompiledHeightfieldSurface {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let z = self.compiled.eval_2d(u, v, &self.params[..self.param_len]);
        Point3::new(u, v, sanitize_surface_coord(z))
    }

    #[verified_engine::verified]
    fn u_domain(&self) -> (f64, f64) {
        (-2.0, 2.0)
    }

    #[verified_engine::verified]
    fn v_domain(&self) -> (f64, f64) {
        (-2.0, 2.0)
    }

    #[verified_engine::verified]
    fn is_u_periodic(&self) -> bool {
        false
    }

    #[verified_engine::verified]
    fn is_v_periodic(&self) -> bool {
        false
    }
}

/// Parametric 3D surface $r(u, v) = (x(u, v), y(u, v), z(u, v))$ backed by [`CompiledExpr`] triples.
#[derive(Debug, Clone)]
pub struct CompiledParametricSurface {
    x_expr: CompiledExpr,
    y_expr: CompiledExpr,
    z_expr: CompiledExpr,
    params: [f64; MAX_SURFACE_PARAMS],
    param_len: usize,
}

impl CompiledParametricSurface {
    /// Constructs a parametric surface from three [`CompiledExpr`] formulas compiled over `["u", "v"]`.
    #[must_use]
    pub fn new(
        x_expr: CompiledExpr,
        y_expr: CompiledExpr,
        z_expr: CompiledExpr,
        params: &[f64],
    ) -> Self {
        let (packed, param_len) = pack_surface_params(params);
        Self {
            x_expr,
            y_expr,
            z_expr,
            params: packed,
            param_len,
        }
    }
}

impl ParametricSurface for CompiledParametricSurface {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let p = &self.params[..self.param_len];
        let x = sanitize_surface_coord(self.x_expr.eval_2d(u, v, p));
        let y = sanitize_surface_coord(self.y_expr.eval_2d(u, v, p));
        let z = sanitize_surface_coord(self.z_expr.eval_2d(u, v, p));
        Point3::new(x, y, z)
    }

    #[verified_engine::verified]
    fn is_u_periodic(&self) -> bool {
        false
    }

    #[verified_engine::verified]
    fn is_v_periodic(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod compiled_surface_tests {
    use super::*;
    use math_commons::ast::parse;

    #[test]
    fn test_compiled_heightfield_saddle_negative_gaussian_curvature() {
        let expr = parse("a * (x^2 - y^2)").expect("parse saddle");
        let compiled = CompiledExpr::compile_with_vars(&expr, &["x", "y"]).expect("compile saddle");
        let surface = CompiledHeightfieldSurface::new(compiled, &[0.5]);

        let pos = surface.position(2.0, 1.0);
        assert!((pos.z - 1.5).abs() < 1e-12);

        // At origin (0, 0), z = a(x^2 - y^2) has K = -4 a^2 = -1.0 for a = 0.5
        let k_origin = surface.gaussian_curvature(0.0, 0.0);
        assert!((k_origin - -1.0).abs() < 1e-5, "got K = {k_origin}");
    }

    #[test]
    fn test_compiled_parametric_triple_torus_parity() {
        let shared = ["r_maj", "r_min"];
        let cx = CompiledExpr::compile_with_vars_and_params(
            &parse("(r_maj + r_min * cos(v)) * cos(u)").unwrap(),
            &["u", "v"],
            &shared,
        )
        .unwrap();
        let cy = CompiledExpr::compile_with_vars_and_params(
            &parse("(r_maj + r_min * cos(v)) * sin(u)").unwrap(),
            &["u", "v"],
            &shared,
        )
        .unwrap();
        let cz = CompiledExpr::compile_with_vars_and_params(
            &parse("r_min * sin(v)").unwrap(),
            &["u", "v"],
            &shared,
        )
        .unwrap();

        let compiled_torus = CompiledParametricSurface::new(cx, cy, cz, &[2.0, 0.6]);
        let analytic_torus = Torus {
            major_radius: 2.0,
            minor_radius: 0.6,
        };

        let p_comp = compiled_torus.position(0.7, 1.1);
        let p_ana = analytic_torus.position(0.7, 1.1);
        assert!((p_comp - p_ana).norm() < 1e-12);
        assert!(
            (compiled_torus.gaussian_curvature(0.7, 1.1)
                - analytic_torus.gaussian_curvature(0.7, 1.1))
            .abs()
                < 1e-6
        );
    }
}
