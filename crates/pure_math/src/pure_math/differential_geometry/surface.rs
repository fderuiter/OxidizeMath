// [cite:modular_polynomials_review]
#[path = "surface_compiled.rs"]
mod surface_compiled;

pub use surface_compiled::{CompiledHeightfieldSurface, CompiledParametricSurface};

use math_scalars::{Dual, HyperDual, Scalar};
use nalgebra::{Matrix2, Point3, Vector3};

/// Maximum inline parameter slots stored on compiled [`ParametricSurface`] evaluators.
pub const MAX_SURFACE_PARAMS: usize = 6;

/// Generic scalar evaluation contract enabling exact zero-allocation [`Dual`] and [`HyperDual`]
/// surface jets without breaking `dyn ParametricSurface` object safety.
pub trait DifferentiableSurface: ParametricSurface {
    /// Evaluates the surface embedding $r(u, v) \in \mathbb{R}^3$ over any algebraic [`Scalar`].
    fn position_scalar<S: Scalar>(&self, u: S, v: S) -> [S; 3];
}

/// Evaluates $\partial_u r(u, v)$ exactly via forward-mode [`Dual<f64>`].
#[inline]
#[verified_engine::verified]
pub fn eval_partial_u_ad<T: DifferentiableSurface + ?Sized>(
    surface: &T,
    u: f64,
    v: f64,
) -> Vector3<f64> {
    let out = surface.position_scalar(Dual::variable(u), Dual::constant(v));
    Vector3::new(out[0].eps, out[1].eps, out[2].eps)
}

/// Evaluates $\partial_v r(u, v)$ exactly via forward-mode [`Dual<f64>`].
#[inline]
#[verified_engine::verified]
pub fn eval_partial_v_ad<T: DifferentiableSurface + ?Sized>(
    surface: &T,
    u: f64,
    v: f64,
) -> Vector3<f64> {
    let out = surface.position_scalar(Dual::constant(u), Dual::variable(v));
    Vector3::new(out[0].eps, out[1].eps, out[2].eps)
}

/// Evaluates exact second partial derivatives $(r_{uu}, r_{uv}, r_{vv})$ via [`HyperDual<f64>`]
/// without subtractive cancellation.
#[inline]
#[verified_engine::verified]
pub fn eval_second_partials_ad<T: DifferentiableSurface + ?Sized>(
    surface: &T,
    u: f64,
    v: f64,
) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let uu = surface.position_scalar(HyperDual::new(u, 1.0, 1.0, 0.0), HyperDual::constant(v));
    let uv = surface.position_scalar(
        HyperDual::new(u, 1.0, 0.0, 0.0),
        HyperDual::new(v, 0.0, 1.0, 0.0),
    );
    let vv = surface.position_scalar(HyperDual::constant(u), HyperDual::new(v, 1.0, 1.0, 0.0));
    (
        Vector3::new(uu[0].eps12, uu[1].eps12, uu[2].eps12),
        Vector3::new(uv[0].eps12, uv[1].eps12, uv[2].eps12),
        Vector3::new(vv[0].eps12, vv[1].eps12, vv[2].eps12),
    )
}

/// Represents a parametric surface $r(u, v)$ in $\mathbb{R}^3$.
pub trait ParametricSurface {
    /// Returns the 3D point at parameters $(u, v)$.
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64>;

    /// Returns the partial derivative $\frac{\partial r}{\partial u}$.
    /// Built-in surfaces override this with exact forward-mode [`Dual<f64>`].
    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let h = f64::EPSILON.sqrt() * u.abs().max(1.0);
        (self.position(u + h, v) - self.position(u - h, v)) / (2.0 * h)
    }

    /// Returns the partial derivative $\frac{\partial r}{\partial v}$.
    /// Built-in surfaces override this with exact forward-mode [`Dual<f64>`].
    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let h = f64::EPSILON.sqrt() * v.abs().max(1.0);
        (self.position(u, v + h) - self.position(u, v - h)) / (2.0 * h)
    }

    /// Returns the second partial derivatives $(r_{uu}, r_{uv}, r_{vv})$ at $(u, v)$.
    /// Built-in surfaces override this with exact [`HyperDual<f64>`] second-order jets.
    #[verified_engine::verified]
    fn second_partials(&self, u: f64, v: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let hu = f64::EPSILON.powf(0.25) * u.abs().max(1.0);
        let hv = f64::EPSILON.powf(0.25) * v.abs().max(1.0);
        let p = self.position(u, v);
        let ruu = (self.position(u + hu, v).coords - p.coords * 2.0
            + self.position(u - hu, v).coords)
            / (hu * hu);
        let rvv = (self.position(u, v + hv).coords - p.coords * 2.0
            + self.position(u, v - hv).coords)
            / (hv * hv);
        let ruv = (self.position(u + hu, v + hv).coords
            - self.position(u - hu, v + hv).coords
            - self.position(u + hu, v - hv).coords
            + self.position(u - hu, v - hv).coords)
            / (4.0 * hu * hv);
        (ruu, ruv, rvv)
    }
}

/// Extension trait providing geometric analysis methods for any `ParametricSurface`.
pub trait SurfaceAnalysis {
    /// Returns the unit normal vector $\mathbf{n} = \frac{r_u \times r_v}{|r_u \times r_v|}$.
    #[verified_engine::verified]
    fn unit_normal(&self, u: f64, v: f64) -> Vector3<f64>;

    /// Computes the First Fundamental Form coefficients $(E, F, G)$.
    #[verified_engine::verified]
    fn first_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64);

    /// Computes the Second Fundamental Form coefficients $(L, M, N)$.
    #[verified_engine::verified]
    fn second_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64);

    /// Computes the Gaussian Curvature $K = \frac{LN - M^2}{EG - F^2}$.
    #[verified_engine::verified]
    fn gaussian_curvature(&self, u: f64, v: f64) -> f64;

    /// Computes the Mean Curvature $H = \frac{EN - 2FM + GL}{2(EG - F^2)}$.
    #[verified_engine::verified]
    fn mean_curvature(&self, u: f64, v: f64) -> f64;

    /// Computes the metric tensor $g_{ij}$ and its inverse $g^{ij}$ as `(g_det, g_inv)`.
    #[verified_engine::verified]
    fn metric_tensor_inverse(&self, u: f64, v: f64) -> (f64, Matrix2<f64>);

    /// Differential area element $\sqrt{EG - F^2}$.
    #[verified_engine::verified]
    fn area_element(&self, u: f64, v: f64) -> f64;
}

impl<T: ?Sized + ParametricSurface> SurfaceAnalysis for T {
    #[verified_engine::verified]
    fn unit_normal(&self, u: f64, v: f64) -> Vector3<f64> {
        let ru = self.partial_u(u, v);
        let rv = self.partial_v(u, v);
        ru.cross(&rv).normalize()
    }

    #[verified_engine::verified]
    fn first_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let ru = self.partial_u(u, v);
        let rv = self.partial_v(u, v);
        (ru.dot(&ru), ru.dot(&rv), rv.dot(&rv))
    }

    #[verified_engine::verified]
    fn second_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let n = self.unit_normal(u, v);
        let (ruu, ruv, rvv) = self.second_partials(u, v);
        (ruu.dot(&n), ruv.dot(&n), rvv.dot(&n))
    }

    #[verified_engine::verified]
    fn gaussian_curvature(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let (l, m, n) = self.second_fundamental_form(u, v);
        (l * n - m * m) / (e * g - f * f)
    }

    #[verified_engine::verified]
    fn mean_curvature(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let (l, m, n) = self.second_fundamental_form(u, v);
        (e * n - 2.0 * f * m + g * l) / (2.0 * (e * g - f * f))
    }

    #[verified_engine::verified]
    fn metric_tensor_inverse(&self, u: f64, v: f64) -> (f64, Matrix2<f64>) {
        let (e, f, g) = self.first_fundamental_form(u, v);
        let det = e * g - f * f;
        (det, Matrix2::new(g, -f, -f, e) / det)
    }

    #[verified_engine::verified]
    fn area_element(&self, u: f64, v: f64) -> f64 {
        let (e, f, g) = self.first_fundamental_form(u, v);
        (e * g - f * f).sqrt()
    }
}

/// A Sphere of radius R.
pub struct Sphere {
    /// Value for `radius` (radius).
    pub radius: f64,
}

impl DifferentiableSurface for Sphere {
    #[inline]
    fn position_scalar<S: Scalar>(&self, u: S, v: S) -> [S; 3] {
        let r = S::from_f64(self.radius);
        let (sv, cv) = v.sin_cos();
        let (su, cu) = u.sin_cos();
        [r * sv * cu, r * sv * su, r * cv]
    }
}

impl ParametricSurface for Sphere {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let [x, y, z] = self.position_scalar(u, v);
        Point3::new(x, y, z)
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        eval_partial_u_ad(self, u, v)
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        eval_partial_v_ad(self, u, v)
    }

    #[verified_engine::verified]
    fn second_partials(&self, u: f64, v: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        eval_second_partials_ad(self, u, v)
    }
}

/// A Torus with major radius R and minor radius r.
pub struct Torus {
    /// Value for `major_radius` (major radius).
    pub major_radius: f64,
    /// Value for `minor_radius` (minor radius).
    pub minor_radius: f64,
}

impl DifferentiableSurface for Torus {
    #[inline]
    fn position_scalar<S: Scalar>(&self, u: S, v: S) -> [S; 3] {
        let r_maj = S::from_f64(self.major_radius);
        let r_min = S::from_f64(self.minor_radius);
        let (sv, cv) = v.sin_cos();
        let (su, cu) = u.sin_cos();
        let tube = r_maj + r_min * cv;
        [tube * cu, tube * su, r_min * sv]
    }
}

impl ParametricSurface for Torus {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let [x, y, z] = self.position_scalar(u, v);
        Point3::new(x, y, z)
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        eval_partial_u_ad(self, u, v)
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        eval_partial_v_ad(self, u, v)
    }

    #[verified_engine::verified]
    fn second_partials(&self, u: f64, v: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        eval_second_partials_ad(self, u, v)
    }
}

/// A Klein Bottle immersion (Figure-8 parametrization) in R^3.
pub struct KleinBottle {
    /// Value for `radius` (radius).
    pub radius: f64,
}

impl DifferentiableSurface for KleinBottle {
    #[inline]
    fn position_scalar<S: Scalar>(&self, u: S, v: S) -> [S; 3] {
        let r = S::from_f64(self.radius);
        let half_r = r * S::HALF;
        let half_u = u * S::HALF;
        let (sv, cv) = v.sin_cos();
        let (su, cu) = u.sin_cos();
        let (shu, chu) = half_u.sin_cos();
        let rad = r + half_r * cv * chu - half_r * sv * shu;
        [rad * cu, rad * su, half_r * sv * chu + half_r * cv * shu]
    }
}

impl ParametricSurface for KleinBottle {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let [x, y, z] = self.position_scalar(u, v);
        Point3::new(x, y, z)
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        eval_partial_u_ad(self, u, v)
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        eval_partial_v_ad(self, u, v)
    }

    #[verified_engine::verified]
    fn second_partials(&self, u: f64, v: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        eval_second_partials_ad(self, u, v)
    }
}

#[cfg(test)]
mod compiled_surface_tests {
    use super::*;
    use math_commons::ast::{CompiledExpr, parse};

    #[test]
    fn test_compiled_heightfield_saddle_negative_gaussian_curvature() {
        let expr = parse("a * (x^2 - y^2)").expect("parse saddle");
        let compiled = CompiledExpr::compile_with_vars(&expr, &["x", "y"]).expect("compile saddle");
        let surface = CompiledHeightfieldSurface::new(compiled, &[0.5]);

        let pos = surface.position(2.0, 1.0);
        assert!((pos.z - 1.5).abs() < 1e-12);

        let k_origin = surface.gaussian_curvature(0.0, 0.0);
        assert!((k_origin - -1.0).abs() < 1e-14, "got K = {k_origin}");
    }

    #[test]
    fn test_torus_and_compiled_parametric_gaussian_curvature_machine_precision() {
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

        let (r_maj, r_min) = (2.0, 0.6);
        let compiled_torus = CompiledParametricSurface::new(cx, cy, cz, &[r_maj, r_min]);
        let analytic_torus = Torus {
            major_radius: r_maj,
            minor_radius: r_min,
        };

        for &(u, v) in &[(0.7, 1.1), (2.3, 0.2), (4.1, 3.8), (1.0, -0.9)] {
            let exact_k = v.cos() / (r_min * (r_maj + r_min * v.cos()));
            let k_torus = analytic_torus.gaussian_curvature(u, v);
            let k_comp = compiled_torus.gaussian_curvature(u, v);
            let rel_err = (k_torus - exact_k).abs() / exact_k.abs().max(1.0);
            assert!(rel_err < 1e-14, "Torus K rel_err={rel_err} at ({u},{v})");
            assert!(
                (k_comp - exact_k).abs() < 1e-14,
                "Compiled Torus K diff={} at ({u},{v})",
                (k_comp - exact_k).abs()
            );
        }
    }
}
