// [cite:modular_polynomials_review]
#[path = "surface_compiled.rs"]
mod surface_compiled;

use engine_symbolic::Jet2;
use nalgebra::{Matrix2, Point3, Vector3};

pub use surface_compiled::{CompiledHeightfieldSurface, CompiledParametricSurface};

/// Maximum inline parameter slots stored on compiled [`ParametricSurface`] evaluators.
pub const MAX_SURFACE_PARAMS: usize = 6;

/// Default 3D viewport coordinate clamp applied at mesh/scene projection boundaries (not inside
/// differential-geometry evaluators, preventing artificial boundary kink curvature spikes).
pub const SURFACE_VIEWPORT_CLAMP: f64 = 50.0;

/// Clamps a 3D surface point to `[-SURFACE_VIEWPORT_CLAMP, SURFACE_VIEWPORT_CLAMP]^3` for GUI
/// mesh/wireframe projection after geometric curvatures have been evaluated on the true manifold.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn clamp_surface_point(p: Point3<f64>) -> Point3<f64> {
    let clamp_c = |c: f64| {
        if c.is_finite() {
            c.clamp(-SURFACE_VIEWPORT_CLAMP, SURFACE_VIEWPORT_CLAMP)
        } else {
            0.0
        }
    };
    Point3::new(clamp_c(p.x), clamp_c(p.y), clamp_c(p.z))
}

#[inline]
#[verified_engine::verified]
pub(super) fn sanitize_surface_coord(val: f64) -> f64 {
    if val.is_finite() { val } else { 0.0 }
}

/// Represents a parametric surface $r(u, v)$ in $\mathbb{R}^3$.
pub trait ParametricSurface {
    /// Returns the 3D point at parameters $(u, v)$.
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64>;

    /// Evaluates second-order Taylor jets $[x(u, v), y(u, v), z(u, v)]$ in a single `Jet2<2>` pass.
    #[verified_engine::verified]
    fn position_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> [Jet2<2>; 3] {
        let (u0, v0) = (u.val, v.val);
        let p = self.position(u0, v0);
        let ru = self.partial_u(u0, v0);
        let rv = self.partial_v(u0, v0);
        let h = 1e-5;
        let ruu = (self.position(u0 + h, v0).coords - p.coords * 2.0
            + self.position(u0 - h, v0).coords)
            / (h * h);
        let rvv = (self.position(u0, v0 + h).coords - p.coords * 2.0
            + self.position(u0, v0 - h).coords)
            / (h * h);
        let ruv = (self.position(u0 + h, v0 + h).coords
            - self.position(u0 - h, v0 + h).coords
            - self.position(u0 + h, v0 - h).coords
            + self.position(u0 - h, v0 - h).coords)
            / (4.0 * h * h);
        [
            compose_coord_jet(p.x, [ru.x, rv.x], [ruu.x, ruv.x, rvv.x], u, v),
            compose_coord_jet(p.y, [ru.y, rv.y], [ruu.y, ruv.y, rvv.y], u, v),
            compose_coord_jet(p.z, [ru.z, rv.z], [ruu.z, ruv.z, rvv.z], u, v),
        ]
    }

    /// Returns the partial derivative $\frac{\partial r}{\partial u}$.
    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let h = math_commons::registry::TOLERANCE_FAST;
        (self.position(u + h, v) - self.position(u - h, v)) / (2.0 * h)
    }

    /// Returns the partial derivative $\frac{\partial r}{\partial v}$.
    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let h = math_commons::registry::TOLERANCE_FAST;
        (self.position(u, v + h) - self.position(u, v - h)) / (2.0 * h)
    }
}

#[verified_engine::verified]
fn compose_coord_jet(
    val: f64,
    first: [f64; 2],
    second: [f64; 3],
    u: Jet2<2>,
    v: Jet2<2>,
) -> Jet2<2> {
    let du = u - u.val;
    let dv = v - v.val;
    Jet2::constant(val)
        + du * first[0]
        + dv * first[1]
        + du * du * (0.5 * second[0])
        + du * dv * second[1]
        + dv * dv * (0.5 * second[2])
}

struct SurfacePartials {
    ru: Vector3<f64>,
    rv: Vector3<f64>,
    ruu: Vector3<f64>,
    ruv: Vector3<f64>,
    rvv: Vector3<f64>,
}

#[verified_engine::verified]
fn eval_surface_jets<T: ParametricSurface + ?Sized>(surf: &T, u: f64, v: f64) -> SurfacePartials {
    let jets = surf.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
    SurfacePartials {
        ru: Vector3::new(jets[0].du(), jets[1].du(), jets[2].du()),
        rv: Vector3::new(jets[0].dv(), jets[1].dv(), jets[2].dv()),
        ruu: Vector3::new(jets[0].duu(), jets[1].duu(), jets[2].duu()),
        ruv: Vector3::new(jets[0].duv(), jets[1].duv(), jets[2].duv()),
        rvv: Vector3::new(jets[0].dvv(), jets[1].dvv(), jets[2].dvv()),
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

    /// Computes the metric tensor $g_{ij}$ and its inverse $g^{ij}$, returning `(g_det, g_inv)`.
    #[verified_engine::verified]
    fn metric_tensor_inverse(&self, u: f64, v: f64) -> (f64, Matrix2<f64>);

    /// Differential area element $\sqrt{EG - F^2}$.
    #[verified_engine::verified]
    fn area_element(&self, u: f64, v: f64) -> f64;
}

impl<T: ?Sized + ParametricSurface> SurfaceAnalysis for T {
    #[verified_engine::verified]
    fn unit_normal(&self, u: f64, v: f64) -> Vector3<f64> {
        let p = eval_surface_jets(self, u, v);
        p.ru.cross(&p.rv).normalize()
    }

    #[verified_engine::verified]
    fn first_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let p = eval_surface_jets(self, u, v);
        (p.ru.dot(&p.ru), p.ru.dot(&p.rv), p.rv.dot(&p.rv))
    }

    #[verified_engine::verified]
    fn second_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let p = eval_surface_jets(self, u, v);
        let n = p.ru.cross(&p.rv).normalize();
        (p.ruu.dot(&n), p.ruv.dot(&n), p.rvv.dot(&n))
    }

    #[verified_engine::verified]
    fn gaussian_curvature(&self, u: f64, v: f64) -> f64 {
        let p = eval_surface_jets(self, u, v);
        let cross = p.ru.cross(&p.rv);
        let det_g = cross.norm_squared();
        let n = cross / det_g.sqrt();
        let (l, m, nn) = (p.ruu.dot(&n), p.ruv.dot(&n), p.rvv.dot(&n));
        (l * nn - m * m) / det_g
    }

    #[verified_engine::verified]
    fn mean_curvature(&self, u: f64, v: f64) -> f64 {
        let p = eval_surface_jets(self, u, v);
        let (e, f, g) = (p.ru.dot(&p.ru), p.ru.dot(&p.rv), p.rv.dot(&p.rv));
        let det_g = e * g - f * f;
        let n = p.ru.cross(&p.rv) / det_g.sqrt();
        let (l, m, nn) = (p.ruu.dot(&n), p.ruv.dot(&n), p.rvv.dot(&n));
        (e * nn - 2.0 * f * m + g * l) / (2.0 * det_g)
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

/// A Sphere of radius $R$.
pub struct Sphere {
    #[allow(missing_docs)]
    pub radius: f64,
}

impl ParametricSurface for Sphere {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let (sin_v, cos_v) = v.sin_cos();
        let (sin_u, cos_u) = u.sin_cos();
        Point3::new(
            self.radius * sin_v * cos_u,
            self.radius * sin_v * sin_u,
            self.radius * cos_v,
        )
    }

    #[verified_engine::verified]
    fn position_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> [Jet2<2>; 3] {
        let r = self.radius;
        let sin_v = v.sin();
        [sin_v * u.cos() * r, sin_v * u.sin() * r, v.cos() * r]
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].du(), jets[1].du(), jets[2].du())
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].dv(), jets[1].dv(), jets[2].dv())
    }
}

/// A Torus with major radius $R$ and minor radius $r$.
pub struct Torus {
    #[allow(missing_docs)]
    pub major_radius: f64,
    #[allow(missing_docs)]
    pub minor_radius: f64,
}

impl ParametricSurface for Torus {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let ring = self.major_radius + self.minor_radius * v.cos();
        Point3::new(ring * u.cos(), ring * u.sin(), self.minor_radius * v.sin())
    }

    #[verified_engine::verified]
    fn position_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> [Jet2<2>; 3] {
        let ring = v.cos() * self.minor_radius + self.major_radius;
        [ring * u.cos(), ring * u.sin(), v.sin() * self.minor_radius]
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].du(), jets[1].du(), jets[2].du())
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].dv(), jets[1].dv(), jets[2].dv())
    }
}

/// A Klein Bottle immersion (Figure-8 parametrization) in $\mathbb{R}^3$.
pub struct KleinBottle {
    #[allow(missing_docs)]
    pub radius: f64,
}

impl ParametricSurface for KleinBottle {
    #[verified_engine::verified]
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        let r = self.radius;
        let half_u = 0.5 * u;
        let tube = r + 0.5 * r * (v.cos() * half_u.cos() - v.sin() * half_u.sin());
        let z = 0.5 * r * (v.sin() * half_u.cos() + v.cos() * half_u.sin());
        Point3::new(tube * u.cos(), tube * u.sin(), z)
    }

    #[verified_engine::verified]
    fn position_jet2(&self, u: Jet2<2>, v: Jet2<2>) -> [Jet2<2>; 3] {
        let r = self.radius;
        let half_r = 0.5 * r;
        let half_u = u * 0.5;
        let (sin_v, cos_v) = (v.sin(), v.cos());
        let (sin_hu, cos_hu) = (half_u.sin(), half_u.cos());
        let tube = (cos_v * cos_hu - sin_v * sin_hu) * half_r + r;
        let z = (sin_v * cos_hu + cos_v * sin_hu) * half_r;
        [tube * u.cos(), tube * u.sin(), z]
    }

    #[verified_engine::verified]
    fn partial_u(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].du(), jets[1].du(), jets[2].du())
    }

    #[verified_engine::verified]
    fn partial_v(&self, u: f64, v: f64) -> Vector3<f64> {
        let jets = self.position_jet2(Jet2::var(u, 0), Jet2::var(v, 1));
        Vector3::new(jets[0].dv(), jets[1].dv(), jets[2].dv())
    }
}

#[cfg(test)]
mod compiled_surface_tests {
    use super::*;
    use math_commons::ast::{CompiledExpr, parse};

    #[test]
    fn test_shallow_saddle_machine_precision_and_no_clamp_kink_spike() {
        // Shallow saddle at z = 100.0 (above viewport clamp 50.0): exact K(0, 0) = -4.0e-8.
        let expr = parse("100.0 + 1e-4 * (x^2 - y^2)").expect("parse shallow saddle");
        let compiled = CompiledExpr::compile_with_vars(&expr, &["x", "y"]).expect("compile");
        let surface = CompiledHeightfieldSurface::new(compiled, &[]);

        let k_origin = surface.gaussian_curvature(0.0, 0.0);
        assert!(
            (k_origin - (-4.0e-8)).abs() < 1e-20,
            "expected -4.0e-8 within 1e-20, got {k_origin:e}"
        );

        // Paraboloid z = x^2 + y^2 at (5, 5) where z = 50.0 previously produced a -1941.5 spike.
        let paraboloid = CompiledHeightfieldSurface::new(
            CompiledExpr::compile_with_vars(&parse("x^2 + y^2").unwrap(), &["x", "y"]).unwrap(),
            &[],
        );
        let k_boundary = paraboloid.gaussian_curvature(5.0, 5.0);
        let exact_k = 4.0 / (1.0_f64 + 4.0 * 25.0 + 4.0 * 25.0).powi(2);
        assert!(
            (k_boundary - exact_k).abs() < 1e-16,
            "expected {exact_k:e} without -1941.5 clamp spike, got {k_boundary:e}"
        );
        let clamped = clamp_surface_point(Point3::new(0.0, 0.0, 120.0));
        assert_eq!(clamped.z, SURFACE_VIEWPORT_CLAMP);
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
                < 1e-14
        );
    }
}
