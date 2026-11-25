// [cite:quasiperfect_alcf]
use math_scalars::{DualVec, Scalar};
use nalgebra::Vector3;

/// Trait for Orthogonal Coordinate Systems.
/// Defines how to convert to/from Cartesian and the scale factors $(h_1, h_2, h_3)$.
pub trait OrthogonalCoordinateSystem {
    /// Converts a point from this coordinate system to Cartesian (x, y, z).
    #[verified_engine::verified]
    fn local_to_cartesian(&self, point: &Vector3<f64>) -> Vector3<f64>;

    /// Converts a point from Cartesian (x, y, z) to this coordinate system.
    #[verified_engine::verified]
    fn cartesian_to_local(&self, cartesian: &Vector3<f64>) -> Vector3<f64>;

    /// Returns the scale factors $(h_1, h_2, h_3)$ at the given point (in this coordinate system).
    #[verified_engine::verified]
    fn scale_factors(&self, point: &Vector3<f64>) -> Vector3<f64>;

    /// Returns the Jacobian of scale factors `partials[m][k] = \partial h_m / \partial q_k`.
    /// Built-in coordinate systems override this with single-pass [`DualVec<f64, 3>`] AD.
    #[verified_engine::verified]
    fn scale_factor_partials(&self, point: &Vector3<f64>) -> [[f64; 3]; 3] {
        let mut out = [[0.0_f64; 3]; 3];
        for k in 0..3 {
            let step = f64::EPSILON.sqrt() * point[k].abs().max(1.0);
            let mut p_plus = *point;
            let mut p_minus = *point;
            p_plus[k] += step;
            p_minus[k] -= step;
            let diff = (self.scale_factors(&p_plus) - self.scale_factors(&p_minus)) / (2.0 * step);
            for m in 0..3 {
                out[m][k] = diff[m];
            }
        }
        out
    }
}

/// Generic scalar extension enabling zero-allocation forward-AD scale factors over [`Scalar`].
pub trait DifferentiableCoordinates: OrthogonalCoordinateSystem {
    /// Evaluates $(h_1, h_2, h_3)$ over any algebraic [`Scalar`] `S`.
    fn scale_factors_scalar<S: Scalar>(&self, point: &[S; 3]) -> [S; 3];
}

#[inline]
#[verified_engine::verified]
fn eval_scale_partials_ad<C: DifferentiableCoordinates + ?Sized>(
    coords: &C,
    point: &Vector3<f64>,
) -> [[f64; 3]; 3] {
    let seed = DualVec::<f64, 3>::jacobian_seed([point[0], point[1], point[2]]);
    let h = coords.scale_factors_scalar(&seed);
    [h[0].eps, h[1].eps, h[2].eps]
}

/// Cartesian Coordinates (x, y, z). Scale factors are all 1.
pub struct Cartesian;

impl DifferentiableCoordinates for Cartesian {
    #[inline]
    fn scale_factors_scalar<S: Scalar>(&self, _point: &[S; 3]) -> [S; 3] {
        [S::ONE, S::ONE, S::ONE]
    }
}

impl OrthogonalCoordinateSystem for Cartesian {
    #[verified_engine::verified]
    fn local_to_cartesian(&self, point: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        *point
    }

    #[verified_engine::verified]
    fn cartesian_to_local(&self, cartesian: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(cartesian.iter().all(|x| x.is_finite()));
        *cartesian
    }

    #[verified_engine::verified]
    fn scale_factors(&self, _point: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(_point.iter().all(|x| x.is_finite()));
        Vector3::new(1.0, 1.0, 1.0)
    }

    #[verified_engine::verified]
    fn scale_factor_partials(&self, point: &Vector3<f64>) -> [[f64; 3]; 3] {
        eval_scale_partials_ad(self, point)
    }
}

/// Cylindrical Coordinates $(\rho, \phi, z)$: $x = \rho \cos\phi, y = \rho \sin\phi, z = z$.
pub struct Cylindrical;

impl DifferentiableCoordinates for Cylindrical {
    #[inline]
    fn scale_factors_scalar<S: Scalar>(&self, point: &[S; 3]) -> [S; 3] {
        [S::ONE, point[0], S::ONE]
    }
}

impl OrthogonalCoordinateSystem for Cylindrical {
    #[verified_engine::verified]
    fn local_to_cartesian(&self, point: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        let (rho, phi, z) = (point[0], point[1], point[2]);
        Vector3::new(rho * phi.cos(), rho * phi.sin(), z)
    }

    #[verified_engine::verified]
    fn cartesian_to_local(&self, cartesian: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(cartesian.iter().all(|x| x.is_finite()));
        let (x, y, z) = (cartesian[0], cartesian[1], cartesian[2]);
        Vector3::new((x * x + y * y).sqrt(), y.atan2(x), z)
    }

    #[verified_engine::verified]
    fn scale_factors(&self, point: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        Vector3::new(1.0, point[0], 1.0)
    }

    #[verified_engine::verified]
    fn scale_factor_partials(&self, point: &Vector3<f64>) -> [[f64; 3]; 3] {
        eval_scale_partials_ad(self, point)
    }
}

/// Spherical Coordinates $(r, \theta, \phi)$: $x = r \sin\theta \cos\phi, y = r \sin\theta \sin\phi, z = r \cos\theta$.
pub struct Spherical;

impl DifferentiableCoordinates for Spherical {
    #[inline]
    fn scale_factors_scalar<S: Scalar>(&self, point: &[S; 3]) -> [S; 3] {
        [S::ONE, point[0], point[0] * point[1].sin()]
    }
}

impl OrthogonalCoordinateSystem for Spherical {
    #[verified_engine::verified]
    fn local_to_cartesian(&self, point: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        let (r, theta, phi) = (point[0], point[1], point[2]);
        let sin_theta = theta.sin();
        Vector3::new(
            r * sin_theta * phi.cos(),
            r * sin_theta * phi.sin(),
            r * theta.cos(),
        )
    }

    #[verified_engine::verified]
    fn cartesian_to_local(&self, cartesian: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(cartesian.iter().all(|x| x.is_finite()));
        let (x, y, z) = (cartesian[0], cartesian[1], cartesian[2]);
        let r = (x * x + y * y + z * z).sqrt();
        if r < 1e-12 {
            return Vector3::zeros();
        }
        Vector3::new(r, (z / r).clamp(-1.0, 1.0).acos(), y.atan2(x))
    }

    #[verified_engine::verified]
    fn scale_factors(&self, point: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        Vector3::new(1.0, point[0], point[0] * point[1].sin())
    }

    #[verified_engine::verified]
    fn scale_factor_partials(&self, point: &Vector3<f64>) -> [[f64; 3]; 3] {
        eval_scale_partials_ad(self, point)
    }
}
