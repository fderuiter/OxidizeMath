// [cite:quasiperfect_alcf]
use engine_symbolic::{Dual, Jet2};
use nalgebra::Vector3;

/// Trait for Orthogonal Coordinate Systems.
/// Defines how to convert to/from Cartesian and the curvilinear scale factors $(h_1, h_2, h_3)$.
pub trait OrthogonalCoordinateSystem {
    /// Converts a point from this coordinate system to Cartesian $(x, y, z)$.
    #[verified_engine::verified]
    fn local_to_cartesian(&self, point: &Vector3<f64>) -> Vector3<f64>;

    /// Converts a point from Cartesian $(x, y, z)$ to this coordinate system.
    #[verified_engine::verified]
    fn cartesian_to_local(&self, cartesian: &Vector3<f64>) -> Vector3<f64>;

    /// Returns the scale factors $(h_1, h_2, h_3)$ at the given point (in this coordinate system).
    #[verified_engine::verified]
    fn scale_factors(&self, point: &Vector3<f64>) -> Vector3<f64>;

    /// Evaluates $(h_1, h_2, h_3)$ over forward-mode [`Dual<f64>`] coordinates.
    #[verified_engine::verified]
    fn scale_factors_dual(&self, point: &[Dual<f64>; 3]) -> [Dual<f64>; 3] {
        let p0 = Vector3::new(point[0].re, point[1].re, point[2].re);
        let base = self.scale_factors(&p0);
        let eps = 1e-6;
        let mut dh = [[0.0_f64; 3]; 3];
        for axis in 0..3 {
            let mut pp = p0;
            let mut pm = p0;
            pp[axis] += eps;
            pm[axis] -= eps;
            let diff = (self.scale_factors(&pp) - self.scale_factors(&pm)) / (2.0 * eps);
            dh[0][axis] = diff[0];
            dh[1][axis] = diff[1];
            dh[2][axis] = diff[2];
        }
        [
            Dual::new(
                base[0],
                dh[0][0] * point[0].eps + dh[0][1] * point[1].eps + dh[0][2] * point[2].eps,
            ),
            Dual::new(
                base[1],
                dh[1][0] * point[0].eps + dh[1][1] * point[1].eps + dh[1][2] * point[2].eps,
            ),
            Dual::new(
                base[2],
                dh[2][0] * point[0].eps + dh[2][1] * point[1].eps + dh[2][2] * point[2].eps,
            ),
        ]
    }

    /// Evaluates $(h_1, h_2, h_3)$ over second-order [`Jet2<3>`] coordinates.
    #[verified_engine::verified]
    fn scale_factors_jet2(&self, point: &[Jet2<3>; 3]) -> [Jet2<3>; 3] {
        let p0 = Vector3::new(point[0].val, point[1].val, point[2].val);
        let mut out = [Jet2::ZERO; 3];
        for axis in 0..3 {
            let mut dpt = [
                Dual::constant(p0[0]),
                Dual::constant(p0[1]),
                Dual::constant(p0[2]),
            ];
            dpt[axis] = Dual::variable(p0[axis]);
            let hd = self.scale_factors_dual(&dpt);
            for c in 0..3 {
                out[c].val = hd[c].re;
                out[c].grad[axis] = hd[c].eps;
            }
        }
        out
    }
}

/// Cartesian Coordinates $(x, y, z)$. Scale factors are all `1`.
pub struct Cartesian;

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
    fn scale_factors_dual(&self, _point: &[Dual<f64>; 3]) -> [Dual<f64>; 3] {
        [Dual::ONE, Dual::ONE, Dual::ONE]
    }

    #[verified_engine::verified]
    fn scale_factors_jet2(&self, _point: &[Jet2<3>; 3]) -> [Jet2<3>; 3] {
        [Jet2::ONE, Jet2::ONE, Jet2::ONE]
    }
}

/// Cylindrical Coordinates $(\rho, \phi, z)$: $x = \rho \cos\phi, y = \rho \sin\phi, z = z$.
pub struct Cylindrical;

impl OrthogonalCoordinateSystem for Cylindrical {
    #[verified_engine::verified]
    fn local_to_cartesian(&self, point: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        Vector3::new(
            point[0] * point[1].cos(),
            point[0] * point[1].sin(),
            point[2],
        )
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
    fn scale_factors_dual(&self, point: &[Dual<f64>; 3]) -> [Dual<f64>; 3] {
        [Dual::ONE, point[0], Dual::ONE]
    }

    #[verified_engine::verified]
    fn scale_factors_jet2(&self, point: &[Jet2<3>; 3]) -> [Jet2<3>; 3] {
        [Jet2::ONE, point[0], Jet2::ONE]
    }
}

/// Spherical Coordinates $(r, \theta, \phi)$: $x = r \sin\theta \cos\phi, y = r \sin\theta \sin\phi, z = r \cos\theta$.
pub struct Spherical;

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
        let theta = (z / r).clamp(-1.0, 1.0).acos();
        Vector3::new(r, theta, y.atan2(x))
    }

    #[verified_engine::verified]
    fn scale_factors(&self, point: &Vector3<f64>) -> Vector3<f64> {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        Vector3::new(1.0, point[0], point[0] * point[1].sin())
    }

    #[verified_engine::verified]
    fn scale_factors_dual(&self, point: &[Dual<f64>; 3]) -> [Dual<f64>; 3] {
        [Dual::ONE, point[0], point[0] * point[1].sin()]
    }

    #[verified_engine::verified]
    fn scale_factors_jet2(&self, point: &[Jet2<3>; 3]) -> [Jet2<3>; 3] {
        [Jet2::ONE, point[0], point[0] * point[1].sin()]
    }
}
