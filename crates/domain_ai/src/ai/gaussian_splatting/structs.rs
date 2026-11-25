pub use math_scalars::{Quaternion, UnitQuaternion};
use nalgebra::{Matrix2, Matrix3, Point2, Point3, Vector3};

/// Represents a 3D Gaussian in the scene.
///
/// Defined by its mean position, covariance (parameterized by scale and rotation),
/// opacity, and color (Spherical Harmonics).
#[derive(Clone, Debug)]
pub struct Gaussian3D {
    /// World-space 3D center position $\boldsymbol{\mu} \in \mathbb{R}^3$.
    pub mean: Point3<f64>,
    /// Principal-axis standard deviations $\mathbf{s} = (s_x, s_y, s_z)$.
    pub scale: Vector3<f64>,
    /// Unit quaternion encoding orientation matrix $\mathbf{R} \in \mathrm{SO}(3)$.
    pub rotation: UnitQuaternion<f64>,
    /// Scalar opacity $\alpha \in [0, 1]$ before kernel attenuation.
    pub opacity: f64,
    /// Simplified color representation (e.g., DC component of SH or RGB)
    pub color: Vector3<f64>,
}

impl Gaussian3D {
    /// Computes the 3D covariance matrix from scale and rotation.
    ///
    /// sigma = R * S * S^T * R^T
    #[verified_engine::verified]
    pub fn compute_covariance(&self) -> Matrix3<f64> {
        let s = Matrix3::from_diagonal(&self.scale);
        let rm = self.rotation.to_rotation_matrix3();
        let r = Matrix3::new(
            rm[0][0], rm[0][1], rm[0][2], rm[1][0], rm[1][1], rm[1][2], rm[2][0], rm[2][1],
            rm[2][2],
        );
        let m = r * s;
        m * m.transpose()
    }

    /// Converts this 3D Gaussian (with an optional positive uniform `scale_multiplier`) into
    /// a portable [`oxidize_core::mesh::GaussianSplatPly`] record for INRIA 3DGS `.ply` serialization.
    #[verified_engine::verified]
    pub fn to_splat_ply(&self, scale_multiplier: f64) -> oxidize_core::mesh::GaussianSplatPly {
        let mult = if scale_multiplier.is_finite() && scale_multiplier > 0.0 {
            scale_multiplier
        } else {
            1.0
        };
        let [w, x, y, z] = self.rotation.as_wxyz();
        let unit_f32 =
            UnitQuaternion::<f32>::from_wxyz_or_identity([w as f32, x as f32, y as f32, z as f32]);
        oxidize_core::mesh::GaussianSplatPly {
            mean: oxidize_core::mesh::Point3D::new(
                self.mean.x as f32,
                self.mean.y as f32,
                self.mean.z as f32,
            ),
            scale: oxidize_core::mesh::Point3D::new(
                (self.scale.x * mult) as f32,
                (self.scale.y * mult) as f32,
                (self.scale.z * mult) as f32,
            ),
            rotation: unit_f32.as_wxyz(),
            opacity: self.opacity as f32,
            color: [
                self.color.x as f32,
                self.color.y as f32,
                self.color.z as f32,
            ],
        }
    }
}

/// Represents the 2D projection of a Gaussian on the image plane.
#[derive(Clone, Debug)]
pub struct Gaussian2D {
    /// Projected screen-space 2D center coordinates.
    pub mean: Point2<f64>,
    /// The conic matrix (inverse covariance with -0.5 factor) for fast evaluation.
    pub conic: Matrix2<f64>,
    /// Scalar opacity $\alpha \in [0, 1]$ used in alpha compositing.
    pub opacity: f64,
    /// Linear RGB radiance contribution of the splat.
    pub color: Vector3<f64>,
    /// Depth is needed for sorting
    pub depth: f64,
}

/// A collection of 3D Gaussians.
pub struct Scene {
    /// Ordered list of 3D Gaussian primitives comprising the scene.
    pub gaussians: Vec<Gaussian3D>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_covariance_computation() {
        let scale = Vector3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::identity();
        let gaussian = Gaussian3D {
            mean: Point3::origin(),
            scale,
            rotation,
            opacity: 1.0,
            color: Vector3::zeros(),
        };

        let cov = gaussian.compute_covariance();
        assert_eq!(cov[(0, 0)], 1.0);
        assert_eq!(cov[(1, 1)], 4.0);
        assert_eq!(cov[(2, 2)], 9.0);
        assert_eq!(cov[(0, 1)], 0.0);
    }
}
