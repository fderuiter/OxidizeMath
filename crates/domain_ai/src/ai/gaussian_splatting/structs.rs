use nalgebra::{Matrix2, Matrix3, Point2, Point3, UnitQuaternion, Vector3};

/// Represents a 3D Gaussian in the scene.
///
/// Defined by its mean position, covariance (parameterized by scale and rotation),
/// opacity, and color (Spherical Harmonics).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gaussian3D {
    #[allow(missing_docs)]
    pub mean: Point3<f64>,
    #[allow(missing_docs)]
    pub scale: Vector3<f64>,
    #[allow(missing_docs)]
    pub rotation: UnitQuaternion<f64>,
    #[allow(missing_docs)]
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
        let r = self.rotation.to_rotation_matrix();
        let m = r * s;
        m * m.transpose()
    }
}

/// Represents the 2D projection of a Gaussian on the image plane.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gaussian2D {
    #[allow(missing_docs)]
    pub mean: Point2<f64>,
    /// The conic matrix (inverse covariance with -0.5 factor) for fast evaluation.
    pub conic: Matrix2<f64>,
    #[allow(missing_docs)]
    pub opacity: f64,
    #[allow(missing_docs)]
    pub color: Vector3<f64>,
    /// Depth is needed for sorting
    pub depth: f64,
}

/// A collection of 3D Gaussians.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Scene {
    #[allow(missing_docs)]
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
