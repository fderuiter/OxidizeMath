//! Surface Meshing and Noise Reduction.
//!
//! Fits a continuous **Bi-Quadratic Polynomial Surface** to the sparse radar point cloud to smooth out "sparkle" noise.
//!
//! # Model
//! $$ z = c_0 + c_1 x + c_2 y + c_3 xy + c_4 x^2 + c_5 y^2 $$
//!
//! Note: The coordinate system here assumes $z$ is height/chest amplitude, and $x, y$ are lateral positions.
//! In the geometry module, we output $(x, y, z)$ where $y$ is depth and $z$ is vertical.
//! Consumers of this module should map their axes accordingly (e.g., using $y$ from geometry as "lateral" if needed,
//! or if the patient is lying down, $z$ (vertical) is the height).

use crate::physics::medical::signal;
use nalgebra::{DMatrix, DVector, Point3};

/// A fitted bi-quadratic surface.
#[derive(Debug, Clone)]
pub struct BiQuadraticSurface {
    /// Coefficients $[c_0, c_1, c_2, c_3, c_4, c_5]$.
    pub coefficients: DVector<f64>,
}

impl BiQuadraticSurface {
    /// Fits a bi-quadratic surface to a set of 3D points using Least Squares.
    ///
    /// Solve for $c$ in $A c = b$.
    ///
    /// # Arguments
    /// * `points` - A slice of Cartesian points $(x, y, z)$.
    ///
    /// # Returns
    /// * `Some(BiQuadraticSurface)` if the fit is successful.
    /// * `None` if the system is under-determined (too few points).
    #[verified_engine::verified]
    pub fn fit(points: &[Point3<f64>]) -> Option<Self> {
        let n = points.len();
        if n < 6 {
            return None; // Need at least 6 points for 6 coefficients
        }

        // Construct Design Matrix A (n x 6) and Vector b (n x 1)
        // Rows of A: [1, x, y, xy, x^2, y^2]
        // Rows of b: [z]

        let mut a_data = Vec::with_capacity(n * 6);
        let mut b_data = Vec::with_capacity(n);

        for p in points {
            a_data.push(1.0);
            a_data.push(p.x);
            a_data.push(p.y);
            a_data.push(p.x * p.y);
            a_data.push(p.x.powi(2));
            a_data.push(p.y.powi(2));

            b_data.push(p.z);
        }

        let a = DMatrix::from_row_slice(n, 6, &a_data);
        let b = DVector::from_column_slice(&b_data);

        // Solve using SVD for numerical stability on potentially ill-conditioned matrices
        // (e.g. if points are collinear or not spread out).
        // A * x = b  =>  x = A_pseudo_inv * b
        // nalgebra's SVD solve handles this.
        let epsilon = math_commons::registry::TOLERANCE_STANDARD;
        match a.svd(true, true).solve(&b, epsilon) {
            Ok(coefficients) => Some(Self { coefficients }),
            Err(_) => None,
        }
    }

    /// Evaluates the surface height $z$ at a given $(x, y)$.
    #[verified_engine::verified]
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        let c = &self.coefficients;
        c[0] + c[1] * x + c[2] * y + c[3] * x * y + c[4] * x.powi(2) + c[5] * y.powi(2)
    }

    /// Generates a dense grid of points from the fitted surface.
    ///
    /// Useful for visualization.
    #[verified_engine::verified]
    pub fn generate_mesh(
        &self,
        x_range: (f64, f64),
        y_range: (f64, f64),
        resolution: usize,
    ) -> Vec<Point3<f64>> {
        let mut mesh = Vec::with_capacity(resolution * resolution);
        let dx = (x_range.1 - x_range.0) / (resolution as f64 - 1.0);
        let dy = (y_range.1 - y_range.0) / (resolution as f64 - 1.0);

        for i in 0..resolution {
            for j in 0..resolution {
                let x = x_range.0 + i as f64 * dx;
                let y = y_range.0 + j as f64 * dy;
                let z = self.evaluate(x, y);
                mesh.push(Point3::new(x, y, z));
            }
        }
        mesh
    }
}

/// Calculates the Weighted Average Height (WAH) for a radar mesh.
///
/// $$ V_{resp}(t) = \frac{1}{\text{Area}} \iint_{ROI} Z_{mesh}(x,y,t) dx dy $$
///
/// Discretized as the mean of Z-coordinates for vertices within the ROI.
///
/// # Arguments
///
/// * `vertices` - List of vertices in the ROI.
///
/// # Returns
///
/// * `f64` - The WAH value.
#[verified_engine::verified]
pub fn weighted_average_height(vertices: &[Point3<f64>]) -> f64 {
    signal::mean_vertex_height(vertices)
}
