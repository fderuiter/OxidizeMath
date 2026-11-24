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
use engine_linalg::{SolverWorkspace, svd_ops_workspace_len, svd_solve_least_squares};
use math_commons::error::MathError;
use nalgebra::{DVector, Point3};

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
    /// * `Some(BiQuadraticSurface)` if the fit is successful and all inputs are finite.
    /// * `None` if the system is under-determined, contains non-finite values, or overflows `usize`.
    #[verified_engine::verified(
        alloc_ok = "allocates least-squares design matrix and fitted coefficient vector"
    )]
    pub fn fit(points: &[Point3<f64>]) -> Option<Self> {
        let n = points.len();
        if n < 6 {
            return None;
        }
        if !points
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        {
            return None;
        }
        let a_cap = n.checked_mul(6)?;
        let mut a_data = Vec::with_capacity(a_cap);
        let mut b_data = Vec::with_capacity(n);

        for p in points {
            let xy = p.x * p.y;
            let x2 = p.x.powi(2);
            let y2 = p.y.powi(2);
            if !xy.is_finite() || !x2.is_finite() || !y2.is_finite() {
                return None;
            }
            a_data.push(1.0);
            a_data.push(p.x);
            a_data.push(p.y);
            a_data.push(xy);
            a_data.push(x2);
            a_data.push(y2);
            b_data.push(p.z);
        }

        let epsilon = math_commons::registry::TOLERANCE_STANDARD;
        let mut coeffs = [0.0_f64; 6];
        let mut scratch = vec![0.0_f64; svd_ops_workspace_len(n, 6)];
        svd_solve_least_squares(
            &a_data,
            n,
            6,
            &b_data,
            epsilon,
            &mut coeffs,
            &mut SolverWorkspace::new(&mut scratch),
        )
        .ok()?;
        if !coeffs.iter().all(|c| c.is_finite()) {
            return None;
        }
        Some(Self {
            coefficients: DVector::from_column_slice(&coeffs),
        })
    }

    /// Evaluates the surface height $z$ at a given $(x, y)$.
    #[verified_engine::verified]
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        if self.coefficients.len() < 6 || !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        let c = &self.coefficients;
        c[0] + c[1] * x + c[2] * y + c[3] * x * y + c[4] * x.powi(2) + c[5] * y.powi(2)
    }

    /// Checked surface mesh generator rejecting zero resolution, non-finite/inverted bounds, and
    /// `resolution * resolution` overflow while handling `resolution == 1` at the domain midpoint
    /// without dividing by zero.
    #[verified_engine::verified(alloc_ok = "allocates sampled 3D surface mesh vector")]
    pub fn try_generate_mesh(
        &self,
        x_range: (f64, f64),
        y_range: (f64, f64),
        resolution: usize,
    ) -> Result<Vec<Point3<f64>>, MathError> {
        if resolution == 0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if !x_range.0.is_finite()
            || !x_range.1.is_finite()
            || x_range.0 > x_range.1
            || !y_range.0.is_finite()
            || !y_range.1.is_finite()
            || y_range.0 > y_range.1
        {
            return Err(MathError::NumericalError {
                reason: "Surface mesh coordinate ranges must be finite and ordered".to_string(),
            });
        }
        let total = resolution
            .checked_mul(resolution)
            .ok_or(MathError::CapacityOverflow {
                context: "BiQuadraticSurface::try_generate_mesh resolution * resolution",
            })?;
        if resolution == 1 {
            let x_mid = x_range.0 + 0.5 * (x_range.1 - x_range.0);
            let y_mid = y_range.0 + 0.5 * (y_range.1 - y_range.0);
            let z = self.evaluate(x_mid, y_mid);
            return Ok(vec![Point3::new(x_mid, y_mid, z)]);
        }

        let mut mesh = Vec::with_capacity(total);
        let denom = (resolution - 1) as f64;
        let dx = (x_range.1 - x_range.0) / denom;
        let dy = (y_range.1 - y_range.0) / denom;

        for i in 0..resolution {
            let x = x_range.0 + i as f64 * dx;
            for j in 0..resolution {
                let y = y_range.0 + j as f64 * dy;
                let z = self.evaluate(x, y);
                mesh.push(Point3::new(x, y, z));
            }
        }
        Ok(mesh)
    }

    /// Generates a dense grid of points from the fitted surface.
    ///
    /// Useful for visualization. Returns an empty vector when `resolution == 0` or ranges are
    /// invalid, and evaluates the domain midpoint when `resolution == 1` to avoid division by zero.
    #[verified_engine::verified(alloc_ok = "allocates sampled 3D surface mesh vector")]
    pub fn generate_mesh(
        &self,
        x_range: (f64, f64),
        y_range: (f64, f64),
        resolution: usize,
    ) -> Vec<Point3<f64>> {
        if resolution == 0 {
            return Vec::new();
        }
        self.try_generate_mesh(x_range, y_range, resolution)
            .unwrap_or_default()
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
