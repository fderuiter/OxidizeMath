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
use engine_linalg::{ByteArena, SolverWorkspace, svd_ops_workspace_len, svd_solve_least_squares};
use math_memory::ArenaLayout;
use nalgebra::{DVector, Point3};

/// Returns the byte capacity required by [`BiQuadraticSurface::fit_with_workspace`] for `num_points` samples.
#[verified_engine::verified]
#[must_use]
pub fn biquadratic_fit_workspace_bytes(num_points: usize) -> usize {
    let n = num_points.max(6);
    let mut layout = ArenaLayout::new();
    let _ = layout.add_slice::<f64>(n.saturating_mul(6));
    let _ = layout.add_slice::<f64>(n);
    let _ = layout.add_slice::<f64>(svd_ops_workspace_len(n, 6));
    layout.total_bytes().saturating_add(64)
}

/// A fitted bi-quadratic surface.
#[derive(Debug, Clone)]
pub struct BiQuadraticSurface {
    /// Coefficients $[c_0, c_1, c_2, c_3, c_4, c_5]$.
    pub coefficients: DVector<f64>,
}

impl BiQuadraticSurface {
    /// Fits a bi-quadratic surface into `out_coeffs` (`[c_0, .., c_5]`) using caller-provided `ws`
    /// with zero heap allocations.
    #[verified_engine::verified]
    pub fn fit_with_workspace<W: SolverWorkspace>(
        points: &[Point3<f64>],
        out_coeffs: &mut [f64; 6],
        ws: &mut W,
    ) -> Option<()> {
        let n = points.len();
        if n < 6 {
            return None;
        }
        let mut scope = ws.reborrow();
        let a_data = scope.alloc_slice::<f64>(n.checked_mul(6)?).ok()?;
        let b_data = scope.alloc_slice::<f64>(n).ok()?;
        for (i, p) in points.iter().enumerate() {
            let base = i * 6;
            a_data[base] = 1.0;
            a_data[base + 1] = p.x;
            a_data[base + 2] = p.y;
            a_data[base + 3] = p.x * p.y;
            a_data[base + 4] = p.x.powi(2);
            a_data[base + 5] = p.y.powi(2);
            b_data[i] = p.z;
        }
        let epsilon = math_commons::registry::TOLERANCE_STANDARD;
        svd_solve_least_squares(a_data, n, 6, b_data, epsilon, out_coeffs, &mut scope).ok()?;
        Some(())
    }

    /// Updates `self.coefficients` in place from `points` using caller-provided `ws` with zero heap allocations.
    #[verified_engine::verified]
    pub fn fit_into_with_workspace<W: SolverWorkspace>(
        &mut self,
        points: &[Point3<f64>],
        ws: &mut W,
    ) -> Option<()> {
        if self.coefficients.len() != 6 {
            return None;
        }
        let mut coeffs = [0.0_f64; 6];
        Self::fit_with_workspace(points, &mut coeffs, ws)?;
        self.coefficients.copy_from_slice(&coeffs);
        Some(())
    }

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
            return None;
        }
        let mut coeffs = [0.0_f64; 6];
        let mut scratch = vec![0_u8; biquadratic_fit_workspace_bytes(n)];
        let mut arena = ByteArena::from_slice(&mut scratch);
        Self::fit_with_workspace(points, &mut coeffs, &mut arena)?;
        Some(Self {
            coefficients: DVector::from_column_slice(&coeffs),
        })
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
