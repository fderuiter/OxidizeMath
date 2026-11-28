//! Surface Meshing and Noise Reduction.
//!
//! Fits a continuous **Bi-Quadratic Polynomial Surface** to the sparse radar point cloud to smooth out "sparkle" noise.
//!
//! # Model
//! $$ z = c_0 + c_1 x + c_2 y + c_3 xy + c_4 x^2 + c_5 y^2 $$

use engine_linalg::{SolverWorkspace, svd_ops_workspace_len, svd_solve_least_squares};
use math_memory::{ArenaLayout, OwnedByteArena};
use nalgebra::{DVector, Point3};

/// Computes the byte capacity required for zero-allocation [`BiQuadraticSurface::fit_with_workspace`].
#[must_use]
#[verified_engine::verified]
pub fn radar_surface_workspace_bytes(num_points: usize) -> usize {
    let mut layout = ArenaLayout::new();
    let _ = layout.reserve::<f64>(num_points.saturating_mul(6));
    let _ = layout.reserve::<f64>(num_points);
    let _ = layout.reserve::<f64>(svd_ops_workspace_len(num_points, 6));
    layout.total_bytes().saturating_add(64)
}

/// A fitted bi-quadratic surface.
#[derive(Debug, Clone)]
pub struct BiQuadraticSurface {
    /// Coefficients $[c_0, c_1, c_2, c_3, c_4, c_5]$.
    pub coefficients: DVector<f64>,
}

impl BiQuadraticSurface {
    /// Preallocates a zeroed 6-coefficient surface container for zero-allocation [`Self::fit_into_with_workspace`].
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            coefficients: DVector::zeros(6),
        }
    }

    /// Fits the 6 bi-quadratic polynomial coefficients using `workspace` with `0` heap allocations.
    #[verified_engine::verified]
    pub fn fit_with_workspace(
        points: &[Point3<f64>],
        workspace: &mut (impl SolverWorkspace + ?Sized),
    ) -> Option<[f64; 6]> {
        let n = points.len();
        if n < 6 {
            return None;
        }
        let mut cur = workspace.cursor();
        let a_data = cur.alloc_slice::<f64>(n.checked_mul(6)?).ok()?;
        let b_data = cur.alloc_slice::<f64>(n).ok()?;
        for (i, p) in points.iter().enumerate() {
            let row = &mut a_data[i * 6..(i + 1) * 6];
            row[0] = 1.0;
            row[1] = p.x;
            row[2] = p.y;
            row[3] = p.x * p.y;
            row[4] = p.x.powi(2);
            row[5] = p.y.powi(2);
            b_data[i] = p.z;
        }
        let epsilon = math_commons::registry::TOLERANCE_STANDARD;
        let mut coeffs = [0.0_f64; 6];
        let mut sub = cur.into_arena();
        svd_solve_least_squares(a_data, n, 6, b_data, epsilon, &mut coeffs, &mut sub).ok()?;
        Some(coeffs)
    }

    /// Fits `points` into `self.coefficients` in-place using `workspace` (`0` heap allocations).
    #[verified_engine::verified]
    pub fn fit_into_with_workspace(
        &mut self,
        points: &[Point3<f64>],
        workspace: &mut (impl SolverWorkspace + ?Sized),
    ) -> bool {
        if self.coefficients.len() != 6 {
            return false;
        }
        let Some(coeffs) = Self::fit_with_workspace(points, workspace) else {
            return false;
        };
        self.coefficients.as_mut_slice().copy_from_slice(&coeffs);
        true
    }

    /// Fits a bi-quadratic surface to a set of 3D points using Least Squares.
    #[verified_engine::verified]
    pub fn fit(points: &[Point3<f64>]) -> Option<Self> {
        let mut arena = OwnedByteArena::with_capacity(radar_surface_workspace_bytes(points.len()));
        let coeffs = Self::fit_with_workspace(points, &mut arena)?;
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
#[verified_engine::verified]
pub fn weighted_average_height(vertices: &[Point3<f64>]) -> f64 {
    pure_math::pure_math::differential_geometry::mean_vertex_height(vertices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_biquadratic_fit_with_workspace_zero_heap_allocations() {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (-1.0, 0.5),
            (0.5, -1.0),
            (2.0, -0.5),
        ];
        let pts: Vec<Point3<f64>> = coords
            .iter()
            .map(|&(x, y)| {
                let z = 1.2 + 0.3 * x - 0.4 * y + 0.2 * x * y + 0.5 * x * x - 0.1 * y * y;
                Point3::new(x, y, z)
            })
            .collect();
        let mut surf = BiQuadraticSurface::zeros();
        let mut arena = OwnedByteArena::with_capacity(radar_surface_workspace_bytes(pts.len()));
        let (ok, _) =
            VerifiedEngine::run_verified(|| surf.fit_into_with_workspace(&pts, &mut arena));
        assert!(ok);
        assert!(
            (surf.evaluate(0.5, -0.5) - (1.2 + 0.15 + 0.2 - 0.05 + 0.125 - 0.025)).abs() < 1e-9
        );
    }
}
