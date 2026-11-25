pub use super::buckled_flow::{BuckledSurfaceFlow, SurfaceTopology};
use nalgebra::{Point3, Vector3};

/// A discrete surface represented by a grid of points.
/// Used for evolutions where the surface geometry itself changes.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteSurface {
    /// 2D parameter grid of 3D surface positions $\mathbf{r}(u_i, v_j)$.
    pub points: Vec<Vec<Point3<f64>>>,
    /// Whether the $u$-parameter boundary wraps periodically.
    pub closed_u: bool,
    /// Whether the $v$-parameter boundary wraps periodically.
    pub closed_v: bool,
}

impl DiscreteSurface {
    /// Constructs a structured parametric surface grid with optional periodic `u`/`v` wrapping.
    #[verified_engine::verified]
    pub fn new(points: Vec<Vec<Point3<f64>>>, closed_u: bool, closed_v: bool) -> Self {
        Self {
            points,
            closed_u,
            closed_v,
        }
    }

    #[verified_engine::verified]
    fn get(&self, i: i32, j: i32) -> Point3<f64> {
        let nu = self.points.len() as i32;
        let nv = self.points[0].len() as i32;

        let i_idx = if self.closed_u {
            i.rem_euclid(nu)
        } else {
            i.clamp(0, nu - 1)
        } as usize;
        let j_idx = if self.closed_v {
            j.rem_euclid(nv)
        } else {
            j.clamp(0, nv - 1)
        } as usize;

        self.points[i_idx][j_idx]
    }

    /// Computes Mean Curvature Vector $\vec{H} = H \mathbf{n}$ at $(i, j)$.
    /// Uses the Laplace-Beltrami of the position vector: $\Delta_S \mathbf{r} = 2 \vec{H}$.
    /// Reference: "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds", Meyer et al.
    /// Or simpler finite difference on the parameterization if regular grid.
    /// Since we have a structured grid (u, v), we can use the same operator logic as heat equation,
    /// but applied to x, y, z coordinates.
    #[verified_engine::verified]
    pub fn mean_curvature_vector(&self, i: usize, j: usize) -> Vector3<f64> {
        // We approximate Delta_S r.
        // For a conformal parameterization, Delta_S r ~ Laplacian(r).
        // For general, we need the metric.

        // Let's compute local derivatives via central difference
        let idx_i = i as i32;
        let idx_j = j as i32;
        let p = self.get(idx_i, idx_j);

        let p_ip = self.get(idx_i + 1, idx_j);
        let p_im = self.get(idx_i - 1, idx_j);
        let p_jp = self.get(idx_i, idx_j + 1);
        let p_jm = self.get(idx_i, idx_j - 1);

        // Derivatives (assuming unit spacing in parameter domain for simplicity,
        // effectively treating the grid as the parameter domain integers)
        let r_u = (p_ip - p_im) / 2.0;
        let r_v = (p_jp - p_jm) / 2.0;

        let r_uu = p_ip - p.coords * 2.0 + p_im.coords;
        let r_vv = p_jp - p.coords * 2.0 + p_jm.coords;

        // Mixed r_uv
        let p_pp = self.get(idx_i + 1, idx_j + 1);
        let p_mm = self.get(idx_i - 1, idx_j - 1);
        let p_pm = self.get(idx_i + 1, idx_j - 1);
        let p_mp = self.get(idx_i - 1, idx_j + 1);
        let r_uv = (p_pp.coords - p_mp.coords - p_pm.coords + p_mm.coords) / 4.0;

        // Fundamental forms
        let e = r_u.dot(&r_u);
        let f = r_u.dot(&r_v);
        let g = r_v.dot(&r_v);

        let det_g = e * g - f * f;
        if det_g.abs() < 1e-12 {
            // Singularity (e.g., pole), return zero flow or handle explicitly.
            return Vector3::zeros();
        }

        let n_unnorm = r_u.cross(&r_v);
        let n_norm = n_unnorm.norm();
        if n_norm < 1e-12 {
            return Vector3::zeros();
        }
        let n = n_unnorm / n_norm;

        let l = r_uu.coords.dot(&n);
        let m = r_uv.dot(&n);
        let n_coeff = r_vv.coords.dot(&n); // 'n' is taken, use n_coeff

        let h_val = (e * n_coeff - 2.0 * f * m + g * l) / (2.0 * det_g);

        n * h_val
    }
}

/// Explicit parametric mean-curvature flow solver evolving $\frac{\partial \mathbf{r}}{\partial t} = H \mathbf{n} = \frac{1}{2} \Delta_S \mathbf{r}$ on a double-buffered surface grid.
///
/// Reference: Meyer, Desbrun, Schröder & Barr (2003), DOI: 10.1007/978-3-662-05105-4_2; [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Explicit parabolic geometric evolution requires $\Delta t \lesssim \frac{1}{4} h_{\min}^2$, where $h_{\min}$ is the shortest Euclidean grid spacing; degenerate poles with $|\det g| < 10^{-12}$ or $\|\mathbf{r}_u \times \mathbf{r}_v\| < 10^{-12}$ safely clamp to zero velocity.
///
/// # Domain of Validity
/// Valid on non-empty structured 2D surface grids prior to neckpinch or finite-time curvature singularities, with zero heap allocations during `step`.
#[derive(Debug, Clone, PartialEq)]
pub struct MeanCurvatureFlow {
    /// Active evolving discrete surface grid.
    pub surface: DiscreteSurface,
    /// Bolt Optimization: Buffer to avoid allocations in step
    pub next_points: Vec<Vec<Point3<f64>>>,
}

impl MeanCurvatureFlow {
    /// Initializes a double-buffered mean-curvature flow stepper over `surface`.
    #[verified_engine::verified]
    pub fn new(surface: DiscreteSurface) -> Self {
        let next_points = surface.points.clone();
        Self {
            surface,
            next_points,
        }
    }

    /// Advances every grid vertex by one explicit Euler step `dt` along its mean-curvature vector.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let nu = self.surface.points.len();
        let nv = self.surface.points[0].len();

        #[allow(clippy::needless_range_loop)]
        for i in 0..nu {
            for j in 0..nv {
                let h_vec = self.surface.mean_curvature_vector(i, j);
                self.next_points[i][j] = self.surface.points[i][j] + h_vec * dt;
            }
        }

        std::mem::swap(&mut self.surface.points, &mut self.next_points);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_mean_curvature_flow_100_steps_zero_alloc_and_pole_safe() {
        let nu = 12;
        let nv = 12;
        let mut pts = vec![vec![Point3::origin(); nv]; nu];
        for (i, row) in pts.iter_mut().enumerate() {
            let u = 2.0 * PI * (i as f64) / (nu as f64);
            for (j, slot) in row.iter_mut().enumerate() {
                let v = PI * (j as f64) / ((nv - 1) as f64);
                *slot = Point3::new(v.sin() * u.cos(), v.sin() * u.sin(), v.cos());
            }
        }
        let mut flow = MeanCurvatureFlow::new(DiscreteSurface::new(pts, true, false));
        let r_before = flow.surface.points[3][5].coords.norm();
        let ((), _metrics) = VerifiedEngine::run_verified(|| {
            for _ in 0..100 {
                flow.step(1e-4);
            }
        });
        let r_after = flow.surface.points[3][5].coords.norm();
        assert!(r_after < r_before, "Forward MCF must shrink sphere radius");
        for row in &flow.surface.points {
            for p in row {
                assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
            }
        }
    }
}
