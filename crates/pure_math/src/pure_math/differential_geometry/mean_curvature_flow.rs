use nalgebra::{Point3, Vector3};

/// A discrete surface represented by a grid of points.
/// Used for evolutions where the surface geometry itself changes.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiscreteSurface {
    #[allow(missing_docs)]
    pub points: Vec<Vec<Point3<f64>>>,
    #[allow(missing_docs)]
    pub closed_u: bool,
    #[allow(missing_docs)]
    pub closed_v: bool,
}

impl DiscreteSurface {
    #[allow(missing_docs)]
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

        let n_unnorm = r_u.cross(&r_v);
        let n = n_unnorm.normalize();

        let l = r_uu.coords.dot(&n);
        let m = r_uv.dot(&n);
        let n_coeff = r_vv.coords.dot(&n); // 'n' is taken, use n_coeff

        // Mean curvature H
        let det_g = e * g - f * f;
        if det_g.abs() < 1e-12 {
            // Singularity (e.g., pole), return zero flow or handle explicitly.
            return Vector3::zeros();
        }
        let h_val = (e * n_coeff - 2.0 * f * m + g * l) / (2.0 * det_g);

        n * h_val
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct MeanCurvatureFlow {
    #[allow(missing_docs)]
    pub surface: DiscreteSurface,
    /// Bolt Optimization: Buffer to avoid allocations in step
    pub next_points: Vec<Vec<Point3<f64>>>,
}

impl MeanCurvatureFlow {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(surface: DiscreteSurface) -> Self {
        let next_points = surface.points.clone();
        Self {
            surface,
            next_points,
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let nu = self.surface.points.len();
        let nv = self.surface.points[0].len();

        #[allow(clippy::needless_range_loop)]
        for i in 0..nu {
            for j in 0..nv {
                // Velocity = Mean Curvature Vector
                // Flow equation: dr/dt = \vec{H} (or 2\vec{H}, conventions vary)
                // Note: Standard MCF is dr/dt = -Hn if H is defined via divergence of normal?
                // Usually dr/dt = \Delta_S r.
                // Our calc `mean_curvature_vector` returns H*n.
                // If we want to shrink a sphere, H is positive (1/R) and n is outward.
                // We want to move inward. So velocity should be -H*n.
                // Wait, if n is outward, H is 1/R. n = (x,y,z)/R. Hn = (x,y,z)/R^2.
                // We want -dr/dt for minimization?
                // Actually, Mean Curvature Vector H_vec is usually defined as pointing in direction of decreasing area (inward for sphere).
                // Let's check our H sign.
                // Sphere: r_u x r_v points OUTWARD (standard param).
                // H = 1/R > 0.
                // H*n points OUTWARD.
                // To shrink, we need -H*n.

                let h_vec = self.surface.mean_curvature_vector(i, j);

                // Update
                // Flow by mean curvature vector: dr/dt = H_vec
                // H_vec points in the direction of steepest area descent (curvature vector).
                self.next_points[i][j] = self.surface.points[i][j] + h_vec * dt;
            }
        }

        std::mem::swap(&mut self.surface.points, &mut self.next_points);
    }
}
