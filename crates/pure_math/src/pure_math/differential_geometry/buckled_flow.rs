// [cite:modular_polynomials_review]
//! Flat-buffer manifold topology descriptors and morphogen-buckled mean-curvature flow.

#[path = "buckled_flow_geometry.rs"]
mod geometry;

use super::surface::ParametricSurface;
use geometry::{StencilNeighbors, build_sampled_geometry};
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::{Point3, Vector3};

/// Intrinsic boundary topology of a 2D parameterized surface grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceTopology {
    /// Doubly periodic manifold $(u, v) \in [u_{\min}, u_{\max}) \times [v_{\min}, v_{\max})$ (e.g. `Torus`).
    PeriodicTorus,
    /// Non-orientable manifold with twisted $u$-seam bijection and periodic $v$ (e.g. `KleinBottle`).
    TwistedKlein,
    /// Spherical manifold with periodic azimuth $u$ and polar cap singularities at $v_{\min}, v_{\max}$.
    PolarSphere,
    /// Open surface patch with homogeneous zero-flux Neumann boundaries (e.g. `CompiledHeightfieldSurface`).
    OpenNeumann,
}

impl SurfaceTopology {
    /// Computes grid step sizes `(du, dv)` for `(nu, nv)` samples on `(range_u, range_v)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn grid_spacings(
        self,
        range_u: (f64, f64),
        range_v: (f64, f64),
        nu: usize,
        nv: usize,
    ) -> (f64, f64) {
        let u_span = range_u.1 - range_u.0;
        let v_span = range_v.1 - range_v.0;
        match self {
            Self::PeriodicTorus | Self::TwistedKlein => {
                (u_span / nu.max(1) as f64, v_span / nv.max(1) as f64)
            }
            Self::PolarSphere => (
                u_span / nu.max(1) as f64,
                v_span / nv.saturating_sub(1).max(1) as f64,
            ),
            Self::OpenNeumann => (
                u_span / nu.saturating_sub(1).max(1) as f64,
                v_span / nv.saturating_sub(1).max(1) as f64,
            ),
        }
    }

    /// Evaluates parameter coordinates `(u, v)` at grid cell `(i, j)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn uv_at(
        self,
        range_u: (f64, f64),
        range_v: (f64, f64),
        nu: usize,
        nv: usize,
        i: usize,
        j: usize,
    ) -> (f64, f64) {
        let (du, dv) = self.grid_spacings(range_u, range_v, nu, nv);
        (range_u.0 + i as f64 * du, range_v.0 + j as f64 * dv)
    }
}

/// Detects the $v$-index bijection across $u_{\min} \leftrightarrow u_{\max}$ on a twisted seam.
#[must_use]
#[verified_engine::verified(alloc_ok = "allocates twisted seam index bijection vectors")]
pub fn detect_twisted_seam_map<S: ParametricSurface + ?Sized>(
    surface: &S,
    range_u: (f64, f64),
    range_v: (f64, f64),
    nv: usize,
) -> (Vec<usize>, Vec<usize>) {
    let dv = (range_v.1 - range_v.0) / nv.max(1) as f64;
    let mut fwd = vec![0_usize; nv];
    let mut rev = vec![0_usize; nv];
    let start_pts: Vec<Point3<f64>> = (0..nv)
        .map(|j| surface.position(range_u.0, range_v.0 + j as f64 * dv))
        .collect();
    let end_pts: Vec<Point3<f64>> = (0..nv)
        .map(|k| surface.position(range_u.1, range_v.0 + k as f64 * dv))
        .collect();

    for j in 0..nv {
        let mut best_k = (nv - j) % nv;
        let mut best_d2 = f64::INFINITY;
        for (k, end_pt) in end_pts.iter().enumerate() {
            let d2 = (start_pts[j] - end_pt).norm_squared();
            if d2 < best_d2 {
                best_d2 = d2;
                best_k = k;
            }
        }
        fwd[j] = if best_d2 < 1e-3 {
            best_k
        } else {
            (nv - j) % nv
        };
    }
    for (j, &k) in fwd.iter().enumerate() {
        rev[k] = j;
    }
    (fwd, rev)
}

/// Pre-allocated, zero-allocation morphogen-buckled mean-curvature flow solver on a flat 2D grid.
#[derive(Clone)]
pub struct BuckledSurfaceFlow {
    nu: usize,
    nv: usize,
    topology: SurfaceTopology,
    du: f64,
    dv: f64,
    min_area_element: f64,
    north_meridional_g: f64,
    south_meridional_g: f64,
    seam_fwd: Vec<usize>,
    neighbors: Vec<StencilNeighbors>,
    ref_points: Vec<Point3<f64>>,
    buckling_normals: Vec<Vector3<f64>>,
    points: Vec<Point3<f64>>,
    next_points: Vec<Point3<f64>>,
    mean_curvature: Vec<f64>,
    displacements: Vec<f64>,
}

impl BuckledSurfaceFlow {
    /// Fallible constructor validating non-zero grid dimensions, finite ranges, and `nu * nv` capacity.
    #[verified_engine::verified(alloc_ok = "pre-allocates flat 2D surface flow workspace")]
    pub fn try_from_parametric<S: ParametricSurface + ?Sized>(
        surface: &S,
        topology: SurfaceTopology,
        range_u: (f64, f64),
        range_v: (f64, f64),
        nu: usize,
        nv: usize,
    ) -> Result<Self, math_commons::error::MathError> {
        if nu < 4 || nv < 4 {
            return Err(math_commons::error::MathError::OutOfBounds {
                value: nu.min(nv) as f64,
                min: 4.0,
                max: usize::MAX as f64,
            });
        }
        if !range_u.0.is_finite()
            || !range_u.1.is_finite()
            || range_u.0 >= range_u.1
            || !range_v.0.is_finite()
            || !range_v.1.is_finite()
            || range_v.0 >= range_v.1
        {
            return Err(math_commons::error::MathError::NumericalError {
                reason: "BuckledSurfaceFlow parameter ranges must be finite and strictly ordered"
                    .to_string(),
            });
        }
        let len = nu
            .checked_mul(nv)
            .ok_or(math_commons::error::MathError::CapacityOverflow {
                context: "BuckledSurfaceFlow::try_from_parametric nu * nv",
            })?;
        let (du, dv) = topology.grid_spacings(range_u, range_v, nu, nv);
        let sampled =
            geometry::try_build_sampled_geometry(surface, topology, range_u, range_v, nu, nv)?;
        Ok(Self {
            nu,
            nv,
            topology,
            du,
            dv,
            min_area_element: sampled.min_area_element,
            north_meridional_g: sampled.north_meridional_g,
            south_meridional_g: sampled.south_meridional_g,
            seam_fwd: sampled.seam_fwd,
            neighbors: sampled.neighbors,
            points: sampled.ref_points.clone(),
            next_points: sampled.ref_points.clone(),
            ref_points: sampled.ref_points,
            buckling_normals: sampled.buckling_normals,
            mean_curvature: vec![0.0; len],
            displacements: vec![0.0; len],
        })
    }

    /// Samples `surface` onto a pre-allocated flat `nu * nv` grid with topology-aware normals and stencils.
    #[must_use]
    #[verified_engine::verified(alloc_ok = "pre-allocates flat 2D surface flow workspace")]
    pub fn from_parametric<S: ParametricSurface + ?Sized>(
        surface: &S,
        topology: SurfaceTopology,
        range_u: (f64, f64),
        range_v: (f64, f64),
        nu: usize,
        nv: usize,
    ) -> Self {
        let nu = nu.max(4);
        let nv = nv.max(4);
        let len = nu
            .checked_mul(nv)
            .expect("BuckledSurfaceFlow nu * nv overflowed usize");
        let (du, dv) = topology.grid_spacings(range_u, range_v, nu, nv);
        let sampled = build_sampled_geometry(surface, topology, range_u, range_v, nu, nv);

        Self {
            nu,
            nv,
            topology,
            du,
            dv,
            min_area_element: sampled.min_area_element,
            north_meridional_g: sampled.north_meridional_g,
            south_meridional_g: sampled.south_meridional_g,
            seam_fwd: sampled.seam_fwd,
            neighbors: sampled.neighbors,
            points: sampled.ref_points.clone(),
            next_points: sampled.ref_points.clone(),
            ref_points: sampled.ref_points,
            buckling_normals: sampled.buckling_normals,
            mean_curvature: vec![0.0; len],
            displacements: vec![0.0; len],
        }
    }

    /// Grid dimensions `(nu, nv)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.nu, self.nv)
    }

    /// Active [`SurfaceTopology`].
    #[must_use]
    #[verified_engine::verified]
    pub fn topology(&self) -> SurfaceTopology {
        self.topology
    }

    /// Minimum non-degenerate differential area element $\min \sqrt{EG - F^2}$ across the grid.
    #[must_use]
    #[verified_engine::verified]
    pub fn min_area_element(&self) -> f64 {
        self.min_area_element
    }

    /// Maximum explicit time step satisfying the geometric mean-curvature CFL bound.
    #[must_use]
    #[verified_engine::verified]
    pub fn max_stable_dt(&self, tension: f64) -> f64 {
        let min_cell = self.min_area_element.max(1e-5);
        let denom = (4.0 * tension.abs()).max(1e-6);
        (0.2 * min_cell / denom).clamp(1e-6, 0.05)
    }

    /// Flat slice of evolved 3D vertex positions indexed by `flatten_2d_index(j, i, nv)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn points(&self) -> &[Point3<f64>] {
        &self.points
    }

    /// Flat slice of unbuckled reference 3D vertex positions.
    #[must_use]
    #[verified_engine::verified]
    pub fn reference_points(&self) -> &[Point3<f64>] {
        &self.ref_points
    }

    /// Flat slice of outward buckling normal vectors.
    #[must_use]
    #[verified_engine::verified]
    pub fn buckling_normals(&self) -> &[Vector3<f64>] {
        &self.buckling_normals
    }

    /// Flat slice of signed scalar normal displacements $(\mathbf{r} - \mathbf{r}_0) \cdot \mathbf{n}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn displacements(&self) -> &[f64] {
        &self.displacements
    }

    /// Flat slice of discrete scalar mean curvatures $H$ recorded on the most recent step.
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_curvatures(&self) -> &[f64] {
        &self.mean_curvature
    }

    /// Twisted seam mapping `j -> j_seam` across $u_{\min} \leftrightarrow u_{\max}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn seam_forward_map(&self) -> &[usize] {
        &self.seam_fwd
    }

    /// Resets evolved vertices, curvatures, and displacements back to the unbuckled reference surface.
    #[verified_engine::verified]
    pub fn reset_to_reference(&mut self) {
        self.points.copy_from_slice(&self.ref_points);
        self.next_points.copy_from_slice(&self.ref_points);
        self.mean_curvature.fill(0.0);
        self.displacements.fill(0.0);
    }

    /// Copies a previously saved vertex buffer back into the solver (used by `FiniteStateGuard` rollback).
    #[verified_engine::verified]
    pub fn restore_points(&mut self, saved: &[Point3<f64>]) -> bool {
        if saved.len() != self.points.len()
            || !saved
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        {
            return false;
        }
        self.points.copy_from_slice(saved);
        self.next_points.copy_from_slice(saved);
        for (idx, pt) in self.points.iter().enumerate() {
            self.displacements[idx] = (pt - self.ref_points[idx]).dot(&self.buckling_normals[idx]);
        }
        true
    }

    /// Advances morphogen-driven normal buckling against mean-curvature tension with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_buckled(
        &mut self,
        morphogen: &[f64],
        baseline: f64,
        buckling_gain: f64,
        tension: f64,
        tether: f64,
        dt: f64,
    ) -> bool {
        if morphogen.len() != self.points.len()
            || !dt.is_finite()
            || dt <= 0.0
            || !baseline.is_finite()
            || !buckling_gain.is_finite()
            || !tension.is_finite()
            || !tether.is_finite()
        {
            return false;
        }
        let eff_dt = dt.min(self.max_stable_dt(tension));
        if !self.compute_interior_step(morphogen, baseline, buckling_gain, tension, tether, eff_dt)
        {
            return false;
        }
        if self.topology == SurfaceTopology::PolarSphere
            && !self.apply_polar_cap_step(
                morphogen,
                baseline,
                buckling_gain,
                tension,
                tether,
                eff_dt,
            )
        {
            return false;
        }
        std::mem::swap(&mut self.points, &mut self.next_points);
        for (idx, pt) in self.points.iter().enumerate() {
            self.displacements[idx] = (pt - self.ref_points[idx]).dot(&self.buckling_normals[idx]);
        }
        true
    }

    fn compute_interior_step(
        &mut self,
        morphogen: &[f64],
        baseline: f64,
        buckling_gain: f64,
        tension: f64,
        tether: f64,
        eff_dt: f64,
    ) -> bool {
        let (nu, nv) = (self.nu, self.nv);
        let j_range = if self.topology == SurfaceTopology::PolarSphere {
            1..(nv - 1)
        } else {
            0..nv
        };

        for i in 0..nu {
            for j in j_range.clone() {
                let idx = flatten_2d_index(j, i, nv);
                let c = morphogen[idx];
                if !c.is_finite() {
                    return false;
                }
                let (h_vec, h_scalar) = self.discrete_mean_curvature_at(idx);
                let n_out = self.buckling_normals[idx];
                let delta_r = self.points[idx] - self.ref_points[idx];
                let vel =
                    n_out * (buckling_gain * (c - baseline)) + h_vec * tension - delta_r * tether;
                let updated = self.points[idx] + vel * eff_dt;
                if !updated.x.is_finite() || !updated.y.is_finite() || !updated.z.is_finite() {
                    return false;
                }
                self.next_points[idx] = updated;
                self.mean_curvature[idx] = h_scalar;
            }
        }
        true
    }

    fn apply_polar_cap_step(
        &mut self,
        morphogen: &[f64],
        baseline: f64,
        buckling_gain: f64,
        tension: f64,
        tether: f64,
        eff_dt: f64,
    ) -> bool {
        let (nu, nv) = (self.nu, self.nv);
        let inv_nu = 1.0 / nu as f64;
        for &(pole_j, ring_j, g_mer) in &[
            (0_usize, 1_usize, self.north_meridional_g),
            (nv - 1, nv - 2, self.south_meridional_g),
        ] {
            let pole_idx0 = flatten_2d_index(pole_j, 0, nv);
            let p_pole = self.points[pole_idx0];
            let mut ring_sum = Vector3::zeros();
            let mut c_sum = 0.0;
            for i in 0..nu {
                ring_sum += self.points[flatten_2d_index(ring_j, i, nv)].coords;
                let c_val = morphogen[flatten_2d_index(pole_j, i, nv)];
                if !c_val.is_finite() {
                    return false;
                }
                c_sum += c_val;
            }
            let ring_avg = Point3::from(ring_sum * inv_nu);
            let h_vec = (ring_avg - p_pole) * (2.0 / (g_mer * self.dv * self.dv).max(1e-8));
            let n_out = self.buckling_normals[pole_idx0];
            let c_pole = c_sum * inv_nu;
            let vel = n_out * (buckling_gain * (c_pole - baseline)) + h_vec * tension
                - (p_pole - self.ref_points[pole_idx0]) * tether;
            let updated = p_pole + vel * eff_dt;
            if !updated.x.is_finite() || !updated.y.is_finite() || !updated.z.is_finite() {
                return false;
            }
            let h_scalar = h_vec.norm();
            for i in 0..nu {
                let idx = flatten_2d_index(pole_j, i, nv);
                self.next_points[idx] = updated;
                self.mean_curvature[idx] = h_scalar;
            }
        }
        true
    }

    fn discrete_mean_curvature_at(&self, idx: usize) -> (Vector3<f64>, f64) {
        let nb = self.neighbors[idx];
        let p = self.points[idx];
        let (p_ip, p_im) = (self.points[nb.ip], self.points[nb.im]);
        let (p_jp, p_jm) = (self.points[nb.jp], self.points[nb.jm]);
        let r_u = (p_ip - p_im) / (2.0 * self.du);
        let r_v = (p_jp - p_jm) / (2.0 * self.dv);
        let r_uu = (p_ip.coords - p.coords * 2.0 + p_im.coords) / (self.du * self.du);
        let r_vv = (p_jp.coords - p.coords * 2.0 + p_jm.coords) / (self.dv * self.dv);
        let r_uv =
            (self.points[nb.pp].coords - self.points[nb.mp].coords - self.points[nb.pm].coords
                + self.points[nb.mm].coords)
                / (4.0 * self.du * self.dv);

        let (e, f, g) = (r_u.dot(&r_u), r_u.dot(&r_v), r_v.dot(&r_v));
        let det_g = e * g - f * f;
        let cross = r_u.cross(&r_v);
        let cross_norm = cross.norm();
        if det_g <= 1e-12 || cross_norm <= 1e-12 {
            return (Vector3::zeros(), 0.0);
        }
        let n = cross / cross_norm;
        let (l, m, n_c) = (r_uu.dot(&n), r_uv.dot(&n), r_vv.dot(&n));
        let h_val = (e * n_c - 2.0 * f * m + g * l) / (2.0 * det_g);
        (n * h_val, h_val)
    }
}

#[cfg(test)]
#[path = "buckled_flow_tests.rs"]
mod tests;
