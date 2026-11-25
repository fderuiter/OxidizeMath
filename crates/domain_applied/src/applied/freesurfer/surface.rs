#[path = "surface/topology.rs"]
mod topology;

use math_scalars::{DualVec, HyperDual, Scalar};
use topology::{
    SURFACE_CHUNK, accumulate_biharmonic_terms, accumulate_chunk_topology,
    evolve_sequential_polyline, mesh_bending_energy, mesh_tension_energy,
    sequential_internal_energy,
};

/// Triangulated three-dimensional cortical surface mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Surface {
    /// Vertex coordinates `[x, y, z]` in millimeters.
    pub vertices: Vec<[f64; 3]>,
    /// Outward-oriented triangle face vertex indices `[i0, i1, i2]`.
    pub faces: Vec<[usize; 3]>,
}

impl Surface {
    /// Constructs a triangulated [`Surface`] from vertex coordinates and triangle indices.
    #[must_use]
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[usize; 3]>) -> Self {
        Self { vertices, faces }
    }

    /// Constructs a vertex-only [`Surface`] with an empty triangle index list.
    #[must_use]
    pub fn from_vertices(vertices: Vec<[f64; 3]>) -> Self {
        Self {
            vertices,
            faces: Vec::new(),
        }
    }
}

/// Three-dimensional structural MRI intensity volume on an isotropic physical grid.
#[derive(Clone, Debug, PartialEq)]
pub struct MriVolume {
    /// Voxel counts `[nx, ny, nz]` along the X, Y, and Z axes.
    pub dims: [usize; 3],
    /// World-space coordinates `[x0, y0, z0]` (mm) of voxel `(0, 0, 0)`.
    pub origin: [f64; 3],
    /// Isotropic voxel spacing (mm).
    pub voxel_size: f64,
    /// Row-major (`(z * ny + y) * nx + x`) intensity samples of length `nx * ny * nz`.
    pub intensities: Vec<f64>,
}

#[inline]
fn clamp_grid_axis<S: Scalar>(
    coord_ref: f64,
    coord_s: S,
    origin: f64,
    inv_h: f64,
    dim: usize,
) -> (usize, usize, S) {
    let max_idx = dim.saturating_sub(1);
    let max_f = max_idx as f64;
    let u_ref = (coord_ref - origin) * inv_h;
    if u_ref <= 0.0 || max_idx == 0 {
        (0, 0, S::ZERO)
    } else if u_ref >= max_f {
        (max_idx, max_idx, S::ZERO)
    } else {
        let i0 = (u_ref.floor() as usize).min(max_idx);
        let i1 = (i0 + 1).min(max_idx);
        let t = (coord_s - S::from_f64(origin)) * S::from_f64(inv_h) - S::from_usize(i0);
        (i0, i1, t)
    }
}

#[inline]
fn bspline_axis_weights<S: Scalar>(
    coord_ref: f64,
    coord_s: S,
    origin: f64,
    inv_h: f64,
    dim: usize,
) -> ([usize; 4], [S; 4]) {
    let max_idx = dim.saturating_sub(1) as isize;
    let u_ref = (coord_ref - origin) * inv_h;
    let base = u_ref.floor() as isize;
    let t = (coord_s - S::from_f64(origin)) * S::from_f64(inv_h) - S::from_f64(base as f64);
    let one_minus_t = S::ONE - t;
    let t2 = t * t;
    let t3 = t2 * t;
    let sixth = S::ONE / S::from_f64(6.0);
    let three = S::from_f64(3.0);
    let four = S::from_f64(4.0);
    let six = S::from_f64(6.0);

    let w0 = one_minus_t * one_minus_t * one_minus_t * sixth;
    let w1 = (three * t3 - six * t2 + four) * sixth;
    let w2 = (-three * t3 + three * t2 + three * t + S::ONE) * sixth;
    let w3 = t3 * sixth;
    let idx = [
        (base - 1).clamp(0, max_idx) as usize,
        base.clamp(0, max_idx) as usize,
        (base + 1).clamp(0, max_idx) as usize,
        (base + 2).clamp(0, max_idx) as usize,
    ];
    (idx, [w0, w1, w2, w3])
}

impl MriVolume {
    /// Creates a validated [`MriVolume`] when dimensions, origin, spacing, and buffer length match.
    pub fn new(
        dims: [usize; 3],
        origin: [f64; 3],
        voxel_size: f64,
        intensities: Vec<f64>,
    ) -> Result<Self, &'static str> {
        if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return Err("MRI volume dimensions must be non-zero");
        }
        if !voxel_size.is_finite() || voxel_size <= 0.0 {
            return Err("MRI voxel_size must be finite and positive");
        }
        if !origin[0].is_finite() || !origin[1].is_finite() || !origin[2].is_finite() {
            return Err("MRI origin coordinates must be finite");
        }
        let expected = dims[0]
            .checked_mul(dims[1])
            .and_then(|xy| xy.checked_mul(dims[2]))
            .ok_or("MRI volume dimensions overflow")?;
        if intensities.len() != expected {
            return Err("MRI intensities length does not match dims product");
        }
        Ok(Self {
            dims,
            origin,
            voxel_size,
            intensities,
        })
    }

    /// Returns the intensity at clamped integer voxel coordinates `(ix, iy, iz)`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn voxel(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        let x = ix.min(self.dims[0].saturating_sub(1));
        let y = iy.min(self.dims[1].saturating_sub(1));
        let z = iz.min(self.dims[2].saturating_sub(1));
        let idx = (z * self.dims[1] + y) * self.dims[0] + x;
        self.intensities.get(idx).copied().unwrap_or(0.0)
    }

    /// Evaluates clamped trilinear MRI interpolation over any algebraic [`Scalar`] `S`.
    #[inline]
    #[verified_engine::verified]
    pub fn sample_trilinear_scalar<S: Scalar>(&self, ref_pt: &[f64; 3], point: &[S; 3]) -> S {
        if self.intensities.is_empty() {
            return S::ZERO;
        }
        let inv_h = 1.0 / self.voxel_size.max(1e-9);
        let (x0, x1, tx) =
            clamp_grid_axis(ref_pt[0], point[0], self.origin[0], inv_h, self.dims[0]);
        let (y0, y1, ty) =
            clamp_grid_axis(ref_pt[1], point[1], self.origin[1], inv_h, self.dims[1]);
        let (z0, z1, tz) =
            clamp_grid_axis(ref_pt[2], point[2], self.origin[2], inv_h, self.dims[2]);

        let ox = S::ONE - tx;
        let oy = S::ONE - ty;
        let oz = S::ONE - tz;
        let c00 =
            S::from_f64(self.voxel(x0, y0, z0)) * ox + S::from_f64(self.voxel(x1, y0, z0)) * tx;
        let c10 =
            S::from_f64(self.voxel(x0, y1, z0)) * ox + S::from_f64(self.voxel(x1, y1, z0)) * tx;
        let c01 =
            S::from_f64(self.voxel(x0, y0, z1)) * ox + S::from_f64(self.voxel(x1, y0, z1)) * tx;
        let c11 =
            S::from_f64(self.voxel(x0, y1, z1)) * ox + S::from_f64(self.voxel(x1, y1, z1)) * tx;
        let c0 = c00 * oy + c10 * ty;
        let c1 = c01 * oy + c11 * ty;
        c0 * oz + c1 * tz
    }

    #[inline]
    #[verified_engine::verified]
    fn sample_bspline_scalar<S: Scalar>(&self, ref_pt: &[f64; 3], point: &[S; 3]) -> S {
        if self.intensities.is_empty() {
            return S::ZERO;
        }
        let inv_h = 1.0 / self.voxel_size.max(1e-9);
        let (ix, wx) =
            bspline_axis_weights(ref_pt[0], point[0], self.origin[0], inv_h, self.dims[0]);
        let (iy, wy) =
            bspline_axis_weights(ref_pt[1], point[1], self.origin[1], inv_h, self.dims[1]);
        let (iz, wz) =
            bspline_axis_weights(ref_pt[2], point[2], self.origin[2], inv_h, self.dims[2]);

        let mut acc = S::ZERO;
        for kz in 0..4 {
            let mut acc_y = S::ZERO;
            for ky in 0..4 {
                let mut acc_x = S::ZERO;
                for kx in 0..4 {
                    acc_x += wx[kx] * S::from_f64(self.voxel(ix[kx], iy[ky], iz[kz]));
                }
                acc_y += wy[ky] * acc_x;
            }
            acc += wz[kz] * acc_y;
        }
        acc
    }

    /// Evaluates $C^2$ Gaussian-smoothed MRI intensity $(G_\sigma * I)(s)$ over any [`Scalar`] `S`.
    #[inline]
    #[verified_engine::verified]
    pub fn smoothed_intensity_scalar<S: Scalar>(
        &self,
        ref_pt: &[f64; 3],
        point: &[S; 3],
        sigma: f64,
    ) -> S {
        if !sigma.is_finite() || sigma <= 1e-6 {
            return self.sample_trilinear_scalar(ref_pt, point);
        }
        let offsets = [-sigma, 0.0, sigma];
        let weights = [0.25, 0.5, 0.25];
        let mut accum = S::ZERO;
        for (ix, &ox) in offsets.iter().enumerate() {
            let ox_s = S::from_f64(ox);
            for (iy, &oy) in offsets.iter().enumerate() {
                let oy_s = S::from_f64(oy);
                let w_xy = weights[ix] * weights[iy];
                for (iz, &oz) in offsets.iter().enumerate() {
                    let q_ref = [ref_pt[0] + ox, ref_pt[1] + oy, ref_pt[2] + oz];
                    let q_s = [point[0] + ox_s, point[1] + oy_s, point[2] + S::from_f64(oz)];
                    accum +=
                        S::from_f64(w_xy * weights[iz]) * self.sample_bspline_scalar(&q_ref, &q_s);
                }
            }
        }
        accum
    }

    /// Samples MRI intensity at world point `[x, y, z]` via clamped trilinear interpolation.
    #[must_use]
    #[verified_engine::verified]
    pub fn sample_trilinear(&self, point: &[f64; 3]) -> f64 {
        self.sample_trilinear_scalar(point, point)
    }

    /// Evaluates Gaussian-smoothed intensity $(G_\sigma * I)(s)$ via a $C^2$ B-spline + 3x3x3 kernel.
    #[must_use]
    #[verified_engine::verified]
    pub fn smoothed_intensity(&self, point: &[f64; 3], sigma: f64) -> f64 {
        self.smoothed_intensity_scalar(point, point, sigma)
    }

    /// Evaluates the exact spatial gradient $\nabla (G_\sigma * I)(s)$ in a single [`DualVec<f64, 3>`] pass.
    #[must_use]
    #[verified_engine::verified]
    pub fn smoothed_gradient(&self, point: &[f64; 3], sigma: f64) -> [f64; 3] {
        let s = if sigma.is_finite() {
            sigma.max(0.0)
        } else {
            0.0
        };
        let seed = DualVec::<f64, 3>::jacobian_seed(*point);
        self.smoothed_intensity_scalar(point, &seed, s).eps
    }
}

/// Evaluates thin-plate internal smoothness energy combining edge tension (`alpha`) and
/// 1-ring umbrella Laplacian bending (`beta`) with zero dynamic heap allocation.
#[verified_engine::verified]
pub fn internal_energy(surface: &Surface, alpha: f64, beta: f64) -> f64 {
    if surface.faces.is_empty() {
        return sequential_internal_energy(&surface.vertices, alpha, beta);
    }
    alpha * mesh_tension_energy(&surface.vertices, &surface.faces)
        + beta * mesh_bending_energy(&surface.vertices, &surface.faces)
}

/// Evaluates scalar external energy assuming a uniform average image gradient magnitude.
#[verified_engine::verified]
pub fn external_energy(surface: &Surface, image_gradient_strength: f64) -> f64 {
    -(surface.vertices.len() as f64) * image_gradient_strength.powi(2)
}

/// Evaluates MRI external boundary energy $-\sum_{v \in S} \|\nabla (G_\sigma * I)(v)\|^2$.
#[verified_engine::verified]
pub fn external_energy_volume(surface: &Surface, volume: &MriVolume, sigma: f64) -> f64 {
    let mut energy = 0.0;
    for v in &surface.vertices {
        let g = volume.smoothed_gradient(v, sigma);
        energy -= g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    }
    energy
}

/// Evolves `surface` one step along outward triangle vertex normals (skipping zero-length normals),
/// or relaxes sequential vertex pairs when `surface.faces` is empty.
#[verified_engine::verified]
pub fn evolve_surface(surface: &mut Surface, learning_rate: f64, gradient_strength: f64) {
    let step = learning_rate * gradient_strength;
    if !step.is_finite() || step == 0.0 {
        return;
    }
    if surface.faces.is_empty() {
        evolve_sequential_polyline(&mut surface.vertices, step);
        return;
    }

    let n = surface.vertices.len();
    let mut nbr_sum = [[0.0_f64; 3]; SURFACE_CHUNK];
    let mut nbr_deg = [0_u32; SURFACE_CHUNK];
    let mut normals = [[0.0_f64; 3]; SURFACE_CHUNK];
    let mut base = 0;
    while base < n {
        let chunk_len = (n - base).min(SURFACE_CHUNK);
        for k in 0..chunk_len {
            nbr_sum[k] = [0.0; 3];
            nbr_deg[k] = 0;
            normals[k] = [0.0; 3];
        }
        accumulate_chunk_topology(
            &surface.vertices,
            &surface.faces,
            base,
            chunk_len,
            &mut nbr_sum,
            &mut nbr_deg,
            &mut normals,
        );
        for (k, nrm) in normals.iter().take(chunk_len).enumerate() {
            let norm = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            if norm > 1e-12 {
                let scale = step / norm;
                surface.vertices[base + k][0] += scale * nrm[0];
                surface.vertices[base + k][1] += scale * nrm[1];
                surface.vertices[base + k][2] += scale * nrm[2];
            }
        }
        base += SURFACE_CHUNK;
    }
}

/// Evaluates the exact outward normal external MRI force $-\partial_n E_{\text{ext}}(x)\hat{n}
/// = 2 (\nabla I_\sigma(x) \cdot \partial_n \nabla I_\sigma(x)) \hat{n}$ via [`HyperDual<f64>`].
#[verified_engine::verified]
pub fn normal_external_force(
    point: &[f64; 3],
    raw_normal: &[f64; 3],
    volume: &MriVolume,
    sigma: f64,
) -> [f64; 3] {
    let norm = (raw_normal[0] * raw_normal[0]
        + raw_normal[1] * raw_normal[1]
        + raw_normal[2] * raw_normal[2])
        .sqrt();
    if norm <= 1e-12 {
        return [0.0; 3];
    }
    let (nx, ny, nz) = (
        raw_normal[0] / norm,
        raw_normal[1] / norm,
        raw_normal[2] / norm,
    );
    let s = if sigma.is_finite() {
        sigma.max(0.0)
    } else {
        0.0
    };

    let mut neg_de_dn = 0.0_f64;
    for axis in 0..3 {
        let pt_hd = [
            HyperDual::new(point[0], f64::from(u8::from(axis == 0)), nx, 0.0),
            HyperDual::new(point[1], f64::from(u8::from(axis == 1)), ny, 0.0),
            HyperDual::new(point[2], f64::from(u8::from(axis == 2)), nz, 0.0),
        ];
        let jet = volume.smoothed_intensity_scalar(point, &pt_hd, s);
        neg_de_dn += 2.0 * jet.eps1 * jet.eps12;
    }
    [neg_de_dn * nx, neg_de_dn * ny, neg_de_dn * nz]
}

#[derive(Clone, Copy)]
struct DeformStepCfg {
    alpha: f64,
    beta: f64,
    sigma: f64,
    learning_rate: f64,
    max_disp: f64,
}

#[verified_engine::verified]
fn step_deformable_chunk(
    surface: &mut Surface,
    volume: &MriVolume,
    base: usize,
    chunk_len: usize,
    cfg: DeformStepCfg,
) {
    let mut nbr_sum = [[0.0_f64; 3]; SURFACE_CHUNK];
    let mut nbr_deg = [0_u32; SURFACE_CHUNK];
    let mut normals = [[0.0_f64; 3]; SURFACE_CHUNK];
    let mut lap = [[0.0_f64; 3]; SURFACE_CHUNK];
    let mut biharm = [[0.0_f64; 3]; SURFACE_CHUNK];
    let mut forces = [[0.0_f64; 3]; SURFACE_CHUNK];

    accumulate_chunk_topology(
        &surface.vertices,
        &surface.faces,
        base,
        chunk_len,
        &mut nbr_sum,
        &mut nbr_deg,
        &mut normals,
    );
    for k in 0..chunk_len {
        if nbr_deg[k] > 0 {
            let inv_d = 1.0 / f64::from(nbr_deg[k]);
            let v = surface.vertices[base + k];
            lap[k] = [
                nbr_sum[k][0] * inv_d - v[0],
                nbr_sum[k][1] * inv_d - v[1],
                nbr_sum[k][2] * inv_d - v[2],
            ];
        }
    }
    accumulate_biharmonic_terms(&surface.faces, base, chunk_len, &lap, &nbr_deg, &mut biharm);

    let mut max_force_sq = 0.0_f64;
    for k in 0..chunk_len {
        let deg_half = 0.5 * f64::from(nbr_deg[k]);
        let f_ext =
            normal_external_force(&surface.vertices[base + k], &normals[k], volume, cfg.sigma);
        for axis in 0..3 {
            forces[k][axis] = 2.0 * cfg.alpha * deg_half * lap[k][axis]
                + 2.0 * cfg.beta * (lap[k][axis] - biharm[k][axis])
                + f_ext[axis];
        }
        let f_sq =
            forces[k][0] * forces[k][0] + forces[k][1] * forces[k][1] + forces[k][2] * forces[k][2];
        if f_sq > max_force_sq {
            max_force_sq = f_sq;
        }
    }
    let max_force = max_force_sq.sqrt();
    let eff_lr = if max_force * cfg.learning_rate > cfg.max_disp && max_force > 1e-12 {
        cfg.max_disp / max_force
    } else {
        cfg.learning_rate
    };
    for (k, f) in forces.iter().take(chunk_len).enumerate() {
        surface.vertices[base + k][0] += eff_lr * f[0];
        surface.vertices[base + k][1] += eff_lr * f[1];
        surface.vertices[base + k][2] += eff_lr * f[2];
    }
}

/// Evolves `surface` by one zero-allocation gradient descent step on
/// $E_{\text{total}} = E_{\text{internal}}(\alpha, \beta) + E_{\text{external}}(G_\sigma * I)$.
#[verified_engine::verified]
pub fn evolve_surface_deformable(
    surface: &mut Surface,
    volume: &MriVolume,
    alpha: f64,
    beta: f64,
    sigma: f64,
    learning_rate: f64,
) {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return;
    }
    if surface.faces.is_empty() {
        evolve_sequential_polyline(&mut surface.vertices, learning_rate * alpha.max(0.0));
        return;
    }
    let n = surface.vertices.len();
    let cfg = DeformStepCfg {
        alpha,
        beta,
        sigma,
        learning_rate,
        max_disp: 0.2 * sigma.max(volume.voxel_size).max(1e-3),
    };
    let mut base = 0;
    while base < n {
        let chunk_len = (n - base).min(SURFACE_CHUNK);
        step_deformable_chunk(surface, volume, base, chunk_len, cfg);
        base += SURFACE_CHUNK;
    }
}

#[cfg(test)]
#[path = "surface_tests.rs"]
mod tests;
