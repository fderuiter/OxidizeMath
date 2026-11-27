#[path = "surface/topology.rs"]
mod topology;

use topology::{
    SURFACE_CHUNK, accumulate_biharmonic_terms, accumulate_chunk_topology,
    evolve_sequential_polyline, mesh_bending_energy, mesh_tension_energy,
    sequential_internal_energy,
};

pub use pure_math::pure_math::differential_geometry::Surface;

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

impl MriVolume {
    /// Creates a validated [`MriVolume`] when dimensions, origin, spacing, and buffer length match.
    pub fn new(
        dims: [usize; 3],
        origin: [f64; 3],
        voxel_size: f64,
        intensities: Vec<f64>,
    ) -> Result<Self, crate::error::EngineeringError> {
        use crate::error::EngineeringError;
        if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return Err(EngineeringError::ZeroVolumeDimension { dims });
        }
        if !voxel_size.is_finite() || voxel_size <= 0.0 {
            return Err(EngineeringError::InvalidVoxelSize { voxel_size });
        }
        if !origin[0].is_finite() || !origin[1].is_finite() || !origin[2].is_finite() {
            return Err(EngineeringError::NonFiniteOrigin { origin });
        }
        let expected = dims[0]
            .checked_mul(dims[1])
            .and_then(|xy| xy.checked_mul(dims[2]))
            .ok_or(EngineeringError::VolumeDimensionsOverflow { dims })?;
        if intensities.len() != expected {
            return Err(EngineeringError::DimensionMismatch {
                context: "MRI volume intensities length",
                expected,
                actual: intensities.len(),
            });
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

    /// Samples MRI intensity at world point `[x, y, z]` via clamped trilinear interpolation.
    #[must_use]
    #[verified_engine::verified]
    pub fn sample_trilinear(&self, point: &[f64; 3]) -> f64 {
        if self.intensities.is_empty() {
            return 0.0;
        }
        let inv_h = 1.0 / self.voxel_size.max(1e-9);
        let gx = ((point[0] - self.origin[0]) * inv_h).clamp(0.0, (self.dims[0] - 1) as f64);
        let gy = ((point[1] - self.origin[1]) * inv_h).clamp(0.0, (self.dims[1] - 1) as f64);
        let gz = ((point[2] - self.origin[2]) * inv_h).clamp(0.0, (self.dims[2] - 1) as f64);

        let (x0, y0, z0) = (
            gx.floor() as usize,
            gy.floor() as usize,
            gz.floor() as usize,
        );
        let (x1, y1, z1) = (
            (x0 + 1).min(self.dims[0] - 1),
            (y0 + 1).min(self.dims[1] - 1),
            (z0 + 1).min(self.dims[2] - 1),
        );
        let (tx, ty, tz) = (gx - x0 as f64, gy - y0 as f64, gz - z0 as f64);

        let c00 = self.voxel(x0, y0, z0) * (1.0 - tx) + self.voxel(x1, y0, z0) * tx;
        let c10 = self.voxel(x0, y1, z0) * (1.0 - tx) + self.voxel(x1, y1, z0) * tx;
        let c01 = self.voxel(x0, y0, z1) * (1.0 - tx) + self.voxel(x1, y0, z1) * tx;
        let c11 = self.voxel(x0, y1, z1) * (1.0 - tx) + self.voxel(x1, y1, z1) * tx;
        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        c0 * (1.0 - tz) + c1 * tz
    }

    /// Evaluates Gaussian-smoothed intensity $(G_\sigma * I)(s)$ via a separable 3x3x3 kernel.
    #[must_use]
    #[verified_engine::verified]
    pub fn smoothed_intensity(&self, point: &[f64; 3], sigma: f64) -> f64 {
        if !sigma.is_finite() || sigma <= 1e-6 {
            return self.sample_trilinear(point);
        }
        let offsets = [-sigma, 0.0, sigma];
        let weights = [0.25, 0.5, 0.25];
        let mut accum = 0.0;
        for (ix, &ox) in offsets.iter().enumerate() {
            for (iy, &oy) in offsets.iter().enumerate() {
                let w_xy = weights[ix] * weights[iy];
                for (iz, &oz) in offsets.iter().enumerate() {
                    let q = [point[0] + ox, point[1] + oy, point[2] + oz];
                    accum += (w_xy * weights[iz]) * self.sample_trilinear(&q);
                }
            }
        }
        accum
    }

    /// Evaluates the spatial gradient $\nabla (G_\sigma * I)(s)$ via central finite differences.
    #[must_use]
    #[verified_engine::verified]
    pub fn smoothed_gradient(&self, point: &[f64; 3], sigma: f64) -> [f64; 3] {
        let s = if sigma.is_finite() {
            sigma.max(0.0)
        } else {
            0.0
        };
        let h = s.max(self.voxel_size).max(1e-3);
        let inv_2h = 0.5 / h;
        let gx = (self.smoothed_intensity(&[point[0] + h, point[1], point[2]], s)
            - self.smoothed_intensity(&[point[0] - h, point[1], point[2]], s))
            * inv_2h;
        let gy = (self.smoothed_intensity(&[point[0], point[1] + h, point[2]], s)
            - self.smoothed_intensity(&[point[0], point[1] - h, point[2]], s))
            * inv_2h;
        let gz = (self.smoothed_intensity(&[point[0], point[1], point[2] + h], s)
            - self.smoothed_intensity(&[point[0], point[1], point[2] - h], s))
            * inv_2h;
        [gx, gy, gz]
    }
}

/// Evaluates thin-plate internal smoothness energy combining edge tension (`alpha`) and
/// 1-ring umbrella Laplacian bending (`beta`) with zero dynamic heap allocation.
///
/// Falls back to sequential polyline first- and second-difference energies when `surface.faces`
/// is empty.
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

#[verified_engine::verified]
fn normal_external_force(
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
    let delta = 0.5 * sigma.max(volume.voxel_size).max(1e-3);
    let gp = volume.smoothed_gradient(
        &[
            point[0] + delta * nx,
            point[1] + delta * ny,
            point[2] + delta * nz,
        ],
        sigma,
    );
    let gm = volume.smoothed_gradient(
        &[
            point[0] - delta * nx,
            point[1] - delta * ny,
            point[2] - delta * nz,
        ],
        sigma,
    );
    let e_plus = -(gp[0] * gp[0] + gp[1] * gp[1] + gp[2] * gp[2]);
    let e_minus = -(gm[0] * gm[0] + gm[1] * gm[1] + gm[2] * gm[2]);
    let neg_de_dn = -(e_plus - e_minus) / (2.0 * delta);
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
/// $E_{\text{total}} = E_{\text{internal}}(\alpha, \beta) + E_{\text{external}}(G_\sigma * I)$,
/// combining 1-ring tension, biharmonic bending, and outward normal MRI gradient attraction.
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
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_internal_energy_100_steps_zero_alloc_and_exact_value() {
        let mut surface =
            Surface::from_vertices(vec![[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [1.0, 2.0, 5.0]]);
        // Tension: (1+4+4) + (0+0+9) = 18.0; bending: (-1)^2 + (-2)^2 + 1^2 = 6.0 -> 0.5*18 + 0.25*6 = 10.5
        let initial = internal_energy(&surface, 0.5, 0.25);
        assert!((initial - 10.5).abs() < 1e-12);

        let (accum, _metrics) = VerifiedEngine::run_verified(|| {
            let mut sum = 0.0;
            for _ in 0..100 {
                evolve_surface(&mut surface, 1e-4, 0.2);
                sum += internal_energy(&surface, 0.5, 0.25) + external_energy(&surface, 0.2);
            }
            sum
        });
        assert!(accum.is_finite());
    }
}
