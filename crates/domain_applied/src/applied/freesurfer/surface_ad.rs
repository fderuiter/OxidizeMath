//! Exact forward-mode automatic differentiation for FreeSurfer MRI volumes and normal forces.

use math_scalars::{DualVec, HyperDual, Scalar};

use super::MriVolume;

/// Maximum 1D Gaussian kernel support (`2 * MAX_GAUSS_RADIUS + 1 = 17` voxels per axis).
const MAX_GAUSS_SUPPORT: usize = 17;
const MAX_GAUSS_RADIUS: isize = 8;

impl MriVolume {
    /// Samples MRI intensity over a generic differentiable scalar `S` (zero heap allocation).
    #[must_use]
    #[verified_engine::verified]
    pub fn sample_trilinear_scalar<S: Scalar>(&self, point: &[S; 3]) -> S {
        if self.intensities.is_empty() {
            return S::ZERO;
        }
        let inv_h = 1.0 / self.voxel_size.max(1e-9);
        let inv_h_s = S::from_f64(inv_h);

        let (x0, x1, tx) =
            axis_cell_weights(point[0], self.origin[0], inv_h, inv_h_s, self.dims[0]);
        let (y0, y1, ty) =
            axis_cell_weights(point[1], self.origin[1], inv_h, inv_h_s, self.dims[1]);
        let (z0, z1, tz) =
            axis_cell_weights(point[2], self.origin[2], inv_h, inv_h_s, self.dims[2]);

        let one_x = S::ONE - tx;
        let c00 =
            S::from_f64(self.voxel(x0, y0, z0)) * one_x + S::from_f64(self.voxel(x1, y0, z0)) * tx;
        let c10 =
            S::from_f64(self.voxel(x0, y1, z0)) * one_x + S::from_f64(self.voxel(x1, y1, z0)) * tx;
        let c01 =
            S::from_f64(self.voxel(x0, y0, z1)) * one_x + S::from_f64(self.voxel(x1, y0, z1)) * tx;
        let c11 =
            S::from_f64(self.voxel(x0, y1, z1)) * one_x + S::from_f64(self.voxel(x1, y1, z1)) * tx;

        let one_y = S::ONE - ty;
        let c0 = c00 * one_y + c10 * ty;
        let c1 = c01 * one_y + c11 * ty;
        c0 * (S::ONE - tz) + c1 * tz
    }

    /// Evaluates smooth $C^\infty$ separable Gaussian-filtered intensity $(G_\sigma * I)(s)$
    /// over a generic [`Scalar`] `S` with zero heap allocation.
    #[must_use]
    #[verified_engine::verified]
    pub fn smoothed_intensity_scalar<S: Scalar>(&self, point: &[S; 3], sigma: f64) -> S {
        if self.intensities.is_empty() {
            return S::ZERO;
        }
        if !sigma.is_finite() || sigma <= 1e-6 {
            return self.sample_trilinear_scalar(point);
        }
        let h = self.voxel_size.max(1e-9);
        let inv_2sig2 = S::from_f64(-0.5 / (sigma * sigma));
        let rad = ((4.0 * sigma / h).ceil() as isize).clamp(2, MAX_GAUSS_RADIUS);

        let mut wx = [S::ZERO; MAX_GAUSS_SUPPORT];
        let mut wy = [S::ZERO; MAX_GAUSS_SUPPORT];
        let mut wz = [S::ZERO; MAX_GAUSS_SUPPORT];
        let (x_lo, nx) = build_axis_gaussian(
            point[0],
            self.origin[0],
            h,
            self.dims[0],
            rad,
            inv_2sig2,
            &mut wx,
        );
        let (y_lo, ny) = build_axis_gaussian(
            point[1],
            self.origin[1],
            h,
            self.dims[1],
            rad,
            inv_2sig2,
            &mut wy,
        );
        let (z_lo, nz) = build_axis_gaussian(
            point[2],
            self.origin[2],
            h,
            self.dims[2],
            rad,
            inv_2sig2,
            &mut wz,
        );

        let mut accum_z = S::ZERO;
        for (kz, &wz_k) in wz.iter().take(nz).enumerate() {
            let iz = z_lo + kz;
            let mut accum_y = S::ZERO;
            for (ky, &wy_k) in wy.iter().take(ny).enumerate() {
                let iy = y_lo + ky;
                let mut accum_x = S::ZERO;
                for (kx, &wx_k) in wx.iter().take(nx).enumerate() {
                    accum_x += wx_k * S::from_f64(self.voxel(x_lo + kx, iy, iz));
                }
                accum_y += wy_k * accum_x;
            }
            accum_z += wz_k * accum_y;
        }
        accum_z
    }

    /// Evaluates the exact spatial gradient $\nabla (G_\sigma * I)(s)$ in a single [`DualVec<f64, 3>`] pass.
    #[must_use]
    #[verified_engine::verified]
    pub fn smoothed_gradient_ad(&self, point: &[f64; 3], sigma: f64) -> [f64; 3] {
        let s = if sigma.is_finite() {
            sigma.max(0.0)
        } else {
            0.0
        };
        let pt_ad = [
            DualVec::<f64, 3>::variable(point[0], 0),
            DualVec::<f64, 3>::variable(point[1], 1),
            DualVec::<f64, 3>::variable(point[2], 2),
        ];
        self.smoothed_intensity_scalar(&pt_ad, s).eps
    }
}

#[inline]
#[verified_engine::verified]
fn build_axis_gaussian<S: Scalar>(
    coord: S,
    origin: f64,
    h: f64,
    dim: usize,
    rad: isize,
    inv_2sig2: S,
    out_weights: &mut [S; MAX_GAUSS_SUPPORT],
) -> (usize, usize) {
    let center = ((coord.real_part() - origin) / h).round() as isize;
    let max_idx = (dim as isize).saturating_sub(1);
    let i_lo = (center - rad).clamp(0, max_idx) as usize;
    let i_hi = (center + rad).clamp(0, max_idx) as usize;
    let count = (i_hi - i_lo + 1).min(MAX_GAUSS_SUPPORT);

    let mut sum = S::ZERO;
    for (k, slot) in out_weights.iter_mut().take(count).enumerate() {
        let pos = origin + h * ((i_lo + k) as f64);
        let diff = coord - S::from_f64(pos);
        let w = (diff * diff * inv_2sig2).exp();
        *slot = w;
        sum += w;
    }
    if sum.real_part() > 0.0 {
        let inv_sum = S::ONE / sum;
        for slot in out_weights.iter_mut().take(count) {
            *slot *= inv_sum;
        }
    }
    (i_lo, count)
}

#[inline]
#[verified_engine::verified]
fn axis_cell_weights<S: Scalar>(
    coord: S,
    origin: f64,
    inv_h: f64,
    inv_h_s: S,
    dim: usize,
) -> (usize, usize, S) {
    if dim <= 1 {
        return (0, 0, S::ZERO);
    }
    let max_idx = (dim - 1) as f64;
    let raw_primal = (coord.real_part() - origin) * inv_h;
    if raw_primal < 0.0 {
        return (0, 0, S::ZERO);
    }
    if raw_primal > max_idx {
        let last = dim - 1;
        return (last, last, S::ZERO);
    }
    let i0 = (raw_primal.floor() as usize).min(dim - 2);
    let raw_s = (coord - S::from_f64(origin)) * inv_h_s;
    let t = raw_s - S::from_f64(i0 as f64);
    (i0, i0 + 1, t)
}

/// Evaluates the exact normal external MRI boundary force
/// $F_{\text{ext}}(s, \hat{n}) = 2 (\nabla I_\sigma(s) \cdot H_{I_\sigma}(s) \hat{n}) \hat{n}$
/// via zero-allocation [`HyperDual<f64>`] directional second derivatives.
#[must_use]
#[verified_engine::verified]
pub fn normal_external_force_ad(
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
    let n_unit = [
        raw_normal[0] / norm,
        raw_normal[1] / norm,
        raw_normal[2] / norm,
    ];
    let s = if sigma.is_finite() {
        sigma.max(0.0)
    } else {
        0.0
    };

    let mut neg_de_dn = 0.0_f64;
    for k in 0..3 {
        let pt_hd = [
            HyperDual::new(point[0], if k == 0 { 1.0 } else { 0.0 }, n_unit[0], 0.0),
            HyperDual::new(point[1], if k == 1 { 1.0 } else { 0.0 }, n_unit[1], 0.0),
            HyperDual::new(point[2], if k == 2 { 1.0 } else { 0.0 }, n_unit[2], 0.0),
        ];
        let jet = volume.smoothed_intensity_scalar(&pt_hd, s);
        neg_de_dn += 2.0 * jet.eps1 * jet.eps12;
    }
    [
        neg_de_dn * n_unit[0],
        neg_de_dn * n_unit[1],
        neg_de_dn * n_unit[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    fn analytical_axis_jet2(
        coord: f64,
        origin: f64,
        h: f64,
        dim: usize,
        sigma: f64,
    ) -> (usize, Vec<(f64, f64, f64)>) {
        let rad = ((4.0 * sigma / h).ceil() as isize).clamp(2, MAX_GAUSS_RADIUS);
        let center = ((coord - origin) / h).round() as isize;
        let max_idx = (dim as isize) - 1;
        let i_lo = (center - rad).clamp(0, max_idx) as usize;
        let i_hi = (center + rad).clamp(0, max_idx) as usize;
        let sig2 = sigma * sigma;
        let mut raw = Vec::new();
        let mut sum = 0.0;
        for idx in i_lo..=i_hi {
            let pos = origin + h * (idx as f64);
            let w = (-0.5 * (coord - pos).powi(2) / sig2).exp();
            raw.push((pos, w));
            sum += w;
        }
        let w_norm: Vec<(f64, f64)> = raw.into_iter().map(|(pos, w)| (pos, w / sum)).collect();
        let mu: f64 = w_norm.iter().map(|(pos, wn)| pos * wn).sum();
        let var: f64 = w_norm.iter().map(|(pos, wn)| (pos - mu).powi(2) * wn).sum();
        let jets = w_norm
            .into_iter()
            .map(|(pos, wn)| {
                let dw = wn * (pos - mu) / sig2;
                let d2w = wn * ((pos - mu).powi(2) - var) / (sig2 * sig2);
                (wn, dw, d2w)
            })
            .collect();
        (i_lo, jets)
    }

    fn sample_test_mri_volume() -> MriVolume {
        let dims = [11, 11, 11];
        let origin = [-2.5, -2.5, -2.5];
        let h = 0.5;
        let mut vox = Vec::with_capacity(11 * 11 * 11);
        for iz in 0..dims[2] {
            let z = origin[2] + h * (iz as f64);
            for iy in 0..dims[1] {
                let y = origin[1] + h * (iy as f64);
                for ix in 0..dims[0] {
                    let x = origin[0] + h * (ix as f64);
                    vox.push(80.0 + 25.0 * (0.8 * x).sin() * (0.6 * y).cos() + 9.5 * x * y * z);
                }
            }
        }
        MriVolume::new(dims, origin, h, vox).expect("valid volume")
    }

    #[test]
    fn test_freesurfer_mri_ad_gradient_and_normal_force_machine_precision() {
        let dims = [11, 11, 11];
        let origin = [-2.5, -2.5, -2.5];
        let h = 0.5;
        let vol = sample_test_mri_volume();
        let pt = [0.23_f64, -0.37_f64, 0.41_f64];
        let raw_n = [1.2_f64, -0.7_f64, 0.9_f64];
        let sigma = 0.65;

        let ((grad_got, force_got), _metrics) = VerifiedEngine::run_verified(|| {
            let g = vol.smoothed_gradient(&pt, sigma);
            let f = normal_external_force_ad(&pt, &raw_n, &vol, sigma);
            (g, f)
        });

        let (x0, jx) = analytical_axis_jet2(pt[0], origin[0], h, dims[0], sigma);
        let (y0, jy) = analytical_axis_jet2(pt[1], origin[1], h, dims[1], sigma);
        let (z0, jz) = analytical_axis_jet2(pt[2], origin[2], h, dims[2], sigma);
        let mut grad_exact = [0.0_f64; 3];
        let mut hess_exact = [[0.0_f64; 3]; 3];
        for (kz, &(wz, dwz, d2wz)) in jz.iter().enumerate() {
            for (ky, &(wy, dwy, d2wy)) in jy.iter().enumerate() {
                for (kx, &(wx, dwx, d2wx)) in jx.iter().enumerate() {
                    let v = vol.voxel(x0 + kx, y0 + ky, z0 + kz);
                    grad_exact[0] += v * dwx * wy * wz;
                    grad_exact[1] += v * wx * dwy * wz;
                    grad_exact[2] += v * wx * wy * dwz;
                    hess_exact[0][0] += v * d2wx * wy * wz;
                    hess_exact[1][1] += v * wx * d2wy * wz;
                    hess_exact[2][2] += v * wx * wy * d2wz;
                    hess_exact[0][1] += v * dwx * dwy * wz;
                    hess_exact[0][2] += v * dwx * wy * dwz;
                    hess_exact[1][2] += v * wx * dwy * dwz;
                }
            }
        }
        hess_exact[1][0] = hess_exact[0][1];
        hess_exact[2][0] = hess_exact[0][2];
        hess_exact[2][1] = hess_exact[1][2];

        for a in 0..3 {
            assert!((grad_got[a] - grad_exact[a]).abs() < 1e-13);
        }
        let n_norm = (raw_n[0] * raw_n[0] + raw_n[1] * raw_n[1] + raw_n[2] * raw_n[2]).sqrt();
        let n_u = [raw_n[0] / n_norm, raw_n[1] / n_norm, raw_n[2] / n_norm];
        let mut neg_de_dn = 0.0;
        for r in 0..3 {
            let hn_r =
                hess_exact[r][0] * n_u[0] + hess_exact[r][1] * n_u[1] + hess_exact[r][2] * n_u[2];
            neg_de_dn += 2.0 * grad_exact[r] * hn_r;
        }
        for a in 0..3 {
            assert!((force_got[a] - neg_de_dn * n_u[a]).abs() < 1e-13);
        }
    }
}
