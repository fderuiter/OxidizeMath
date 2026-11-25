//! Generic [`math_scalars::Scalar`] trilinear and cubic B-spline MRI voxel interpolation kernels.

use super::MriVolume;
use math_scalars::Scalar;

#[inline]
pub(super) fn clamp_grid_axis<S: Scalar>(
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
pub(super) fn bspline_axis_weights<S: Scalar>(
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
    pub(super) fn sample_bspline_scalar<S: Scalar>(&self, ref_pt: &[f64; 3], point: &[S; 3]) -> S {
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
}
