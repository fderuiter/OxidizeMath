//! Precomputed 2D discrete Laplacian/biharmonic Fourier symbol tables and circularly averaged
//! radial structure-factor shell accumulation for [`super::SemiImplicitSpectral2D`].

// [cite:spde]

use std::f64::consts::PI;

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;

use super::RadialSpectrumSummary;

pub(super) fn precompute_spectral_tables(
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>) {
    let total = width * height;
    let num_bins = (width.min(height) / 2) + 1;
    let mut q_sq = vec![0.0; total];
    let mut q_fourth = vec![0.0; total];
    let mut radial_bin_index = vec![0usize; total];
    let mut radial_bin_k = vec![0.0; num_bins];
    fill_spectral_tables_in_place(
        width,
        height,
        dx,
        dy,
        &mut q_sq,
        &mut q_fourth,
        &mut radial_bin_index,
        &mut radial_bin_k,
    );
    (q_sq, q_fourth, radial_bin_index, radial_bin_k)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fill_spectral_tables_in_place(
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    q_sq: &mut [f64],
    q_fourth: &mut [f64],
    radial_bin_index: &mut [usize],
    radial_bin_k: &mut [f64],
) {
    let inv_dx_sq = 1.0 / (dx * dx);
    let inv_dy_sq = 1.0 / (dy * dy);
    let dk_x = 2.0 * PI / ((width as f64) * dx);
    let dk_y = 2.0 * PI / ((height as f64) * dy);
    let dk_fundamental = dk_x.min(dk_y);
    let max_bin = radial_bin_k.len().saturating_sub(1);

    for (b, slot) in radial_bin_k.iter_mut().enumerate() {
        *slot = (b as f64) * dk_fundamental;
    }

    for kx in 0..width {
        let sx = (PI * (kx as f64) / (width as f64)).sin();
        let qx2 = 4.0 * sx * sx * inv_dx_sq;
        let signed_kx = if kx <= width / 2 {
            kx as isize
        } else {
            (kx as isize) - (width as isize)
        };
        let phys_kx = (signed_kx as f64) * dk_x;

        for ky in 0..height {
            let sy = (PI * (ky as f64) / (height as f64)).sin();
            let qy2 = 4.0 * sy * sy * inv_dy_sq;
            let q2 = qx2 + qy2;
            let idx = flatten_2d_index(ky, kx, height);
            q_sq[idx] = q2;
            q_fourth[idx] = q2 * q2;

            let signed_ky = if ky <= height / 2 {
                ky as isize
            } else {
                (ky as isize) - (height as isize)
            };
            let phys_ky = (signed_ky as f64) * dk_y;
            let k_mag = (phys_kx * phys_kx + phys_ky * phys_ky).sqrt();
            let bin = (k_mag / dk_fundamental).round() as usize;
            radial_bin_index[idx] = bin.min(max_bin);
        }
    }
}

pub(super) fn accumulate_radial_bins(
    transposed_spec: &[Complex<f64>],
    radial_bin_index: &[usize],
    radial_bin_k: &[f64],
    norm_wh: f64,
    k_bins_out: &mut [f64],
    s_k_out: &mut [f64],
    bin_counts: &mut [usize],
) -> RadialSpectrumSummary {
    let n_bins = radial_bin_k.len();
    k_bins_out.copy_from_slice(radial_bin_k);
    s_k_out.fill(0.0);
    bin_counts.fill(0);

    let inv_wh = 1.0 / norm_wh;
    for (spec_val, &bin) in transposed_spec.iter().zip(radial_bin_index.iter()) {
        if bin < n_bins {
            s_k_out[bin] += spec_val.norm_sqr() * inv_wh;
            bin_counts[bin] += 1;
        }
    }

    let mut k_max = 0.0;
    let mut s_max = 0.0;
    let mut weighted_k_sum = 0.0;
    let mut weight_sum = 0.0;

    for b in 0..n_bins {
        if bin_counts[b] > 0 {
            s_k_out[b] /= bin_counts[b] as f64;
        }
        if b > 0 {
            let s_val = s_k_out[b];
            let k_val = k_bins_out[b];
            if s_val > s_max {
                s_max = s_val;
                k_max = k_val;
            }
            weighted_k_sum += k_val * s_val;
            weight_sum += s_val;
        }
    }

    RadialSpectrumSummary {
        num_bins: n_bins,
        k_max,
        s_max,
        k_first_moment: if weight_sum > 0.0 {
            weighted_k_sum / weight_sum
        } else {
            0.0
        },
    }
}
