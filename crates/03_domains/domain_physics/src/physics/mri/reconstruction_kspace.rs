//! Zero-allocation masked 2D k-space Fourier reconstruction workspace, Parseval radial PSNR,
//! and spatial error evaluators.

use super::{Fft2dWorkspace, simulate_signal_2d};
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::DMatrix;
use num_complex::Complex;

/// Cached forward 2D FFT spectrum of a square spatial phantom.
#[derive(Clone, Debug)]
pub struct ForwardKSpaceCache {
    /// Square grid side length $N$.
    pub dim: usize,
    /// Unshifted complex $N \times N$ forward 2D FFT output from [`simulate_signal_2d`].
    pub raw_kspace: DMatrix<Complex<f64>>,
    /// Centered $N \times N$ log-magnitude spectrum $\log_{10}(1 + |S(k)|)$.
    pub centered_log_kspace: Vec<f64>,
    /// Exact Parseval cumulative keyhole PSNR curve `[cutoff_radius_bins, psnr_db]`.
    pub radial_psnr_curve: Vec<[f64; 2]>,
}

/// Complete output of a square masked inverse 2D FFT reconstruction pass.
#[derive(Clone, Debug)]
pub struct ReconOutput {
    /// Centered $N \times N$ log-magnitude full k-space spectrum $\log_{10}(1 + |S(k)|)$.
    pub centered_log_kspace: Vec<f64>,
    /// Reconstructed spatial magnitude $|\hat{\rho}(x, y)|$.
    pub reconstructed: Vec<f64>,
    /// Pixel-wise absolute reconstruction error $||\hat{\rho}(x, y)| - \rho(x, y)|$.
    pub abs_error: Vec<f64>,
    /// Peak Signal-to-Noise Ratio in decibels (`0.0..=99.9` dB).
    pub psnr_db: f64,
    /// Relative $L_2$ norm error $\|\hat{\rho} - \rho\|_2 / \|\rho\|_2$.
    pub normalized_l2_error: f64,
    /// Maximum pixel-wise absolute error.
    pub max_abs_error: f64,
    /// Number of active (`true`) k-space bins.
    pub sampled_bins: usize,
    /// Percentage of k-space bins sampled (`0.0..=100.0`).
    pub sampling_pct: f64,
    /// Radial mean log-power curve `[radius_bins, mean_log10_power]`.
    pub radial_power_curve: Vec<[f64; 2]>,
    /// Radial sampling fraction curve `[radius_bins, sampled_fraction]`.
    pub radial_sampling_curve: Vec<[f64; 2]>,
    /// Cumulative keyhole PSNR curve `[cutoff_radius_bins, psnr_db]`.
    pub radial_psnr_curve: Vec<[f64; 2]>,
}

impl ReconOutput {
    /// Preallocates all output vectors for a `dim x dim` k-space scanner grid.
    #[must_use]
    pub fn with_dim(dim: usize) -> Self {
        let len = dim.saturating_mul(dim);
        let max_r = dim / 2;
        Self {
            centered_log_kspace: vec![0.0; len],
            reconstructed: vec![0.0; len],
            abs_error: vec![0.0; len],
            psnr_db: 0.0,
            normalized_l2_error: 0.0,
            max_abs_error: 0.0,
            sampled_bins: 0,
            sampling_pct: 0.0,
            radial_power_curve: vec![[0.0, 0.0]; max_r + 1],
            radial_sampling_curve: vec![[0.0, 0.0]; max_r + 1],
            radial_psnr_curve: vec![[0.0, 0.0]; max_r + 1],
        }
    }
}

#[inline]
#[verified_engine::verified]
const fn shift_idx(idx: usize, dim: usize) -> usize {
    (idx + dim / 2) % dim
}

#[inline]
#[verified_engine::verified]
fn infer_square_dim(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let root = (len as f64).sqrt().round() as usize;
    if root.saturating_mul(root) == len {
        root
    } else {
        0
    }
}

/// Computes and caches the forward 2D FFT (`simulate_signal_2d`), centered log spectrum,
/// and Parseval cumulative keyhole PSNR curve for a square `ground_truth` slice.
#[must_use]
#[verified_engine::verified]
pub fn compute_forward_kspace(ground_truth: &[f64]) -> ForwardKSpaceCache {
    let dim = infer_square_dim(ground_truth.len());
    if dim == 0 {
        return ForwardKSpaceCache {
            dim: 0,
            raw_kspace: DMatrix::zeros(0, 0),
            centered_log_kspace: Vec::new(),
            radial_psnr_curve: Vec::new(),
        };
    }
    let len = dim * dim;
    let center = dim / 2;
    let density = DMatrix::from_fn(dim, dim, |r, c| {
        Complex::new(ground_truth[flatten_2d_index(c, r, dim)], 0.0)
    });
    let raw_kspace = simulate_signal_2d(&density);
    let mut centered_log_kspace = vec![0.0_f64; len];
    let mut ring_energy = vec![0.0_f64; center + 2];
    let scale = len as f64;

    for disp_r in 0..dim {
        let dr = (disp_r as f64) - (center as f64);
        let fft_r = shift_idx(disp_r, dim);
        for disp_c in 0..dim {
            let dc = (disp_c as f64) - (center as f64);
            let fft_c = shift_idx(disp_c, dim);
            let sample = raw_kspace[(fft_r, fft_c)];
            centered_log_kspace[flatten_2d_index(disp_c, disp_r, dim)] =
                (1.0 + sample.norm()).log10();
            let rad = (dr.hypot(dc).ceil() as usize).min(center + 1);
            ring_energy[rad] += sample.norm_sqr() / scale;
        }
    }

    let gt_energy: f64 = ground_truth.iter().map(|v| v * v).sum();
    let peak_sq = ground_truth
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1.0)
        .powi(2);
    let mut radial_psnr_curve = Vec::with_capacity(center + 1);
    let mut captured_energy = 0.0_f64;
    for (r, &energy) in ring_energy.iter().enumerate().take(center + 1) {
        captured_energy += energy;
        let mse = (gt_energy - captured_energy).max(0.0) / scale;
        let psnr = if mse <= 1e-12 {
            99.9
        } else {
            (10.0 * (peak_sq / mse).log10()).clamp(0.0, 99.9)
        };
        radial_psnr_curve.push([r as f64, psnr]);
    }

    ForwardKSpaceCache {
        dim,
        raw_kspace,
        centered_log_kspace,
        radial_psnr_curve,
    }
}

/// Preallocated workspace for zero-allocation masked 2D k-space reconstruction.
///
/// Reuses `masked_raw` and `inv_buf` complex matrices alongside radial accumulator vectors so
/// [`Self::reconstruct_from_forward_cache_into`] performs zero `DMatrix` or `Vec` allocations
/// per call inside `VerifiedEngine::run_verified`.
pub struct KSpaceReconWorkspace {
    dim: usize,
    fft: Fft2dWorkspace,
    masked_raw: DMatrix<Complex<f64>>,
    inv_buf: DMatrix<Complex<f64>>,
    power_sums: Vec<f64>,
    sampled_counts: Vec<usize>,
    bin_counts: Vec<usize>,
}

impl KSpaceReconWorkspace {
    /// Creates a preallocated k-space reconstruction workspace for `dim x dim` images.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        let max_r = dim / 2;
        Self {
            dim,
            fft: Fft2dWorkspace::new(dim, dim),
            masked_raw: DMatrix::zeros(dim, dim),
            inv_buf: DMatrix::zeros(dim, dim),
            power_sums: vec![0.0; max_r + 1],
            sampled_counts: vec![0; max_r + 1],
            bin_counts: vec![0; max_r + 1],
        }
    }

    /// Reallocates internal scratch buffers only if `dim` differs from the current grid size.
    pub fn ensure_dim(&mut self, dim: usize) {
        if self.dim != dim {
            *self = Self::new(dim);
        }
    }

    /// Executes masked inverse 2D FFT reconstruction into preallocated `out` with zero heap allocations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_physics::__doctest_facade as math_explorer;
    /// use math_explorer::physics::mri::reconstruction::{
    ///     KSpaceReconWorkspace, ReconOutput, compute_forward_kspace,
    /// };
    ///
    /// let gt = vec![1.0_f64; 16 * 16];
    /// let fwd = compute_forward_kspace(&gt);
    /// let mask = vec![true; 16 * 16];
    /// let mut ws = KSpaceReconWorkspace::new(16);
    /// let mut out = ReconOutput::with_dim(16);
    /// ws.reconstruct_from_forward_cache_into(&gt, &fwd, &mask, &mut out);
    /// assert!(out.psnr_db >= 99.0);
    /// ```
    #[verified_engine::verified]
    pub fn reconstruct_from_forward_cache_into(
        &mut self,
        ground_truth: &[f64],
        forward: &ForwardKSpaceCache,
        mask: &[bool],
        out: &mut ReconOutput,
    ) {
        let dim = forward.dim;
        let len = dim.saturating_mul(dim);
        if dim == 0
            || self.dim != dim
            || ground_truth.len() != len
            || mask.len() != len
            || out.reconstructed.len() != len
            || out.abs_error.len() != len
            || out.centered_log_kspace.len() != len
        {
            return;
        }
        let sampled_bins = self.populate_masked_kspace(forward, mask);
        self.fft
            .reconstruct_normalized_2d_into(&self.masked_raw, &mut self.inv_buf);
        for r in 0..dim {
            for c in 0..dim {
                out.reconstructed[flatten_2d_index(c, r, dim)] = self.inv_buf[(r, c)].norm();
            }
        }
        compute_psnr_and_errors_into(ground_truth, out);
        self.compute_radial_curves_into(&forward.centered_log_kspace, mask, out);
        out.centered_log_kspace
            .copy_from_slice(&forward.centered_log_kspace);
        if out.radial_psnr_curve.len() == forward.radial_psnr_curve.len() {
            out.radial_psnr_curve
                .copy_from_slice(&forward.radial_psnr_curve);
        }
        out.sampled_bins = sampled_bins;
        out.sampling_pct = 100.0 * (sampled_bins as f64) / (len as f64);
    }

    #[verified_engine::verified]
    fn populate_masked_kspace(&mut self, forward: &ForwardKSpaceCache, mask: &[bool]) -> usize {
        let dim = self.dim;
        self.masked_raw.fill(Complex::new(0.0, 0.0));
        let mut sampled_bins = 0_usize;
        for disp_r in 0..dim {
            let fft_r = shift_idx(disp_r, dim);
            for disp_c in 0..dim {
                if mask[flatten_2d_index(disp_c, disp_r, dim)] {
                    let fft_c = shift_idx(disp_c, dim);
                    self.masked_raw[(fft_r, fft_c)] = forward.raw_kspace[(fft_r, fft_c)];
                    sampled_bins += 1;
                }
            }
        }
        sampled_bins
    }

    #[verified_engine::verified]
    fn compute_radial_curves_into(
        &mut self,
        centered_log: &[f64],
        mask: &[bool],
        out: &mut ReconOutput,
    ) {
        let dim = self.dim;
        let max_r = dim / 2;
        self.power_sums.fill(0.0);
        self.sampled_counts.fill(0);
        self.bin_counts.fill(0);

        for row in 0..dim {
            let dr = (row as f64) - (max_r as f64);
            for col in 0..dim {
                let dc = (col as f64) - (max_r as f64);
                let r_bin = dr.hypot(dc).round() as usize;
                if r_bin <= max_r {
                    let idx = flatten_2d_index(col, row, dim);
                    self.power_sums[r_bin] += centered_log[idx];
                    if mask[idx] {
                        self.sampled_counts[r_bin] += 1;
                    }
                    self.bin_counts[r_bin] += 1;
                }
            }
        }
        if out.radial_power_curve.len() == max_r + 1 && out.radial_sampling_curve.len() == max_r + 1
        {
            for r in 0..=max_r {
                let denom = self.bin_counts[r].max(1) as f64;
                out.radial_power_curve[r] = [r as f64, self.power_sums[r] / denom];
                out.radial_sampling_curve[r] = [r as f64, (self.sampled_counts[r] as f64) / denom];
            }
        }
    }
}

#[verified_engine::verified]
fn compute_psnr_and_errors_into(ground_truth: &[f64], out: &mut ReconOutput) {
    let mut sq_err_sum = 0.0_f64;
    let mut gt_sq_sum = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut max_err = 0.0_f64;

    for ((&gt, &rec), slot) in ground_truth
        .iter()
        .zip(out.reconstructed.iter())
        .zip(out.abs_error.iter_mut())
    {
        let diff = (rec - gt).abs();
        *slot = diff;
        sq_err_sum += diff * diff;
        gt_sq_sum += gt * gt;
        peak = peak.max(gt.abs());
        max_err = max_err.max(diff);
    }

    let mse = sq_err_sum / (ground_truth.len().max(1) as f64);
    let ref_peak = peak.max(1.0);
    out.psnr_db = if mse <= 1e-12 {
        99.9
    } else {
        (10.0 * ((ref_peak * ref_peak) / mse).log10()).clamp(0.0, 99.9)
    };
    out.normalized_l2_error = if gt_sq_sum > 1e-12 {
        (sq_err_sum / gt_sq_sum).sqrt()
    } else {
        sq_err_sum.sqrt()
    };
    out.max_abs_error = max_err;
}

/// Allocating convenience wrapper around [`KSpaceReconWorkspace::reconstruct_from_forward_cache_into`].
#[must_use]
pub fn reconstruct_from_forward_cache(
    ground_truth: &[f64],
    forward: &ForwardKSpaceCache,
    mask: &[bool],
    workspace: &mut KSpaceReconWorkspace,
) -> ReconOutput {
    workspace.ensure_dim(forward.dim);
    let mut out = ReconOutput::with_dim(forward.dim);
    workspace.reconstruct_from_forward_cache_into(ground_truth, forward, mask, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_kspace_recon_workspace_zero_alloc_in_verified_engine() {
        let dim = 64;
        let len = dim * dim;
        let mut gt = vec![0.0_f64; len];
        for r in 16..48 {
            for c in 20..44 {
                gt[flatten_2d_index(c, r, dim)] = 1.0;
            }
        }
        let fwd = compute_forward_kspace(&gt);
        let mask = vec![true; len];
        let mut ws = KSpaceReconWorkspace::new(dim);
        let mut out = ReconOutput::with_dim(dim);

        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        VerifiedEngine::run_verified(|| {
            ws.reconstruct_from_forward_cache_into(&gt, &fwd, &mask, &mut out);
        });
        assert!(out.psnr_db >= 99.0);
        assert!(out.normalized_l2_error < 1e-12);
        assert_eq!(out.sampled_bins, len);
    }
}
