//! 2D Fourier signal simulation (`simulate_signal_2d`), cached forward k-space spectra,
//! inverse DFT reconstruction (`inverse_dft_2d`), PSNR metrics, and `ColorImage` rendering.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{KSPACE_CENTER, KSPACE_DIM, KSPACE_LEN};
use eframe::egui::{Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::mri::reconstruction::{inverse_dft_2d, simulate_signal_2d};
use nalgebra::DMatrix;
use num_complex::Complex;

/// Cached forward 2D FFT spectrum of the current spatial phantom so tactile k-space
/// brush strokes only execute the masked inverse 2D FFT (`inverse_dft_2d`).
#[derive(Clone, Debug)]
pub struct ForwardKSpaceCache {
    /// Unshifted complex `64x64` forward 2D FFT output from `simulate_signal_2d`.
    pub raw_kspace: DMatrix<Complex<f64>>,
    /// Centered `64x64` log-magnitude spectrum $\log_{10}(1 + |S(k)|)$.
    pub centered_log_kspace: Vec<f64>,
    /// Exact Parseval cumulative keyhole PSNR curve `[cutoff_radius_bins, psnr_db]`.
    pub radial_psnr_curve: Vec<[f64; 2]>,
}

/// Complete output of a `64x64` masked inverse 2D FFT reconstruction pass.
#[derive(Clone, Debug)]
pub struct ReconOutput {
    /// Centered `64x64` log-magnitude full k-space spectrum $\log_{10}(1 + |S(k)|)$.
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
    /// Number of active (`true`) k-space bins out of `4096`.
    pub sampled_bins: usize,
    /// Percentage of `4096` k-space bins sampled (`0.0..=100.0`).
    pub sampling_pct: f64,
    /// Radial mean log-power curve `[radius_bins, mean_log10_power]`.
    pub radial_power_curve: Vec<[f64; 2]>,
    /// Radial sampling fraction curve `[radius_bins, sampled_fraction]`.
    pub radial_sampling_curve: Vec<[f64; 2]>,
    /// Cumulative keyhole PSNR curve `[cutoff_radius_bins, psnr_db]`.
    pub radial_psnr_curve: Vec<[f64; 2]>,
}

#[inline]
const fn shift_idx(idx: usize) -> usize {
    (idx + KSPACE_CENTER) % KSPACE_DIM
}

/// Computes and caches the forward 2D FFT (`simulate_signal_2d`), centered log spectrum,
/// and Parseval cumulative keyhole PSNR curve for `ground_truth`.
#[must_use]
pub fn compute_forward_kspace(ground_truth: &[f64]) -> ForwardKSpaceCache {
    assert_eq!(ground_truth.len(), KSPACE_LEN);
    let density = DMatrix::from_fn(KSPACE_DIM, KSPACE_DIM, |r, c| {
        Complex::new(ground_truth[flatten_2d_index(c, r, KSPACE_DIM)], 0.0)
    });
    let raw_kspace = simulate_signal_2d(&density);
    let mut centered_log_kspace = vec![0.0_f64; KSPACE_LEN];
    let max_r = KSPACE_CENTER;
    let mut ring_energy = vec![0.0_f64; max_r + 2];
    let scale = KSPACE_LEN as f64;

    for disp_r in 0..KSPACE_DIM {
        let dr = (disp_r as f64) - (KSPACE_CENTER as f64);
        let fft_r = shift_idx(disp_r);
        for disp_c in 0..KSPACE_DIM {
            let dc = (disp_c as f64) - (KSPACE_CENTER as f64);
            let fft_c = shift_idx(disp_c);
            let sample = raw_kspace[(fft_r, fft_c)];
            centered_log_kspace[flatten_2d_index(disp_c, disp_r, KSPACE_DIM)] =
                (1.0 + sample.norm()).log10();
            let rad = (dr.hypot(dc).ceil() as usize).min(max_r + 1);
            ring_energy[rad] += sample.norm_sqr() / scale;
        }
    }

    let gt_energy: f64 = ground_truth.iter().map(|v| v * v).sum();
    let peak_sq = ground_truth
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1.0)
        .powi(2);
    let mut radial_psnr_curve = Vec::with_capacity(max_r + 1);
    let mut captured_energy = 0.0_f64;
    for (r, &energy) in ring_energy.iter().enumerate().take(max_r + 1) {
        captured_energy += energy;
        let residual_sq = (gt_energy - captured_energy).max(0.0);
        let mse = residual_sq / scale;
        let psnr = if mse <= 1e-12 {
            99.9
        } else {
            (10.0 * (peak_sq / mse).log10()).clamp(0.0, 99.9)
        };
        radial_psnr_curve.push([r as f64, psnr]);
    }

    ForwardKSpaceCache {
        raw_kspace,
        centered_log_kspace,
        radial_psnr_curve,
    }
}

fn compute_psnr_and_errors(
    ground_truth: &[f64],
    reconstructed: &[f64],
) -> (Vec<f64>, f64, f64, f64) {
    let mut abs_err = Vec::with_capacity(KSPACE_LEN);
    let mut sq_err_sum = 0.0_f64;
    let mut gt_sq_sum = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut max_err = 0.0_f64;

    for (&gt, &rec) in ground_truth.iter().zip(reconstructed.iter()) {
        let diff = (rec - gt).abs();
        abs_err.push(diff);
        sq_err_sum += diff * diff;
        gt_sq_sum += gt * gt;
        peak = peak.max(gt.abs());
        max_err = max_err.max(diff);
    }

    let mse = sq_err_sum / (KSPACE_LEN as f64);
    let ref_peak = peak.max(1.0);
    let psnr_db = if mse <= 1e-12 {
        99.9
    } else {
        (10.0 * ((ref_peak * ref_peak) / mse).log10()).clamp(0.0, 99.9)
    };
    let norm_l2 = if gt_sq_sum > 1e-12 {
        (sq_err_sum / gt_sq_sum).sqrt()
    } else {
        sq_err_sum.sqrt()
    };
    (abs_err, psnr_db, norm_l2, max_err)
}

fn compute_radial_sampling_and_power(
    centered_log: &[f64],
    mask: &[bool],
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let max_r = KSPACE_CENTER;
    let mut power_sums = vec![0.0_f64; max_r + 1];
    let mut sampled_counts = vec![0_usize; max_r + 1];
    let mut bin_counts = vec![0_usize; max_r + 1];

    for row in 0..KSPACE_DIM {
        let dr = (row as f64) - (KSPACE_CENTER as f64);
        for col in 0..KSPACE_DIM {
            let dc = (col as f64) - (KSPACE_CENTER as f64);
            let r_bin = dr.hypot(dc).round() as usize;
            if r_bin <= max_r {
                let idx = flatten_2d_index(col, row, KSPACE_DIM);
                power_sums[r_bin] += centered_log[idx];
                if mask[idx] {
                    sampled_counts[r_bin] += 1;
                }
                bin_counts[r_bin] += 1;
            }
        }
    }

    let mut power_curve = Vec::with_capacity(max_r + 1);
    let mut sampling_curve = Vec::with_capacity(max_r + 1);
    for r in 0..=max_r {
        let denom = bin_counts[r].max(1) as f64;
        power_curve.push([r as f64, power_sums[r] / denom]);
        sampling_curve.push([r as f64, (sampled_counts[r] as f64) / denom]);
    }
    (power_curve, sampling_curve)
}

/// Applies `mask` to a cached [`ForwardKSpaceCache`] and executes the `64x64` inverse 2D FFT (`inverse_dft_2d`).
#[must_use]
pub fn reconstruct_from_forward_cache(
    ground_truth: &[f64],
    forward: &ForwardKSpaceCache,
    mask: &[bool],
) -> ReconOutput {
    assert_eq!(ground_truth.len(), KSPACE_LEN);
    assert_eq!(mask.len(), KSPACE_LEN);

    let mut masked_raw = DMatrix::zeros(KSPACE_DIM, KSPACE_DIM);
    let mut sampled_bins = 0_usize;
    for disp_r in 0..KSPACE_DIM {
        let fft_r = shift_idx(disp_r);
        for disp_c in 0..KSPACE_DIM {
            if mask[flatten_2d_index(disp_c, disp_r, KSPACE_DIM)] {
                let fft_c = shift_idx(disp_c);
                masked_raw[(fft_r, fft_c)] = forward.raw_kspace[(fft_r, fft_c)];
                sampled_bins += 1;
            }
        }
    }

    let inv = inverse_dft_2d(&masked_raw);
    let scale = KSPACE_LEN as f64;
    let mut reconstructed = vec![0.0_f64; KSPACE_LEN];
    for r in 0..KSPACE_DIM {
        for c in 0..KSPACE_DIM {
            reconstructed[flatten_2d_index(c, r, KSPACE_DIM)] = (inv[(r, c)] / scale).norm();
        }
    }

    let (abs_error, psnr_db, normalized_l2_error, max_abs_error) =
        compute_psnr_and_errors(ground_truth, &reconstructed);
    let (radial_power_curve, radial_sampling_curve) =
        compute_radial_sampling_and_power(&forward.centered_log_kspace, mask);

    ReconOutput {
        centered_log_kspace: forward.centered_log_kspace.clone(),
        reconstructed,
        abs_error,
        psnr_db,
        normalized_l2_error,
        max_abs_error,
        sampled_bins,
        sampling_pct: 100.0 * (sampled_bins as f64) / scale,
        radial_power_curve,
        radial_sampling_curve,
        radial_psnr_curve: forward.radial_psnr_curve.clone(),
    }
}

/// Renders a `64x64` anatomical magnitude image (`Ground-Truth` or `Reconstructed`) in clinical grayscale.
#[must_use]
pub fn render_grayscale_panel(values: &[f64], high_contrast: bool) -> ColorImage {
    let peak = values
        .iter()
        .fold(0.0_f64, |acc, &v| if v.is_finite() { acc.max(v) } else { acc })
        .max(1e-6);
    let floor = if high_contrast { 24.0 } else { 8.0 };
    let mut pixels = Vec::with_capacity(KSPACE_LEN);
    for &v in values {
        let norm = if v.is_finite() {
            (v / peak).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let lum = (floor + norm * (255.0 - floor)).round() as u8;
        pixels.push(Color32::from_rgb(lum, lum, lum));
    }
    ColorImage {
        size: [KSPACE_DIM, KSPACE_DIM],
        pixels,
        ..Default::default()
    }
}

/// Renders the centered `64x64` log-magnitude k-space spectrum highlighting sampled vs erased bins.
#[must_use]
pub fn render_kspace_panel(log_power: &[f64], mask: &[bool], high_contrast: bool) -> ColorImage {
    let max_log = log_power
        .iter()
        .fold(0.0_f64, |acc, &v| if v.is_finite() { acc.max(v) } else { acc })
        .max(1e-6);
    let mut pixels = Vec::with_capacity(KSPACE_LEN);
    for (&lp, &sampled) in log_power.iter().zip(mask.iter()) {
        let norm = if lp.is_finite() {
            (lp / max_log).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if sampled {
            let rgb = if high_contrast {
                heatmap_color_high_contrast(norm)
            } else {
                let r = (35.0 + 215.0 * norm.powf(1.4)) as u8;
                let g = (65.0 + 185.0 * norm) as u8;
                let b = (115.0 + 140.0 * norm.sqrt()) as u8;
                [r, g, b]
            };
            pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        } else {
            let ghost = (10.0 + 26.0 * norm) as u8;
            pixels.push(Color32::from_rgb(ghost, ghost / 2, ghost / 2));
        }
    }
    ColorImage {
        size: [KSPACE_DIM, KSPACE_DIM],
        pixels,
        ..Default::default()
    }
}

/// Renders the `64x64` absolute reconstruction error heatmap.
#[must_use]
pub fn render_error_panel(abs_error: &[f64], high_contrast: bool) -> ColorImage {
    let scale = abs_error
        .iter()
        .fold(0.0_f64, |acc, &v| if v.is_finite() { acc.max(v) } else { acc })
        .max(0.05);
    let mut pixels = Vec::with_capacity(KSPACE_LEN);
    for &err in abs_error {
        let norm = if err.is_finite() {
            (err / scale).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let rgb = if high_contrast {
            heatmap_color_high_contrast(norm)
        } else {
            heatmap_color(norm)
        };
        pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    }
    ColorImage {
        size: [KSPACE_DIM, KSPACE_DIM],
        pixels,
        ..Default::default()
    }
}

/// Stitches all four `64x64` panels into a `2x2` composite `ColorImage` for PNG/SVG/GIF figure export.
#[must_use]
pub fn compose_quad_export_image(panels: [&ColorImage; 4], high_contrast: bool) -> ColorImage {
    let gap = 4_usize;
    let side = KSPACE_DIM * 2 + gap;
    let theme = if high_contrast {
        crate::app::ThemeMode::HighContrast
    } else {
        crate::app::ThemeMode::Dark
    };
    let border = crate::app::SemanticPalette::for_theme(theme).surface_fill;
    let mut pixels = vec![border; side * side];
    let offsets = [
        (0, 0),
        (0, KSPACE_DIM + gap),
        (KSPACE_DIM + gap, 0),
        (KSPACE_DIM + gap, KSPACE_DIM + gap),
    ];
    for (panel, (row_off, col_off)) in panels.into_iter().zip(offsets) {
        for r in 0..KSPACE_DIM {
            for c in 0..KSPACE_DIM {
                let dst = flatten_2d_index(col_off + c, row_off + r, side);
                pixels[dst] = panel.pixels[flatten_2d_index(c, r, KSPACE_DIM)];
            }
        }
    }
    ColorImage {
        size: [side, side],
        pixels,
        ..Default::default()
    }
}
