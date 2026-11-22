//! Concentric ring kernel synthesis for spectral Lenia convolution.

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use rustfft::Fft;
use std::sync::Arc;

/// Radial bump profile used for each concentric ring of a Lenia convolution kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeniaKernelCore {
    /// Smooth compact exponential bump $K_C(a) = \exp(4 - 1 / (a(1 - a)))$ from Chan (2019).
    #[default]
    ExponentialBump,
    /// Gaussian shell profile $K_C(a) = \exp(-(a - 0.5)^2 / (2 \cdot 0.15^2))$ truncated to $(0, 1)$.
    GaussianShell,
}

impl LeniaKernelCore {
    /// Evaluates the unnormalized radial bump $K_C(a)$ for $a \in [0, 1]$.
    #[verified_engine::verified]
    pub fn evaluate(self, local_r: f64) -> f64 {
        if local_r <= 1e-9 || local_r >= 1.0 - 1e-9 {
            return 0.0;
        }
        match self {
            Self::ExponentialBump => {
                let denom = local_r * (1.0 - local_r);
                (4.0 - 1.0 / denom).exp()
            }
            Self::GaussianShell => {
                let diff = (local_r - 0.5) / 0.15;
                (-0.5 * diff * diff).exp()
            }
        }
    }
}

/// Evaluates the multi-ring shell weight $K_S(r)$ for normalized radius $r = d / R$.
#[verified_engine::verified]
pub fn evaluate_shell_kernel(r: f64, ring_weights: &[f64], core: LeniaKernelCore) -> f64 {
    if !(0.0..1.0).contains(&r) || ring_weights.is_empty() {
        return 0.0;
    }
    let num_rings = ring_weights.len();
    let scaled = (num_rings as f64) * r;
    let ring_idx = (scaled.floor() as usize).min(num_rings - 1);
    let local_r = scaled - (ring_idx as f64);
    let weight = ring_weights[ring_idx].max(0.0);
    weight * core.evaluate(local_r)
}

/// Builds a wrap-around normalized spatial kernel on `(width, height)` centered at `(0, 0)`.
#[verified_engine::verified]
pub fn build_spatial_kernel(
    width: usize,
    height: usize,
    kernel_radius: usize,
    ring_weights: &[f64],
    core: LeniaKernelCore,
    out_kernel: &mut [f64],
) {
    let radius_f = (kernel_radius.max(1)) as f64;
    let half_w = width / 2;
    let half_h = height / 2;
    let mut sum = 0.0;

    for y in 0..height {
        let dy = if y <= half_h {
            y as f64
        } else {
            (y as f64) - (height as f64)
        };
        for x in 0..width {
            let dx = if x <= half_w {
                x as f64
            } else {
                (x as f64) - (width as f64)
            };
            let norm_r = dx.hypot(dy) / radius_f;
            let val = evaluate_shell_kernel(norm_r, ring_weights, core);
            let idx = flatten_2d_index(x, y, width);
            out_kernel[idx] = val;
            sum += val;
        }
    }

    if sum > 1e-15 {
        let inv_sum = 1.0 / sum;
        for v in out_kernel.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Transposes a contiguous `rows x cols` complex matrix into `cols x rows` layout in-place into `dst`.
#[verified_engine::verified]
pub(crate) fn transpose_complex(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        for c in 0..cols {
            let src_idx = flatten_2d_index(c, r, cols);
            let dst_idx = flatten_2d_index(r, c, rows);
            dst[dst_idx] = src[src_idx];
        }
    }
}

/// Transforms a spatial `(width, height)` real buffer into transposed `(width, height)` frequency space.
#[verified_engine::verified]
pub(crate) fn forward_fft_2d_transposed(
    real_input: &[f64],
    dims: (usize, usize),
    fft_row_fwd: &Arc<dyn Fft<f64>>,
    fft_col_fwd: &Arc<dyn Fft<f64>>,
    spatial_buf: &mut [Complex<f64>],
    transposed_buf: &mut [Complex<f64>],
    scratch_buf: &mut [Complex<f64>],
) {
    let (width, height) = dims;
    for (dst, &val) in spatial_buf.iter_mut().zip(real_input.iter()) {
        *dst = Complex::new(val, 0.0);
    }
    fft_row_fwd.process_with_scratch(spatial_buf, scratch_buf);
    transpose_complex(spatial_buf, transposed_buf, height, width);
    fft_col_fwd.process_with_scratch(transposed_buf, scratch_buf);
}
