//! Zero-allocation 2D Discrete Fourier Transform analyzer built on pre-planned `rustfft` passes.
//!
//! Pre-allocates complex work buffers, transpose storage, Hann window weights, and `rustfft`
//! scratch slices at initialization so [`SpectralAnalyzer2D::compute_spectrum`] executes with
//! zero dynamic heap allocations under `VerifiedAllocator::lock_allocations()`.

// [cite:advanced_linear_algebra]

use std::f64::consts::PI;
use std::sync::Arc;

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Dominant non-DC spatial-frequency peak in a centered 2D magnitude spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpectralPeak2D {
    /// Signed horizontal wavenumber index $k_x \in [-W/2, W/2)$.
    pub kx: isize,
    /// Signed vertical wavenumber index $k_y \in [-H/2, H/2)$.
    pub ky: isize,
    /// Radial wavenumber magnitude $\sqrt{k_x^2 + k_y^2}$.
    pub k_radial: f64,
    /// Normalized spectral magnitude $|F(k_x, k_y)| / (W \cdot H)$.
    pub magnitude: f64,
}

/// Pre-planned 2D FFT spectral analyzer with reusable transpose and scratch buffers.
#[derive(Clone)]
pub struct SpectralAnalyzer2D {
    width: usize,
    height: usize,
    row_fft: Arc<dyn Fft<f64>>,
    col_fft: Arc<dyn Fft<f64>>,
    spatial_complex: Vec<Complex<f64>>,
    transposed: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    hann_x: Vec<f64>,
    hann_y: Vec<f64>,
    magnitude_spectrum: Vec<f64>,
    use_hann_window: bool,
}

impl std::fmt::Debug for SpectralAnalyzer2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralAnalyzer2D")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("use_hann_window", &self.use_hann_window)
            .finish()
    }
}

impl SpectralAnalyzer2D {
    /// Plans forward 1D FFTs along rows and columns and pre-allocates all intermediate buffers.
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width >= 2 && height >= 2, "2D FFT dimensions must be >= 2");
        let mut planner = FftPlanner::new();
        let row_fft = planner.plan_fft_forward(width);
        let col_fft = planner.plan_fft_forward(height);
        let scratch_len = row_fft
            .get_inplace_scratch_len()
            .max(col_fft.get_inplace_scratch_len())
            .max(1);
        let len = width * height;
        let hann_x = build_hann_weights(width);
        let hann_y = build_hann_weights(height);

        Self {
            width,
            height,
            row_fft,
            col_fft,
            spatial_complex: vec![Complex::new(0.0, 0.0); len],
            transposed: vec![Complex::new(0.0, 0.0); len],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            hann_x,
            hann_y,
            magnitude_spectrum: vec![0.0; len],
            use_hann_window: true,
        }
    }

    /// Configures whether separable 2D Hann windowing is applied prior to the forward FFT.
    #[verified_engine::verified]
    pub fn set_hann_window(&mut self, enabled: bool) {
        self.use_hann_window = enabled;
    }

    /// Spectrum width $W$.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Spectrum height $H$.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Centered (`fftshift`) normalized 2D magnitude spectrum from the latest pass.
    #[verified_engine::verified]
    pub fn spectrum(&self) -> &[f64] {
        &self.magnitude_spectrum
    }

    /// Computes the `fftshift`-centered 2D magnitude spectrum of `field` with zero heap allocations.
    #[verified_engine::verified]
    pub fn compute_spectrum(&mut self, field: &[f64]) -> &[f64] {
        let w = self.width;
        let h = self.height;
        assert_eq!(field.len(), w * h, "Input field size must match W*H");

        self.load_windowed_input(field);
        self.row_fft
            .process_with_scratch(&mut self.spatial_complex, &mut self.scratch);
        self.transpose_rows_to_cols();
        self.col_fft
            .process_with_scratch(&mut self.transposed, &mut self.scratch);
        self.write_shifted_magnitudes();

        &self.magnitude_spectrum
    }

    #[verified_engine::verified]
    fn load_windowed_input(&mut self, field: &[f64]) {
        let w = self.width;
        let h = self.height;
        let mean = compute_finite_mean(field);
        for y in 0..h {
            let wy = if self.use_hann_window {
                self.hann_y[y]
            } else {
                1.0
            };
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                let wx = if self.use_hann_window {
                    self.hann_x[x]
                } else {
                    1.0
                };
                let val = if field[idx].is_finite() {
                    (field[idx] - mean) * wx * wy
                } else {
                    0.0
                };
                self.spatial_complex[idx] = Complex::new(val, 0.0);
            }
        }
    }

    #[verified_engine::verified]
    fn transpose_rows_to_cols(&mut self) {
        let w = self.width;
        let h = self.height;
        for y in 0..h {
            for x in 0..w {
                let src_idx = flatten_2d_index(x, y, w);
                let dst_idx = flatten_2d_index(y, x, h);
                self.transposed[dst_idx] = self.spatial_complex[src_idx];
            }
        }
    }

    #[verified_engine::verified]
    fn write_shifted_magnitudes(&mut self) {
        let w = self.width;
        let h = self.height;
        let half_w = w / 2;
        let half_h = h / 2;
        let inv_norm = 1.0 / ((w * h) as f64);
        for x in 0..w {
            let sx = (x + half_w) % w;
            for y in 0..h {
                let sy = (y + half_h) % h;
                let src_idx = flatten_2d_index(y, x, h);
                let dst_idx = flatten_2d_index(sx, sy, w);
                self.magnitude_spectrum[dst_idx] = self.transposed[src_idx].norm() * inv_norm;
            }
        }
    }

    /// Identifies the strongest non-DC peak in the centered 2D magnitude spectrum.
    #[verified_engine::verified]
    pub fn dominant_peak(&self) -> SpectralPeak2D {
        let w = self.width;
        let h = self.height;
        let cx = (w / 2) as isize;
        let cy = (h / 2) as isize;
        let mut best = SpectralPeak2D::default();

        for y in 0..h {
            let ky = (y as isize) - cy;
            for x in 0..w {
                let kx = (x as isize) - cx;
                if kx * kx + ky * ky <= 4 {
                    continue;
                }
                if ky < 0 || (ky == 0 && kx < 0) {
                    continue;
                }
                let mag = self.magnitude_spectrum[flatten_2d_index(x, y, w)];
                if mag > best.magnitude {
                    let kr = ((kx * kx + ky * ky) as f64).sqrt();
                    best = SpectralPeak2D {
                        kx,
                        ky,
                        k_radial: kr,
                        magnitude: mag,
                    };
                }
            }
        }
        best
    }

    /// Accumulates azimuthally averaged 1D radial spectrum $S(|k|)$ into pre-allocated `out_bins` and `counts`.
    #[verified_engine::verified]
    pub fn compute_radial_profile(&self, out_bins: &mut [f64], counts: &mut [usize]) {
        assert_eq!(out_bins.len(), counts.len());
        out_bins.fill(0.0);
        counts.fill(0);
        if out_bins.is_empty() {
            return;
        }
        let w = self.width;
        let h = self.height;
        let cx = (w / 2) as f64;
        let cy = (h / 2) as f64;
        let max_bin = out_bins.len() - 1;
        for y in 0..h {
            let dy = (y as f64) - cy;
            for x in 0..w {
                let dx = (x as f64) - cx;
                let kr = (dx * dx + dy * dy).sqrt().round() as usize;
                if kr <= max_bin {
                    out_bins[kr] += self.magnitude_spectrum[flatten_2d_index(x, y, w)];
                    counts[kr] += 1;
                }
            }
        }
        for (bin, &cnt) in out_bins.iter_mut().zip(counts.iter()) {
            if cnt > 0 {
                *bin /= cnt as f64;
            }
        }
    }
}

#[verified_engine::verified]
fn compute_finite_mean(field: &[f64]) -> f64 {
    if field.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for &v in field {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count > 0 { sum / (count as f64) } else { 0.0 }
}

#[verified_engine::verified]
fn build_hann_weights(n: usize) -> Vec<f64> {
    assert!(n >= 2, "Window length must be >= 2");
    let denom = (n - 1) as f64;
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let w = 0.5 * (1.0 - (2.0 * PI * (i as f64) / denom).cos());
        weights.push(w);
    }
    weights
}
