//! Zero-allocation 2D Discrete Fourier Transform analyzer built on pre-planned `rustfft` passes.
//!
//! Pre-allocates complex work buffers, transpose storage, Hann window weights, and `rustfft`
//! scratch slices at initialization so [`SpectralAnalyzer2D::compute_spectrum`] executes with
//! zero dynamic heap allocations under `VerifiedAllocator::lock_allocations()`.

// [cite:advanced_linear_algebra]

mod helpers;

use std::sync::Arc;

use self::helpers::{
    build_hann_weights, compute_finite_mean, ensure_eq_math, ensure_math, insert_ranked_mode,
    insert_sorted_peak,
};
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

/// Ranked non-DC 2D spatial Fourier mode with complex phase from a centered spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpectralMode2D {
    /// Signed horizontal wavenumber index $k_x \in [-W/2, W/2)$.
    pub kx: isize,
    /// Signed vertical wavenumber index $k_y \in [-H/2, H/2)$.
    pub ky: isize,
    /// Radial wavenumber magnitude $\sqrt{k_x^2 + k_y^2}$.
    pub k_radial: f64,
    /// Normalized spectral magnitude $|F(k_x, k_y)| / (W \cdot H)$.
    pub magnitude: f64,
    /// Principal complex Fourier phase $\operatorname{atan2}(\Im F, \Re F) \in [-\pi, \pi]$.
    pub phase: f64,
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
    /// Fallible companion to [`new`](Self::new).
    pub fn try_new(
        width: usize,
        height: usize,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
        ensure_math(
            width >= 2 && height >= 2,
            "new: precondition or invariant violated",
        )?;
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
        debug_assert_eq!(hann_x.len(), width);
        debug_assert_eq!(hann_y.len(), height);

        Ok(Self {
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
        })
    }

    /// Plans forward 1D FFTs along rows and columns and pre-allocates all intermediate buffers.
    pub fn new(width: usize, height: usize) -> Self {
        Self::try_new(width, height).expect("new preconditions must hold")
    }

    /// Configures whether separable 2D Hann windowing is applied prior to the forward FFT.
    #[verified_engine::verified]
    pub fn set_hann_window(&mut self, enabled: bool) {
        self.use_hann_window = enabled;
        debug_assert_eq!(self.use_hann_window, enabled);
    }

    /// Spectrum width $W$.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        debug_assert!(self.width >= 2);
        self.width
    }

    /// Spectrum height $H$.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        debug_assert!(self.height >= 2);
        self.height
    }

    /// Centered (`fftshift`) normalized 2D magnitude spectrum from the latest pass.
    #[verified_engine::verified]
    pub fn spectrum(&self) -> &[f64] {
        debug_assert_eq!(self.magnitude_spectrum.len(), self.width * self.height);
        &self.magnitude_spectrum
    }

    /// Fallible companion to [`compute_spectrum`](Self::compute_spectrum).
    #[verified_engine::verified]
    pub fn try_compute_spectrum(
        &mut self,
        field: &[f64],
    ) -> core::result::Result<&[f64], math_commons::error::MathError> {
        let w = self.width;
        let h = self.height;
        ensure_eq_math(
            &(field.len()),
            &(w * h),
            "compute_spectrum: equality invariant violated",
        )?;

        self.load_windowed_input(field);
        self.row_fft
            .process_with_scratch(&mut self.spatial_complex, &mut self.scratch);
        self.transpose_rows_to_cols();
        self.col_fft
            .process_with_scratch(&mut self.transposed, &mut self.scratch);
        self.write_shifted_magnitudes();
        debug_assert_eq!(self.magnitude_spectrum.len(), w * h);
        Ok(&self.magnitude_spectrum)
    }

    /// Computes the `fftshift`-centered 2D magnitude spectrum of `field` with zero heap allocations.
    #[verified_engine::verified]
    pub fn compute_spectrum(&mut self, field: &[f64]) -> &[f64] {
        self.try_compute_spectrum(field)
            .expect("compute_spectrum preconditions must hold")
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
        debug_assert_eq!(self.spatial_complex.len(), w * h);
    }

    #[verified_engine::verified]
    fn transpose_rows_to_cols(&mut self) {
        let w = self.width;
        let h = self.height;
        debug_assert_eq!(self.transposed.len(), w * h);
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
        debug_assert!(inv_norm > 0.0);
    }

    /// Identifies the strongest non-DC peak in the centered 2D magnitude spectrum.
    #[verified_engine::verified]
    pub fn dominant_peak(&self) -> SpectralPeak2D {
        let mut top = [SpectralPeak2D::default(); 1];
        let count = self.top_peaks(&mut top);
        debug_assert!(count <= 1);
        top[0]
    }

    /// Populates `out` with up to `out.len()` strongest distinct non-DC upper-half-plane local
    /// maxima in descending `magnitude` order with zero heap allocations, returning the active count.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use std::f64::consts::TAU;
    /// use math_explorer::pure_math::analysis::pde::wave_2d::{SpectralAnalyzer2D, SpectralPeak2D};
    ///
    /// let n = 32;
    /// let mut field = vec![0.0; n * n];
    /// for y in 0..n {
    ///     for x in 0..n {
    ///         let mode_a = (TAU * 3.0 * (x as f64) / (n as f64)).cos()
    ///             * (TAU * 2.0 * (y as f64) / (n as f64)).cos();
    ///         let mode_b = 0.5 * (TAU * 6.0 * (x as f64) / (n as f64)).cos()
    ///             * (TAU * 5.0 * (y as f64) / (n as f64)).cos();
    ///         field[y * n + x] = mode_a + mode_b;
    ///     }
    /// }
    /// let mut fft = SpectralAnalyzer2D::new(n, n);
    /// fft.compute_spectrum(&field);
    /// let mut peaks = [SpectralPeak2D::default(); 4];
    /// let found = fft.top_peaks(&mut peaks);
    /// assert!(found >= 2);
    /// assert!(peaks[0].magnitude >= peaks[1].magnitude);
    /// ```
    #[verified_engine::verified]
    pub fn top_peaks(&self, out: &mut [SpectralPeak2D]) -> usize {
        out.fill(SpectralPeak2D::default());
        if out.is_empty() {
            return 0;
        }
        let w = self.width;
        let h = self.height;
        let cx = (w / 2) as isize;
        let cy = (h / 2) as isize;
        let mut count = 0_usize;

        for y in 0..h {
            let ky = (y as isize) - cy;
            for x in 0..w {
                let kx = (x as isize) - cx;
                if kx * kx + ky * ky <= 4 || ky < 0 || (ky == 0 && kx < 0) {
                    continue;
                }
                let mag = self.magnitude_spectrum[flatten_2d_index(x, y, w)];
                if mag <= 0.0 || !self.is_non_dc_local_max(x, y, mag) {
                    continue;
                }
                let candidate = SpectralPeak2D {
                    kx,
                    ky,
                    k_radial: ((kx * kx + ky * ky) as f64).sqrt(),
                    magnitude: mag,
                };
                count = insert_sorted_peak(out, count, candidate);
            }
        }
        debug_assert!(count <= out.len());
        count
    }

    #[verified_engine::verified]
    fn is_non_dc_local_max(&self, x: usize, y: usize, mag: f64) -> bool {
        let w = self.width;
        let h = self.height;
        let cx = (w / 2) as isize;
        let cy = (h / 2) as isize;
        let y_lo = y.saturating_sub(1);
        let y_hi = (y + 1).min(h - 1);
        let x_lo = x.saturating_sub(1);
        let x_hi = (x + 1).min(w - 1);

        for ny in y_lo..=y_hi {
            let nky = (ny as isize) - cy;
            for nx in x_lo..=x_hi {
                if nx == x && ny == y {
                    continue;
                }
                let nkx = (nx as isize) - cx;
                if nkx * nkx + nky * nky <= 4 {
                    continue;
                }
                let nmag = self.magnitude_spectrum[flatten_2d_index(nx, ny, w)];
                if nmag > mag || (nmag == mag && (ny, nx) < (y, x)) {
                    return false;
                }
            }
        }
        true
    }

    /// Extracts up to `out.len()` strongest upper-half-plane local-maximum 2D Fourier modes
    /// (sorted by descending `magnitude`) with complex phases and zero heap allocations.
    #[verified_engine::verified]
    pub fn extract_top_modes(&self, out: &mut [SpectralMode2D]) -> usize {
        out.fill(SpectralMode2D::default());
        if out.is_empty() {
            return 0;
        }
        let w = self.width;
        let h = self.height;
        let cx = (w / 2) as isize;
        let cy = (h / 2) as isize;
        let mut count = 0_usize;

        for sy in 0..h {
            let ky = (sy as isize) - cy;
            for sx in 0..w {
                let kx = (sx as isize) - cx;
                if kx * kx + ky * ky <= 1 || ky < 0 || (ky == 0 && kx <= 0) {
                    continue;
                }
                let mag = self.magnitude_spectrum[flatten_2d_index(sx, sy, w)];
                if !mag.is_finite() || mag <= 1.0e-12 || !self.is_upper_local_maximum(sx, sy, mag) {
                    continue;
                }
                let mode = SpectralMode2D {
                    kx,
                    ky,
                    k_radial: ((kx * kx + ky * ky) as f64).sqrt(),
                    magnitude: mag,
                    phase: self.shifted_bin_phase(sx, sy),
                };
                insert_ranked_mode(out, &mut count, mode);
            }
        }
        debug_assert!(count <= out.len());
        count
    }

    #[verified_engine::verified]
    fn is_upper_local_maximum(&self, sx: usize, sy: usize, mag: f64) -> bool {
        let w = self.width as isize;
        let h = self.height as isize;
        let cx = w / 2;
        let cy = h / 2;
        for dy in -1_isize..=1 {
            let ny = (sy as isize) + dy;
            if ny < 0 || ny >= h {
                continue;
            }
            for dx in -1_isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = (sx as isize) + dx;
                if nx < 0 || nx >= w {
                    continue;
                }
                let nkx = nx - cx;
                let nky = ny - cy;
                if nkx * nkx + nky * nky <= 1 {
                    continue;
                }
                let n_mag =
                    self.magnitude_spectrum[flatten_2d_index(nx as usize, ny as usize, self.width)];
                let precedes = (ny, nx) < (sy as isize, sx as isize);
                if (precedes && n_mag >= mag) || (!precedes && n_mag > mag) {
                    return false;
                }
            }
        }
        true
    }

    #[verified_engine::verified]
    fn shifted_bin_phase(&self, sx: usize, sy: usize) -> f64 {
        let w = self.width;
        let h = self.height;
        let fx = (sx + w - (w / 2)) % w;
        let fy = (sy + h - (h / 2)) % h;
        let coeff = self.transposed[flatten_2d_index(fy, fx, h)];
        if coeff.re.is_finite() && coeff.im.is_finite() {
            coeff.im.atan2(coeff.re)
        } else {
            0.0
        }
    }

    /// Fallible companion to [`compute_radial_profile`](Self::compute_radial_profile).
    #[verified_engine::verified]
    pub fn try_compute_radial_profile(
        &self,
        out_bins: &mut [f64],
        counts: &mut [usize],
    ) -> core::result::Result<(), math_commons::error::MathError> {
        ensure_eq_math(
            &(out_bins.len()),
            &(counts.len()),
            "compute_radial_profile: equality invariant violated",
        )?;
        out_bins.fill(0.0);
        counts.fill(0);
        if out_bins.is_empty() {
            return Ok(());
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
        debug_assert_eq!(out_bins.len(), counts.len());
        Ok(())
    }

    /// Accumulates azimuthally averaged 1D radial spectrum $S(|k|)$ into pre-allocated `out_bins` and `counts`.
    #[verified_engine::verified]
    pub fn compute_radial_profile(&self, out_bins: &mut [f64], counts: &mut [usize]) {
        self.try_compute_radial_profile(out_bins, counts)
            .expect("compute_radial_profile preconditions must hold");
    }
}
