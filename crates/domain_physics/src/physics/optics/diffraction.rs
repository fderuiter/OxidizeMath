//! Zero-allocation 2D Helmholtz angular-spectrum diffraction propagator.
//!
//! Pre-plans forward and inverse 1D `rustfft` passes alongside transpose, scratch, and
//! longitudinal wavenumber buffers so [`AngularSpectrumPropagator::propagate_into`] and
//! [`AngularSpectrumPropagator::propagate_axial_carpet_into`] execute with zero heap
//! allocations inside `VerifiedEngine::run_verified`.

// [cite:graph_parameters_rust]

use std::f64::consts::PI;
use std::sync::Arc;

use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

#[path = "apertures.rs"]
mod apertures;
pub use apertures::*;

/// Computes the dimensionless Fresnel number $N_F = a^2 / (\lambda |z|)$.
#[must_use]
#[verified_engine::verified]
pub fn fresnel_number(aperture_radius: f64, wavelength: f64, distance_z: f64) -> f64 {
    let denom = wavelength * distance_z.abs();
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    (aperture_radius * aperture_radius) / denom
}

/// Computes the paraxial Talbot self-imaging length $z_T = 2 p^2 / \lambda$.
#[must_use]
#[verified_engine::verified]
pub fn talbot_length(period: f64, wavelength: f64) -> f64 {
    if wavelength <= 0.0 || !wavelength.is_finite() {
        return 0.0;
    }
    2.0 * period * period / wavelength
}

/// Computes the exact non-paraxial Rayleigh-Talbot length $z_{T,\text{exact}} = \lambda / (1 - \sqrt{1 - (\lambda / p)^2})$.
///
/// Falls back to [`talbot_length`] if `period <= wavelength`.
#[must_use]
#[verified_engine::verified]
pub fn rayleigh_talbot_length(period: f64, wavelength: f64) -> f64 {
    if wavelength <= 0.0 || period <= wavelength {
        return talbot_length(period, wavelength);
    }
    let ratio = wavelength / period;
    let denom = 1.0 - (1.0 - ratio * ratio).sqrt();
    if denom <= 1.0e-15 {
        talbot_length(period, wavelength)
    } else {
        wavelength / denom
    }
}

/// Integrates total optical power $P = \sum_{x, y} |U(x, y)|^2 \Delta x \Delta y$.
#[must_use]
#[verified_engine::verified]
pub fn optical_power(field: &[Complex<f64>], dx: f64, dy: f64) -> f64 {
    let mut sum = 0.0_f64;
    for c in field {
        sum += c.norm_sqr();
    }
    sum * dx * dy
}

/// Writes optical intensity $I(x, y) = |U(x, y)|^2$ into `out_intensity` without allocation.
#[verified_engine::verified]
pub fn field_intensity_into(field: &[Complex<f64>], out_intensity: &mut [f64]) {
    assert_eq!(field.len(), out_intensity.len(), "Intensity slice mismatch");
    for (dst, src) in out_intensity.iter_mut().zip(field.iter()) {
        *dst = src.norm_sqr();
    }
}

/// Writes wrapped wavefront phase $\arg U(x, y) \in (-\pi, \pi]$ into `out_phase` without allocation.
#[verified_engine::verified]
pub fn wrapped_phase_into(field: &[Complex<f64>], out_phase: &mut [f64]) {
    assert_eq!(field.len(), out_phase.len(), "Phase slice mismatch");
    for (dst, src) in out_phase.iter_mut().zip(field.iter()) {
        *dst = src.im.atan2(src.re);
    }
}

/// Pre-planned 2D Helmholtz angular-spectrum wave propagator with evanescent mode filtering.
pub struct AngularSpectrumPropagator {
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    wavelength: f64,
    evanescent_filter: bool,
    fwd_row: Arc<dyn Fft<f64>>,
    fwd_col: Arc<dyn Fft<f64>>,
    inv_row: Arc<dyn Fft<f64>>,
    inv_col: Arc<dyn Fft<f64>>,
    kz_transposed: Vec<f64>,
    prop_mask: Vec<u8>,
    row_buf: Vec<Complex<f64>>,
    col_buf: Vec<Complex<f64>>,
    spectrum_cache: Vec<Complex<f64>>,
    line_buf: Vec<Complex<f64>>,
    y_twiddle: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl std::fmt::Debug for AngularSpectrumPropagator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AngularSpectrumPropagator")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("dx", &self.dx)
            .field("dy", &self.dy)
            .field("wavelength", &self.wavelength)
            .field("evanescent_filter", &self.evanescent_filter)
            .finish()
    }
}

impl Clone for AngularSpectrumPropagator {
    fn clone(&self) -> Self {
        let mut cloned = Self::new(self.width, self.height, self.dx, self.dy, self.wavelength);
        cloned.evanescent_filter = self.evanescent_filter;
        cloned
    }
}

impl AngularSpectrumPropagator {
    /// Plans 2D `rustfft` passes and precomputes Helmholtz longitudinal wavenumbers $k_z(f_x, f_y)$.
    #[must_use]
    pub fn new(width: usize, height: usize, dx: f64, dy: f64, wavelength: f64) -> Self {
        assert!(width >= 2 && height >= 2, "Propagator grid must be >= 2x2");
        assert!(
            dx > 0.0 && dy > 0.0 && wavelength > 0.0,
            "Grid pitch and wavelength must be > 0"
        );
        let mut planner = FftPlanner::new();
        let fwd_row = planner.plan_fft_forward(width);
        let fwd_col = planner.plan_fft_forward(height);
        let inv_row = planner.plan_fft_inverse(width);
        let inv_col = planner.plan_fft_inverse(height);
        let scratch_len = fwd_row
            .get_inplace_scratch_len()
            .max(fwd_col.get_inplace_scratch_len())
            .max(inv_row.get_inplace_scratch_len())
            .max(inv_col.get_inplace_scratch_len())
            .max(1);
        let total = width * height;
        let mut prop = Self {
            width,
            height,
            dx,
            dy,
            wavelength,
            evanescent_filter: true,
            fwd_row,
            fwd_col,
            inv_row,
            inv_col,
            kz_transposed: vec![0.0; total],
            prop_mask: vec![1; total],
            row_buf: vec![Complex::new(0.0, 0.0); total],
            col_buf: vec![Complex::new(0.0, 0.0); total],
            spectrum_cache: vec![Complex::new(0.0, 0.0); total],
            line_buf: vec![Complex::new(0.0, 0.0); width],
            y_twiddle: vec![Complex::new(0.0, 0.0); height],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        };
        prop.rebuild_kz_table();
        prop
    }

    /// Grid width $W$.
    #[must_use]
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height $H$.
    #[must_use]
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Horizontal sample spacing $\Delta x$.
    #[must_use]
    #[verified_engine::verified]
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// Vertical sample spacing $\Delta y$.
    #[must_use]
    #[verified_engine::verified]
    pub fn dy(&self) -> f64 {
        self.dy
    }

    /// Optical wavelength $\lambda$.
    #[must_use]
    #[verified_engine::verified]
    pub fn wavelength(&self) -> f64 {
        self.wavelength
    }

    /// Enables or disables hard filtering ($H = 0$) of evanescent modes ($f_x^2 + f_y^2 > \lambda^{-2}$).
    #[verified_engine::verified]
    pub fn set_evanescent_filter(&mut self, enabled: bool) {
        self.evanescent_filter = enabled;
    }

    /// Updates `(dx, dy, wavelength)` and recomputes $k_z(f_x, f_y)$ in-place without heap allocation.
    #[verified_engine::verified]
    pub fn reconfigure(&mut self, dx: f64, dy: f64, wavelength: f64) {
        assert!(
            dx > 0.0 && dy > 0.0 && wavelength > 0.0,
            "Grid pitch and wavelength must be > 0"
        );
        self.dx = dx;
        self.dy = dy;
        self.wavelength = wavelength;
        self.rebuild_kz_table();
    }

    #[verified_engine::verified]
    fn rebuild_kz_table(&mut self) {
        let inv_lam2 = 1.0 / (self.wavelength * self.wavelength);
        let inv_lx = 1.0 / ((self.width as f64) * self.dx);
        let inv_ly = 1.0 / ((self.height as f64) * self.dy);

        for nx in 0..self.width {
            let sx = if nx < self.width / 2 {
                nx as isize
            } else {
                (nx as isize) - (self.width as isize)
            };
            let fx = (sx as f64) * inv_lx;
            for ny in 0..self.height {
                let sy = if ny < self.height / 2 {
                    ny as isize
                } else {
                    (ny as isize) - (self.height as isize)
                };
                let fy = (sy as f64) * inv_ly;
                let idx = flatten_2d_index(ny, nx, self.height);
                let fz2 = inv_lam2 - fx * fx - fy * fy;
                if fz2 >= 0.0 {
                    self.prop_mask[idx] = 1;
                    self.kz_transposed[idx] = 2.0 * PI * fz2.sqrt();
                } else {
                    self.prop_mask[idx] = 0;
                    self.kz_transposed[idx] = 2.0 * PI * (-fz2).sqrt();
                }
            }
        }
    }

    /// Propagates complex field `aperture` across axial distance `distance_z` into `out_field`
    /// with zero dynamic heap allocations.
    #[verified_engine::verified]
    pub fn propagate_into(
        &mut self,
        aperture: &[Complex<f64>],
        distance_z: f64,
        out_field: &mut [Complex<f64>],
    ) {
        let total = self.width * self.height;
        assert_eq!(aperture.len(), total, "Aperture slice length mismatch");
        assert_eq!(out_field.len(), total, "Output field slice length mismatch");

        self.forward_to_col_spectrum(aperture);
        self.apply_helmholtz_transfer(distance_z);
        self.inverse_from_col_spectrum(out_field);
    }

    #[verified_engine::verified]
    fn forward_to_col_spectrum(&mut self, input: &[Complex<f64>]) {
        self.row_buf.copy_from_slice(input);
        self.fwd_row
            .process_with_scratch(&mut self.row_buf, &mut self.scratch);
        transpose_grid(&self.row_buf, &mut self.col_buf, self.width, self.height);
        self.fwd_col
            .process_with_scratch(&mut self.col_buf, &mut self.scratch);
    }

    #[verified_engine::verified]
    fn apply_helmholtz_transfer(&mut self, distance_z: f64) {
        for ((bin, &kz), &mask) in self
            .col_buf
            .iter_mut()
            .zip(self.kz_transposed.iter())
            .zip(self.prop_mask.iter())
        {
            if mask != 0 {
                let phase = kz * distance_z;
                *bin *= Complex::new(phase.cos(), phase.sin());
            } else if self.evanescent_filter || distance_z < 0.0 {
                *bin = Complex::new(0.0, 0.0);
            } else {
                *bin *= (-kz * distance_z).exp();
            }
        }
    }

    #[verified_engine::verified]
    fn inverse_from_col_spectrum(&mut self, out_field: &mut [Complex<f64>]) {
        self.inv_col
            .process_with_scratch(&mut self.col_buf, &mut self.scratch);
        transpose_grid(&self.col_buf, out_field, self.height, self.width);
        self.inv_row
            .process_with_scratch(out_field, &mut self.scratch);
        let scale = 1.0 / ((self.width * self.height) as f64);
        for c in out_field.iter_mut() {
            *c *= scale;
        }
    }

    /// Computes a centered unitary 2D forward or inverse FFT into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn centered_unitary_fft2d_into(
        &mut self,
        input: &[Complex<f64>],
        inverse: bool,
        out: &mut [Complex<f64>],
    ) {
        let total = self.width * self.height;
        assert_eq!(input.len(), total);
        assert_eq!(out.len(), total);

        shift_quadrants_into(input, &mut self.row_buf, self.width, self.height, true);
        if inverse {
            self.inv_row
                .process_with_scratch(&mut self.row_buf, &mut self.scratch);
            transpose_grid(&self.row_buf, &mut self.col_buf, self.width, self.height);
            self.inv_col
                .process_with_scratch(&mut self.col_buf, &mut self.scratch);
        } else {
            self.fwd_row
                .process_with_scratch(&mut self.row_buf, &mut self.scratch);
            transpose_grid(&self.row_buf, &mut self.col_buf, self.width, self.height);
            self.fwd_col
                .process_with_scratch(&mut self.col_buf, &mut self.scratch);
        }
        transpose_grid(&self.col_buf, &mut self.row_buf, self.height, self.width);
        shift_quadrants_into(&self.row_buf, out, self.width, self.height, false);
        let norm = 1.0 / ((total as f64).sqrt());
        for c in out.iter_mut() {
            *c *= norm;
        }
    }

    /// Evaluates an axial `(x, z)` intensity carpet along transverse slice `y_row` across
    /// `z_steps` planes spanning `[z_min, z_max]` with zero heap allocations.
    #[verified_engine::verified]
    pub fn propagate_axial_carpet_into(
        &mut self,
        aperture: &[Complex<f64>],
        y_row: usize,
        z_min: f64,
        z_max: f64,
        z_steps: usize,
        out_carpet: &mut [f64],
    ) {
        assert_eq!(aperture.len(), self.width * self.height);
        assert_eq!(out_carpet.len(), self.width * z_steps);
        if z_steps == 0 {
            return;
        }
        let clamped_y = y_row.min(self.height - 1);
        self.forward_to_col_spectrum(aperture);
        self.spectrum_cache.copy_from_slice(&self.col_buf);
        self.populate_y_twiddle(clamped_y);

        let inv_norm = 1.0 / ((self.width * self.height) as f64);
        let dz = if z_steps > 1 {
            (z_max - z_min) / ((z_steps - 1) as f64)
        } else {
            0.0
        };
        for z_idx in 0..z_steps {
            let z = z_min + (z_idx as f64) * dz;
            self.contract_slice_spectrum_at_z(z);
            self.inv_row
                .process_with_scratch(&mut self.line_buf, &mut self.scratch);
            for x in 0..self.width {
                let amp = self.line_buf[x] * inv_norm;
                out_carpet[flatten_2d_index(x, z_idx, self.width)] = amp.norm_sqr();
            }
        }
    }

    #[verified_engine::verified]
    fn populate_y_twiddle(&mut self, y_row: usize) {
        let h_f64 = self.height as f64;
        for ny in 0..self.height {
            let angle = 2.0 * PI * ((ny * y_row) as f64) / h_f64;
            self.y_twiddle[ny] = Complex::new(angle.cos(), angle.sin());
        }
    }

    #[verified_engine::verified]
    fn contract_slice_spectrum_at_z(&mut self, distance_z: f64) {
        for nx in 0..self.width {
            let mut acc = Complex::new(0.0, 0.0);
            for ny in 0..self.height {
                let idx = flatten_2d_index(ny, nx, self.height);
                if self.prop_mask[idx] != 0 {
                    let phase = self.kz_transposed[idx] * distance_z;
                    let transfer = Complex::new(phase.cos(), phase.sin());
                    acc += self.spectrum_cache[idx] * transfer * self.y_twiddle[ny];
                }
            }
            self.line_buf[nx] = acc;
        }
    }
}

#[verified_engine::verified]
fn transpose_grid(src: &[Complex<f64>], dst: &mut [Complex<f64>], src_w: usize, src_h: usize) {
    for y in 0..src_h {
        for x in 0..src_w {
            dst[flatten_2d_index(y, x, src_h)] = src[flatten_2d_index(x, y, src_w)];
        }
    }
}

#[verified_engine::verified]
fn shift_quadrants_into(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    width: usize,
    height: usize,
    inverse_shift: bool,
) {
    let sx = if inverse_shift {
        width - (width / 2)
    } else {
        width / 2
    };
    let sy = if inverse_shift {
        height - (height / 2)
    } else {
        height / 2
    };
    for y in 0..height {
        let dy = (y + sy) % height;
        for x in 0..width {
            let dx = (x + sx) % width;
            dst[flatten_2d_index(dx, dy, width)] = src[flatten_2d_index(x, y, width)];
        }
    }
}
