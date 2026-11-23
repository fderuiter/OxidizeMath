//! Zero-allocation $2\times 2$ plaquette phase-winding vortex detector and Parseval invariant
//! diagnostics for [`super::GrossPitaevskiiSolver2D`].

// [cite:quantum_mechanics]

use std::f64::consts::{PI, TAU};

use math_commons::math_kernel::types::flatten_2d_index;

use super::GrossPitaevskiiSolver2D;
use super::config::{GpeInvariants2D, VortexChargeSummary, VortexCore2D};

/// Wraps a phase difference $\Delta \theta$ onto the principal interval $(-\pi, \pi]$.
///
/// # Example
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::gross_pitaevskii::wrap_phase_difference;
/// use std::f64::consts::PI;
///
/// assert!((wrap_phase_difference(1.5 * PI) - (-0.5 * PI)).abs() < 1e-12);
/// assert!((wrap_phase_difference(-1.5 * PI) - (0.5 * PI)).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn wrap_phase_difference(dtheta: f64) -> f64 {
    (dtheta + PI).rem_euclid(TAU) - PI
}

impl GrossPitaevskiiSolver2D {
    /// Refreshes cached spatial density $|\psi(x, y)|^2$ and principal phase $\arg\psi(x, y) \in [-\pi, \pi]$
    /// buffers in-place with zero heap allocations, returning the peak condensate density.
    #[verified_engine::verified]
    pub fn refresh_density_and_phase(&mut self) -> f64 {
        let mut max_rho = 0.0_f64;
        for i in 0..self.psi.len() {
            let z = self.psi[i];
            let rho = z.norm_sqr();
            self.density_buf[i] = rho;
            self.phase_buf[i] = if rho > 1.0e-28 { z.im.atan2(z.re) } else { 0.0 };
            if rho.is_finite() && rho > max_rho {
                max_rho = rho;
            }
        }
        max_rho
    }

    /// Scans all interior $2\times 2$ grid plaquettes inside the condensate bulk
    /// (`avg_density >= density_cutoff_ratio * max_density`), summing wrapped phase differences
    /// around each counter-clockwise loop to classify quantized $+1$ (`+2*pi`) and $-1$ (`-2*pi`) vortices.
    #[verified_engine::verified]
    pub fn detect_vortices(&mut self, density_cutoff_ratio: f64) -> VortexChargeSummary {
        let max_rho = self.refresh_density_and_phase();
        self.detected_vortices.clear();
        if max_rho <= 1.0e-18 {
            return VortexChargeSummary::default();
        }

        let ratio = density_cutoff_ratio.clamp(1.0e-4, 0.5);
        let rho_threshold = ratio * max_rho;
        let w = self.width;
        let h = self.height;
        let mut pos_count = 0_usize;
        let mut neg_count = 0_usize;
        let mut net_q = 0_i32;

        for y in 0..(h - 1) {
            for x in 0..(w - 1) {
                let Some((charge, cx, cy)) = self.inspect_plaquette(x, y, rho_threshold) else {
                    continue;
                };
                if charge > 0 {
                    pos_count += 1;
                } else {
                    neg_count += 1;
                }
                net_q += charge;
                if self.detected_vortices.len() < self.detected_vortices.capacity() {
                    self.detected_vortices.push(VortexCore2D {
                        x: cx,
                        y: cy,
                        grid_x: x,
                        grid_y: y,
                        charge,
                        circulation: (charge as f64) * TAU,
                    });
                }
            }
        }

        VortexChargeSummary {
            positive_count: pos_count,
            negative_count: neg_count,
            net_charge: net_q,
            net_circulation: (net_q as f64) * TAU,
        }
    }

    /// Computes conserved Gross-Pitaevskii norm $M$, momentum $(P_x, P_y)$, energy $H$, and chemical
    /// potential $\mu$ with zero heap allocations using spectral Parseval identities.
    #[verified_engine::verified]
    pub fn compute_invariants(&mut self, interaction_g: f64) -> GpeInvariants2D {
        let w = self.width;
        let h = self.height;
        let cell_area = self.dx * self.dy;

        let mut sum_norm = 0.0_f64;
        let mut sum_pot = 0.0_f64;
        let mut sum_int = 0.0_f64;
        for idx in 0..(w * h) {
            let rho = self.psi[idx].norm_sqr();
            sum_norm += rho;
            sum_pot += self.potential[idx] * rho;
            sum_int += rho * rho;
        }

        self.kspace.copy_from_slice(&self.psi);
        self.fft_row_fwd
            .process_with_scratch(&mut self.kspace, &mut self.fft_scratch);
        self.transpose_kspace_to_cols();
        self.fft_col_fwd
            .process_with_scratch(&mut self.transposed, &mut self.fft_scratch);

        let mut spec_kin = 0.0_f64;
        let mut spec_px = 0.0_f64;
        let mut spec_py = 0.0_f64;
        for x in 0..w {
            let kx = self.kx_deriv[x];
            let kx2 = self.kx_sq[x];
            for y in 0..h {
                let idx = flatten_2d_index(y, x, h);
                let mag_sq = self.transposed[idx].norm_sqr();
                spec_kin += 0.5 * (kx2 + self.ky_sq[y]) * mag_sq;
                spec_px += kx * mag_sq;
                spec_py += self.ky_deriv[y] * mag_sq;
            }
        }

        let parseval = cell_area / ((w * h) as f64);
        let e_kin = spec_kin * parseval;
        let e_pot = sum_pot * cell_area;
        let e_int_full = interaction_g * sum_int * cell_area;
        let norm = sum_norm * cell_area;
        let energy = e_kin + e_pot + 0.5 * e_int_full;
        let mu = if norm > 1.0e-15 {
            (e_kin + e_pot + e_int_full) / norm
        } else {
            0.0
        };

        GpeInvariants2D {
            norm,
            momentum_x: spec_px * parseval,
            momentum_y: spec_py * parseval,
            energy,
            chemical_potential: mu,
        }
    }

    fn inspect_plaquette(&self, x: usize, y: usize, rho_threshold: f64) -> Option<(i32, f64, f64)> {
        let w = self.width;
        // Reject perimeter wrap artifacts within 2 cells of periodic box boundary.
        if x < 2 || y < 2 || x + 3 >= w || y + 3 >= self.height {
            return None;
        }
        let i00 = flatten_2d_index(x, y, w);
        let i10 = flatten_2d_index(x + 1, y, w);
        let i11 = flatten_2d_index(x + 1, y + 1, w);
        let i01 = flatten_2d_index(x, y + 1, w);

        let d00 = self.density_buf[i00];
        let d10 = self.density_buf[i10];
        let d11 = self.density_buf[i11];
        let d01 = self.density_buf[i01];
        let min_corner = d00.min(d10).min(d11).min(d01);
        let mut neighborhood_peak = d00.max(d10).max(d11).max(d01);
        for ny in (y - 1)..=(y + 2) {
            for nx in (x - 1)..=(x + 2) {
                neighborhood_peak =
                    neighborhood_peak.max(self.density_buf[flatten_2d_index(nx, ny, w)]);
            }
        }
        if neighborhood_peak < rho_threshold || min_corner <= 1.0e-12 * rho_threshold {
            return None;
        }

        let p00 = self.phase_buf[i00];
        let p10 = self.phase_buf[i10];
        let p11 = self.phase_buf[i11];
        let p01 = self.phase_buf[i01];
        let loop_sum = wrap_phase_difference(p10 - p00)
            + wrap_phase_difference(p11 - p10)
            + wrap_phase_difference(p01 - p11)
            + wrap_phase_difference(p00 - p01);
        let charge = (loop_sum / TAU).round() as i32;
        if charge == 0 {
            return None;
        }

        let w00 = 1.0 / (d00.sqrt() + 1.0e-6);
        let w10 = 1.0 / (d10.sqrt() + 1.0e-6);
        let w11 = 1.0 / (d11.sqrt() + 1.0e-6);
        let w01 = 1.0 / (d01.sqrt() + 1.0e-6);
        let w_sum = w00 + w10 + w11 + w01;
        let x0 = self.x_coords[x];
        let x1 = self.x_coords[x + 1];
        let y0 = self.y_coords[y];
        let y1 = self.y_coords[y + 1];
        let cx = ((w00 + w01) * x0 + (w10 + w11) * x1) / w_sum;
        let cy = ((w00 + w10) * y0 + (w01 + w11) * y1) / w_sum;
        Some((charge, cx, cy))
    }
}
