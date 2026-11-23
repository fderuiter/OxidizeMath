//! Pre-planned zero-allocation 2D `rustfft` Bragg diffraction and rotational symmetry analyzer.

// [cite:solid_state]

use crate::error::SolidStateError;
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::Vector2;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::sync::Arc;

/// Evaluates the exact kinematic structure factor $S(\mathbf{k}) = \frac{1}{N}\sum_{j=1}^N e^{-i\mathbf{k}\cdot\mathbf{r}_j}$.
#[must_use]
#[verified_engine::verified]
pub fn kinematic_structure_factor(sites: &[[f64; 2]], k: Vector2<f64>) -> Complex<f64> {
    if sites.is_empty() {
        return Complex::new(0.0, 0.0);
    }
    let mut sum = Complex::new(0.0, 0.0);
    for &[x, y] in sites {
        let (s, c) = (-(k.x * x + k.y * y)).sin_cos();
        sum += Complex::new(c, s);
    }
    sum / (sites.len() as f64)
}

/// Dominant non-DC Bragg reflection extracted from the 2D diffraction intensity map.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BraggPeak2D {
    /// Sub-pixel refined reciprocal wavevector $(k_x, k_y)$ in $\text{length}^{-1}$.
    pub k: [f64; 2],
    /// Normalized 2D FFT intensity $I(\mathbf{k}) / I_{\max} \in [0, 1]$.
    pub normalized_intensity: f64,
    /// Exact kinematic structure-factor amplitude $|S(\mathbf{k})|$.
    pub structure_factor_amp: f64,
    /// Structure-factor phase $\arg S(\mathbf{k})$ in radians.
    pub structure_factor_phase: f64,
}

/// Rotational Pearson correlation profile across crystallographic and forbidden orders.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RotationalSymmetryReport {
    /// Two-fold centrosymmetric Friedel correlation $C_2$.
    pub c2: f64,
    /// Rotational correlations $[C_4, C_5, C_6, C_7, C_8, C_9, C_{10}, C_{11}, C_{12}]$.
    pub c4_to_c12: [f64; 9],
    /// Reciprocal-space grid spacing $\Delta k$ ($\text{rad} / \text{length}$) per FFT pixel.
    pub delta_k: f64,
    /// Peak non-DC intensity observed in the shifted spectrum.
    pub max_non_dc_intensity: f64,
}

impl Default for RotationalSymmetryReport {
    fn default() -> Self {
        Self {
            c2: 0.0,
            c4_to_c12: [0.0; 9],
            delta_k: 1.0,
            max_non_dc_intensity: 0.0,
        }
    }
}

impl RotationalSymmetryReport {
    /// Returns the rotational Pearson correlation $C_n$ for $n = 2$ or $n \in 4..=12$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn correlation(&self, order: usize) -> f64 {
        if order == 2 {
            self.c2
        } else if order >= 4 && order <= 12 {
            self.c4_to_c12[order - 4]
        } else {
            0.0
        }
    }
}

/// Pre-planned zero-allocation 2D `rustfft` Bragg diffraction and symmetry correlation workspace.
pub struct BraggDiffractionWorkspace {
    size: usize,
    fft_plan: Arc<dyn Fft<f64>>,
    spatial_buf: Vec<Complex<f64>>,
    transpose_buf: Vec<Complex<f64>>,
    fft_scratch: Vec<Complex<f64>>,
    intensity_shifted: Vec<f64>,
    peaks: Vec<BraggPeak2D>,
    last_report: RotationalSymmetryReport,
}

impl BraggDiffractionWorkspace {
    /// Creates a pre-allocated $N \times N$ 2D `rustfft` diffraction workspace (standard $N = 256$).
    pub fn new(size: usize) -> Result<Self, SolidStateError> {
        if size < 32 || !size.is_power_of_two() {
            return Err(SolidStateError::InvalidParameter(
                "FFT grid size must be a power of two >= 32".into(),
            ));
        }
        let mut planner = FftPlanner::new();
        let fft_plan = planner.plan_fft_forward(size);
        let scratch_len = fft_plan.get_inplace_scratch_len();
        let total = size * size;
        Ok(Self {
            size,
            fft_plan,
            spatial_buf: vec![Complex::new(0.0, 0.0); total],
            transpose_buf: vec![Complex::new(0.0, 0.0); total],
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            intensity_shifted: vec![0.0; total],
            peaks: Vec::with_capacity(32),
            last_report: RotationalSymmetryReport::default(),
        })
    }

    /// Grid side length $N$ (e.g., `256` for a `256x256` Bragg map).
    #[must_use]
    #[verified_engine::verified]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Centered (`fftshift`) $N \times N$ Bragg intensity map $|F(\mathbf{k})|^2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn intensity_map(&self) -> &[f64] {
        &self.intensity_shifted
    }

    /// Dominant non-DC Bragg peaks with exact kinematic structure factors from the latest pass.
    #[must_use]
    #[verified_engine::verified]
    pub fn peaks(&self) -> &[BraggPeak2D] {
        &self.peaks
    }

    /// Latest rotational symmetry report ($C_2$ and $C_4 \dots C_{12}$).
    #[must_use]
    #[verified_engine::verified]
    pub const fn report(&self) -> RotationalSymmetryReport {
        self.last_report
    }

    /// Executes Gaussian site deposition, 2D `rustfft`, `fftshift`, $C_4 \dots C_{12}$, and peak extraction with zero heap allocations.
    #[verified_engine::verified]
    pub fn compute_diffraction_into(
        &mut self,
        sites: &[[f64; 2]],
        edge_length: f64,
        aperture_radius: f64,
    ) -> Result<RotationalSymmetryReport, SolidStateError> {
        if !edge_length.is_finite()
            || edge_length <= 0.0
            || !aperture_radius.is_finite()
            || aperture_radius <= 0.0
        {
            return Err(SolidStateError::InvalidParameter(
                "edge_length and aperture_radius must be finite and positive".into(),
            ));
        }
        let n_f = self.size as f64;
        let target_r = aperture_radius.max(27.5 * edge_length);
        let px_per_unit = (0.42 * n_f / target_r).max(3.8 / edge_length);
        let eff_radius = aperture_radius.min(0.42 * n_f / px_per_unit);
        let delta_k = 2.0 * PI * px_per_unit / n_f;

        self.deposit_sites(sites, px_per_unit, eff_radius);
        self.execute_2d_fft_and_shift();

        let raw_max = self.max_non_dc_intensity();
        self.extract_peaks(sites, delta_k, raw_max);
        self.apply_detector_psf();

        let max_non_dc = self.max_non_dc_intensity();
        let c2 = self.rotational_correlation(2);
        let mut c4_to_c12 = [0.0_f64; 9];
        for order in 4..=12 {
            c4_to_c12[order - 4] = self.rotational_correlation(order);
        }

        let report = RotationalSymmetryReport {
            c2,
            c4_to_c12,
            delta_k,
            max_non_dc_intensity: max_non_dc,
        };
        self.last_report = report;
        Ok(report)
    }

    #[verified_engine::verified]
    fn deposit_sites(&mut self, sites: &[[f64; 2]], px_per_unit: f64, eff_radius: f64) {
        self.spatial_buf.fill(Complex::new(0.0, 0.0));
        let n = self.size as isize;
        let center = 0.5 * (self.size as f64);
        let r_sq_max = eff_radius * eff_radius;
        let inv_r = 1.0 / eff_radius.max(1e-12);
        let inv_two_atom_sq = 1.0 / (2.0 * 1.35_f64 * 1.35_f64);

        for &[x, y] in sites {
            let r2 = x * x + y * y;
            if r2 > r_sq_max {
                continue;
            }
            let env = 0.5 * (1.0 + (PI * r2.sqrt() * inv_r).cos());
            let (gx, gy) = (center + x * px_per_unit, center + y * px_per_unit);
            let (ix0, iy0) = (gx.round() as isize, gy.round() as isize);
            for dy in -4_isize..=4 {
                let py = iy0 + dy;
                if py < 0 || py >= n {
                    continue;
                }
                let ddy = (py as f64) - gy;
                for dx in -4_isize..=4 {
                    let px = ix0 + dx;
                    if px >= 0 && px < n {
                        let ddx = (px as f64) - gx;
                        let d2 = ddx * ddx + ddy * ddy;
                        if d2 <= 16.0 {
                            let w = env * (-d2 * inv_two_atom_sq).exp();
                            let idx = flatten_2d_index(px as usize, py as usize, self.size);
                            self.spatial_buf[idx].re += w;
                        }
                    }
                }
            }
        }
    }

    #[verified_engine::verified]
    fn apply_detector_psf(&mut self) {
        const KERNEL: [f64; 7] = [
            0.056_188, 0.118_349, 0.198_412, 0.254_102, 0.198_412, 0.118_349, 0.056_188,
        ];
        let n = self.size;
        let max_i = (n - 1) as isize;
        for y in 0..n {
            for x in 0..n {
                let mut acc = 0.0;
                for (k, &wk) in KERNEL.iter().enumerate() {
                    let sx = ((x as isize) + (k as isize) - 3).clamp(0, max_i) as usize;
                    acc += wk * self.intensity_shifted[flatten_2d_index(sx, y, n)];
                }
                self.spatial_buf[flatten_2d_index(x, y, n)].re = acc;
            }
        }
        for y in 0..n {
            for x in 0..n {
                let mut acc = 0.0;
                for (k, &wk) in KERNEL.iter().enumerate() {
                    let sy = ((y as isize) + (k as isize) - 3).clamp(0, max_i) as usize;
                    acc += wk * self.spatial_buf[flatten_2d_index(x, sy, n)].re;
                }
                self.intensity_shifted[flatten_2d_index(x, y, n)] = acc;
            }
        }
    }

    #[verified_engine::verified]
    fn execute_2d_fft_and_shift(&mut self) {
        let n = self.size;
        for row in self.spatial_buf.chunks_exact_mut(n) {
            self.fft_plan
                .process_with_scratch(row, &mut self.fft_scratch);
        }
        for y in 0..n {
            for x in 0..n {
                let dst = flatten_2d_index(y, x, n);
                self.transpose_buf[dst] = self.spatial_buf[flatten_2d_index(x, y, n)];
            }
        }
        for col in self.transpose_buf.chunks_exact_mut(n) {
            self.fft_plan
                .process_with_scratch(col, &mut self.fft_scratch);
        }
        let half = n / 2;
        for ky in 0..n {
            let sy = (ky + half) % n;
            for kx in 0..n {
                let sx = (kx + half) % n;
                let dst = flatten_2d_index(sx, sy, n);
                self.intensity_shifted[dst] =
                    self.transpose_buf[flatten_2d_index(ky, kx, n)].norm_sqr();
            }
        }
    }

    #[verified_engine::verified]
    fn max_non_dc_intensity(&self) -> f64 {
        let n = self.size;
        let c = (n / 2) as isize;
        let mut max_val = 0.0_f64;
        for y in 0..n {
            let dy = (y as isize) - c;
            for x in 0..n {
                let dx = (x as isize) - c;
                if dx * dx + dy * dy >= 64 {
                    max_val = max_val.max(self.intensity_shifted[flatten_2d_index(x, y, n)]);
                }
            }
        }
        max_val
    }

    #[verified_engine::verified]
    fn rotational_correlation(&self, order: usize) -> f64 {
        let n = self.size;
        let half = 0.5 * (n as f64);
        let (r_min_sq, r_max_sq) = (100.0_f64, (0.42 * (n as f64)).powi(2));
        let (sin_a, cos_a) = (2.0 * PI / (order as f64)).sin_cos();
        let (mut count, mut sum_a, mut sum_b) = (0.0, 0.0, 0.0);
        let (mut sum_aa, mut sum_bb, mut sum_ab) = (0.0, 0.0, 0.0);

        for iy in 3..(n - 3) {
            let dy = (iy as f64) - half;
            for ix in 3..(n - 3) {
                let dx = (ix as f64) - half;
                let r2 = dx * dx + dy * dy;
                if !(r_min_sq..=r_max_sq).contains(&r2) {
                    continue;
                }
                let rx = half + dx * cos_a - dy * sin_a;
                let ry = half + dx * sin_a + dy * cos_a;
                let va = self.intensity_shifted[flatten_2d_index(ix, iy, n)];
                let vb = bicubic_sample(&self.intensity_shifted, n, rx, ry);
                count += 1.0;
                sum_a += va;
                sum_b += vb;
                sum_aa += va * va;
                sum_bb += vb * vb;
                sum_ab += va * vb;
            }
        }
        if count < 1.0 {
            return 0.0;
        }
        let cov = sum_ab - (sum_a * sum_b) / count;
        let var_a = (sum_aa - (sum_a * sum_a) / count).max(0.0);
        let var_b = (sum_bb - (sum_b * sum_b) / count).max(0.0);
        let denom = (var_a * var_b).sqrt();
        if denom > 1e-18 {
            (cov / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    #[verified_engine::verified]
    fn extract_peaks(&mut self, sites: &[[f64; 2]], delta_k: f64, max_non_dc: f64) {
        self.peaks.clear();
        if max_non_dc <= 1e-18 {
            return;
        }
        let n = self.size;
        let half = (n / 2) as isize;
        let threshold = 0.04 * max_non_dc;
        let max_r2 = ((n as isize) * 42 / 100).pow(2);

        for y in 2..(n - 2) {
            let dy = (y as isize) - half;
            for x in 2..(n - 2) {
                let dx = (x as isize) - half;
                let r2 = dx * dx + dy * dy;
                let val = self.intensity_shifted[flatten_2d_index(x, y, n)];
                if !(64..=max_r2).contains(&r2)
                    || val < threshold
                    || !is_local_peak(&self.intensity_shifted, n, x, y, val)
                {
                    continue;
                }
                let sub_x = parabolic_offset(
                    self.intensity_shifted[flatten_2d_index(x - 1, y, n)],
                    val,
                    self.intensity_shifted[flatten_2d_index(x + 1, y, n)],
                );
                let sub_y = parabolic_offset(
                    self.intensity_shifted[flatten_2d_index(x, y - 1, n)],
                    val,
                    self.intensity_shifted[flatten_2d_index(x, y + 1, n)],
                );
                let kx = ((dx as f64) + sub_x) * delta_k;
                let ky = ((dy as f64) + sub_y) * delta_k;
                insert_top_peak(&mut self.peaks, [kx, ky], val / max_non_dc, 16);
            }
        }
        for peak in &mut self.peaks {
            let sf = kinematic_structure_factor(sites, Vector2::new(peak.k[0], peak.k[1]));
            peak.structure_factor_amp = sf.norm();
            peak.structure_factor_phase = sf.arg();
        }
    }
}

#[verified_engine::verified]
fn parabolic_offset(left: f64, center: f64, right: f64) -> f64 {
    let denom = 2.0 * (left - 2.0 * center + right);
    if denom.abs() > 1e-18 {
        ((left - right) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

#[verified_engine::verified]
fn cubic_weight(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ]
}

#[verified_engine::verified]
fn bicubic_sample(grid: &[f64], n: usize, x: f64, y: f64) -> f64 {
    let (ix, iy) = (x.floor() as isize, y.floor() as isize);
    let (wx, wy) = (cubic_weight(x - (ix as f64)), cubic_weight(y - (iy as f64)));
    let max_i = (n - 1) as isize;
    let mut val = 0.0;
    for (jj, &wy_j) in wy.iter().enumerate() {
        let yy = (iy - 1 + (jj as isize)).clamp(0, max_i) as usize;
        let mut row_val = 0.0;
        for (ii, &wx_i) in wx.iter().enumerate() {
            let xx = (ix - 1 + (ii as isize)).clamp(0, max_i) as usize;
            row_val += wx_i * grid[flatten_2d_index(xx, yy, n)];
        }
        val += wy_j * row_val;
    }
    val.max(0.0)
}

#[verified_engine::verified]
fn is_local_peak(grid: &[f64], n: usize, x: usize, y: usize, val: f64) -> bool {
    for oy in (y - 1)..=(y + 1) {
        for ox in (x - 1)..=(x + 1) {
            if ox == x && oy == y {
                continue;
            }
            let neighbor = grid[flatten_2d_index(ox, oy, n)];
            if neighbor > val || (neighbor == val && (oy < y || (oy == y && ox < x))) {
                return false;
            }
        }
    }
    true
}

#[verified_engine::verified]
fn insert_top_peak(peaks: &mut Vec<BraggPeak2D>, k: [f64; 2], norm_i: f64, cap: usize) {
    let candidate = BraggPeak2D {
        k,
        normalized_intensity: norm_i,
        structure_factor_amp: 0.0,
        structure_factor_phase: 0.0,
    };
    if peaks.len() < cap && peaks.len() < peaks.capacity() {
        peaks.push(candidate);
    } else if let Some((min_idx, min_p)) = peaks.iter().enumerate().min_by(|a, b| {
        a.1.normalized_intensity
            .total_cmp(&b.1.normalized_intensity)
    }) && norm_i > min_p.normalized_intensity
    {
        peaks[min_idx] = candidate;
    }
    peaks.sort_unstable_by(|a, b| b.normalized_intensity.total_cmp(&a.normalized_intensity));
}
