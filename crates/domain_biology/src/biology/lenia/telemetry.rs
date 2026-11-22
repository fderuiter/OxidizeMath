//! Zero-allocation toroidal center-of-mass, velocity, and rotational symmetry telemetry for Lenia.

use math_commons::math_kernel::types::flatten_2d_index;
use std::f64::consts::TAU;

/// Live morphological and kinematic invariants computed on each Lenia simulation step.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LeniaTelemetry {
    /// Total integrated mass $M = \sum_{x, y} A(x, y)$ across the grid.
    pub total_mass: f64,
    /// Toroidal center-of-mass coordinates `[x_cm, y_cm]` in `[0, width) x [0, height)`.
    pub center_of_mass: [f64; 2],
    /// Instantaneous translational velocity `[vx, vy]` in grid cells per unit time, unwrapped across periodic boundaries.
    pub velocity: [f64; 2],
    /// Scalar translational speed $\sqrt{v_x^2 + v_y^2}$.
    pub speed: f64,
    /// Rotational/circular harmonic symmetry score in `[0.0, 1.0]`.
    pub rotational_symmetry: f64,
    /// Dominant angular harmonic symmetry fold $m \in \{1, \dots, 6\}$.
    pub dominant_symmetry_fold: usize,
}

/// Precomputed trigonometric tables for zero-allocation toroidal moment evaluation.
#[derive(Debug, Clone)]
pub(crate) struct ToroidalBasis {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) cos_x: Vec<f64>,
    pub(crate) sin_x: Vec<f64>,
    pub(crate) cos_y: Vec<f64>,
    pub(crate) sin_y: Vec<f64>,
}

impl ToroidalBasis {
    #[verified_engine::verified]
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let w_f = (width.max(1)) as f64;
        let h_f = (height.max(1)) as f64;
        let cos_x = (0..width).map(|x| (TAU * (x as f64) / w_f).cos()).collect();
        let sin_x = (0..width).map(|x| (TAU * (x as f64) / w_f).sin()).collect();
        let cos_y = (0..height)
            .map(|y| (TAU * (y as f64) / h_f).cos())
            .collect();
        let sin_y = (0..height)
            .map(|y| (TAU * (y as f64) / h_f).sin())
            .collect();
        Self {
            width,
            height,
            cos_x,
            sin_x,
            cos_y,
            sin_y,
        }
    }

    #[verified_engine::verified]
    pub(crate) fn compute_mass_and_com(&self, grid: &[f64]) -> (f64, [f64; 2]) {
        let mut mass = 0.0;
        let mut sum_cx = 0.0;
        let mut sum_sx = 0.0;
        let mut sum_cy = 0.0;
        let mut sum_sy = 0.0;

        for y in 0..self.height {
            let cy = self.cos_y[y];
            let sy = self.sin_y[y];
            for x in 0..self.width {
                let idx = flatten_2d_index(x, y, self.width);
                let a = grid[idx];
                if a > 0.0 {
                    mass += a;
                    sum_cx += a * self.cos_x[x];
                    sum_sx += a * self.sin_x[x];
                    sum_cy += a * cy;
                    sum_sy += a * sy;
                }
            }
        }

        if mass <= 1e-12 {
            return (0.0, [0.0, 0.0]);
        }

        let angle_x = sum_sx.atan2(sum_cx).rem_euclid(TAU);
        let angle_y = sum_sy.atan2(sum_cy).rem_euclid(TAU);
        let cm_x = angle_x * (self.width as f64) / TAU;
        let cm_y = angle_y * (self.height as f64) / TAU;
        (mass, [cm_x, cm_y])
    }

    #[verified_engine::verified]
    pub(crate) fn compute_angular_symmetry(
        &self,
        grid: &[f64],
        mass: f64,
        com: [f64; 2],
    ) -> (f64, usize) {
        if mass <= 1e-12 {
            return (1.0, 1);
        }
        let w_f = self.width as f64;
        let h_f = self.height as f64;
        let mut re_harmonics = [0.0_f64; 6];
        let mut im_harmonics = [0.0_f64; 6];
        let mut radial_weight = 0.0;

        for y in 0..self.height {
            let dy = shortest_toroidal_delta(y as f64, com[1], h_f);
            for x in 0..self.width {
                let idx = flatten_2d_index(x, y, self.width);
                let a = grid[idx];
                if a <= 1e-9 {
                    continue;
                }
                let dx = shortest_toroidal_delta(x as f64, com[0], w_f);
                let r = dx.hypot(dy);
                if r <= 1e-6 {
                    continue;
                }
                radial_weight += a;
                let c1 = dx / r;
                let s1 = dy / r;
                let mut cm = 1.0;
                let mut sm = 0.0;
                for m in 0..6 {
                    let next_c = cm * c1 - sm * s1;
                    let next_s = cm * s1 + sm * c1;
                    cm = next_c;
                    sm = next_s;
                    re_harmonics[m] += a * cm;
                    im_harmonics[m] += a * sm;
                }
            }
        }

        if radial_weight <= 1e-12 {
            return (1.0, 1);
        }

        let inv_w = 1.0 / radial_weight;
        let c1_dipole = re_harmonics[0].hypot(im_harmonics[0]) * inv_w;
        let mut best_fold = 1usize;
        let mut best_higher = 0.0_f64;
        for m in 1..6 {
            let amp = re_harmonics[m].hypot(im_harmonics[m]) * inv_w;
            if amp > best_higher {
                best_higher = amp;
                best_fold = m + 1;
            }
        }
        let symmetry = (1.0 - c1_dipole).clamp(0.0, 1.0);
        if c1_dipole > best_higher {
            best_fold = 1;
        }
        (symmetry, best_fold)
    }
}

/// Computes the signed shortest displacement `(target - origin)` on a periodic domain of length `period`.
#[verified_engine::verified]
pub fn shortest_toroidal_delta(target: f64, origin: f64, period: f64) -> f64 {
    if period <= 0.0 {
        return 0.0;
    }
    let half = 0.5 * period;
    (target - origin + half).rem_euclid(period) - half
}
