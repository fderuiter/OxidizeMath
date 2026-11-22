//! Canonical continuous cellular automaton presets (`Orbium`, `Hydrogeminium`) for Lenia.

use super::LeniaParams;
use super::kernel::LeniaKernelCore;
use math_commons::math_kernel::types::flatten_2d_index;

/// Canonical lifeform presets in Lenia parameter space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeniaPreset {
    /// `Orbium unicaudatus`: single-ring smooth glider ($\mu = 0.15, \sigma = 0.015, R = 13$).
    #[default]
    Orbium,
    /// `Hydrogeminium natans`: three-ring wobbling and self-dividing organism ($\mu = 0.26, \sigma = 0.036, R = 18$).
    Hydrogeminium,
}

/// Canonical $20 \times 20$ `Orbium unicaudatus` seed matrix from Chan (2019), calibrated at $R = 13$.
const ORBIUM_TEMPLATE_20X20: [[f64; 20]; 20] = [
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.14, 0.1, 0.0, 0.0, 0.03, 0.03, 0.0, 0.0, 0.3, 0.0,
        0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2,
        0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.34, 0.44, 0.49, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18,
        0.45, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0.0, 0.0, 0.0, 0.02, 0.16,
        0.68, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.18, 0.42, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.82, 0.0, 0.0,
    ],
    [
        0.27, 0.0, 0.16, 0.12, 0.0, 0.0, 0.0, 0.25, 0.38, 0.44, 0.45, 0.34, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.22, 0.17, 0.0,
    ],
    [
        0.0, 0.07, 0.2, 0.02, 0.0, 0.0, 0.0, 0.31, 0.48, 0.57, 0.6, 0.57, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.49, 0.0,
    ],
    [
        0.0, 0.59, 0.19, 0.0, 0.0, 0.0, 0.0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.36, 0.0,
    ],
    [
        0.0, 0.58, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0.0, 0.0, 0.0,
        0.0, 0.22, 0.07,
    ],
    [
        0.0, 0.0, 0.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.93, 1.0, 1.0, 1.0, 0.61, 0.0, 0.0, 0.0,
        0.0, 0.18, 0.11,
    ],
    [
        0.0, 0.0, 0.82, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47, 1.0, 1.0, 0.98, 1.0, 0.96, 0.27, 0.0, 0.0,
        0.0, 0.19, 0.1,
    ],
    [
        0.0, 0.0, 0.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 1.0, 0.84, 0.92, 0.97, 0.54, 0.14,
        0.04, 0.1, 0.21, 0.05,
    ],
    [
        0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.09, 0.8, 1.0, 0.82, 0.8, 0.85, 0.63, 0.31, 0.18,
        0.19, 0.2, 0.01,
    ],
    [
        0.0, 0.0, 0.0, 0.36, 0.1, 0.0, 0.0, 0.0, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.6, 0.39,
        0.28, 0.24, 0.13, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.01, 0.3, 0.07, 0.0, 0.0, 0.08, 0.36, 0.64, 0.7, 0.64, 0.6, 0.51, 0.39,
        0.29, 0.19, 0.04, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.1, 0.24, 0.14, 0.1, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46, 0.4, 0.31,
        0.21, 0.08, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33, 0.26, 0.18,
        0.09, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18, 0.13, 0.05,
        0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ],
];

impl LeniaPreset {
    /// Returns the canonical parameter set for the selected lifeform preset.
    #[verified_engine::verified]
    pub fn params(self) -> LeniaParams {
        match self {
            Self::Orbium => LeniaParams {
                mu: 0.15,
                sigma: 0.015,
                dt: 0.1,
                kernel_radius: 13,
                ring_weights: vec![1.0],
                kernel_core: LeniaKernelCore::ExponentialBump,
            },
            Self::Hydrogeminium => LeniaParams {
                mu: 0.26,
                sigma: 0.036,
                dt: 0.1,
                kernel_radius: 18,
                ring_weights: vec![0.5, 1.0, 2.0 / 3.0],
                kernel_core: LeniaKernelCore::ExponentialBump,
            },
        }
    }

    /// Stamps the preset organism centered at `(center_x, center_y)` (wrapping periodically).
    #[verified_engine::verified]
    pub fn stamp_into_grid(
        self,
        grid: &mut [f64],
        width: usize,
        height: usize,
        kernel_radius: usize,
        center_x: usize,
        center_y: usize,
    ) {
        grid.fill(0.0);
        match self {
            Self::Orbium => {
                stamp_orbium(grid, width, height, kernel_radius, center_x, center_y);
            }
            Self::Hydrogeminium => {
                stamp_hydrogeminium(grid, width, height, kernel_radius, center_x, center_y);
            }
        }
    }
}

#[verified_engine::verified]
fn sample_orbium_bilinear(u: f64, v: f64) -> f64 {
    if !(0.0..=19.0).contains(&u) || !(0.0..=19.0).contains(&v) {
        return 0.0;
    }
    let x0 = (u.floor() as usize).min(18);
    let y0 = (v.floor() as usize).min(18);
    let tx = (u - (x0 as f64)).clamp(0.0, 1.0);
    let ty = (v - (y0 as f64)).clamp(0.0, 1.0);
    let c00 = ORBIUM_TEMPLATE_20X20[y0][x0];
    let c10 = ORBIUM_TEMPLATE_20X20[y0][x0 + 1];
    let c01 = ORBIUM_TEMPLATE_20X20[y0 + 1][x0];
    let c11 = ORBIUM_TEMPLATE_20X20[y0 + 1][x0 + 1];
    let top = c00 * (1.0 - tx) + c10 * tx;
    let bot = c01 * (1.0 - tx) + c11 * tx;
    top * (1.0 - ty) + bot * ty
}

#[verified_engine::verified]
fn stamp_orbium(
    grid: &mut [f64],
    width: usize,
    height: usize,
    kernel_radius: usize,
    center_x: usize,
    center_y: usize,
) {
    let scale = (kernel_radius.max(4) as f64) / 13.0;
    let extent = ((20.0 * scale).ceil() as isize).max(20);
    let half = extent / 2;

    for dy in -half..=half {
        for dx in -half..=half {
            let u = ((dx as f64) / scale) + 9.5;
            let v = ((dy as f64) / scale) + 9.5;
            let val = sample_orbium_bilinear(u, v);
            if val > 0.0 {
                let gx = (center_x as isize + dx).rem_euclid(width as isize) as usize;
                let gy = (center_y as isize + dy).rem_euclid(height as isize) as usize;
                let idx = flatten_2d_index(gx, gy, width);
                grid[idx] = val.clamp(0.0, 1.0);
            }
        }
    }
}

#[verified_engine::verified]
fn stamp_hydrogeminium(
    grid: &mut [f64],
    width: usize,
    height: usize,
    kernel_radius: usize,
    center_x: usize,
    center_y: usize,
) {
    let r_f = (kernel_radius.max(6)) as f64;
    let extent = ((1.45 * r_f).ceil() as isize).max(12);
    for dy in -extent..=extent {
        for dx in -extent..=extent {
            let xf = (dx as f64) / r_f;
            let yf = (dy as f64) / r_f;
            let d1 = ((xf + 0.32).hypot(yf - 0.08)) / 0.78;
            let d2 = ((xf - 0.34).hypot(yf + 0.08)) / 0.74;
            let ring_outer = (-((xf.hypot(yf) - 0.88) / 0.22).powi(2)).exp() * 0.58;
            let lobe_a = if d1 < 1.0 {
                (1.0 - d1 * d1).powi(2) * 0.92
            } else {
                0.0
            };
            let lobe_b = if d2 < 1.0 {
                (1.0 - d2 * d2).powi(2) * 0.84
            } else {
                0.0
            };
            let val = (lobe_a + lobe_b + ring_outer).clamp(0.0, 1.0);
            if val > 1e-4 {
                let gx = (center_x as isize + dx).rem_euclid(width as isize) as usize;
                let gy = (center_y as isize + dy).rem_euclid(height as isize) as usize;
                let idx = flatten_2d_index(gx, gy, width);
                grid[idx] = val;
            }
        }
    }
}
