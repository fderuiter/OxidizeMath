#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Analytical benchmark potentials, real-time 32-bump Gaussian sculpting,
//! 3D surface wireframe projection, and 2D marching-squares contour helpers.

pub use super::evaluator::SurfaceEvaluator;
use crate::camera::Camera3D;
use eframe::egui::{self, Color32};
use nalgebra::Vector2;

pub const MAX_SCULPT_BUMPS: usize = 32;
pub const SCULPT_AMP_MIN: f64 = -20.0;
pub const SCULPT_AMP_MAX: f64 = 20.0;
pub const SCULPT_SIGMA_MIN: f64 = 0.1;
pub const SCULPT_SIGMA_MAX: f64 = 5.0;
pub const ARENA_3D_HALF_EXTENT: f64 = 1.65;

pub(super) const LINEAR_MSE_X: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
pub(super) const LINEAR_MSE_Y: [f64; 5] = [1.0, 3.0, 5.0, 7.0, 9.0];

#[inline]
fn round4(val: f64) -> f64 {
    (val * 10_000.0).round() / 10_000.0
}

#[inline]
pub fn lerp_channel(low: u8, high: u8, t: f64) -> u8 {
    let clamped = t.clamp(0.0, 1.0);
    (f64::from(low) + (f64::from(high) - f64::from(low)) * clamped).round() as u8
}

#[inline]
pub fn lerp_color(low: Color32, high: Color32, t: f64) -> Color32 {
    Color32::from_rgb(
        lerp_channel(low.r(), high.r(), t),
        lerp_channel(low.g(), high.g(), t),
        lerp_channel(low.b(), high.b(), t),
    )
}

/// Interactive isotropic 2D Gaussian hill ($A > 0$) or well ($A < 0$) sculpted on $(w, b)$.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SculptBump {
    pub w: f64,
    pub b: f64,
    #[serde(rename = "a", alias = "amplitude")]
    pub amplitude: f64,
    #[serde(rename = "s", alias = "sigma")]
    pub sigma: f64,
}

impl SculptBump {
    #[must_use]
    pub fn new(w: f64, b: f64, amplitude: f64, sigma: f64) -> Self {
        let safe_amp = if amplitude.is_finite() {
            amplitude.clamp(SCULPT_AMP_MIN, SCULPT_AMP_MAX)
        } else {
            1.0
        };
        let safe_sigma = if sigma.is_finite() {
            sigma.clamp(SCULPT_SIGMA_MIN, SCULPT_SIGMA_MAX)
        } else {
            0.45
        };
        Self {
            w: round4(w.clamp(-100.0, 100.0)),
            b: round4(b.clamp(-100.0, 100.0)),
            amplitude: round4(safe_amp),
            sigma: round4(safe_sigma),
        }
    }

    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.w.is_finite()
            && self.b.is_finite()
            && (-100.0..=100.0).contains(&self.w)
            && (-100.0..=100.0).contains(&self.b)
            && self.amplitude.is_finite()
            && (SCULPT_AMP_MIN..=SCULPT_AMP_MAX).contains(&self.amplitude)
            && self.sigma.is_finite()
            && (SCULPT_SIGMA_MIN..=SCULPT_SIGMA_MAX).contains(&self.sigma)
    }

    #[inline]
    #[must_use]
    pub fn evaluate(&self, w: f64, b: f64) -> f64 {
        let (dw, db) = (w - self.w, b - self.b);
        self.amplitude * (-(dw * dw + db * db) / (2.0 * self.sigma * self.sigma)).exp()
    }

    #[inline]
    #[must_use]
    pub fn gradient(&self, w: f64, b: f64) -> Vector2<f64> {
        let (dw, db) = (w - self.w, b - self.b);
        let var = self.sigma * self.sigma;
        let factor = -self.amplitude * (-(dw * dw + db * db) / (2.0 * var)).exp() / var;
        Vector2::new(factor * dw, factor * db)
    }
}

/// Appends `bump` while enforcing [`MAX_SCULPT_BUMPS`] via FIFO eviction.
pub fn push_sculpt_bump(bumps: &mut Vec<SculptBump>, bump: SculptBump) {
    let clean = SculptBump::new(bump.w, bump.b, bump.amplitude, bump.sigma);
    if bumps.len() >= MAX_SCULPT_BUMPS {
        bumps.drain(0..(bumps.len() + 1 - MAX_SCULPT_BUMPS));
    }
    bumps.push(clean);
}

/// Benchmark 2D optimization potential surfaces $L(w, b)$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum LandscapeSurface {
    #[default]
    LinearMse,
    Rosenbrock,
    Rastrigin,
    MonkeySaddle,
}

impl LandscapeSurface {
    pub const ALL: [Self; 4] = [
        Self::LinearMse,
        Self::Rosenbrock,
        Self::Rastrigin,
        Self::MonkeySaddle,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LinearMse => "Linear Regression MSE",
            Self::Rosenbrock => "Rosenbrock Valley",
            Self::Rastrigin => "Rastrigin Egg-Crate",
            Self::MonkeySaddle => "Monkey Saddle",
        }
    }

    #[must_use]
    pub const fn default_bounds(self) -> (f64, f64, f64, f64) {
        match self {
            Self::LinearMse => (0.0, 4.0, -1.0, 3.0),
            Self::Rosenbrock => (-2.0, 2.0, -1.0, 3.0),
            Self::Rastrigin => (-4.0, 4.0, -4.0, 4.0),
            Self::MonkeySaddle => (-2.2, 2.2, -2.2, 2.2),
        }
    }

    #[must_use]
    pub const fn global_minima(self) -> &'static [[f64; 2]] {
        match self {
            Self::LinearMse => &[[2.0, 1.0]],
            Self::Rosenbrock => &[[1.0, 1.0]],
            Self::Rastrigin => &[[0.0, 0.0]],
            Self::MonkeySaddle => &[[-1.5, 0.0], [0.75, 1.299_038_1], [0.75, -1.299_038_1]],
        }
    }

    #[must_use]
    pub fn base_loss(self, w: f64, b: f64) -> f64 {
        match self {
            Self::LinearMse => {
                let mut sum_sq = 0.0;
                for i in 0..LINEAR_MSE_X.len() {
                    let err = (w * LINEAR_MSE_X[i] + b) - LINEAR_MSE_Y[i];
                    sum_sq += err * err;
                }
                sum_sq / (LINEAR_MSE_X.len() as f64)
            }
            Self::Rosenbrock => {
                let (dw, valley) = (1.0 - w, b - w * w);
                dw * dw + 100.0 * valley * valley
            }
            Self::Rastrigin => {
                let pi2 = std::f64::consts::TAU;
                20.0 + (w * w - 10.0 * (pi2 * w).cos()) + (b * b - 10.0 * (pi2 * b).cos())
            }
            Self::MonkeySaddle => {
                let r2 = w * w + b * b;
                (w * w * w - 3.0 * w * b * b) + 0.5 * r2 * r2 + 1.6875
            }
        }
    }

    #[must_use]
    pub fn base_gradient(self, w: f64, b: f64) -> Vector2<f64> {
        match self {
            Self::LinearMse => {
                let (mut dw_sum, mut db_sum) = (0.0, 0.0);
                for i in 0..LINEAR_MSE_X.len() {
                    let err = (w * LINEAR_MSE_X[i] + b) - LINEAR_MSE_Y[i];
                    dw_sum += err * LINEAR_MSE_X[i];
                    db_sum += err;
                }
                let scale = 2.0 / (LINEAR_MSE_X.len() as f64);
                Vector2::new(scale * dw_sum, scale * db_sum)
            }
            Self::Rosenbrock => {
                let valley = b - w * w;
                Vector2::new(-2.0 * (1.0 - w) - 400.0 * w * valley, 200.0 * valley)
            }
            Self::Rastrigin => {
                let pi2 = std::f64::consts::TAU;
                let k = 10.0 * pi2;
                Vector2::new(2.0 * w + k * (pi2 * w).sin(), 2.0 * b + k * (pi2 * b).sin())
            }
            Self::MonkeySaddle => {
                let r2 = w * w + b * b;
                Vector2::new(
                    3.0 * (w * w - b * b) + 2.0 * w * r2,
                    -6.0 * w * b + 2.0 * b * r2,
                )
            }
        }
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub fn evaluate(self, bumps: &[SculptBump], w: f64, b: f64) -> f64 {
        let mut total = self.base_loss(w, b);
        for bump in bumps {
            total += bump.evaluate(w, b);
        }
        total
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub fn gradient(self, bumps: &[SculptBump], w: f64, b: f64) -> Vector2<f64> {
        let mut grad = self.base_gradient(w, b);
        for bump in bumps {
            grad += bump.gradient(w, b);
        }
        grad
    }
}

/// Sampled 2D surface heightfield for 3D wireframes and 2D iso-contours.
pub struct SurfaceGrid {
    pub rows: Vec<Vec<[f64; 3]>>,
    pub min_loss: f64,
    pub max_loss: f64,
    pub bounds: (f64, f64, f64, f64),
}

impl SurfaceGrid {
    #[must_use]
    pub fn sample_eval(
        eval: SurfaceEvaluator<'_>,
        bounds: (f64, f64, f64, f64),
        resolution: usize,
    ) -> Self {
        let res = resolution.clamp(10, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
        let (w_min, w_max, b_min, b_max) = bounds;
        let denom = (res - 1).max(1) as f64;
        let (w_step, b_step) = ((w_max - w_min) / denom, (b_max - b_min) / denom);
        let (mut min_loss, mut max_loss) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut rows = Vec::with_capacity(res);

        for j in 0..res {
            let b = b_min + (j as f64) * b_step;
            let mut row = Vec::with_capacity(res);
            for i in 0..res {
                let w = w_min + (i as f64) * w_step;
                let loss = eval.evaluate(w, b);
                if loss.is_finite() {
                    min_loss = min_loss.min(loss);
                    max_loss = max_loss.max(loss);
                }
                row.push([w, b, loss]);
            }
            rows.push(row);
        }
        if !min_loss.is_finite() || !max_loss.is_finite() {
            min_loss = 0.0;
            max_loss = 1.0;
        }
        Self {
            rows,
            min_loss,
            max_loss,
            bounds,
        }
    }

    #[inline]
    #[must_use]
    pub fn normalized_height(&self, loss: f64) -> f64 {
        let span = self.max_loss - self.min_loss;
        if span <= 1e-12 || !loss.is_finite() {
            return 0.0;
        }
        let shifted = (loss - self.min_loss).max(0.0);
        if span > 25.0 {
            (shifted.ln_1p() / span.ln_1p()).clamp(0.0, 1.0)
        } else {
            (shifted / span).clamp(0.0, 1.0)
        }
    }

    #[inline]
    #[must_use]
    pub fn world_point(&self, w: f64, b: f64, loss: f64, z_offset: f64) -> [f64; 3] {
        let (w_min, w_max, b_min, b_max) = self.bounds;
        let wx = ((w - w_min) / (w_max - w_min).max(1e-9)) * 2.0 - 1.0;
        let by = ((b - b_min) / (b_max - b_min).max(1e-9)) * 2.0 - 1.0;
        let hz = -0.55 + 1.3 * self.normalized_height(loss) + z_offset;
        [wx.clamp(-1.05, 1.05), by.clamp(-1.05, 1.05), hz]
    }

    #[must_use]
    pub fn contour_segments(&self, num_levels: usize) -> Vec<(f64, Vec<[[f64; 2]; 2]>)> {
        let levels = num_levels.clamp(2, 16);
        let mut result = Vec::with_capacity(levels);
        if self.rows.len() < 2 || self.rows[0].len() < 2 {
            return result;
        }
        for lvl_idx in 1..=levels {
            let iso = (lvl_idx as f64) / ((levels + 1) as f64);
            let mut segs = Vec::new();
            for r in 0..(self.rows.len() - 1) {
                for c in 0..(self.rows[r].len() - 1) {
                    let cells = [
                        self.rows[r][c],
                        self.rows[r][c + 1],
                        self.rows[r + 1][c + 1],
                        self.rows[r + 1][c],
                    ];
                    let h = cells.map(|pt| self.normalized_height(pt[2]));
                    collect_cell_iso_segments(cells, h, iso, &mut segs);
                }
            }
            if !segs.is_empty() {
                result.push((iso, segs));
            }
        }
        result
    }
}

fn collect_cell_iso_segments(
    pts: [[f64; 3]; 4],
    h: [f64; 4],
    iso: f64,
    out: &mut Vec<[[f64; 2]; 2]>,
) {
    let mut hits = [[0.0_f64; 2]; 4];
    let mut count = 0;
    for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        if (h[i] < iso && h[j] >= iso) || (h[j] < iso && h[i] >= iso) {
            let t = ((iso - h[i]).abs() / (h[j] - h[i]).abs().max(1e-12)).clamp(0.0, 1.0);
            hits[count] = [
                pts[i][0] + t * (pts[j][0] - pts[i][0]),
                pts[i][1] + t * (pts[j][1] - pts[i][1]),
            ];
            count += 1;
        }
    }
    if count >= 2 {
        out.push([hits[0], hits[1]]);
    }
    if count == 4 {
        out.push([hits[2], hits[3]]);
    }
}

#[must_use]
pub fn unproject_3d_rect_pos(
    camera: &Camera3D,
    rect: egui::Rect,
    pos: egui::Pos2,
    bounds: (f64, f64, f64, f64),
) -> [f64; 2] {
    let (width, height) = (rect.width().max(1.0) as f64, rect.height().max(1.0) as f64);
    let aspect = (width / height).max(1.0);
    let u = ((pos.x - rect.min.x) as f64 / width) * 2.0 - 1.0;
    let v = 1.0 - ((pos.y - rect.min.y) as f64 / height) * 2.0;
    let (plot_x, plot_y) = (u * ARENA_3D_HALF_EXTENT * aspect, v * ARENA_3D_HALF_EXTENT);

    let zoom = (camera.zoom as f64).max(0.05);
    let (sy, cy) = (camera.yaw as f64).sin_cos();
    let (sp, cp) = (camera.pitch as f64).sin_cos();
    let safe_cp = if cp.abs() < 0.15 {
        0.15_f64.copysign(if cp >= 0.0 { 1.0 } else { -1.0 })
    } else {
        cp
    };
    let x1 = plot_x / zoom;
    let y1 = (plot_y / zoom - 0.25 * sp) / safe_cp;
    let wx = (x1 * cy + y1 * sy).clamp(-1.0, 1.0);
    let by = (-x1 * sy + y1 * cy).clamp(-1.0, 1.0);
    let (w_min, w_max, b_min, b_max) = bounds;
    [
        w_min + (wx + 1.0) * 0.5 * (w_max - w_min),
        b_min + (by + 1.0) * 0.5 * (b_max - b_min),
    ]
}

#[must_use]
pub fn map_2d_rect_pos(
    rect: egui::Rect,
    pos: egui::Pos2,
    bounds: (f64, f64, f64, f64),
) -> [f64; 2] {
    let tx = (((pos.x - rect.min.x) / rect.width().max(1.0)) as f64).clamp(0.0, 1.0);
    let ty = (1.0 - ((pos.y - rect.min.y) / rect.height().max(1.0)) as f64).clamp(0.0, 1.0);
    let (w_min, w_max, b_min, b_max) = bounds;
    [w_min + tx * (w_max - w_min), b_min + ty * (b_max - b_min)]
}
