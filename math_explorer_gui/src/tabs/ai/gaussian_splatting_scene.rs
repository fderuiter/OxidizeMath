#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::Camera3D;
use eframe::egui;
use math_explorer::ai::gaussian_splatting::{
    optimization::{determine_density_action, DensityAction},
    rendering, Gaussian2D, Gaussian3D, Scene,
};
use nalgebra::{Matrix2, Point2, Point3, UnitQuaternion, Vector3};

pub(crate) const SPLAT_GRID_SIZE: usize = 96;
pub(super) const VIEWPORT_HALF_EXTENT: f64 = 2.5;
pub(super) const CAMERA_DISTANCE: f64 = 5.5;
const ELLIPSE_SEGMENTS: usize = 24;
pub(super) const ELLIPSE_POINTS: usize = ELLIPSE_SEGMENTS + 1;
const ELLIPSE_SIGMA: f64 = 1.5;
pub(super) const DEFAULT_GRAD_THRESHOLD: f64 = 0.10;
pub(super) const GRAD_THRESHOLD_MIN: f64 = 0.01;
pub(super) const GRAD_THRESHOLD_MAX: f64 = 0.40;
const SCALE_THRESHOLD: f64 = 0.45;
pub(super) const DEFAULT_OPACITY_THRESHOLD: f64 = 0.05;
pub(super) const OPACITY_THRESHOLD_MIN: f64 = 0.005;
pub(super) const OPACITY_THRESHOLD_MAX: f64 = 0.95;

type SplatSpec = ([f64; 3], [f64; 3], [f64; 3], f64, [f64; 3], f64);

const CANONICAL_SCENE: [SplatSpec; 8] = [
    (
        [-0.65, 0.45, -0.70],
        [0.28, 0.16, 0.20],
        [0.0, 0.0, 0.45],
        0.90,
        [0.96, 0.32, 0.28],
        0.24,
    ),
    (
        [0.60, 0.42, -0.50],
        [0.18, 0.30, 0.22],
        [0.2, -0.1, -0.55],
        0.86,
        [0.22, 0.82, 0.56],
        0.19,
    ),
    (
        [0.0, -0.15, -1.00],
        [0.36, 0.24, 0.26],
        [0.15, 0.25, 0.10],
        0.82,
        [0.28, 0.58, 0.98],
        0.31,
    ),
    (
        [-0.48, -0.58, -0.10],
        [0.58, 0.32, 0.28],
        [-0.2, 0.1, 0.35],
        0.78,
        [0.95, 0.76, 0.22],
        0.16,
    ),
    (
        [0.55, -0.52, 0.10],
        [0.25, 0.22, 0.20],
        [0.0, 0.2, -0.30],
        0.80,
        [0.78, 0.36, 0.94],
        0.06,
    ),
    (
        [0.02, 0.68, 0.40],
        [0.34, 0.20, 0.24],
        [0.1, -0.15, 0.05],
        0.74,
        [0.20, 0.86, 0.90],
        0.04,
    ),
    (
        [-0.15, 0.08, 0.80],
        [0.65, 0.52, 0.40],
        [0.05, 0.1, -0.20],
        0.52,
        [0.92, 0.46, 0.68],
        0.07,
    ),
    (
        [0.85, 0.82, 1.30],
        [0.22, 0.18, 0.18],
        [0.0, 0.0, 0.0],
        0.02,
        [0.50, 0.55, 0.65],
        0.14,
    ),
];

/// Summary counts of adaptive Gaussian density control decisions across the canonical 3D scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DensityActionCounts {
    /// Splats exceeding both gradient and world-space scale thresholds.
    pub split: usize,
    /// Splats exceeding gradient threshold while remaining compact.
    pub clone: usize,
    /// Converged splats below gradient threshold with sufficient opacity.
    pub keep: usize,
    /// Nearly transparent splats below the pruning opacity threshold.
    pub prune: usize,
}

pub(super) fn build_default_scene() -> Scene {
    let gaussians = CANONICAL_SCENE
        .into_iter()
        .map(|(pos, base_scale, axis_angle, opacity, color, _)| Gaussian3D {
            mean: Point3::new(pos[0], pos[1], pos[2]),
            scale: Vector3::new(base_scale[0], base_scale[1], base_scale[2]),
            rotation: UnitQuaternion::new(Vector3::new(
                axis_angle[0],
                axis_angle[1],
                axis_angle[2],
            )),
            opacity,
            color: Vector3::new(color[0], color[1], color[2]),
        })
        .collect();
    Scene { gaussians }
}

pub(super) fn evaluate_density_counts_with_thresholds(
    scale: f64,
    grad_threshold: f64,
    opacity_threshold: f64,
) -> DensityActionCounts {
    let safe_scale = if scale.is_finite() {
        scale.clamp(0.1, 10.0)
    } else {
        1.0
    };
    let safe_grad = if grad_threshold.is_finite() {
        grad_threshold.clamp(GRAD_THRESHOLD_MIN, GRAD_THRESHOLD_MAX)
    } else {
        DEFAULT_GRAD_THRESHOLD
    };
    let safe_opacity = if opacity_threshold.is_finite() {
        opacity_threshold.clamp(OPACITY_THRESHOLD_MIN, OPACITY_THRESHOLD_MAX)
    } else {
        DEFAULT_OPACITY_THRESHOLD
    };
    let mut counts = DensityActionCounts::default();
    for &(_, base_scale, _, opacity, _, grad_mag) in &CANONICAL_SCENE {
        let max_base = base_scale[0].max(base_scale[1]).max(base_scale[2]);
        let action = determine_density_action(
            grad_mag,
            max_base * safe_scale,
            opacity,
            safe_grad,
            SCALE_THRESHOLD,
            safe_opacity,
        );
        match action {
            DensityAction::Split => counts.split += 1,
            DensityAction::Clone => counts.clone += 1,
            DensityAction::Keep => counts.keep += 1,
            DensityAction::Prune => counts.prune += 1,
        }
    }
    counts
}

pub(super) fn covariance_ellipse_ring(
    mean: Point2<f64>,
    cov2d: &Matrix2<f64>,
) -> [[f64; 2]; ELLIPSE_POINTS] {
    let s_xx = cov2d[(0, 0)];
    let s_xy = cov2d[(0, 1)];
    let s_yy = cov2d[(1, 1)];
    let half_trace = 0.5 * (s_xx + s_yy);
    let half_diff = 0.5 * (s_xx - s_yy);
    let disc = (half_diff * half_diff + s_xy * s_xy).max(0.0).sqrt();
    let major = ELLIPSE_SIGMA * (half_trace + disc).max(1e-6).sqrt();
    let minor = ELLIPSE_SIGMA * (half_trace - disc).max(1e-6).sqrt();
    let (sin_p, cos_p) = (0.5 * (2.0 * s_xy).atan2(s_xx - s_yy)).sin_cos();

    let mut ring = [[0.0_f64; 2]; ELLIPSE_POINTS];
    for (idx, slot) in ring.iter_mut().enumerate() {
        let theta = (idx as f64) * std::f64::consts::TAU / (ELLIPSE_SEGMENTS as f64);
        let (sin_t, cos_t) = theta.sin_cos();
        let lx = major * cos_t;
        let ly = minor * sin_t;
        *slot = [
            mean.x + lx * cos_p - ly * sin_p,
            mean.y + lx * sin_p + ly * cos_p,
        ];
    }
    ring
}

pub(super) fn lift_ring_to_world(
    camera: &Camera3D,
    ring_2d: &[[f64; 2]; ELLIPSE_POINTS],
    mean2d: Point2<f64>,
    cam_coords: [f64; 3],
    proj_scale: f64,
) -> [Point3<f64>; ELLIPSE_POINTS] {
    let (sy, cy) = f64::from(camera.yaw).sin_cos();
    let (sp, cp) = f64::from(camera.pitch).sin_cos();
    let inv_scale = if proj_scale.abs() > 1e-9 {
        1.0 / proj_scale
    } else {
        1.0
    };
    let mut ring_3d = [Point3::origin(); ELLIPSE_POINTS];
    for (idx, &[rx, ry]) in ring_2d.iter().enumerate() {
        let x2 = cam_coords[0] + (rx - mean2d.x) * inv_scale;
        let y2 = cam_coords[1] + (ry - mean2d.y) * inv_scale;
        let z2 = cam_coords[2];
        let y1 = y2 * cp + z2 * sp;
        let z1 = -y2 * sp + z2 * cp;
        let x = x2 * cy + y1 * sy;
        let y = -x2 * sy + y1 * cy;
        ring_3d[idx] = Point3::new(x, y, z1);
    }
    ring_3d
}

pub(super) fn splat_stroke_color(rgb: &Vector3<f64>, opacity: f64) -> egui::Color32 {
    let a = (opacity as f32).clamp(0.12, 0.85);
    let r = (rgb.x as f32).clamp(0.0, 1.0) * a;
    let g = (rgb.y as f32).clamp(0.0, 1.0) * a;
    let b = (rgb.z as f32).clamp(0.0, 1.0) * a;
    egui::Color32::from(egui::Rgba::from_rgba_premultiplied(r, g, b, a))
}

pub(super) fn rasterize_splats_into(
    splats: &[Gaussian2D],
    x_half: f64,
    y_half: f64,
    pixels: &mut [egui::Color32],
) {
    let safe_x_half = if x_half.is_finite() && x_half > 0.1 {
        x_half
    } else {
        VIEWPORT_HALF_EXTENT
    };
    let safe_y_half = if y_half.is_finite() && y_half > 0.1 {
        y_half
    } else {
        VIEWPORT_HALF_EXTENT
    };
    let span_x = 2.0 * safe_x_half;
    let span_y = 2.0 * safe_y_half;
    let inv_dim = 1.0 / (SPLAT_GRID_SIZE as f64);
    let (rows, _) = pixels.as_chunks_mut::<SPLAT_GRID_SIZE>();
    for (row_idx, row_pixels) in rows.iter_mut().enumerate() {
        let y = safe_y_half - span_y * ((row_idx as f64 + 0.5) * inv_dim);
        for (col_idx, pixel) in row_pixels.iter_mut().enumerate() {
            let x = -safe_x_half + span_x * ((col_idx as f64 + 0.5) * inv_dim);
            let rgb = rendering::blend_gaussians(splats, &Point2::new(x, y));
            let r = (rgb.x as f32).clamp(0.0, 1.0);
            let g = (rgb.y as f32).clamp(0.0, 1.0);
            let b = (rgb.z as f32).clamp(0.0, 1.0);
            let alpha = r.max(g).max(b);
            *pixel = egui::Color32::from(egui::Rgba::from_rgba_premultiplied(r, g, b, alpha));
        }
    }
}
