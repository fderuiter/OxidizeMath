#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

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
const GRAD_THRESHOLD: f64 = 0.10;
const SCALE_THRESHOLD: f64 = 0.45;
pub(super) const OPACITY_THRESHOLD: f64 = 0.05;

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

pub(super) fn evaluate_density_counts(scale: f64) -> DensityActionCounts {
    let safe_scale = if scale.is_finite() {
        scale.clamp(0.1, 10.0)
    } else {
        1.0
    };
    let mut counts = DensityActionCounts::default();
    for &(_, base_scale, _, opacity, _, grad_mag) in &CANONICAL_SCENE {
        let max_base = base_scale[0].max(base_scale[1]).max(base_scale[2]);
        let action = determine_density_action(
            grad_mag,
            max_base * safe_scale,
            opacity,
            GRAD_THRESHOLD,
            SCALE_THRESHOLD,
            OPACITY_THRESHOLD,
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

pub(super) fn splat_stroke_color(rgb: &Vector3<f64>, opacity: f64) -> egui::Color32 {
    let a = (opacity as f32).clamp(0.12, 0.85);
    let r = (rgb.x as f32).clamp(0.0, 1.0) * a;
    let g = (rgb.y as f32).clamp(0.0, 1.0) * a;
    let b = (rgb.z as f32).clamp(0.0, 1.0) * a;
    egui::Color32::from(egui::Rgba::from_rgba_premultiplied(r, g, b, a))
}

pub(super) fn rasterize_splats_into(splats: &[Gaussian2D], pixels: &mut [egui::Color32]) {
    let span = 2.0 * VIEWPORT_HALF_EXTENT;
    let inv_dim = 1.0 / (SPLAT_GRID_SIZE as f64);
    let (rows, _) = pixels.as_chunks_mut::<SPLAT_GRID_SIZE>();
    for (row_idx, row_pixels) in rows.iter_mut().enumerate() {
        let y = VIEWPORT_HALF_EXTENT - span * ((row_idx as f64 + 0.5) * inv_dim);
        for (col_idx, pixel) in row_pixels.iter_mut().enumerate() {
            let x = -VIEWPORT_HALF_EXTENT + span * ((col_idx as f64 + 0.5) * inv_dim);
            let rgb = rendering::blend_gaussians(splats, &Point2::new(x, y));
            let r = (rgb.x as f32).clamp(0.0, 1.0);
            let g = (rgb.y as f32).clamp(0.0, 1.0);
            let b = (rgb.z as f32).clamp(0.0, 1.0);
            let alpha = r.max(g).max(b);
            *pixel = egui::Color32::from(egui::Rgba::from_rgba_premultiplied(r, g, b, alpha));
        }
    }
}
