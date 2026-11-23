#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Scene delegation layer connecting `GaussianSplattingTool` to shared `crate::volumetric_scene`.
// [cite:gaussian_splatting]

pub use crate::volumetric_scene::presets::{
    build_preset_scene_and_gradients, VolumetricSplatPreset,
};
pub use crate::volumetric_scene::{
    covariance_ellipse_ring, splat_stroke_color, DensityActionCounts, ELLIPSE_POINTS,
};

use crate::framework::{Camera3D, ProjectionMode};
use eframe::egui;
use math_explorer::ai::gaussian_splatting::{
    optimization::{determine_density_action, DensityAction},
    projection, rendering, Gaussian2D, Gaussian3D, Scene,
};
use nalgebra::{Matrix2, Matrix2x3, Point2, Point3, Vector3};

pub(super) type ProjectedSplatRing = (
    [[f64; 2]; ELLIPSE_POINTS],
    [Point3<f64>; ELLIPSE_POINTS],
    egui::Color32,
);

pub(crate) const SPLAT_GRID_SIZE: usize = 96;
pub(super) const VIEWPORT_HALF_EXTENT: f64 = 2.5;
pub(super) const CAMERA_DISTANCE: f64 = 5.5;
pub(super) const DEFAULT_GRAD_THRESHOLD: f64 = crate::volumetric_scene::GRAD_THRESHOLD;
pub(super) const GRAD_THRESHOLD_MIN: f64 = 0.01;
pub(super) const GRAD_THRESHOLD_MAX: f64 = 0.40;
pub(super) const DEFAULT_OPACITY_THRESHOLD: f64 = crate::volumetric_scene::OPACITY_THRESHOLD;
pub(super) const OPACITY_THRESHOLD_MIN: f64 = 0.005;
pub(super) const OPACITY_THRESHOLD_MAX: f64 = 0.95;

pub(super) fn evaluate_scene_density_counts_with_thresholds(
    scene: &Scene,
    grads: &[f64],
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
    for (g, &grad) in scene.gaussians.iter().zip(grads.iter()) {
        let max_base = g.scale.x.max(g.scale.y).max(g.scale.z);
        let action = determine_density_action(
            grad,
            max_base * safe_scale,
            g.opacity,
            safe_grad,
            crate::volumetric_scene::SCALE_THRESHOLD,
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

#[cfg(test)]
pub(super) fn evaluate_density_counts_with_thresholds(
    scale: f64,
    grad_threshold: f64,
    opacity_threshold: f64,
) -> DensityActionCounts {
    let (scene, grads) = build_preset_scene_and_gradients(VolumetricSplatPreset::Canonical);
    evaluate_scene_density_counts_with_thresholds(
        &scene,
        &grads,
        scale,
        grad_threshold,
        opacity_threshold,
    )
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

pub(super) fn project_single_splat(
    tool: &super::GaussianSplattingTool,
    base: &Gaussian3D,
    scale_factor: f64,
    dist: f64,
    view_matrix: &nalgebra::Matrix4<f64>,
    want_ring: bool,
) -> Option<(Gaussian2D, Option<ProjectedSplatRing>)> {
    let camera = &tool.camera;
    let scaled = Gaussian3D {
        mean: base.mean,
        scale: base.scale * scale_factor,
        rotation: base.rotation,
        opacity: base.opacity,
        color: base.color,
    };
    let zoom = f64::from(camera.zoom);
    let focal = dist * zoom;
    let [tx, ty, vz] =
        camera.rotate_to_camera_space(&[scaled.mean.x, scaled.mean.y, scaled.mean.z]);
    let tz = (dist - vz).max(0.25);
    let (jacobian, proj_scale) = match camera.projection {
        ProjectionMode::Parallel => (Matrix2x3::new(zoom, 0.0, 0.0, 0.0, zoom, 0.0), zoom),
        ProjectionMode::Perspective => (
            projection::compute_jacobian(&Vector3::new(tx, ty, tz), focal, focal),
            focal / tz,
        ),
    };
    let cov2d =
        projection::project_covariance(&scaled.compute_covariance(), view_matrix, &jacobian)
            + Matrix2::from_diagonal_element(1e-4);
    let inv_cov = cov2d.try_inverse()?;
    let [mx, my] = tool.project_splat_point(scaled.mean);
    let mean2d = Point2::new(mx, my);
    let maybe_ring = want_ring.then(|| {
        let ring_2d = covariance_ellipse_ring(mean2d, &cov2d);
        let ring_3d = lift_ring_to_world(camera, &ring_2d, mean2d, [tx, ty, vz], proj_scale);
        let color = splat_stroke_color(&scaled.color, scaled.opacity);
        (ring_2d, ring_3d, color)
    });
    let splat_2d = Gaussian2D {
        mean: mean2d,
        conic: -0.5 * inv_cov,
        opacity: scaled.opacity,
        color: scaled.color,
        depth: tz,
    };
    Some((splat_2d, maybe_ring))
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
