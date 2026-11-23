#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Scene delegation layer connecting `GaussianSplattingTool` to shared `crate::volumetric_scene`.
// [cite:gaussian_splatting]

pub use crate::volumetric_scene::presets::{
    build_preset_scene_and_gradients, VolumetricSplatPreset,
};
pub use crate::volumetric_scene::{
    covariance_ellipse_ring, splat_stroke_color, DensityActionCounts, ELLIPSE_POINTS,
    OPACITY_THRESHOLD,
};

use crate::framework::Camera3D;
use eframe::egui;
use math_explorer::ai::gaussian_splatting::{
    optimization::{determine_density_action, DensityAction},
    Gaussian2D, Scene,
};
use nalgebra::{Point2, Point3};

pub(crate) const SPLAT_GRID_SIZE: usize = 96;
pub(super) const VIEWPORT_HALF_EXTENT: f64 = 2.5;
pub(super) const CAMERA_DISTANCE: f64 = 5.5;

pub(super) fn evaluate_scene_density_counts(
    scene: &Scene,
    grads: &[f64],
    scale: f64,
) -> DensityActionCounts {
    let safe_scale = if scale.is_finite() {
        scale.clamp(0.1, 10.0)
    } else {
        1.0
    };
    let mut counts = DensityActionCounts::default();
    for (g, &grad) in scene.gaussians.iter().zip(grads) {
        let max_base = g.scale.x.max(g.scale.y).max(g.scale.z);
        let action = determine_density_action(
            grad,
            max_base * safe_scale,
            g.opacity,
            crate::volumetric_scene::GRAD_THRESHOLD,
            crate::volumetric_scene::SCALE_THRESHOLD,
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

#[cfg(test)]
pub(super) fn evaluate_density_counts(scale: f64) -> DensityActionCounts {
    let (scene, grads) = build_preset_scene_and_gradients(VolumetricSplatPreset::Canonical);
    evaluate_scene_density_counts(&scene, &grads, scale)
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

pub(super) fn rasterize_splats_into(
    splats: &[Gaussian2D],
    pan: [f64; 2],
    pixels: &mut [egui::Color32],
) {
    crate::volumetric_scene::rasterize_splats_with_pan_into(
        splats,
        VIEWPORT_HALF_EXTENT,
        pan,
        pixels,
    );
}
