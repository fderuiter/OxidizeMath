#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Scene delegation layer connecting `GaussianSplattingTool` to shared `crate::volumetric_scene`.
// [cite:gaussian_splatting]

pub use crate::volumetric_scene::presets::{
    VolumetricSplatPreset, build_preset_scene_and_gradients,
};
pub use crate::volumetric_scene::{
    DensityActionCounts, ELLIPSE_POINTS, OPACITY_THRESHOLD, covariance_ellipse_ring,
    splat_stroke_color,
};

use eframe::egui;
use math_explorer::ai::gaussian_splatting::{
    Gaussian2D, Scene,
    optimization::{DensityAction, determine_density_action},
};

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
    let len = scene.gaussians.len().min(grads.len());
    for i in 0..len {
        let g = &scene.gaussians[i];
        let max_base = g.scale.x.max(g.scale.y).max(g.scale.z);
        let action = determine_density_action(
            grads[i],
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

pub(super) fn rasterize_splats_into(splats: &[Gaussian2D], pixels: &mut [egui::Color32]) {
    crate::volumetric_scene::rasterize_splats_into(splats, VIEWPORT_HALF_EXTENT, pixels);
}

pub(super) fn project_tool_scene_into(
    scene: &Scene,
    scale: f64,
    camera: &crate::camera::Camera3D,
    out_splats: &mut Vec<Gaussian2D>,
    out_rings: &mut Vec<([[f64; 2]; ELLIPSE_POINTS], egui::Color32)>,
    out_centers: &mut Vec<[f64; 2]>,
) {
    use math_explorer::ai::gaussian_splatting::{Gaussian3D, projection};
    use nalgebra::{Matrix2, Point2, Vector3};

    out_splats.clear();
    out_rings.clear();
    out_centers.clear();
    let scale_factor = if scale.is_finite() {
        scale.clamp(0.1, 10.0)
    } else {
        1.0
    };
    let view_matrix = camera.view_matrix(CAMERA_DISTANCE);
    let focal = CAMERA_DISTANCE * (camera.zoom as f64);
    let mut rings_pushed = 0usize;

    for base in &scene.gaussians {
        let scaled = Gaussian3D {
            mean: base.mean,
            scale: base.scale * scale_factor,
            rotation: base.rotation,
            opacity: base.opacity,
            color: base.color,
        };
        let [tx, ty, vz] =
            camera.rotate_to_camera_space(&[scaled.mean.x, scaled.mean.y, scaled.mean.z]);
        let tz = (CAMERA_DISTANCE - vz).max(0.25);
        let cam_pos = Vector3::new(tx, ty, tz);
        let jacobian = projection::compute_jacobian(&cam_pos, focal, focal);
        let cov3d = scaled.compute_covariance();
        let cov2d = projection::project_covariance(&cov3d, &view_matrix, &jacobian)
            + Matrix2::from_diagonal_element(1e-4);
        let Some(inv_cov) = cov2d.try_inverse() else {
            continue;
        };
        let mean2d = Point2::new(focal * tx / tz, focal * ty / tz);
        if rings_pushed < 48 {
            out_rings.push((
                covariance_ellipse_ring(mean2d, &cov2d),
                splat_stroke_color(&scaled.color, scaled.opacity),
            ));
            rings_pushed += 1;
        }
        out_centers.push([mean2d.x, mean2d.y]);
        out_splats.push(Gaussian2D {
            mean: mean2d,
            conic: -0.5 * inv_cov,
            opacity: scaled.opacity,
            color: scaled.color,
            depth: tz,
        });
    }
    out_splats.sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
}

