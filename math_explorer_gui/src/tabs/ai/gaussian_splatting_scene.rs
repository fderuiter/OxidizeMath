#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Scene delegation layer connecting `GaussianSplattingTool` to shared `crate::volumetric_scene`.
// [cite:gaussian_splatting]

pub use crate::volumetric_scene::lensing::{
    build_circle_ring, kpc_to_viewport, viewport_to_kpc, LensedFrameTelemetry, SplatLensConfig,
    SplatLensMode,
};
pub use crate::volumetric_scene::presets::{
    build_preset_scene_and_gradients, VolumetricSplatPreset,
};
pub use crate::volumetric_scene::{
    covariance_ellipse_ring, splat_stroke_color, DensityActionCounts, ELLIPSE_POINTS,
    MAX_DYNAMIC_SPLATS,
};

use crate::framework::{Camera3D, ProjectionMode};
use crate::volumetric_scene::lensing::{compute_splat_beaming_weight, evaluate_lens_mode_outcome};
use eframe::egui;
use math_explorer::ai::gaussian_splatting::{
    optimization::{determine_density_action, DensityAction},
    projection, rendering, Gaussian2D, Gaussian3D, Scene,
};
use math_explorer::constants::si::SOLAR_MASS;
use math_explorer::physics::astrophysics::{
    tangential_critical_radius_kpc, validate_christoffel_radial_connection, HaloProfile,
};
use math_explorer::physics::high_energy::general_relativity::SchwarzschildBlackHole;
use math_scalars::ViewportLength;
use nalgebra::{Matrix2, Matrix2x3, Point2, Point3, Vector2, Vector3};

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

pub(super) fn project_lensed_into(
    tool: &super::GaussianSplattingTool,
    out_splats: &mut Vec<Gaussian2D>,
    out_rings: &mut Vec<([[f64; 2]; ELLIPSE_POINTS], egui::Color32)>,
    out_rings_3d: &mut Vec<([Point3<f64>; ELLIPSE_POINTS], egui::Color32)>,
    out_centers: &mut Vec<[f64; 2]>,
) -> LensedFrameTelemetry {
    out_rings.clear();
    out_rings_3d.clear();
    out_centers.clear();

    let dist = tool.camera.perspective_distance();
    let view_matrix = tool.camera.view_matrix(dist);
    let zoom = f64::from(tool.camera.zoom);
    let focal = dist * zoom;
    let cam_proj = projection::SplatCameraProjection {
        view_matrix,
        focal: (focal, focal),
        scale_multiplier: tool.scale,
    };
    let bh = SchwarzschildBlackHole::new(1.0e8 * SOLAR_MASS.get()).expect("valid black hole");
    let cam_forward = Vector3::new(
        view_matrix[(2, 0)],
        view_matrix[(2, 1)],
        view_matrix[(2, 2)],
    );
    let (nfw_halo, burk_halo, sigma_crit) = tool.lens.matched_halos();
    let lens_center = Vector2::new(tool.lens.center[0], tool.lens.center[1]);
    let mut masked_rays = 0usize;
    let mut max_boost = 1.0_f64;

    projection::project_scene_lensed_into(
        &tool.scene,
        &cam_proj,
        |_idx, g, _cam_vec, _raw_beta| {
            let [mx, my] = tool.project_splat_point(g.mean);
            let beta_pt = Point2::new(mx, my);
            let rad_w = compute_splat_beaming_weight(&bh, &g.mean.coords, &cam_forward, &tool.lens);
            max_boost = max_boost.max(rad_w);
            evaluate_lens_mode_outcome(
                &bh,
                &tool.lens,
                lens_center,
                &nfw_halo,
                &burk_halo,
                sigma_crit,
                beta_pt,
                rad_w,
                &mut masked_rays,
            )
        },
        out_splats,
    );

    populate_tool_ellipse_rings(
        tool,
        dist,
        zoom,
        focal,
        out_splats,
        out_rings,
        out_rings_3d,
        out_centers,
    );
    build_lensed_telemetry(
        &bh,
        &tool.lens,
        (&nfw_halo, &burk_halo, sigma_crit),
        (out_splats.len(), masked_rays, max_boost),
    )
}

fn build_lensed_telemetry(
    bh: &SchwarzschildBlackHole,
    lens: &SplatLensConfig,
    (nfw_halo, burk_halo, sigma_crit): (&HaloProfile, &HaloProfile, f64),
    (emitted_images, shadow_masked_rays, max_radiance_boost): (usize, usize, f64),
) -> LensedFrameTelemetry {
    let shadow_radius = if lens.mode == SplatLensMode::Schwarzschild {
        1.5 * lens.viewport_rs.max(0.0)
    } else {
        0.0
    };
    let (christoffel_gamma_r00, _) =
        validate_christoffel_radial_connection(bh, 6.0).unwrap_or((0.0, 0.0));
    let r_search_kpc = viewport_to_kpc(ViewportLength::new(6.0), lens.viewport_kpc_scale);
    LensedFrameTelemetry {
        emitted_images,
        shadow_masked_rays,
        shadow_radius,
        nfw_critical_radius: tangential_critical_radius_kpc(
            None,
            nfw_halo,
            sigma_crit,
            r_search_kpc,
        )
        .map(|r_kpc| kpc_to_viewport(r_kpc, lens.viewport_kpc_scale).get()),
        burkert_critical_radius: tangential_critical_radius_kpc(
            None,
            burk_halo,
            sigma_crit,
            r_search_kpc,
        )
        .map(|r_kpc| kpc_to_viewport(r_kpc, lens.viewport_kpc_scale).get()),
        max_radiance_boost,
        christoffel_gamma_r00,
    }
}

#[allow(clippy::too_many_arguments)]
fn populate_tool_ellipse_rings(
    tool: &super::GaussianSplattingTool,
    dist: f64,
    zoom: f64,
    focal: f64,
    out_splats: &[Gaussian2D],
    out_rings: &mut Vec<([[f64; 2]; ELLIPSE_POINTS], egui::Color32)>,
    out_rings_3d: &mut Vec<([Point3<f64>; ELLIPSE_POINTS], egui::Color32)>,
    out_centers: &mut Vec<[f64; 2]>,
) {
    let ring_cap = 48.min(out_rings.capacity()).min(out_rings_3d.capacity());
    for splat in out_splats {
        if out_centers.len() < out_centers.capacity() {
            out_centers.push([splat.mean.x, splat.mean.y]);
        }
        if out_rings.len() >= ring_cap {
            continue;
        }
        let neg_two_conic = -2.0 * splat.conic;
        if let Some(cov2d) = neg_two_conic.try_inverse() {
            let ring_2d = covariance_ellipse_ring(splat.mean, &cov2d);
            let proj_scale = match tool.camera.projection {
                ProjectionMode::Parallel => zoom,
                ProjectionMode::Perspective => focal / splat.depth.max(0.25),
            };
            let vz = dist - splat.depth;
            let tx = splat.mean.x / proj_scale.max(1.0e-6);
            let ty = splat.mean.y / proj_scale.max(1.0e-6);
            let ring_3d =
                lift_ring_to_world(&tool.camera, &ring_2d, splat.mean, [tx, ty, vz], proj_scale);
            let color = splat_stroke_color(&splat.color, splat.opacity);
            out_rings.push((ring_2d, color));
            out_rings_3d.push((ring_3d, color));
        }
    }
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
