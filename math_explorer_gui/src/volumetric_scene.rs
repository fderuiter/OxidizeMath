#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Shared zero-allocation volumetric `Gaussian3D` extraction, EWA projection, and rasterization
//! shared across `QuantumTab`, `AiTab`, and `MorphogenesisTab`.
// [cite:gaussian_splatting]
// [cite:isosurface_extraction]

#[path = "volumetric_scene_presets.rs"]
pub mod presets;

use crate::camera::Camera3D;
use eframe::egui;
use math_explorer::ai::gaussian_splatting::{
    optimization::{determine_density_action, DensityAction},
    projection::{project_scene_into, SplatCameraProjection},
    rendering, Gaussian2D, Gaussian3D, Scene,
};
use math_explorer::applied::isosurface::{CentralDifferenceEstimator, Point3D, VoxelGrid};
use nalgebra::{Matrix2, Point2, Point3, UnitQuaternion, Vector3};

/// Raster resolution for real-time CPU/WASM Gaussian splat alpha compositing.
pub const SPLAT_GRID_SIZE: usize = 72;
/// Maximum number of active volumetric splats retained per frame for zero-allocation bounds.
pub const MAX_DYNAMIC_SPLATS: usize = 384;
/// Number of segments per `1.5σ` projected covariance ellipse ring.
const ELLIPSE_SEGMENTS: usize = 24;
/// Number of polyline vertices per closed covariance ellipse ring.
pub const ELLIPSE_POINTS: usize = ELLIPSE_SEGMENTS + 1;
const ELLIPSE_SIGMA: f64 = 1.5;
/// Positional gradient threshold for `determine_density_action`.
pub const GRAD_THRESHOLD: f64 = 0.10;
/// World-space scale threshold for `determine_density_action`.
pub const SCALE_THRESHOLD: f64 = 0.45;
/// Minimum opacity threshold before `determine_density_action` prunes a splat.
pub const OPACITY_THRESHOLD: f64 = 0.05;

/// Summary counts of adaptive Gaussian density control decisions across a 3D scene.
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

/// Half-space cutting plane filter applied during volumetric extraction (`coord[axis] <= offset` kept).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeCutFilter {
    /// Axis index (`0 = X`, `1 = Y`, `2 = Z`).
    pub axis: usize,
    /// World-space coordinate along `axis`; points with `coord > max_coord` are clipped away.
    pub max_coord: f64,
}

/// Pre-allocated buffers for zero-allocation volumetric `Gaussian3D` extraction and rasterization.
pub struct VolumetricSplatWorkspace {
    /// Active 3D Gaussian scene populated in-place up to `MAX_DYNAMIC_SPLATS`.
    pub scene: Scene,
    /// Per-splat scalar field gradient magnitudes paired with `scene.gaussians`.
    pub grad_magnitudes: Vec<f64>,
    /// Depth-sorted 2D projected splats.
    pub projected_splats: Vec<Gaussian2D>,
    /// Optional `1.5σ` covariance ellipse rings for top-opacity splats.
    pub ellipse_rings: Vec<([[f64; 2]; ELLIPSE_POINTS], egui::Color32)>,
    /// Projected 2D splat centers.
    pub splat_centers: Vec<[f64; 2]>,
    /// Pre-allocated RGBA raster buffer (`SPLAT_GRID_SIZE * SPLAT_GRID_SIZE`).
    pub pixel_buffer: Vec<egui::Color32>,
    /// Latest adaptive density control summary counts.
    pub density_counts: DensityActionCounts,
}

impl VolumetricSplatWorkspace {
    /// Pre-allocates all vectors for up to `max_splats` with zero runtime reallocation.
    pub fn with_capacity(max_splats: usize) -> Self {
        let cap = max_splats.max(16);
        Self {
            scene: Scene {
                gaussians: Vec::with_capacity(cap),
            },
            grad_magnitudes: Vec::with_capacity(cap),
            projected_splats: Vec::with_capacity(cap),
            ellipse_rings: Vec::with_capacity(cap),
            splat_centers: Vec::with_capacity(cap),
            pixel_buffer: vec![egui::Color32::TRANSPARENT; SPLAT_GRID_SIZE * SPLAT_GRID_SIZE],
            density_counts: DensityActionCounts::default(),
        }
    }

    /// Evaluates `determine_density_action` across `self.scene` and `self.grad_magnitudes` in-place.
    pub fn recompute_density_counts(&mut self, scale_multiplier: f64) -> DensityActionCounts {
        let safe_scale = if scale_multiplier.is_finite() {
            scale_multiplier.clamp(0.1, 10.0)
        } else {
            1.0
        };
        let mut counts = DensityActionCounts::default();
        let len = self.scene.gaussians.len().min(self.grad_magnitudes.len());
        for i in 0..len {
            let g = &self.scene.gaussians[i];
            let max_s = g.scale.x.max(g.scale.y).max(g.scale.z) * safe_scale;
            match determine_density_action(
                self.grad_magnitudes[i],
                max_s,
                g.opacity,
                GRAD_THRESHOLD,
                SCALE_THRESHOLD,
                OPACITY_THRESHOLD,
            ) {
                DensityAction::Split => counts.split += 1,
                DensityAction::Clone => counts.clone += 1,
                DensityAction::Keep => counts.keep += 1,
                DensityAction::Prune => counts.prune += 1,
            }
        }
        self.density_counts = counts;
        counts
    }
}

/// Aligns a local Z-axis to a unit surface normal `n` for tangentially flattened splat disks.
pub fn rotation_from_normal(normal: Point3D) -> UnitQuaternion<f64> {
    let target = Vector3::new(normal.x as f64, normal.y as f64, normal.z as f64);
    let norm = target.norm();
    if norm < 1e-6 {
        return UnitQuaternion::identity();
    }
    let unit = target / norm;
    let z_axis = Vector3::z();
    if unit.dot(&z_axis) < -0.999_9 {
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI)
    } else {
        UnitQuaternion::rotation_between(&z_axis, &unit).unwrap_or_else(UnitQuaternion::identity)
    }
}

/// Extracts gradient-aligned `Gaussian3D` splats from `grid` into `ws` with zero heap allocations.
pub fn extract_voxel_splats_into(
    grid: &VoxelGrid,
    min_abs_value: f32,
    signed_palette: bool,
    cut: Option<VolumeCutFilter>,
    ws: &mut VolumetricSplatWorkspace,
) {
    ws.scene.gaussians.clear();
    ws.grad_magnitudes.clear();
    let cap = ws
        .scene
        .gaussians
        .capacity()
        .min(ws.grad_magnitudes.capacity());
    if cap == 0 {
        return;
    }
    let (w, h, d) = (grid.width(), grid.height(), grid.depth());
    let total = w * h * d;
    if total == 0 {
        return;
    }
    let mut peak = 1e-6_f32;
    for &v in grid.data() {
        peak = peak.max(v.abs());
    }
    let threshold = min_abs_value.max(peak * 0.14);
    let stride = ((total / cap.max(1)).max(1) as f64).cbrt().ceil() as usize;
    let step = stride.max(1);
    let vs = grid.voxel_size();
    let orig = grid.origin();
    let base_r = ((vs.x.max(vs.y).max(vs.z)) as f64) * (step as f64) * 0.72;
    let estimator = CentralDifferenceEstimator;

    for z in (0..d).step_by(step) {
        let wz = (orig.z + (z as f32) * vs.z) as f64;
        for y in (0..h).step_by(step) {
            let wy = (orig.y + (y as f32) * vs.y) as f64;
            for x in (0..w).step_by(step) {
                if ws.scene.gaussians.len() >= cap {
                    return;
                }
                let wx = (orig.x + (x as f32) * vs.x) as f64;
                let coords = [wx, wy, wz];
                if let Some(c) = cut {
                    if coords[c.axis.min(2)] > c.max_coord {
                        continue;
                    }
                }
                let val = grid.get(x, y, z);
                if val.abs() < threshold {
                    continue;
                }
                let (normal, grad_mag) = estimator.world_normal_and_magnitude(grid, x, y, z);
                let rel = (val.abs() / peak).clamp(0.0, 1.0) as f64;
                let color = map_volumetric_splat_color(val, rel, signed_palette);
                let normal_compress = if grad_mag > 1e-3 { 0.45 } else { 0.85 };
                ws.scene.gaussians.push(Gaussian3D {
                    mean: Point3::new(wx, wy, wz),
                    scale: Vector3::new(base_r, base_r, base_r * normal_compress),
                    rotation: rotation_from_normal(normal),
                    opacity: (0.12 + 0.76 * rel).clamp(0.03, 0.92),
                    color,
                });
                ws.grad_magnitudes.push((grad_mag as f64) / (peak as f64));
            }
        }
    }
}

fn map_volumetric_splat_color(val: f32, rel: f64, signed_palette: bool) -> Vector3<f64> {
    if signed_palette {
        if val >= 0.0 {
            Vector3::new(0.18 + 0.25 * rel, 0.62 + 0.32 * rel, 0.96)
        } else {
            Vector3::new(0.96, 0.42 + 0.30 * rel, 0.20 + 0.18 * rel)
        }
    } else {
        Vector3::new(0.16 + 0.72 * rel, 0.82 - 0.18 * rel, 0.58 + 0.36 * rel)
    }
}

/// Projects `ws.scene` and rasterizes alpha-blended radiance pixels in-place with zero heap allocations.
pub fn project_and_rasterize_workspace(
    ws: &mut VolumetricSplatWorkspace,
    camera: &Camera3D,
    camera_distance: f64,
    viewport_half_extent: f64,
    scale_multiplier: f64,
    max_rings: usize,
) {
    let view_matrix = camera.view_matrix(camera_distance);
    let focal = camera_distance * (camera.zoom as f64);
    let proj = SplatCameraProjection {
        view_matrix,
        focal: (focal, focal),
        scale_multiplier,
    };
    project_scene_into(&ws.scene, &proj, &mut ws.projected_splats);
    ws.ellipse_rings.clear();
    ws.splat_centers.clear();
    let ring_limit = max_rings
        .min(ws.ellipse_rings.capacity())
        .min(ws.projected_splats.len());
    for splat in ws.projected_splats.iter().take(ring_limit) {
        if ws.splat_centers.len() < ws.splat_centers.capacity() {
            ws.splat_centers.push([splat.mean.x, splat.mean.y]);
        }
        let neg_two_conic = -2.0 * splat.conic;
        if let Some(cov2d) = neg_two_conic.try_inverse() {
            ws.ellipse_rings.push((
                covariance_ellipse_ring(splat.mean, &cov2d),
                splat_stroke_color(&splat.color, splat.opacity),
            ));
        }
    }
    rasterize_splats_into(
        &ws.projected_splats,
        viewport_half_extent,
        &mut ws.pixel_buffer,
    );
}

/// Computes a `1.5σ` 2D covariance contour polyline around `mean`.
pub fn covariance_ellipse_ring(
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

/// Converts normalized RGB + opacity into a premultiplied stroke `Color32`.
pub fn splat_stroke_color(rgb: &Vector3<f64>, opacity: f64) -> egui::Color32 {
    let a = (opacity as f32).clamp(0.12, 0.85);
    let r = (rgb.x as f32).clamp(0.0, 1.0) * a;
    let g = (rgb.y as f32).clamp(0.0, 1.0) * a;
    let b = (rgb.z as f32).clamp(0.0, 1.0) * a;
    egui::Color32::from(egui::Rgba::from_rgba_premultiplied(r, g, b, a))
}

/// Rasterizes depth-sorted 2D Gaussians into `pixels` (`grid_dim x grid_dim`, inferred from length).
pub fn rasterize_splats_into(
    splats: &[Gaussian2D],
    viewport_half_extent: f64,
    pixels: &mut [egui::Color32],
) {
    let grid_dim = (pixels.len() as f64).sqrt() as usize;
    if grid_dim == 0 || grid_dim * grid_dim != pixels.len() {
        return;
    }
    let span = 2.0 * viewport_half_extent;
    let inv_dim = 1.0 / (grid_dim as f64);
    for row_idx in 0..grid_dim {
        let y = viewport_half_extent - span * ((row_idx as f64 + 0.5) * inv_dim);
        let row_base = row_idx * grid_dim;
        for col_idx in 0..grid_dim {
            let x = -viewport_half_extent + span * ((col_idx as f64 + 0.5) * inv_dim);
            let rgb = rendering::blend_gaussians(splats, &Point2::new(x, y));
            let r = (rgb.x as f32).clamp(0.0, 1.0);
            let g = (rgb.y as f32).clamp(0.0, 1.0);
            let b = (rgb.z as f32).clamp(0.0, 1.0);
            let alpha = r.max(g).max(b);
            pixels[row_base + col_idx] =
                egui::Color32::from(egui::Rgba::from_rgba_premultiplied(r, g, b, alpha));
        }
    }
}
