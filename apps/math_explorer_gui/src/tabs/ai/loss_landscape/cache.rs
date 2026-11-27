#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Cached surface heightfield, 2D raster/contour/tabular slices, and 3D `Scene3DBuilder`
//! state for [`super::LossLandscapeTool`].

#[path = "scene_3d.rs"]
mod scene_3d;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::app::SemanticPalette;
use crate::camera::{Camera3D, Scene3DBuilder};
use eframe::egui::{self, Color32};
use egui_plot::{PlotBounds, PlotPoint, SharedPoints};

use super::surfaces::{lerp_color, LandscapeSurface, SurfaceGrid};
use super::{LossLandscapeTool, PredictorMode};
use scene_3d::{populate_3d_overlays, populate_3d_wireframe};

pub(super) const NUM_CONTOUR_LEVELS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LossSurfaceKey {
    surface_discriminant: u8,
    bounds_bits: [u64; 4],
    resolution: usize,
    empirical_hash: u64,
    bumps_hash: u64,
}

#[derive(Clone)]
pub(super) struct ContourLevelBatch {
    pub(super) iso: f64,
    pub(super) segments: Arc<Vec<Vec<PlotPoint>>>,
    pub(super) bounds: PlotBounds,
}

#[derive(Clone)]
pub(super) struct LossSliceSeries {
    pub(super) name: String,
    pub(super) points: SharedPoints,
    pub(super) t: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LossScene3DKey {
    surface_key: LossSurfaceKey,
    camera_pose: u64,
    step_count: u64,
    dynamic_hash: u64,
    palette_sig: ([u8; 4], [u8; 4], [u8; 4], [u8; 4]),
}

#[derive(Default)]
pub(crate) struct LossLandscapeCache {
    surface_key: Option<LossSurfaceKey>,
    pub(crate) grid: Option<SurfaceGrid>,
    pub(crate) grid_eval_count: usize,
    pub(super) contours: Vec<ContourLevelBatch>,
    pub(super) tabular_slices: Vec<LossSliceSeries>,
    raster_key: Option<(LossSurfaceKey, [u8; 4], [u8; 4])>,
    raster_image: Option<Arc<egui::ColorImage>>,
    raster_texture: Option<egui::TextureHandle>,
    scene_3d_key: Option<LossScene3DKey>,
    pub(crate) scene_builder: Scene3DBuilder,
    pub(crate) last_2d_contour_item_count: usize,
    pub(crate) last_2d_has_plot_image: bool,
    pub(crate) last_3d_plot_item_count: usize,
}

fn surface_discriminant(surface: LandscapeSurface) -> u8 {
    match surface {
        LandscapeSurface::LinearMse => 0,
        LandscapeSurface::Rosenbrock => 1,
        LandscapeSurface::Rastrigin => 2,
        LandscapeSurface::MonkeySaddle => 3,
    }
}

fn hash_empirical_state(tool: &LossLandscapeTool) -> u64 {
    if tool.surface != LandscapeSurface::LinearMse {
        return 0;
    }
    let mut h = DefaultHasher::new();
    (tool.predictor_mode as u8).hash(&mut h);
    (tool.target_mode as u8).hash(&mut h);
    (tool.loss_metric as u8).hash(&mut h);
    tool.huber_delta.to_bits().hash(&mut h);
    tool.cached_targets.len().hash(&mut h);
    for &[x, y] in &tool.cached_targets {
        x.to_bits().hash(&mut h);
        y.to_bits().hash(&mut h);
    }
    if tool.predictor_mode == PredictorMode::Custom {
        tool.math_input.raw_text.hash(&mut h);
        tool.math_input.active_compiled().is_some().hash(&mut h);
        for &p in tool.math_input.param_slice() {
            p.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

fn hash_sculpt_bumps(tool: &LossLandscapeTool) -> u64 {
    let mut h = DefaultHasher::new();
    tool.sculpt_bumps.len().hash(&mut h);
    for bump in &tool.sculpt_bumps {
        bump.w.to_bits().hash(&mut h);
        bump.b.to_bits().hash(&mut h);
        bump.amplitude.to_bits().hash(&mut h);
        bump.sigma.to_bits().hash(&mut h);
    }
    h.finish()
}

fn build_contour_batches(grid: &SurfaceGrid, levels: usize) -> Vec<ContourLevelBatch> {
    let raw_levels = grid.contour_segments_fixed(levels);
    let mut out = Vec::with_capacity(raw_levels.len());
    for (iso, segs) in raw_levels {
        let mut bounds = PlotBounds::NOTHING;
        let mut batch = Vec::with_capacity(segs.len());
        for [p0, p1] in segs {
            let a = PlotPoint::new(p0[0], p0[1]);
            let b = PlotPoint::new(p1[0], p1[1]);
            bounds.extend_with(&a);
            bounds.extend_with(&b);
            batch.push(vec![a, b]);
        }
        out.push(ContourLevelBatch {
            iso,
            segments: Arc::new(batch),
            bounds,
        });
    }
    out
}

fn build_shared_w_slices(grid: &SurfaceGrid) -> Vec<LossSliceSeries> {
    let num_rows = grid.rows.len();
    if num_rows == 0 {
        return Vec::new();
    }
    let (_, _, b_min, b_max) = grid.bounds;
    let b_step = (b_max - b_min).abs() / (num_rows.saturating_sub(1).max(1) as f64);
    let decimals = if b_step >= 0.05 { 2 } else { 4 };
    let denom = num_rows.saturating_sub(1).max(1) as f64;
    let mut slices = Vec::with_capacity(num_rows);
    for (j, row) in grid.rows.iter().enumerate() {
        let b_val = row.first().map_or(b_min, |pt| pt[1]);
        let pts: Vec<[f64; 2]> = row.iter().map(|&[w, _, loss]| [w, loss]).collect();
        slices.push(LossSliceSeries {
            name: format!("b = {b_val:.decimals$}"),
            points: SharedPoints::from(pts),
            t: (j as f64) / denom,
        });
    }
    slices
}

fn bilinear_grid_loss(grid: &SurfaceGrid, u: f64, v: f64) -> f64 {
    let rows = grid.rows.len();
    let cols = grid.rows.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let fx = (u.clamp(0.0, 1.0) * (cols.saturating_sub(1) as f64)).max(0.0);
    let fy = (v.clamp(0.0, 1.0) * (rows.saturating_sub(1) as f64)).max(0.0);
    let c0 = (fx.floor() as usize).min(cols - 1);
    let r0 = (fy.floor() as usize).min(rows - 1);
    let c1 = (c0 + 1).min(cols - 1);
    let r1 = (r0 + 1).min(rows - 1);
    let tx = fx - (c0 as f64);
    let ty = fy - (r0 as f64);
    let top = grid.rows[r0][c0][2] + tx * (grid.rows[r0][c1][2] - grid.rows[r0][c0][2]);
    let bot = grid.rows[r1][c0][2] + tx * (grid.rows[r1][c1][2] - grid.rows[r1][c0][2]);
    top + ty * (bot - top)
}

fn rasterize_surface_grid(grid: &SurfaceGrid, low: Color32, high: Color32) -> egui::ColorImage {
    let side = (grid.rows.len() * 4).clamp(64, 200);
    let denom = side.saturating_sub(1).max(1) as f64;
    let mut pixels = Vec::with_capacity(side * side);
    for py in 0..side {
        let v = 1.0 - (py as f64) / denom;
        for px in 0..side {
            let u = (px as f64) / denom;
            let loss = bilinear_grid_loss(grid, u, v);
            pixels.push(lerp_color(low, high, grid.normalized_height(loss)));
        }
    }
    egui::ColorImage::new([side, side], pixels)
}

fn hash_dynamic_3d_state(tool: &LossLandscapeTool) -> u64 {
    let mut h = DefaultHasher::new();
    tool.arena_config.enabled_racers.hash(&mut h);
    for racer in &tool.arena.racers {
        racer.phase.q.x.to_bits().hash(&mut h);
        racer.phase.q.y.to_bits().hash(&mut h);
        racer.potential_energy.to_bits().hash(&mut h);
        racer.trail.len().hash(&mut h);
    }
    if let Some([ax, ay]) = tool.slingshot_anchor {
        1u8.hash(&mut h);
        ax.to_bits().hash(&mut h);
        ay.to_bits().hash(&mut h);
    }
    if let Some([cx, cy]) = tool.slingshot_current {
        2u8.hash(&mut h);
        cx.to_bits().hash(&mut h);
        cy.to_bits().hash(&mut h);
    }
    h.finish()
}

impl LossLandscapeTool {
    fn compute_surface_key(&self) -> LossSurfaceKey {
        LossSurfaceKey {
            surface_discriminant: surface_discriminant(self.surface),
            bounds_bits: [
                self.w_min.to_bits(),
                self.w_max.to_bits(),
                self.b_min.to_bits(),
                self.b_max.to_bits(),
            ],
            resolution: self.resolution.clamp(10, 50),
            empirical_hash: hash_empirical_state(self),
            bumps_hash: hash_sculpt_bumps(self),
        }
    }

    pub(crate) fn ensure_surface_cache(&mut self) {
        let key = self.compute_surface_key();
        if self.cache.surface_key == Some(key) && self.cache.grid.is_some() {
            return;
        }
        let grid = SurfaceGrid::sample_eval(
            self.surface_evaluator(),
            self.domain_bounds(),
            self.resolution,
        );
        self.cache.contours = build_contour_batches(&grid, NUM_CONTOUR_LEVELS);
        self.cache.tabular_slices = build_shared_w_slices(&grid);
        self.cache.grid = Some(grid);
        self.cache.surface_key = Some(key);
        self.cache.grid_eval_count += 1;
        self.cache.raster_key = None;
        self.cache.scene_3d_key = None;
    }

    pub(super) fn ensure_2d_raster(
        &mut self,
        ctx: &egui::Context,
        low: Color32,
        high: Color32,
    ) -> Option<(egui::TextureId, Arc<egui::ColorImage>)> {
        self.ensure_surface_cache();
        let surface_key = self.cache.surface_key?;
        let desired_key = (surface_key, low.to_array(), high.to_array());
        if self.cache.raster_key != Some(desired_key)
            || self.cache.raster_image.is_none()
            || self.cache.raster_texture.is_none()
        {
            let grid = self.cache.grid.as_ref()?;
            let img = rasterize_surface_grid(grid, low, high);
            let arc_img = Arc::new(img.clone());
            match self.cache.raster_texture.as_mut() {
                Some(tex) => tex.set(img, egui::TextureOptions::LINEAR),
                None => {
                    self.cache.raster_texture = Some(ctx.load_texture(
                        "loss_landscape_2d_heatmap",
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
            self.cache.raster_image = Some(arc_img);
            self.cache.raster_key = Some(desired_key);
        }
        let tid = self.cache.raster_texture.as_ref()?.id();
        let img = Arc::clone(self.cache.raster_image.as_ref()?);
        Some((tid, img))
    }

    pub(super) fn ensure_3d_scene(&mut self, cam: Camera3D, palette: &SemanticPalette) {
        self.ensure_surface_cache();
        let Some(surface_key) = self.cache.surface_key else {
            return;
        };
        let desired = LossScene3DKey {
            surface_key,
            camera_pose: cam.pose_fingerprint(),
            step_count: self.arena.step_count,
            dynamic_hash: hash_dynamic_3d_state(self),
            palette_sig: (
                palette.axis_z.to_array(),
                palette.axis_x.to_array(),
                palette.success.to_array(),
                palette.warning.to_array(),
            ),
        };
        if self.cache.scene_3d_key == Some(desired) {
            return;
        }
        let mut scene = std::mem::take(&mut self.cache.scene_builder);
        scene.begin(cam);
        if let Some(ref grid) = self.cache.grid {
            populate_3d_wireframe(&mut scene, grid, palette);
            populate_3d_overlays(self, &mut scene, grid, palette);
        }
        scene.ensure_sorted();
        self.cache.scene_builder = scene;
        self.cache.scene_3d_key = Some(desired);
    }

    pub(super) fn cold_datasets_2d(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if !self.cache.tabular_slices.is_empty() {
            return self
                .cache
                .tabular_slices
                .iter()
                .map(|s| (s.name.clone(), s.points.as_slice().to_vec()))
                .collect();
        }
        let grid = SurfaceGrid::sample_eval(
            self.surface_evaluator(),
            self.domain_bounds(),
            self.resolution,
        );
        build_shared_w_slices(&grid)
            .into_iter()
            .map(|s| (s.name, s.points.as_slice().to_vec()))
            .collect()
    }

    pub(super) fn cold_datasets_3d(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        if let Some(ref grid) = self.cache.grid {
            let pts: Vec<[f64; 3]> = grid.rows.iter().flatten().copied().collect();
            return vec![("Loss Surface".to_owned(), pts)];
        }
        let grid = SurfaceGrid::sample_eval(
            self.surface_evaluator(),
            self.domain_bounds(),
            self.resolution,
        );
        let pts: Vec<[f64; 3]> = grid.rows.iter().flatten().copied().collect();
        vec![("Loss Surface".to_owned(), pts)]
    }

    pub(super) fn resolve_export_datasets_2d(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let live = egui_plot::get_last_plot_datasets();
        if !live.is_empty() {
            live
        } else {
            self.cold_datasets_2d()
        }
    }

    pub(super) fn resolve_export_datasets_3d(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        if self.view_mode == super::LandscapeViewMode::Contour2D {
            return Vec::new();
        }
        let live = egui_plot::get_last_plot_datasets_3d();
        if !live.is_empty() {
            live
        } else {
            self.cold_datasets_3d()
        }
    }
}
