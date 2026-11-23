#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Dirty-cached loss surface evaluation, raster heatmap texture, shared-X [`TabularSeries`],
//! and camera-invariant 3D vertex cache for [`super::LossLandscapeTool`].

#[path = "cache_overlays.rs"]
mod overlays;

use std::hash::{Hash as _, Hasher as _};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;

use crate::app::SemanticPalette;
use crate::camera::Scene3DBuilder;
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};
use eframe::egui::{self, Color32};

const LOSS_OBJ_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "OBJ File",
    extensions: &["obj"],
    fallback_filename: "loss_landscape.obj",
    write_error_prefix: "Failed to save OBJ mesh",
};
use egui_plot::{ColoredSegment, PlotBounds, TabularSeries};
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::Point3;
use oxidize_core::mesh::{export_mesh_to_obj_string, Mesh, Point3D};

use super::surfaces::{lerp_color, LandscapeSurface, SurfaceGrid};
use super::{LossLandscapeTool, PredictorMode};

type ColorPairKey = ([u8; 4], [u8; 4]);
type SlingshotKey = [u64; 4];

#[derive(Default)]
pub(crate) struct LossLandscapeCache {
    grid_key: Option<u64>,
    cached_grid: Option<SurfaceGrid>,
    pub(super) minima_3d: Vec<Point3<f64>>,
    pub(super) contours: Vec<(f64, Vec<[[f64; 2]; 2]>)>,
    tabular_key: Option<(u64, ColorPairKey)>,
    tabular_series: Arc<Vec<TabularSeries>>,
    texture_key: Option<(u64, ColorPairKey)>,
    backing_image: Option<Arc<egui::ColorImage>>,
    texture: Option<egui::TextureHandle>,
    racer_3d_key: Option<(u64, u64)>,
    pub(super) racer_trails_3d: [Vec<Point3<f64>>; 5],
    pub(super) racer_heads_3d: [Option<Point3<f64>>; 5],
    slingshot_3d_key: Option<(u64, SlingshotKey)>,
    pub(super) slingshot_3d: Option<[Point3<f64>; 2]>,
    scene_3d_key: Option<(u64, u64, SlingshotKey, u64, ColorPairKey, bool)>,
    pub(super) scene_builder: Scene3DBuilder,
    overlay_2d_key: Option<(u64, u64, SlingshotKey, ColorPairKey, bool)>,
    pub(super) overlay_2d_segments: Arc<Vec<ColoredSegment>>,
    pub(super) overlay_2d_bounds: Option<PlotBounds>,
    pub(super) obj_export_rx: Option<Receiver<Result<String, String>>>,
    pub(super) obj_export_status: Option<(String, bool)>,
    pub(crate) eval_count: usize,
    pub(crate) grid_rebuild_count: usize,
    pub(crate) texture_upload_count: usize,
    pub(crate) projection_3d_count: usize,
}

#[inline]
pub(super) fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0
    } else {
        val.to_bits()
    }
}

#[inline]
fn sanitize_loss(val: f64) -> f64 {
    if val.is_finite() {
        val
    } else {
        0.0
    }
}

impl LossLandscapeTool {
    pub(super) fn compute_grid_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.surface as u8).hash(&mut h);
        for v in [self.w_min, self.w_max, self.b_min, self.b_max] {
            canonical_bits(v).hash(&mut h);
        }
        self.resolution.hash(&mut h);
        self.sculpt_bumps.len().hash(&mut h);
        for bump in &self.sculpt_bumps {
            for v in [bump.w, bump.b, bump.amplitude, bump.sigma] {
                canonical_bits(v).hash(&mut h);
            }
        }
        if self.surface == LandscapeSurface::LinearMse {
            self.hash_empirical_key(&mut h);
        }
        h.finish().max(1)
    }

    fn hash_empirical_key(&self, h: &mut std::collections::hash_map::DefaultHasher) {
        (self.predictor_mode as u8).hash(h);
        (self.target_mode as u8).hash(h);
        (self.loss_metric as u8).hash(h);
        canonical_bits(self.huber_delta).hash(h);
        self.cached_targets.len().hash(h);
        for &[x, y] in &self.cached_targets {
            canonical_bits(x).hash(h);
            canonical_bits(y).hash(h);
        }
        if self.predictor_mode == PredictorMode::Custom {
            self.math_input.raw_text.hash(h);
            for &p in self.math_input.param_slice() {
                canonical_bits(p).hash(h);
            }
        }
    }

    pub(super) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.compute_grid_key().hash(&mut h);
        (self.view_mode as u8).hash(&mut h);
        self.camera.hash_state(&mut h);
        self.arena_config.enabled_racers.hash(&mut h);
        for v in [
            self.arena_config.mass,
            self.arena_config.damping,
            self.arena_config.temperature,
            self.arena_config.dt,
            self.arena_config.learning_rate,
            self.arena_config.restitution,
            self.arena_config.spawn[0],
            self.arena_config.spawn[1],
            self.arena_config.slingshot_velocity[0],
            self.arena_config.slingshot_velocity[1],
            self.sculpt_amplitude,
            self.sculpt_sigma,
        ] {
            canonical_bits(v).hash(&mut h);
        }
        self.arena_config.rng_seed.hash(&mut h);
        h.finish().max(1)
    }

    pub(super) fn ensure_cached_grid(&mut self) -> u64 {
        let key = self.compute_grid_key();
        if self.cache.grid_key == Some(key) && self.cache.cached_grid.is_some() {
            return key;
        }
        let res = self
            .resolution
            .clamp(10, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
        let bounds = self.domain_bounds();
        let grid = {
            let eval = self.surface_evaluator();
            SurfaceGrid::sample_eval(eval, bounds, res)
        };
        let minima_3d = {
            let eval = self.surface_evaluator();
            let minima = eval.global_minima();
            let mut pts = Vec::with_capacity(minima.len());
            for &[mw, mb] in minima {
                let mloss = eval.evaluate(mw, mb);
                let wp = grid.world_point(mw, mb, mloss, 0.03);
                pts.push(Point3::new(wp[0], wp[1], wp[2]));
            }
            pts
        };
        let eval_calls = res * res + minima_3d.len();
        self.cache.contours = grid.contour_segments(8);
        self.cache.cached_grid = Some(grid);
        self.cache.minima_3d = minima_3d;
        self.cache.grid_key = Some(key);
        self.cache.eval_count += eval_calls;
        self.cache.grid_rebuild_count += 1;
        key
    }

    pub(super) fn ensure_tabular_series(
        &mut self,
        grid_key: u64,
        palette: &SemanticPalette,
    ) -> Arc<Vec<TabularSeries>> {
        let colors = (palette.axis_z.to_array(), palette.axis_x.to_array());
        if self.cache.tabular_key == Some((grid_key, colors))
            && !self.cache.tabular_series.is_empty()
        {
            return Arc::clone(&self.cache.tabular_series);
        }
        let grid = self.cache.cached_grid.as_ref().expect("populated grid");
        let mut series = Vec::with_capacity(grid.rows.len());
        for row in &grid.rows {
            let Some(&[_, b_val, _]) = row.first() else {
                continue;
            };
            let mut h_sum = 0.0;
            let mut pts = Vec::with_capacity(row.len());
            for &[w, _b, loss] in row {
                let clean = sanitize_loss(loss);
                h_sum += grid.normalized_height(clean);
                pts.push([w, clean]);
            }
            let avg_h = h_sum / (row.len().max(1) as f64);
            let color = lerp_color(palette.axis_z, palette.axis_x, avg_h);
            series.push(TabularSeries::new(
                format!("Loss (b = {b_val:.4})"),
                pts,
                color,
            ));
        }
        let shared = Arc::new(series);
        self.cache.tabular_series = Arc::clone(&shared);
        self.cache.tabular_key = Some((grid_key, colors));
        shared
    }

    pub(super) fn ensure_heatmap_texture(
        &mut self,
        ctx: &egui::Context,
        grid_key: u64,
        palette: &SemanticPalette,
    ) -> (egui::TextureId, Arc<egui::ColorImage>) {
        let colors = (palette.axis_z.to_array(), palette.axis_x.to_array());
        if self.cache.texture_key == Some((grid_key, colors)) {
            if let (Some(ref tex), Some(ref img)) = (&self.cache.texture, &self.cache.backing_image)
            {
                return (tex.id(), Arc::clone(img));
            }
        }
        let grid = self.cache.cached_grid.as_ref().expect("populated grid");
        let img = Arc::new(build_heatmap_color_image(
            grid,
            palette.axis_z,
            palette.axis_x,
        ));
        let handle = ctx.load_texture(
            "loss_landscape_heatmap",
            (*img).clone(),
            egui::TextureOptions::LINEAR,
        );
        let id = handle.id();
        self.cache.texture = Some(handle);
        self.cache.backing_image = Some(Arc::clone(&img));
        self.cache.texture_key = Some((grid_key, colors));
        self.cache.texture_upload_count += 1;
        (id, img)
    }

    /// Serializes the active loss heightfield as a Wavefront `.obj` triangle mesh.
    pub fn build_obj_mesh_string(&self) -> Result<String, String> {
        let res = self
            .resolution
            .clamp(10, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
        let grid = SurfaceGrid::sample_eval(self.surface_evaluator(), self.domain_bounds(), res);
        let mesh = surface_grid_to_mesh(&grid);
        export_mesh_to_obj_string(&mesh).map_err(|e| format!("Failed to serialize OBJ mesh: {e}"))
    }

    pub(super) fn trigger_obj_export(&mut self) {
        match self.build_obj_mesh_string() {
            Ok(obj_content) => {
                let (tx, rx) = mpsc::channel();
                self.cache.obj_export_rx = Some(rx);
                self.cache.obj_export_status = None;
                spawn_save_as_file_dialog(
                    tx,
                    LOSS_OBJ_SAVE_CONFIG,
                    "loss_landscape.obj",
                    obj_content,
                    |path, default_name| {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(default_name)
                            .to_string()
                    },
                );
            }
            Err(err) => self.cache.obj_export_status = Some((err, true)),
        }
    }

    pub(super) fn poll_obj_export(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.cache.obj_export_rx else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(saved_name)) => {
                let msg = format!("Saved OBJ mesh to {saved_name}");
                crate::accessibility::announce_live_status(ctx, &msg);
                self.cache.obj_export_status = Some((msg, false));
                self.cache.obj_export_rx = None;
            }
            Ok(Err(err)) => {
                crate::accessibility::announce_live_assertive_status(ctx, &err);
                self.cache.obj_export_status = Some((err, true));
                self.cache.obj_export_rx = None;
            }
            Err(TryRecvError::Empty) => ctx.request_repaint(),
            Err(TryRecvError::Disconnected) => self.cache.obj_export_rx = None,
        }
    }
}

fn build_heatmap_color_image(grid: &SurfaceGrid, low: Color32, high: Color32) -> egui::ColorImage {
    let h = grid.rows.len().max(1);
    let w = grid.rows.first().map_or(1, Vec::len).max(1);
    let mut pixels = vec![low; w * h];
    for (j, row) in grid.rows.iter().enumerate() {
        let img_y = (h - 1).saturating_sub(j);
        for (i, &[_, _, loss]) in row.iter().enumerate() {
            let t = grid.normalized_height(sanitize_loss(loss));
            pixels[flatten_2d_index(i, img_y, w)] = lerp_color(low, high, t);
        }
    }
    egui::ColorImage::new([w, h], pixels)
}

fn surface_grid_to_mesh(grid: &SurfaceGrid) -> Mesh {
    let rows = grid.rows.len();
    let cols = grid.rows.first().map_or(0, Vec::len);
    let mut vertices = Vec::with_capacity(rows * cols);
    let mut normals = Vec::with_capacity(rows * cols);
    for row in &grid.rows {
        for &[w, b, loss] in row {
            let clean = sanitize_loss(loss);
            vertices.push(Point3D::new(w as f32, b as f32, clean as f32));
            normals.push(Point3D::new(0.0, 0.0, 1.0));
        }
    }
    let mut indices = Vec::with_capacity(rows.saturating_sub(1) * cols.saturating_sub(1) * 6);
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols.saturating_sub(1) {
            let (i00, i01) = (flatten_2d_index(c, r, cols), flatten_2d_index(c + 1, r, cols));
            let (i10, i11) = (
                flatten_2d_index(c, r + 1, cols),
                flatten_2d_index(c + 1, r + 1, cols),
            );
            indices.extend_from_slice(&[i00, i10, i01, i10, i11, i01]);
        }
    }
    Mesh {
        vertices,
        normals,
        indices,
    }
}
