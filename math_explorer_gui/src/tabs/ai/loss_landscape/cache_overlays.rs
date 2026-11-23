#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Consolidated 3D (`Scene3DBuilder`) and 2D (`ColoredSegment`) overlay builders for [`super::super::LossLandscapeTool`].

use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{ColoredSegment, LineStyle, MarkerShape, PlotBounds, PlotPoint};
use nalgebra::Point3;

use super::canonical_bits;
use super::super::surfaces::{lerp_color, SurfaceGrid};
use super::super::LossLandscapeTool;

impl LossLandscapeTool {
    fn arena_fingerprint(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.arena.step_count.hash(&mut h);
        self.arena_config.enabled_racers.hash(&mut h);
        for racer in &self.arena.racers {
            canonical_bits(racer.phase.q.x).hash(&mut h);
            canonical_bits(racer.phase.q.y).hash(&mut h);
            canonical_bits(racer.potential_energy).hash(&mut h);
            racer.trail.len().hash(&mut h);
        }
        h.finish().max(1)
    }

    fn slingshot_fingerprint(&self) -> [u64; 4] {
        match (self.slingshot_anchor, self.slingshot_current) {
            (Some(a), Some(c)) => [
                canonical_bits(a[0]).max(1),
                canonical_bits(a[1]),
                canonical_bits(c[0]),
                canonical_bits(c[1]),
            ],
            _ => [0; 4],
        }
    }

    fn ensure_3d_racer_and_slingshot_vertices(&mut self, grid_key: u64, arena_fp: u64) {
        if self.cache.racer_3d_key != Some((grid_key, arena_fp)) {
            let mut eval_calls = 0usize;
            let mut trails: [Vec<Point3<f64>>; 5] = Default::default();
            let mut heads: [Option<Point3<f64>>; 5] = [None; 5];
            {
                let eval = self.surface_evaluator();
                let grid = self.cache.cached_grid.as_ref().expect("populated grid");
                for (idx, racer) in self.arena.racers.iter().enumerate() {
                    if !self.arena_config.enabled_racers[idx] || !racer.kind.is_available_in_build()
                    {
                        continue;
                    }
                    let mut pts = Vec::with_capacity(racer.trail.len());
                    for [w, b] in racer.trail.iter() {
                        let loss = eval.evaluate(w, b);
                        eval_calls += 1;
                        let wp = grid.world_point(w, b, loss, 0.035);
                        pts.push(Point3::new(wp[0], wp[1], wp[2]));
                    }
                    trails[idx] = pts;
                    let hp = grid.world_point(
                        racer.phase.q.x,
                        racer.phase.q.y,
                        racer.potential_energy,
                        0.045,
                    );
                    heads[idx] = Some(Point3::new(hp[0], hp[1], hp[2]));
                }
            }
            self.cache.racer_trails_3d = trails;
            self.cache.racer_heads_3d = heads;
            self.cache.racer_3d_key = Some((grid_key, arena_fp));
            self.cache.eval_count += eval_calls;
        }

        let sling_fp = self.slingshot_fingerprint();
        if self.cache.slingshot_3d_key != Some((grid_key, sling_fp)) {
            if let (Some(a), Some(c)) = (self.slingshot_anchor, self.slingshot_current) {
                let (pa, pc) = {
                    let eval = self.surface_evaluator();
                    let grid = self.cache.cached_grid.as_ref().expect("populated grid");
                    let la = eval.evaluate(a[0], a[1]);
                    let lc = eval.evaluate(c[0], c[1]);
                    let wa = grid.world_point(a[0], a[1], la, 0.05);
                    let wc = grid.world_point(c[0], c[1], lc, 0.05);
                    (
                        Point3::new(wa[0], wa[1], wa[2]),
                        Point3::new(wc[0], wc[1], wc[2]),
                    )
                };
                self.cache.slingshot_3d = Some([pa, pc]);
                self.cache.eval_count += 2;
            } else {
                self.cache.slingshot_3d = None;
            }
            self.cache.slingshot_3d_key = Some((grid_key, sling_fp));
        }
    }

    pub(crate) fn ensure_3d_scene(
        &mut self,
        grid_key: u64,
        palette: &SemanticPalette,
        dark_mode: bool,
    ) {
        let arena_fp = self.arena_fingerprint();
        self.ensure_3d_racer_and_slingshot_vertices(grid_key, arena_fp);
        let sling_fp = self.slingshot_fingerprint();
        let colors = (palette.axis_z.to_array(), palette.axis_x.to_array());
        let scene_key = (
            grid_key,
            arena_fp,
            sling_fp,
            self.camera.pose_fingerprint(),
            colors,
            dark_mode,
        );
        if self.cache.scene_3d_key == Some(scene_key) {
            return;
        }
        self.rebuild_3d_scene_from_cache(palette);
        self.cache.scene_3d_key = Some(scene_key);
        self.cache.projection_3d_count += 1;
    }

    fn rebuild_3d_scene_from_cache(&mut self, palette: &SemanticPalette) {
        let cam = self.camera;
        self.cache.scene_builder.begin(cam);
        let grid = self.cache.cached_grid.as_ref().expect("populated grid");
        append_wireframe_to_scene(&mut self.cache.scene_builder, grid, palette);

        if !self.cache.minima_3d.is_empty() {
            self.cache.scene_builder.add_points(
                "Global Minimum",
                &self.cache.minima_3d,
                5.5_f32,
                palette.success,
                MarkerShape::Diamond,
                false,
            );
        }
        for (idx, racer) in self.arena.racers.iter().enumerate() {
            if !self.arena_config.enabled_racers[idx] || !racer.kind.is_available_in_build() {
                continue;
            }
            let color = racer.kind.color(palette);
            let trail_3d = &self.cache.racer_trails_3d[idx];
            if trail_3d.len() >= 2 {
                self.cache.scene_builder.add_polyline(
                    racer.kind.label(),
                    trail_3d,
                    egui::Stroke::new(2.2_f32, color),
                    LineStyle::Solid,
                    false,
                );
            }
            if let Some(head) = self.cache.racer_heads_3d[idx] {
                self.cache.scene_builder.add_points(
                    racer.kind.label(),
                    &[head],
                    5.0_f32,
                    color,
                    MarkerShape::Circle,
                    false,
                );
            }
        }
        if let Some([pa, pc]) = self.cache.slingshot_3d {
            self.cache.scene_builder.add_styled_segment(
                "Slingshot Vector",
                [pa, pc],
                egui::Stroke::new(2.5_f32, palette.warning),
                LineStyle::Solid,
                false,
                false,
            );
        }
        self.cache.scene_builder.ensure_sorted();
    }

    pub(crate) fn ensure_2d_overlay(
        &mut self,
        grid_key: u64,
        palette: &SemanticPalette,
        dark_mode: bool,
    ) -> (Arc<Vec<ColoredSegment>>, PlotBounds) {
        let arena_fp = self.arena_fingerprint();
        let sling_fp = self.slingshot_fingerprint();
        let colors = (palette.axis_z.to_array(), palette.axis_x.to_array());
        let key = (grid_key, arena_fp, sling_fp, colors, dark_mode);
        if self.cache.overlay_2d_key == Some(key) {
            if let Some(bounds) = self.cache.overlay_2d_bounds {
                return (Arc::clone(&self.cache.overlay_2d_segments), bounds);
            }
        }
        let (segs, bounds) = self.build_2d_overlay_segments(palette);
        let shared = Arc::new(segs);
        self.cache.overlay_2d_segments = Arc::clone(&shared);
        self.cache.overlay_2d_bounds = Some(bounds);
        self.cache.overlay_2d_key = Some(key);
        (shared, bounds)
    }

    fn build_2d_overlay_segments(
        &self,
        palette: &SemanticPalette,
    ) -> (Vec<ColoredSegment>, PlotBounds) {
        let mut segs = Vec::new();
        let (w_min, w_max, b_min, b_max) = self.domain_bounds();
        let mut bounds = PlotBounds::from_min_max([w_min, b_min], [w_max, b_max]);
        for (iso, iso_segs) in &self.cache.contours {
            let c = lerp_color(palette.axis_z, palette.axis_x, *iso);
            for &[p0, p1] in iso_segs {
                push_colored_polyline(&mut segs, &mut bounds, &[p0, p1], c, 1.2_f32);
            }
        }
        let rx = (w_max - w_min).abs().max(1e-6) * 0.016;
        let ry = (b_max - b_min).abs().max(1e-6) * 0.016;
        for &[mw, mb] in self.surface_evaluator().global_minima() {
            let diamond = [
                [mw, mb + ry],
                [mw + rx, mb],
                [mw, mb - ry],
                [mw - rx, mb],
                [mw, mb + ry],
            ];
            push_colored_polyline(&mut segs, &mut bounds, &diamond, palette.success, 2.0_f32);
        }
        self.append_2d_racers_and_slingshot(&mut segs, &mut bounds, palette, rx * 0.85, ry * 0.85);
        (segs, bounds)
    }

    fn append_2d_racers_and_slingshot(
        &self,
        segs: &mut Vec<ColoredSegment>,
        bounds: &mut PlotBounds,
        palette: &SemanticPalette,
        rx: f64,
        ry: f64,
    ) {
        for (idx, racer) in self.arena.racers.iter().enumerate() {
            if !self.arena_config.enabled_racers[idx] || !racer.kind.is_available_in_build() {
                continue;
            }
            let color = racer.kind.color(palette);
            if racer.trail.len() >= 2 {
                let trail: Vec<[f64; 2]> = racer.trail.iter().collect();
                push_colored_polyline(segs, bounds, &trail, color, 2.2_f32);
            }
            let (qx, qy) = (racer.phase.q.x, racer.phase.q.y);
            let s = std::f64::consts::FRAC_1_SQRT_2;
            let oct = [
                [qx + rx, qy],
                [qx + rx * s, qy + ry * s],
                [qx, qy + ry],
                [qx - rx * s, qy + ry * s],
                [qx - rx, qy],
                [qx - rx * s, qy - ry * s],
                [qx, qy - ry],
                [qx + rx * s, qy - ry * s],
                [qx + rx, qy],
            ];
            push_colored_polyline(segs, bounds, &oct, color, 2.2_f32);
        }
        if let (Some(a), Some(c)) = (self.slingshot_anchor, self.slingshot_current) {
            push_colored_polyline(segs, bounds, &[a, c], palette.warning, 2.5_f32);
        }
    }
}

fn append_wireframe_to_scene(
    scene: &mut crate::camera::Scene3DBuilder,
    grid: &SurfaceGrid,
    palette: &SemanticPalette,
) {
    let num_rows = grid.rows.len();
    if num_rows == 0 {
        return;
    }
    let num_cols = grid.rows[0].len();
    let mut line_pts = Vec::with_capacity(num_cols.max(num_rows));
    for row in &grid.rows {
        line_pts.clear();
        let mut h_sum = 0.0;
        for &[w, b, loss] in row {
            h_sum += grid.normalized_height(loss);
            let wp = grid.world_point(w, b, loss, 0.0);
            line_pts.push(Point3::new(wp[0], wp[1], wp[2]));
        }
        let color = lerp_color(palette.axis_z, palette.axis_x, h_sum / (num_cols.max(1) as f64));
        scene.add_polyline(
            "Surface Mesh",
            &line_pts,
            egui::Stroke::new(1.1_f32, color),
            LineStyle::Solid,
            false,
        );
    }
    for col in 0..num_cols {
        line_pts.clear();
        let mut h_sum = 0.0;
        for row in &grid.rows {
            let [w, b, loss] = row[col];
            h_sum += grid.normalized_height(loss);
            let wp = grid.world_point(w, b, loss, 0.0);
            line_pts.push(Point3::new(wp[0], wp[1], wp[2]));
        }
        let color = lerp_color(palette.axis_z, palette.axis_x, h_sum / (num_rows.max(1) as f64));
        scene.add_polyline(
            "Surface Mesh",
            &line_pts,
            egui::Stroke::new(1.1_f32, color),
            LineStyle::Solid,
            false,
        );
    }
}

fn push_colored_polyline(
    segs: &mut Vec<ColoredSegment>,
    bounds: &mut PlotBounds,
    pts: &[[f64; 2]],
    color: egui::Color32,
    width: f32,
) {
    let mut plot_pts = Vec::with_capacity(pts.len());
    for &[x, y] in pts {
        if x.is_finite() && y.is_finite() {
            let pp = PlotPoint::new(x, y);
            bounds.extend_with(&pp);
            plot_pts.push(pp);
        }
    }
    if plot_pts.len() >= 2 {
        segs.push(ColoredSegment::new(plot_pts, color, width));
    }
}
