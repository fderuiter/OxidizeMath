#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Direct-space tiling canvas, reciprocal-space 256x256 Bragg canvas, and `.png`/`.svg`/`.csv` export builders.

// [cite:solid_state]

use super::{DiffractionColormap, FFT_GRID_SIZE, QuasicrystalStudio, StudioPatternMode};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui;
use math_commons::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color_high_contrast, mapped_gradient,
};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::solid_state::quasicrystal::{ProjectedTile2D, QuasicrystalTileKind};
use std::f64::consts::PI;

impl QuasicrystalStudio {
    pub(super) fn draw_dual_canvases(&mut self, ui: &mut egui::Ui) {
        let theme = ThemeMode::from_visuals(ui.visuals());
        if self.texture_dirty || self.bragg_texture.is_none() {
            let img = self.build_bragg_color_image(theme);
            if let Some(tex) = &mut self.bragg_texture {
                tex.set(img, egui::TextureOptions::LINEAR);
            } else {
                self.bragg_texture = Some(ui.ctx().load_texture(
                    "quasicrystal_bragg_256",
                    img,
                    egui::TextureOptions::LINEAR,
                ));
            }
            self.texture_dirty = false;
        }

        let avail = ui.available_size();
        let stacked = avail.x < 560.0;
        if stacked {
            let pane_h = (0.48 * avail.y).max(160.0);
            self.draw_direct_space_canvas(ui, egui::vec2(avail.x, pane_h));
            ui.add_space(6.0);
            self.draw_reciprocal_space_canvas(ui, egui::vec2(avail.x, pane_h));
        } else {
            let pane_w = ((avail.x - 10.0) * 0.5).max(160.0);
            let pane_h = avail.y.max(200.0);
            ui.horizontal(|ui| {
                self.draw_direct_space_canvas(ui, egui::vec2(pane_w, pane_h));
                ui.add_space(6.0);
                self.draw_reciprocal_space_canvas(ui, egui::vec2(pane_w, pane_h));
            });
        }
    }

    fn direct_space_a11y_summary(&self) -> String {
        match self.state.mode {
            StudioPatternMode::Quasicrystal => format!(
                "{}, {} tiles, {} flipped, w = ({:.3}, {:.3})",
                self.state.quasicrystal.label(),
                self.qc_workspace.stats().tile_count,
                self.qc_workspace.stats().flipped_tile_count,
                self.state.phason_x,
                self.state.phason_y
            ),
            StudioPatternMode::PeriodicBravais => format!(
                "{}, {} sites",
                self.state.bravais.label(),
                self.periodic_sites.len()
            ),
        }
    }

    fn draw_direct_space_canvas(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let summary = self.direct_space_a11y_summary();
        register_canvas_accessibility(
            ui,
            &resp,
            egui::accesskit::Role::GraphicsDocument,
            "Direct-Space 2D Tiling and Phason Flip Canvas",
            Some(&summary),
        );
        if resp.clicked() || resp.dragged() {
            resp.request_focus();
        }
        lock_canvas_arrow_keys(ui, &resp);
        let is_focused = resp.has_focus() || ui.ctx().memory(|m| m.has_focus(resp.id));
        if is_focused {
            self.focused_canvas = Some(resp.id);
            self.handle_canvas_arrow_keys(ui);
        } else if self.focused_canvas == Some(resp.id) {
            self.focused_canvas = None;
        }
        if resp.dragged() && self.state.mode == StudioPatternMode::Quasicrystal {
            let d = resp.drag_delta();
            if d.length_sq() > 0.0 {
                self.step_phason_zero_alloc([f64::from(d.x) * 0.008, -f64::from(d.y) * 0.008]);
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Inside,
        );
        let scale = 0.45 * rect.width().min(rect.height()) / (self.state.aperture_radius as f32);
        let center = rect.center();
        match self.state.mode {
            StudioPatternMode::Quasicrystal => self.paint_quasicrystal_tiles(&painter, center, scale, palette),
            StudioPatternMode::PeriodicBravais => self.paint_bravais_sites(&painter, center, scale, palette),
        }
        painter.text(
            rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            "Direct Space E_parallel (Drag / Arrows shift Phason w)",
            egui::FontId::proportional(12.0),
            palette.node_stroke,
        );
        paint_canvas_focus_ring(ui, &resp);
    }

    fn handle_canvas_arrow_keys(&mut self, ui: &mut egui::Ui) {
        if self.state.mode != StudioPatternMode::Quasicrystal {
            return;
        }
        let step = 0.035;
        let ctx = ui.ctx().clone();
        let (left, right, up, down) = ctx.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
            )
        });
        if left || right || up || down {
            ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
        }
        let delta = [
            (if right { step } else { 0.0 }) - (if left { step } else { 0.0 }),
            (if up { step } else { 0.0 }) - (if down { step } else { 0.0 }),
        ];
        if delta[0] != 0.0 || delta[1] != 0.0 {
            self.step_phason_zero_alloc(delta);
            announce_live_status(
                ui.ctx(),
                format!(
                    "Phason shifted to ({:.3}, {:.3}), {} flipped tiles",
                    self.state.phason_x,
                    self.state.phason_y,
                    self.qc_workspace.stats().flipped_tile_count
                ),
            );
        }
    }

    fn paint_quasicrystal_tiles(
        &self,
        painter: &egui::Painter,
        center: egui::Pos2,
        scale: f32,
        palette: SemanticPalette,
    ) {
        let to_screen = |pt: [f64; 2]| -> egui::Pos2 {
            egui::pos2(
                center.x + (pt[0] as f32) * scale,
                center.y - (pt[1] as f32) * scale,
            )
        };
        for tile in self.qc_workspace.tiles() {
            let fill = self.tile_fill_color(tile, palette);
            let stroke = if self.state.highlight_flipped_tiles && tile.phason_flipped {
                egui::Stroke::new(1.5_f32, palette.accent)
            } else {
                egui::Stroke::new(0.8_f32, palette.graph_edge)
            };
            let n = usize::from(tile.vertex_count);
            let pts: Vec<egui::Pos2> = tile
                .vertices
                .iter()
                .take(n)
                .map(|&v| to_screen(v))
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
    }

    fn tile_fill_color(&self, tile: &ProjectedTile2D, palette: SemanticPalette) -> egui::Color32 {
        if self.state.highlight_flipped_tiles && tile.phason_flipped {
            return palette.warning.gamma_multiply(0.72);
        }
        let base = match tile.kind {
            QuasicrystalTileKind::PenroseThickRhomb
            | QuasicrystalTileKind::AmmannRhomb
            | QuasicrystalTileKind::StampfliTriangle => palette.series[0],
            QuasicrystalTileKind::PenroseThinRhomb
            | QuasicrystalTileKind::AmmannSquare
            | QuasicrystalTileKind::StampfliSquare => palette.series[2],
            QuasicrystalTileKind::StampfliShield => palette.series[4],
        };
        base.gamma_multiply(0.34)
    }

    fn paint_bravais_sites(
        &self,
        painter: &egui::Painter,
        center: egui::Pos2,
        scale: f32,
        palette: SemanticPalette,
    ) {
        let to_screen = |x: f64, y: f64| -> egui::Pos2 {
            egui::pos2(center.x + (x as f32) * scale, center.y - (y as f32) * scale)
        };
        for &[x, y] in &self.periodic_sites {
            painter.circle_filled(to_screen(x, y), 2.6, palette.info);
        }
        if let Ok(lat) = self.current_bravais_params() {
            let [a1, a2] = lat.direct_basis();
            let o = to_screen(0.0, 0.0);
            let p1 = to_screen(a1.x, a1.y);
            let p2 = to_screen(a2.x, a2.y);
            let p12 = to_screen(a1.x + a2.x, a1.y + a2.y);
            painter.add(egui::Shape::line(
                vec![o, p1, p12, p2, o],
                egui::Stroke::new(2.0_f32, palette.accent),
            ));
            painter.line_segment([o, p1], egui::Stroke::new(2.4_f32, palette.axis_x));
            painter.line_segment([o, p2], egui::Stroke::new(2.4_f32, palette.axis_y));
        }
    }

    fn draw_reciprocal_space_canvas(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::focusable_noninteractive());
        let rep = self.fft_workspace.report();
        let summary = format!(
            "256x256 Bragg FFT, C8={:.3}, C10={:.3}, C12={:.3}",
            rep.correlation(8),
            rep.correlation(10),
            rep.correlation(12)
        );
        register_canvas_accessibility(
            ui,
            &resp,
            egui::accesskit::Role::GraphicsDocument,
            "Reciprocal-Space 256x256 Bragg Diffraction and Brillouin Zone Canvas",
            Some(&summary),
        );

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::BLACK);
        let side = rect.width().min(rect.height());
        let img_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(side, side));
        if let Some(tex) = &self.bragg_texture {
            painter.image(
                tex.id(),
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        self.paint_reciprocal_overlays(&painter, img_rect, palette);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            "Reciprocal Space k (256x256 Bragg |F(k)|^2)",
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
        paint_canvas_focus_ring(ui, &resp);
    }

    fn paint_reciprocal_overlays(
        &self,
        painter: &egui::Painter,
        img_rect: egui::Rect,
        palette: SemanticPalette,
    ) {
        let dk = self.fft_workspace.report().delta_k.max(1e-9);
        let px_scale = img_rect.width() / (FFT_GRID_SIZE as f32);
        let k_to_screen = |kx: f64, ky: f64| -> egui::Pos2 {
            egui::pos2(
                img_rect.center().x + ((kx / dk) as f32) * px_scale,
                img_rect.center().y - ((ky / dk) as f32) * px_scale,
            )
        };

        if self.state.mode == StudioPatternMode::PeriodicBravais && self.state.show_brillouin_zone {
            if let Ok(lat) = self.current_bravais_params() {
                let bz = lat.first_brillouin_zone();
                let mut pts: Vec<egui::Pos2> =
                    bz.vertices().iter().map(|v| k_to_screen(v[0], v[1])).collect();
                if let Some(&first) = pts.first() {
                    pts.push(first);
                    painter.add(egui::Shape::line(
                        pts,
                        egui::Stroke::new(2.0_f32, lift_dark_overlay(palette.accent)),
                    ));
                }
                let [b1, b2] = lat.reciprocal_basis();
                let o = img_rect.center();
                painter.line_segment(
                    [o, k_to_screen(b1.x, b1.y)],
                    egui::Stroke::new(1.8_f32, lift_dark_overlay(palette.axis_x)),
                );
                painter.line_segment(
                    [o, k_to_screen(b2.x, b2.y)],
                    egui::Stroke::new(1.8_f32, lift_dark_overlay(palette.axis_y)),
                );
            }
        }

        if self.state.show_bragg_rings {
            if let Some(top) = self.fft_workspace.peaks().first() {
                let kr = (top.k[0].hypot(top.k[1]) / dk) as f32 * px_scale;
                let ring_col = lift_dark_overlay(palette.info);
                let peak_col = lift_dark_overlay(palette.accent);
                painter.circle_stroke(img_rect.center(), kr, egui::Stroke::new(1.1_f32, ring_col));
                for peak in self.fft_workspace.peaks().iter().take(12) {
                    let p = k_to_screen(peak.k[0], peak.k[1]);
                    if img_rect.contains(p) {
                        painter.circle_stroke(p, 4.0, egui::Stroke::new(1.4_f32, peak_col));
                    }
                }
            }
        }
    }

    pub(super) fn build_bragg_color_image(&self, theme: ThemeMode) -> egui::ColorImage {
        let n = self.fft_workspace.size();
        let map = self.fft_workspace.intensity_map();
        let max_val = self.fft_workspace.report().max_non_dc_intensity.max(1e-12);
        let gain_scale = 10.0_f64.powf(self.state.log_gain);
        let denom = (1.0 + gain_scale).ln();
        let mut pixels = Vec::with_capacity(n * n);

        for y in 0..n {
            let src_y = n - 1 - y;
            for x in 0..n {
                let norm = (map[flatten_2d_index(x, src_y, n)] / max_val).clamp(0.0, 1.5);
                let t = (((1.0 + gain_scale * norm).ln()) / denom).clamp(0.0, 1.0);
                pixels.push(map_diffraction_color(t, self.state.colormap, theme));
            }
        }
        egui::ColorImage {
            size: [n, n],
            source_size: egui::vec2(n as f32, n as f32),
            pixels,
        }
    }

    pub(super) fn build_export_overlay_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        let dk = self.fft_workspace.report().delta_k.max(1e-9);
        let half = 0.5 * (FFT_GRID_SIZE as f64);
        let k_to_px = |kx: f64, ky: f64| -> [f64; 2] { [half + kx / dk, half + ky / dk] };
        let mut series = Vec::new();

        if self.state.mode == StudioPatternMode::PeriodicBravais {
            if let Ok(lat) = self.current_bravais_params() {
                let bz = lat.first_brillouin_zone();
                let mut pts: Vec<[f64; 2]> =
                    bz.vertices().iter().map(|v| k_to_px(v[0], v[1])).collect();
                if let Some(&first) = pts.first() {
                    pts.push(first);
                }
                series.push(("First Brillouin Zone".to_string(), pts));
            }
        } else if let Some(top) = self.fft_workspace.peaks().first() {
            let kr = top.k[0].hypot(top.k[1]);
            let ring: Vec<[f64; 2]> = (0..=64)
                .map(|m| {
                    let (sn, cs) = (2.0 * PI * (m as f64) / 64.0).sin_cos();
                    k_to_px(kr * cs, kr * sn)
                })
                .collect();
            series.push(("Primary Bragg Ring".to_string(), ring));
        }
        egui_plot::export_primitives::datasets_to_primitives(&series)
    }

    pub(super) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::with_capacity(5);
        out.push((
            "Direct-Space Sites (x, y)".to_string(),
            self.active_sites().to_vec(),
        ));
        let peaks: Vec<[f64; 2]> = self.fft_workspace.peaks().iter().map(|p| p.k).collect();
        if !peaks.is_empty() {
            out.push(("Dominant Bragg Peaks (kx, ky)".to_string(), peaks));
            let sf: Vec<[f64; 2]> = self
                .fft_workspace
                .peaks()
                .iter()
                .map(|p| [p.k[0].hypot(p.k[1]), p.structure_factor_amp])
                .collect();
            out.push(("Kinematic Structure Factor (|k|, |S(k)|)".to_string(), sf));
        }
        let rep = self.fft_workspace.report();
        let mut corr = vec![[2.0, rep.c2]];
        for order in 4..=12 {
            corr.push([order as f64, rep.correlation(order)]);
        }
        out.push(("Rotational Symmetry Correlation (n, C_n)".to_string(), corr));
        if self.state.mode == StudioPatternMode::PeriodicBravais {
            if let Ok(lat) = self.current_bravais_params() {
                out.push((
                    "First Brillouin Zone (kx, ky)".to_string(),
                    lat.first_brillouin_zone().vertices().to_vec(),
                ));
            }
        }
        out
    }
}

pub(super) fn lift_dark_overlay(c: egui::Color32) -> egui::Color32 {
    let [r, g, b] = ensure_dark_canvas_contrast([c.r(), c.g(), c.b()]);
    egui::Color32::from_rgb(r, g, b)
}

pub(super) fn map_diffraction_color(
    t: f64,
    cm: DiffractionColormap,
    theme: ThemeMode,
) -> egui::Color32 {
    let cutoff = if theme == ThemeMode::HighContrast {
        0.04
    } else {
        0.015
    };
    if !t.is_finite() || t <= cutoff {
        return egui::Color32::BLACK;
    }
    let u = if theme == ThemeMode::HighContrast {
        t.powf(0.75).clamp(0.0, 1.0)
    } else {
        t.clamp(0.0, 1.0)
    };
    let [r, g, b] = match cm {
        DiffractionColormap::Inferno => heatmap_color_high_contrast(u),
        DiffractionColormap::Viridis => ensure_dark_canvas_contrast(mapped_gradient(u, 0.0, 1.0)),
        DiffractionColormap::Monochrome => {
            let gray = egui::Color32::WHITE.gamma_multiply(u as f32);
            ensure_dark_canvas_contrast([gray.r(), gray.g(), gray.b()])
        }
    };
    egui::Color32::from_rgb(r, g, b)
}
