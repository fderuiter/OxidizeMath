#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::interaction::compute_split_layout;
use super::state::{ColormapMode, PropagationMode};
use super::ExcitableSheetTool;
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::app::palette::{SemanticPalette, DARK_PALETTE};
use crate::app::theme::ThemeMode;
use eframe::egui::{self, Color32, ColorImage, FontId, Pos2, Rect, Stroke, Vec2};
use math_explorer::biology::neuroscience::DEFAULT_SHEET_DIM;
use math_explorer::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_explorer::math_kernel::types::flatten_2d_index;

impl ExcitableSheetTool {
    /// Renders the interactive tissue canvas, virtual electrodes, and 3-lead waveforms.
    pub fn draw_canvas(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        painter: &egui::Painter,
    ) {
        if self.cfg.running {
            self.advance_frame_within_budget();
            ui.ctx().request_repaint();
        }
        self.focused_canvas = response.has_focus().then_some(response.id);

        let summary = self.accessibility_value_summary();
        register_canvas_accessibility(
            ui,
            response,
            egui::accesskit::Role::GraphicsDocument,
            "2D Excitable Sheet & Virtual Electrodes Visualization",
            Some(&summary),
        );
        lock_canvas_arrow_keys(ui, response);

        let palette = ThemeMode::from_visuals(ui.visuals()).palette();
        let bg = ui.visuals().extreme_bg_color;
        let layout = compute_split_layout(response.rect);
        self.upload_and_paint_heatmap(ui.ctx(), painter, layout.sheet_rect, &palette);
        self.paint_leads_and_cursor(painter, layout.sheet_rect, &palette, bg);
        self.paint_three_lead_traces(painter, layout.ecg_rect, &palette, bg);
        paint_canvas_focus_ring(ui, response);
    }

    /// Builds an upscaled `384x384` `ColorImage` of the current voltage field for PNG/SVG/GIF export.
    pub fn build_export_raster_image(&self) -> ColorImage {
        let base = self.sample_base_color_image(&DARK_PALETTE);
        let scale = 4_usize;
        let out_w = DEFAULT_SHEET_DIM * scale;
        let out_h = DEFAULT_SHEET_DIM * scale;
        let mut pixels = vec![Color32::BLACK; out_w * out_h];
        for y in 0..out_h {
            let sy = y / scale;
            for x in 0..out_w {
                let sx = x / scale;
                let src_idx = flatten_2d_index(sx, sy, DEFAULT_SHEET_DIM);
                let dst_idx = flatten_2d_index(x, y, out_w);
                pixels[dst_idx] = base.pixels[src_idx];
            }
        }
        ColorImage {
            size: [out_w, out_h],
            source_size: Vec2::new(out_w as f32, out_h as f32),
            pixels,
        }
    }

    /// Builds tabular `(time, potential)` datasets for all 3 leads (`ExportCapabilities::plot()`).
    pub fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let names = self.lead_channel_names();
        names
            .into_iter()
            .enumerate()
            .map(|(k, label)| {
                let pts: Vec<[f64; 2]> = self
                    .trace_history
                    .iter()
                    .map(|&(t, vals)| [t, vals[k]])
                    .collect();
                (label.to_string(), pts)
            })
            .collect()
    }

    fn upload_and_paint_heatmap(
        &mut self,
        ctx: &egui::Context,
        painter: &egui::Painter,
        sheet_rect: Rect,
        palette: &SemanticPalette,
    ) {
        let img = self.sample_base_color_image(palette);
        match &mut self.sheet_texture {
            Some(tex) => tex.set(img, egui::TextureOptions::NEAREST),
            None => {
                self.sheet_texture = Some(ctx.load_texture(
                    "excitable_sheet_heatmap",
                    img,
                    egui::TextureOptions::NEAREST,
                ));
            }
        }
        if let Some(tex) = &self.sheet_texture {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), sheet_rect, uv, Color32::WHITE);
        }
        painter.rect_stroke(
            sheet_rect,
            2.0_f32,
            Stroke::new(1.2_f32, palette.graph_edge),
            egui::StrokeKind::Inside,
        );
        if self.cfg.propagation_mode == PropagationMode::HodgkinHuxleyCable1D {
            self.paint_cable_voltage_overlay(painter, sheet_rect, palette);
        }
    }

    fn sample_base_color_image(&self, palette: &SemanticPalette) -> ColorImage {
        let n = DEFAULT_SHEET_DIM;
        let mut pixels = vec![Color32::BLACK; n * n];
        if self.cfg.propagation_mode == PropagationMode::FitzHughNagumo2D {
            let v = self.sheet.voltage();
            let scars = self.sheet.scar_mask();
            for y in 0..n {
                for x in 0..n {
                    let idx = flatten_2d_index(x, y, n);
                    pixels[idx] = if scars[idx] != 0 {
                        Self::scar_pixel_color(x, y, palette)
                    } else {
                        self.map_voltage_color(((v[idx] + 1.5) / 3.5).clamp(0.0, 1.0))
                    };
                }
            }
        } else {
            let scars = self.cable.scar_mask();
            for y in 0..n {
                for (x, &scar) in scars.iter().enumerate().take(n) {
                    let idx = flatten_2d_index(x, y, n);
                    pixels[idx] = if scar != 0 {
                        Self::scar_pixel_color(x, y, palette)
                    } else {
                        let norm = ((self.cable_kymograph[idx] + 80.0) / 120.0).clamp(0.0, 1.0);
                        self.map_voltage_color(norm)
                    };
                }
            }
        }
        ColorImage {
            size: [n, n],
            source_size: Vec2::new(n as f32, n as f32),
            pixels,
        }
    }

    fn map_voltage_color(&self, norm: f64) -> Color32 {
        let [r, g, b] = match self.cfg.colormap {
            ColormapMode::Perceptual => heatmap_color(norm),
            ColormapMode::HighContrast => heatmap_color_high_contrast(norm),
        };
        Color32::from_rgb(r, g, b)
    }

    fn scar_pixel_color(x: usize, y: usize, palette: &SemanticPalette) -> Color32 {
        if (x + y).is_multiple_of(4) {
            palette.warning
        } else {
            palette.surface_fill
        }
    }

    fn paint_cable_voltage_overlay(
        &self,
        painter: &egui::Painter,
        sheet_rect: Rect,
        palette: &SemanticPalette,
    ) {
        let v = self.cable.voltage();
        if v.len() < 2 {
            return;
        }
        let pts: Vec<Pos2> = v
            .iter()
            .enumerate()
            .map(|(i, &mv)| {
                let ux = i as f32 / (v.len() - 1) as f32;
                let uy = 1.0 - (((mv + 80.0) / 125.0).clamp(0.0, 1.0) as f32);
                egui::pos2(
                    sheet_rect.left() + ux * sheet_rect.width(),
                    sheet_rect.top() + (0.12 + 0.76 * uy) * sheet_rect.height(),
                )
            })
            .collect();
        painter.add(egui::Shape::line(
            pts,
            Stroke::new(2.2_f32, palette.node_stroke),
        ));
    }

    fn paint_leads_and_cursor(
        &self,
        painter: &egui::Painter,
        sheet_rect: Rect,
        palette: &SemanticPalette,
        bg: Color32,
    ) {
        let p0 = Self::grid_to_screen(self.cfg.leads[0].x, self.cfg.leads[0].y, sheet_rect);
        let p1 = Self::grid_to_screen(self.cfg.leads[1].x, self.cfg.leads[1].y, sheet_rect);
        let p2 = Self::grid_to_screen(self.cfg.leads[2].x, self.cfg.leads[2].y, sheet_rect);
        let triangle_stroke = Stroke::new(1.0_f32, palette.graph_edge);
        painter.line_segment([p0, p1], triangle_stroke);
        painter.line_segment([p1, p2], triangle_stroke);
        painter.line_segment([p2, p0], triangle_stroke);

        let tags = ["L1", "L2", "L3"];
        for (i, pos) in [p0, p1, p2].into_iter().enumerate() {
            let active = self.cfg.active_lead == i;
            let rad = if active { 8.5_f32 } else { 6.8_f32 };
            painter.circle_filled(pos, rad, bg);
            painter.circle_stroke(pos, rad, Stroke::new(2.2_f32, palette.series[i]));
            painter.text(
                pos + egui::vec2(10.0, -8.0),
                egui::Align2::LEFT_CENTER,
                tags[i],
                FontId::proportional(11.5),
                Color32::WHITE,
            );
        }

        let cur = Self::grid_to_screen(self.cursor_grid.0, self.cursor_grid.1, sheet_rect);
        let r_px = (self.cfg.brush_radius as f32 / DEFAULT_SHEET_DIM as f32)
            * sheet_rect.width().max(4.0);
        painter.circle_stroke(cur, r_px.max(4.0), Stroke::new(1.4_f32, palette.node_stroke));
    }

    fn paint_three_lead_traces(
        &self,
        painter: &egui::Painter,
        ecg_rect: Rect,
        palette: &SemanticPalette,
        bg: Color32,
    ) {
        if ecg_rect.height() < 20.0 || ecg_rect.width() < 40.0 {
            return;
        }
        painter.rect_filled(ecg_rect, 4.0_f32, bg);
        painter.rect_stroke(
            ecg_rect,
            4.0_f32,
            Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Inside,
        );

        let names = self.lead_channel_names();
        let peak = self
            .trace_history
            .iter()
            .flat_map(|(_, v)| v.iter().map(|x| x.abs()))
            .fold(0.25_f64, f64::max)
            * 1.15;
        let lane_h = ecg_rect.height() / 3.0;
        let n_pts = self.trace_history.len();

        for (k, &trace_col) in palette.series.iter().take(3).enumerate() {
            let y_top = ecg_rect.top() + k as f32 * lane_h;
            let y_mid = y_top + lane_h * 0.5;
            painter.line_segment(
                [
                    egui::pos2(ecg_rect.left() + 4.0, y_mid),
                    egui::pos2(ecg_rect.right() - 4.0, y_mid),
                ],
                Stroke::new(0.7_f32, palette.surface_fill),
            );
            if n_pts >= 2 {
                let pts: Vec<Pos2> = self
                    .trace_history
                    .iter()
                    .enumerate()
                    .map(|(idx, &(_, vals))| {
                        let ux = idx as f32 / (n_pts - 1) as f32;
                        let norm = (vals[k] / peak).clamp(-1.0, 1.0) as f32;
                        egui::pos2(
                            ecg_rect.left() + 6.0 + ux * (ecg_rect.width() - 12.0),
                            y_mid - norm * (lane_h * 0.40),
                        )
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, Stroke::new(1.8_f32, trace_col)));
            }
            let latest = self.trace_history.back().map_or(0.0, |(_, v)| v[k]);
            painter.text(
                egui::pos2(ecg_rect.left() + 8.0, y_top + 3.0),
                egui::Align2::LEFT_TOP,
                format!("{}: {:+.3}", names[k], latest),
                FontId::monospace(11.0),
                trace_col,
            );
        }
    }

    pub(super) fn lead_channel_names(&self) -> [&'static str; 3] {
        if self.cfg.readout_bipolar {
            ["Lead I (LA-RA)", "Lead II (LL-RA)", "Lead III (LL-LA)"]
        } else {
            ["LFP Lead 1", "LFP Lead 2", "LFP Lead 3"]
        }
    }

    fn accessibility_value_summary(&self) -> String {
        let latest = self.trace_history.back().map_or([0.0; 3], |(_, v)| *v);
        let mode_str = match self.cfg.propagation_mode {
            PropagationMode::FitzHughNagumo2D => "2D FitzHugh-Nagumo 96x96",
            PropagationMode::HodgkinHuxleyCable1D => "1D Hodgkin-Huxley 96-compartment cable",
        };
        format!(
            "{mode_str}; cursor ({:.0}, {:.0}); traces [{:+.3}, {:+.3}, {:+.3}]",
            self.cursor_grid.0, self.cursor_grid.1, latest[0], latest[1], latest[2]
        )
    }
}
