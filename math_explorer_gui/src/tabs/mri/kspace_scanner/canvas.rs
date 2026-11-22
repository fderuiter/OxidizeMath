//! Quad-panel `ColorImage` canvas painter, tactile pointer brush, and WCAG keyboard reticle
//! for [`super::tool::MriKSpaceScannerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::tool::MriKSpaceScannerTool;
use super::trajectory::evaluate_bin_scanner_physics;
use super::{KSPACE_CENTER, KSPACE_DIM, KSPACE_LEN};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;

impl MriKSpaceScannerTool {
    pub(crate) fn ensure_textures(&mut self, ctx: &egui::Context) {
        if self.textures.is_some() {
            return;
        }
        let names = [
            "mri_kspace_gt",
            "mri_kspace_spectrum",
            "mri_kspace_recon",
            "mri_kspace_error",
        ];
        self.textures = Some(std::array::from_fn(|idx| {
            ctx.load_texture(
                names[idx],
                self.cached_images[idx].clone(),
                egui::TextureOptions::NEAREST,
            )
        }));
    }

    fn handle_kspace_pointer(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if !(response.clicked() || response.dragged()) {
            return;
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        let rect = response.rect;
        if rect.width() <= 1.0_f32 || rect.height() <= 1.0_f32 {
            return;
        }
        let rel_x = ((pos.x - rect.min.x) / rect.width()).clamp(0.0_f32, 0.999_9_f32);
        let rel_y = ((pos.y - rect.min.y) / rect.height()).clamp(0.0_f32, 0.999_9_f32);
        let col = (rel_x * (KSPACE_DIM as f32)) as usize;
        let row = (rel_y * (KSPACE_DIM as f32)) as usize;
        self.reticle_row = row.min(KSPACE_DIM - 1);
        self.reticle_col = col.min(KSPACE_DIM - 1);
        response.request_focus();
        if self.paint_at_bin(self.reticle_row, self.reticle_col) {
            ui.ctx().request_repaint();
        }
    }

    fn handle_kspace_keyboard(&mut self, ui: &egui::Ui, response: &egui::Response) {
        lock_canvas_arrow_keys(ui, response);
        if !response.has_focus() {
            return;
        }
        let (dr, dc, toggle) = ui.input(|i| {
            let step: isize = if i.modifiers.shift { 4 } else { 1 };
            let mut dr = 0_isize;
            let mut dc = 0_isize;
            if i.key_pressed(egui::Key::ArrowUp) {
                dr -= step;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                dr += step;
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                dc -= step;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                dc += step;
            }
            let toggle = i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter);
            (dr, dc, toggle)
        });
        if dr != 0 || dc != 0 {
            let max_idx = (KSPACE_DIM - 1) as isize;
            self.reticle_row = (self.reticle_row as isize + dr).clamp(0, max_idx) as usize;
            self.reticle_col = (self.reticle_col as isize + dc).clamp(0, max_idx) as usize;
            let phys = evaluate_bin_scanner_physics(self.reticle_row, self.reticle_col);
            let idx = flatten_2d_index(self.reticle_col, self.reticle_row, KSPACE_DIM);
            let status = if self.active_mask[idx] {
                "sampled"
            } else {
                "erased"
            };
            announce_live_status(
                ui.ctx(),
                format!(
                    "k-space reticle at row {}, column {}, {status}, |k| {:.1} cycles per meter",
                    self.reticle_row, self.reticle_col, phys.k_mag_per_m
                ),
            );
        }
        if toggle {
            let now_sampled = self.toggle_reticle_bin();
            let verb = if now_sampled { "sampled" } else { "erased" };
            announce_live_status(
                ui.ctx(),
                format!(
                    "k-space bin ({}, {}) {verb} — PSNR {:.1} dB, coverage {:.1}%",
                    self.reticle_row, self.reticle_col, self.recon.psnr_db, self.recon.sampling_pct
                ),
            );
        }
    }

    fn paint_kspace_canvas(&self, ui: &egui::Ui, rect: egui::Rect, tex_id: egui::TextureId) {
        let uv = egui::Rect::from_min_max(egui::pos2(0.0_f32, 0.0_f32), egui::pos2(1.0_f32, 1.0_f32));
        ui.painter()
            .image(tex_id, rect, uv, egui::Color32::WHITE);
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let dc_frac = (KSPACE_CENTER as f32 + 0.5_f32) / (KSPACE_DIM as f32);
        let dc_x = rect.min.x + dc_frac * rect.width();
        let dc_y = rect.min.y + dc_frac * rect.height();
        let guide = egui::Stroke::new(1.0_f32, palette.graph_edge.gamma_multiply(0.45_f32));
        ui.painter()
            .line_segment([egui::pos2(dc_x, rect.min.y), egui::pos2(dc_x, rect.max.y)], guide);
        ui.painter()
            .line_segment([egui::pos2(rect.min.x, dc_y), egui::pos2(rect.max.x, dc_y)], guide);

        let cell_w = rect.width() / (KSPACE_DIM as f32);
        let cell_h = rect.height() / (KSPACE_DIM as f32);
        let r_min = egui::pos2(
            rect.min.x + (self.reticle_col as f32) * cell_w,
            rect.min.y + (self.reticle_row as f32) * cell_h,
        );
        let reticle_rect =
            egui::Rect::from_min_size(r_min, egui::vec2(cell_w.max(4.0_f32), cell_h.max(4.0_f32)))
                .expand(1.5_f32);
        ui.painter().rect_stroke(
            reticle_rect,
            1.0_f32,
            egui::Stroke::new(1.8_f32, palette.accent),
            egui::StrokeKind::Outside,
        );
        ui.painter().rect_stroke(
            rect,
            2.0_f32,
            egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
            egui::StrokeKind::Inside,
        );
    }

    fn draw_static_panel(
        ui: &mut egui::Ui,
        title: &str,
        caption: &str,
        tex_id: egui::TextureId,
        side: f32,
        a11y_label: &str,
        a11y_value: &str,
    ) {
        ui.vertical(|ui| {
            ui.strong(title);
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(side, side),
                egui::Sense::focusable_noninteractive(),
            );
            let uv =
                egui::Rect::from_min_max(egui::pos2(0.0_f32, 0.0_f32), egui::pos2(1.0_f32, 1.0_f32));
            ui.painter()
                .image(tex_id, rect, uv, egui::Color32::WHITE);
            ui.painter().rect_stroke(
                rect,
                2.0_f32,
                egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
                egui::StrokeKind::Inside,
            );
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::Image,
                a11y_label,
                Some(a11y_value),
            );
            paint_canvas_focus_ring(ui, &response);
            ui.small(caption);
        });
    }

    fn draw_interactive_kspace_panel(
        &mut self,
        ui: &mut egui::Ui,
        side: f32,
        tex_id: egui::TextureId,
    ) {
        ui.vertical(|ui| {
            ui.strong("2. Masked 2D k-Space (Sculpt)");
            let sense =
                egui::Sense::click_and_drag().union(egui::Sense::focusable_noninteractive());
            let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), sense);
            self.kspace_canvas_id = Some(response.id);
            self.handle_kspace_pointer(ui, &response);
            self.handle_kspace_keyboard(ui, &response);
            if response.has_focus() {
                self.focused_kspace_id = Some(response.id);
            } else if self.focused_kspace_id == Some(response.id) {
                self.focused_kspace_id = None;
            }
            let refreshed_tex = self.textures.as_ref().map_or(tex_id, |t| t[1].id());
            self.paint_kspace_canvas(ui, rect, refreshed_tex);
            let val = format!(
                "{:.1}% sampled ({}/{} bins), reticle ({}, {})",
                self.recon.sampling_pct,
                self.recon.sampled_bins,
                KSPACE_LEN,
                self.reticle_row,
                self.reticle_col
            );
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::Canvas,
                "Interactive 2D k-Space Frequency Canvas",
                Some(&val),
            );
            paint_canvas_focus_ring(ui, &response);
            ui.small(format!(
                "Coverage: {:.1}% • {} ({})",
                self.recon.sampling_pct,
                self.state.trajectory.label(),
                self.state.brush_mode.label()
            ));
        });
    }

    fn draw_top_quad_row(&mut self, ui: &mut egui::Ui, side: f32) {
        let Some((gt_tex, ks_tex)) = self.textures.as_ref().map(|t| (t[0].id(), t[1].id())) else {
            return;
        };
        let gt_val = format!(
            "{} (TR={:.2}s, TE={:.3}s)",
            self.state.phantom.label(),
            self.state.tr_s,
            self.state.te_s
        );
        ui.horizontal(|ui| {
            Self::draw_static_panel(
                ui,
                "1. Ground-Truth Phantom",
                &gt_val,
                gt_tex,
                side,
                "Ground-Truth Spin-Echo MRI Phantom",
                &gt_val,
            );
            ui.add_space(12.0_f32);
            self.draw_interactive_kspace_panel(ui, side, ks_tex);
        });
    }

    fn draw_bottom_quad_row(&self, ui: &mut egui::Ui, side: f32) {
        let Some((rec_tex, err_tex)) = self.textures.as_ref().map(|t| (t[2].id(), t[3].id()))
        else {
            return;
        };
        let recon_val = format!(
            "PSNR {:.2} dB • Rel L2 {:.2e}",
            self.recon.psnr_db, self.recon.normalized_l2_error
        );
        let err_val = format!(
            "Max |Δρ| {:.4} • {:.2} ms FFT",
            self.recon.max_abs_error, self.last_recon_ms
        );
        ui.horizontal(|ui| {
            Self::draw_static_panel(
                ui,
                "3. Reconstructed Magnitude",
                &recon_val,
                rec_tex,
                side,
                "Reconstructed 2D Inverse FFT Magnitude Image",
                &recon_val,
            );
            ui.add_space(12.0_f32);
            Self::draw_static_panel(
                ui,
                "4. Absolute Error Map",
                &err_val,
                err_tex,
                side,
                "Spatial Reconstruction Absolute Error Heatmap",
                &err_val,
            );
        });
    }

    pub(crate) fn draw_central_quad(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("2D Fourier k-Space Encoding & Reconstruction");
            ui.small(
                "Drag across Panel 2 to paint/erase spatial frequencies, or Tab + Arrow keys + Space/Enter to toggle bins.",
            );
            let avail = ui.available_size();
            let side = ((avail.x - 28.0_f32) * 0.5_f32)
                .min((avail.y - 100.0_f32) * 0.5_f32)
                .clamp(112.0_f32, 290.0_f32);

            egui::ScrollArea::both().show(ui, |ui| {
                self.draw_top_quad_row(ui, side);
                ui.add_space(8.0_f32);
                self.draw_bottom_quad_row(ui, side);
            });
        });
    }
}
