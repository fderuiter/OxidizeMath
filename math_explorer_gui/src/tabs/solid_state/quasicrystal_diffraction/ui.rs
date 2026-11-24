#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:solid_state]

use super::{DiffractionStructure, QuasicrystalDiffractionTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, TextureOptions, Vec2};
use egui_plot::{Legend, Line, MarkerShape, Plot, Points};
use math_explorer::physics::solid_state::quasicrystal::acceptance_window_polygon;
use math_explorer::physics::solid_state::reciprocal::{
    first_brillouin_zone_2d, pseudo_brillouin_zone_polygon, reciprocal_basis_2d,
};

impl QuasicrystalDiffractionTool {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "quasicrystal_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .resizable(false)
            .min_width(265.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Cut-and-Project & Bragg");
                    ui.separator();
                    let mut changed = self.draw_structure_controls(ui);
                    ui.separator();
                    changed |= self.draw_phason_controls(ui);
                    ui.separator();
                    changed |= self.draw_diffraction_controls(ui);
                    if changed {
                        self.mark_changed();
                    }
                    ui.separator();
                    self.draw_live_metrics(ui);
                });
            });
    }

    fn draw_structure_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Lattice / Quasicrystal Family:");
        let prev = self.state.structure;
        egui::ComboBox::from_id_salt("qc_structure_combo")
            .selected_text(self.state.structure.label())
            .show_ui(ui, |ui| {
                for s in [
                    DiffractionStructure::Fibonacci1D,
                    DiffractionStructure::Penrose5Fold,
                    DiffractionStructure::AmmannBeenker8Fold,
                    DiffractionStructure::Penrose10Fold,
                    DiffractionStructure::Stampfli12Fold,
                    DiffractionStructure::Square2D,
                    DiffractionStructure::Bcc100Slice,
                    DiffractionStructure::Fcc100Slice,
                ] {
                    ui.selectable_value(&mut self.state.structure, s, s.label());
                }
            });
        changed |= self.state.structure != prev;
        changed |= ui
            .add(egui::Slider::new(&mut self.state.patch_radius, 1.5..=5.2).text("Patch Radius R"))
            .changed();
        if self.state.structure.quasicrystal_symmetry().is_some() {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.state.window_scale, 0.65..=1.3)
                        .text("Window Scale"),
                )
                .changed();
        }
        changed
    }

    fn draw_phason_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Perpendicular Phason Shift φ_⊥:");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.phason_x, -1.5..=1.5).text("φ_x"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.phason_y, -1.5..=1.5).text("φ_y"))
            .changed();
        ui.horizontal(|ui| {
            if ui
                .small_button("Zero Phason")
                .accessible_hover_text("Reset perpendicular phason shift to origin")
                .clicked()
            {
                self.state.phason_x = 0.0;
                self.state.phason_y = 0.0;
                changed = true;
            }
            ui.checkbox(&mut self.state.show_perp_window, "Show r_⊥ Window");
        });
        changed
    }

    fn draw_diffraction_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Reciprocal-Space Detector:");
        changed |= ui
            .add(egui::Slider::new(&mut self.state.q_max, 5.0..=22.0).text("q_max"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.grid_res, 32..=128).text("Grid N_q"))
            .changed();
        if ui
            .add(egui::Slider::new(&mut self.state.gamma, 0.2..=1.2).text("Contrast γ"))
            .changed()
        {
            self.dirty = true;
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        ui.checkbox(&mut self.state.show_tile_edges, "Show Tile Edges");
        ui.checkbox(
            &mut self.state.show_bz_overlay,
            "Overlay Brillouin / Pseudo-BZ",
        );
        changed
    }

    fn draw_live_metrics(&self, ui: &mut egui::Ui) {
        let site_count = if self.state.structure.quasicrystal_symmetry().is_some() {
            self.qc_ws.sites().len()
        } else {
            self.periodic_coords.len()
        };
        ui.small(format!("Projected Sites N = {site_count}"));
        if self.state.structure.quasicrystal_symmetry().is_some() {
            ui.small(format!(
                "Phason Flips (+in / -out) = +{} / -{}",
                self.qc_ws.flipped_in_count, self.qc_ws.flipped_out_count
            ));
        }
        ui.small(format!(
            "Peak Bragg Intensity S_max = {:.2}",
            self.bragg_ws.peak_intensity()
        ));
    }

    pub(super) fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            if ui.available_width() > 620.0 {
                ui.columns(2, |cols| {
                    self.draw_spatial_plot(&mut cols[0], palette);
                    self.draw_bragg_canvas(&mut cols[1], palette);
                });
            } else {
                self.draw_spatial_plot(ui, palette);
                ui.add_space(6.0);
                self.draw_bragg_canvas(ui, palette);
            }
        });
    }

    fn draw_spatial_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        if self.state.show_perp_window {
            if let Some(sym) = self.state.structure.quasicrystal_symmetry() {
                self.draw_perpendicular_window_plot(ui, palette, sym);
                return;
            }
        }
        self.draw_real_space_plot(ui, palette);
    }

    fn draw_real_space_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.strong("Real-Space Patch & Phason Tile Flips");
        let mut stable = Vec::new();
        let mut flipped = Vec::new();
        if self.state.structure.quasicrystal_symmetry().is_some() {
            for s in self.qc_ws.sites() {
                if s.flipped {
                    flipped.push(s.pos_parallel);
                } else {
                    stable.push(s.pos_parallel);
                }
            }
        } else {
            stable.extend_from_slice(&self.periodic_coords);
        }

        Plot::new("qc_real_space_plot")
            .data_aspect(1.0)
            .x_axis_label("x_∥")
            .y_axis_label("y_∥")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                if self.state.show_tile_edges
                    && self.state.structure.quasicrystal_symmetry().is_some()
                {
                    for &(a, b) in self.qc_ws.edges() {
                        plot_ui.line(Line::new("Tile Edge", vec![a, b]).color(palette.graph_edge));
                    }
                }
                plot_ui.points(
                    Points::new("Atomic Sites", stable)
                        .color(palette.axis_x)
                        .shape(MarkerShape::Circle)
                        .radius(3.2_f32),
                );
                if !flipped.is_empty() {
                    plot_ui.points(
                        Points::new("Phason-Flipped Sites", flipped)
                            .color(palette.warning)
                            .shape(MarkerShape::Diamond)
                            .radius(4.8_f32),
                    );
                }
            });
    }

    fn draw_perpendicular_window_plot(
        &self,
        ui: &mut egui::Ui,
        palette: SemanticPalette,
        sym: math_explorer::physics::solid_state::quasicrystal::QuasicrystalSymmetry,
    ) {
        ui.strong("Perpendicular Acceptance Window Ω_⊥");
        let mut poly = [[0.0_f64; 2]; 14];
        let count = acceptance_window_polygon(
            sym,
            self.state.window_scale,
            [self.state.phason_x, self.state.phason_y],
            &mut poly,
        );
        let pts: Vec<[f64; 2]> = self.qc_ws.sites().iter().map(|s| s.pos_perp).collect();
        Plot::new("qc_perp_window_plot")
            .data_aspect(1.0)
            .x_axis_label("x_⊥")
            .y_axis_label("y_⊥")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Acceptance Window Ω", poly[..count].to_vec())
                        .color(palette.accent)
                        .width(2.0_f32),
                );
                plot_ui.points(
                    Points::new("Internal Coordinates r_⊥", pts)
                        .color(palette.series_color(2))
                        .radius(2.8_f32),
                );
            });
    }

    fn draw_bragg_canvas(&mut self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.strong("Kinematic Bragg Diffraction S(q_x, q_y)");
        if self.dirty || self.texture.is_none() {
            let hc = ThemeMode::from_visuals(ui.visuals()) == ThemeMode::HighContrast;
            let img = self.bragg_color_image(hc);
            self.texture = Some(ui.ctx().load_texture(
                "qc_bragg_diffraction",
                img,
                TextureOptions::NEAREST,
            ));
            self.dirty = false;
        }
        let side = ui
            .available_width()
            .min(ui.available_height() - 28.0)
            .clamp(160.0, 520.0);
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
        self.focused_canvas = response.has_focus().then_some(response.id);
        let rect = response.rect;
        let summary = format!(
            "{}, phason ({:.2}, {:.2}), flipped vertices {}, peak S {:.2}",
            self.state.structure.label(),
            self.state.phason_x,
            self.state.phason_y,
            self.qc_ws.flipped_in_count + self.qc_ws.flipped_out_count,
            self.bragg_ws.peak_intensity()
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Reciprocal-Space Bragg Diffraction Map",
            Some(&summary),
        );
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);
        self.handle_bragg_canvas_input(ui, &response);

        if let Some(tex) = &self.texture {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), rect, uv, Color32::WHITE);
        }
        if self.state.show_bz_overlay {
            self.paint_bz_overlay(&painter, rect, palette);
        }
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        ui.small("Drag or use Arrow keys when focused to shift phason φ_⊥ (sharp Bragg peaks stay invariant)");
    }

    fn handle_bragg_canvas_input(&mut self, ui: &mut egui::Ui, response: &egui::Response) {
        let ctx = ui.ctx().clone();
        if response.has_focus() {
            let (l, r, d, u) = ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                )
            });
            if l || r || d || u {
                ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
                let dx = (if r { 0.04 } else { 0.0 }) - (if l { 0.04 } else { 0.0 });
                let dy = (if u { 0.04 } else { 0.0 }) - (if d { 0.04 } else { 0.0 });
                self.state.phason_x = (self.state.phason_x + dx).clamp(-1.5, 1.5);
                self.state.phason_y = (self.state.phason_y + dy).clamp(-1.5, 1.5);
                self.mark_changed();
                crate::accessibility::announce_live_status(
                    &ctx,
                    format!(
                        "Phason shift ({:.2}, {:.2}), {} flipped vertices",
                        self.state.phason_x,
                        self.state.phason_y,
                        self.qc_ws.flipped_in_count + self.qc_ws.flipped_out_count
                    ),
                );
            }
        }
        if response.dragged() && response.rect.width() > 1.0 {
            let delta = response.drag_delta();
            self.state.phason_x =
                (self.state.phason_x + f64::from(delta.x) * 0.005).clamp(-1.5, 1.5);
            self.state.phason_y =
                (self.state.phason_y - f64::from(delta.y) * 0.005).clamp(-1.5, 1.5);
            self.mark_changed();
        }
    }

    fn paint_bz_overlay(&self, painter: &egui::Painter, rect: Rect, palette: SemanticPalette) {
        let mut poly = [[0.0_f64; 2]; 16];
        let count = if let Some(sym) = self.state.structure.quasicrystal_symmetry() {
            pseudo_brillouin_zone_polygon(
                sym.rotational_order(),
                sym.fundamental_wavevector(),
                &mut poly,
            )
        } else if let Some(recip) =
            reciprocal_basis_2d(&self.state.structure.periodic_direct_basis_2d())
        {
            first_brillouin_zone_2d(&recip, &mut poly)
        } else {
            0
        };
        if count < 2 {
            return;
        }
        let q_max = self.state.q_max.max(1.0);
        let to_pos = |q: [f64; 2]| -> Pos2 {
            let nx = ((q[0] / q_max) * 0.5 + 0.5) as f32;
            let ny = (0.5 - (q[1] / q_max) * 0.5) as f32;
            Pos2::new(
                rect.left() + nx * rect.width(),
                rect.top() + ny * rect.height(),
            )
        };
        let stroke = Stroke::new(1.6_f32, palette.node_stroke);
        for pair in poly[..count].windows(2) {
            let p0 = to_pos(pair[0]);
            let p1 = to_pos(pair[1]);
            if rect.contains(p0) && rect.contains(p1) {
                painter.line_segment([p0, p1], stroke);
            }
        }
    }
}
