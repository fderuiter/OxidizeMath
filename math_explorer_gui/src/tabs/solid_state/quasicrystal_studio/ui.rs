#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, phason window gizmo, playback transport, and Bragg telemetry for [`super::QuasicrystalStudio`].

// [cite:solid_state]

use super::{DiffractionColormap, QuasicrystalStudio, StudioPatternMode};
use crate::accessibility::{
    announce_live_status, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::app::{SemanticPalette, ThemeMode};
use crate::framework::{
    bounded_drag_value, InteractiveTool, PaneSidePanel, PlaybackAction, PlaybackTransport,
    scoped_panel_id,
};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::physics::solid_state::quasicrystal::QuasicrystalKind;
use math_explorer::physics::solid_state::reciprocal::BravaisLattice2D;
use std::f64::consts::PI;

impl QuasicrystalStudio {
    pub(super) fn render_studio(&mut self, parent_ui: &mut egui::Ui) {
        self.tick_phason_playback(parent_ui.ctx());
        let theme = ThemeMode::from_visuals(parent_ui.visuals());
        if self.last_theme != Some(theme) {
            self.last_theme = Some(theme);
            self.texture_dirty = true;
        }
        if self.recompute_dirty {
            self.recompute_physics();
        }

        let panel_id = scoped_panel_id(parent_ui, "quasicrystal_studio_controls");
        PaneSidePanel::left(panel_id)
            .default_width(320.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.draw_controls_panel(ui);
                });
            });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.draw_dual_canvases(ui);
        });
    }

    fn tick_phason_playback(&mut self, ctx: &egui::Context) {
        let now = PlatformInstant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f64().clamp(0.0, 0.08);
        self.last_tick = now;
        if self.state.is_playing && self.state.mode == StudioPatternMode::Quasicrystal {
            let (s, c) = self.state.phason_drift_angle_deg.to_radians().sin_cos();
            let dist = dt.max(1.0 / 120.0) * self.state.phason_drift_speed;
            self.step_phason_zero_alloc([dist * c, dist * s]);
            ctx.request_repaint();
        }
    }

    fn draw_controls_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("2D Quasicrystal & Bragg Studio");
        ui.separator();

        self.draw_pattern_selector(ui);
        ui.separator();

        if self.state.mode == StudioPatternMode::Quasicrystal {
            self.draw_phason_section(ui);
            ui.separator();
        }

        self.draw_display_section(ui);
        ui.separator();

        self.draw_symmetry_telemetry(ui);
        ui.separator();

        self.draw_bragg_peaks_table(ui);
    }

    fn draw_pattern_selector(&mut self, ui: &mut egui::Ui) {
        ui.label("Structure Class");
        egui::ComboBox::from_id_salt("qc_studio_mode_combo")
            .selected_text(self.state.mode.label())
            .show_ui(ui, |ui| {
                for m in [
                    StudioPatternMode::Quasicrystal,
                    StudioPatternMode::PeriodicBravais,
                ] {
                    if ui.selectable_value(&mut self.state.mode, m, m.label()).changed() {
                        if m == StudioPatternMode::PeriodicBravais {
                            self.state.is_playing = false;
                        }
                        self.mark_dirty();
                    }
                }
            });

        match self.state.mode {
            StudioPatternMode::Quasicrystal => self.draw_quasicrystal_params(ui),
            StudioPatternMode::PeriodicBravais => self.draw_bravais_params(ui),
        }
    }

    fn draw_quasicrystal_params(&mut self, ui: &mut egui::Ui) {
        ui.label("Hyperlattice Family (Z^N -> E_par + E_perp)");
        egui::ComboBox::from_id_salt("qc_kind_combo")
            .selected_text(self.state.quasicrystal.label())
            .show_ui(ui, |ui| {
                for kind in [
                    QuasicrystalKind::Penrose5D,
                    QuasicrystalKind::AmmannBeenker4D,
                    QuasicrystalKind::Stampfli6D,
                ] {
                    if ui
                        .selectable_value(&mut self.state.quasicrystal, kind, kind.label())
                        .changed()
                    {
                        self.mark_dirty();
                    }
                }
            });

        ui.horizontal(|ui| {
            ui.label("Tile Edge (a):");
            if ui
                .add(bounded_drag_value(&mut self.state.edge_length, 0.4..=4.0, 0.02))
                .changed()
            {
                self.mark_dirty();
            }
            ui.label("Aperture (R):");
            if ui
                .add(bounded_drag_value(
                    &mut self.state.aperture_radius,
                    12.0..=34.0,
                    0.2,
                ))
                .changed()
            {
                self.mark_dirty();
            }
        });
    }

    fn draw_bravais_params(&mut self, ui: &mut egui::Ui) {
        ui.label("2D Bravais Net");
        egui::ComboBox::from_id_salt("bravais_2d_combo")
            .selected_text(self.state.bravais.label())
            .show_ui(ui, |ui| {
                for net in [
                    BravaisLattice2D::Oblique,
                    BravaisLattice2D::Rectangular,
                    BravaisLattice2D::CenteredRectangular,
                    BravaisLattice2D::Square,
                    BravaisLattice2D::Hexagonal,
                ] {
                    if ui
                        .selectable_value(&mut self.state.bravais, net, net.label())
                        .changed()
                    {
                        self.mark_dirty();
                    }
                }
            });

        ui.horizontal(|ui| {
            ui.label("a:");
            if ui
                .add(bounded_drag_value(&mut self.state.edge_length, 0.4..=4.0, 0.02))
                .changed()
            {
                self.mark_dirty();
            }
            if matches!(
                self.state.bravais,
                BravaisLattice2D::Oblique
                    | BravaisLattice2D::Rectangular
                    | BravaisLattice2D::CenteredRectangular
            ) {
                ui.label("b:");
                if ui
                    .add(bounded_drag_value(&mut self.state.lattice_b, 0.4..=6.0, 0.02))
                    .changed()
                {
                    self.mark_dirty();
                }
            }
        });
        if self.state.bravais == BravaisLattice2D::Oblique
            && ui
                .add(egui::Slider::new(&mut self.state.gamma_deg, 20.0..=160.0).text("gamma (deg)"))
                .changed()
        {
            self.mark_dirty();
        }
        if ui
            .add(egui::Slider::new(&mut self.state.aperture_radius, 12.0..=34.0).text("Aperture R"))
            .changed()
        {
            self.mark_dirty();
        }
    }

    fn draw_phason_section(&mut self, ui: &mut egui::Ui) {
        ui.strong("Perpendicular Space E_perp Phason");
        if let Some(action) = PlaybackTransport::in_thread(
            "quasicrystal_phason_transport",
            self.name(),
            self.state.is_playing,
        )
        .play_label("▶ Drift")
        .step_label("⏭ Step Phason")
        .reset_label("↺ Zero Phason")
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.state.is_playing = true,
                PlaybackAction::Pause => self.state.is_playing = false,
                PlaybackAction::Step => self.step_simulation(),
                PlaybackAction::Reset => {
                    self.state.is_playing = false;
                    self.state.phason_x = 0.0;
                    self.state.phason_y = 0.0;
                    self.state.reference_phason_x = 0.0;
                    self.state.reference_phason_y = 0.0;
                    self.mark_dirty();
                }
            }
        }

        if ui
            .add(egui::Slider::new(&mut self.state.phason_x, -4.0..=4.0).text("w_x (E_perp)"))
            .changed()
            || ui
                .add(egui::Slider::new(&mut self.state.phason_y, -4.0..=4.0).text("w_y (E_perp)"))
                .changed()
        {
            self.mark_dirty();
        }

        ui.horizontal(|ui| {
            if ui.button("Pin Reference w_0").clicked() {
                self.state.reference_phason_x = self.state.phason_x;
                self.state.reference_phason_y = self.state.phason_y;
                self.mark_dirty();
                announce_live_status(ui.ctx(), "Pinned reference perpendicular phason w_0");
            }
            ui.add(
                bounded_drag_value(&mut self.state.phason_drift_speed, 0.01..=1.5, 0.02)
                    .prefix("v: "),
            );
        });

        self.draw_perp_window_gizmo(ui);
    }

    fn draw_perp_window_gizmo(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(220.0, 96.0), egui::Sense::click_and_drag());
        let summary = format!(
            "w = ({:.3}, {:.3}), flipped tiles = {}",
            self.state.phason_x,
            self.state.phason_y,
            self.qc_workspace.stats().flipped_tile_count
        );
        register_canvas_accessibility(
            ui,
            &resp,
            egui::accesskit::Role::GraphicsDocument,
            "Perpendicular Acceptance Window Gizmo",
            Some(&summary),
        );
        if resp.dragged() || resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let half_h = 0.44 * rect.height().max(1.0);
                self.state.phason_x =
                    f64::from((pos.x - rect.center().x) / half_h * 2.0).clamp(-4.0, 4.0);
                self.state.phason_y =
                    f64::from((rect.center().y - pos.y) / half_h * 2.0).clamp(-4.0, 4.0);
                self.mark_dirty();
            }
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, palette.surface_fill);
        let sides = self.state.quasicrystal.rotational_order();
        let rad = 0.36 * rect.height();
        let center = rect.center();
        let mut win_pts = Vec::with_capacity(sides + 1);
        for k in 0..=sides {
            let (sn, cs) = (2.0 * PI * (k as f64) / (sides as f64)).sin_cos();
            win_pts.push(egui::pos2(center.x + rad * (cs as f32), center.y - rad * (sn as f32)));
        }
        painter.add(egui::Shape::line(
            win_pts,
            egui::Stroke::new(1.5_f32, palette.info),
        ));
        let px = center.x + (self.state.phason_x as f32) * 0.22 * rect.height();
        let py = center.y - (self.state.phason_y as f32) * 0.22 * rect.height();
        let marker = egui::pos2(
            px.clamp(rect.left() + 4.0, rect.right() - 4.0),
            py.clamp(rect.top() + 4.0, rect.bottom() - 4.0),
        );
        painter.circle_filled(marker, 4.5, palette.accent);
        painter.circle_stroke(marker, 4.5, egui::Stroke::new(1.2_f32, palette.node_stroke));
        paint_canvas_focus_ring(ui, &resp);
    }

    fn draw_display_section(&mut self, ui: &mut egui::Ui) {
        ui.strong("Visualization & Colormap");
        if self.state.mode == StudioPatternMode::Quasicrystal {
            ui.checkbox(
                &mut self.state.highlight_flipped_tiles,
                "Highlight Phason-Flipped Tiles",
            );
        } else {
            ui.checkbox(
                &mut self.state.show_brillouin_zone,
                "Overlay First Brillouin Zone & b1, b2",
            );
        }
        ui.checkbox(&mut self.state.show_bragg_rings, "Overlay Bragg Peak Callouts");

        egui::ComboBox::from_id_salt("qc_colormap_combo")
            .selected_text(self.state.colormap.label())
            .show_ui(ui, |ui| {
                for cm in [
                    DiffractionColormap::Inferno,
                    DiffractionColormap::Viridis,
                    DiffractionColormap::Monochrome,
                ] {
                    if ui
                        .selectable_value(&mut self.state.colormap, cm, cm.label())
                        .changed()
                    {
                        self.texture_dirty = true;
                    }
                }
            });
        if ui
            .add(egui::Slider::new(&mut self.state.log_gain, 1.0..=8.0).text("Log Contrast"))
            .changed()
        {
            self.texture_dirty = true;
        }
    }

    fn draw_symmetry_telemetry(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let rep = self.fft_workspace.report();
        ui.strong("Scaling Ratios & Rotational Symmetry");

        if self.state.mode == StudioPatternMode::Quasicrystal {
            let st = self.qc_workspace.stats();
            let err_pct = ((st.empirical_ratio - st.theoretical_ratio).abs()
                / st.theoretical_ratio.max(1e-9))
                * 100.0;
            ui.label(format!(
                "Dim N = {} | Sites: {} | Tiles: {} (Flipped: {})",
                self.state.quasicrystal.hyperlattice_dim(),
                st.vertex_count,
                st.tile_count,
                st.flipped_tile_count
            ));
            ui.colored_label(
                palette.success,
                format!(
                    "Tile Ratio: {:.5} vs {:.5} ({:.2}% err)",
                    st.empirical_ratio, st.theoretical_ratio, err_pct
                ),
            );
        } else if let Ok(lat) = self.current_bravais_params() {
            let bz = lat.first_brillouin_zone();
            ui.label(format!(
                "Sites: {} | BZ Vertices: {} | BZ Area: {:.3}",
                self.periodic_sites.len(),
                bz.len(),
                bz.area()
            ));
        }

        for order in [4_usize, 5, 6, 8, 10, 12] {
            let val = rep.correlation(order).clamp(0.0, 1.0);
            let forbidden = matches!(order, 5 | 8 | 10 | 12);
            let color = if val > 0.90 && forbidden {
                palette.accent
            } else if val > 0.90 {
                palette.success
            } else {
                palette.muted_text
            };
            ui.horizontal(|ui| {
                ui.colored_label(color, format!("C_{order:<2}: {val:>6.4}"));
                ui.add(
                    egui::ProgressBar::new(val as f32)
                        .desired_width(120.0)
                        .fill(SemanticPalette::progress_bar_fill(ui.visuals(), 1.0 - val)),
                );
            });
        }
    }

    fn draw_bragg_peaks_table(&self, ui: &mut egui::Ui) {
        ui.strong("Dominant Bragg Peaks & |S(k)|");
        egui::Grid::new("qc_bragg_peaks_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.label("k_x");
                ui.label("k_y");
                ui.label("I / I_max");
                ui.label("|S(k)|");
                ui.end_row();
                for peak in self.fft_workspace.peaks().iter().take(6) {
                    ui.monospace(format!("{:>6.2}", peak.k[0]));
                    ui.monospace(format!("{:>6.2}", peak.k[1]));
                    ui.monospace(format!("{:>5.3}", peak.normalized_intensity));
                    ui.monospace(format!("{:>5.3}", peak.structure_factor_amp));
                    ui.end_row();
                }
            });
    }
}
