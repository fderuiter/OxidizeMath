#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BifurcationDiagram, BifurcationMapMode};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;

impl BifurcationDiagram {
    pub(super) fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "bifurcation_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("1D Chaos Explorer");
            ui.separator();
            let _ = self.atlas.show_toggle_button(ui);

            let mut sweep_changed = self.draw_custom_map_controls(ui);
            ui.separator();

            let (lo_bound, hi_bound) = match self.map_mode {
                BifurcationMapMode::Logistic => (0.0, 4.0),
                BifurcationMapMode::Custom => (-6.0, 6.0),
            };
            sweep_changed |= ui
                .add(
                    AnimatedSlider::new(&mut self.r_min, lo_bound..=(hi_bound - 0.01))
                        .text("Min r")
                        .id_salt("bifurcation_r_min"),
                )
                .changed();
            if self.r_max <= self.r_min {
                self.r_max = (self.r_min + 0.01).min(hi_bound);
            }
            sweep_changed |= ui
                .add(
                    AnimatedSlider::new(
                        &mut self.r_max,
                        (self.r_min + 0.01).min(hi_bound)..=hi_bound,
                    )
                    .text("Max r")
                    .id_salt("bifurcation_r_max"),
                )
                .changed();
            sweep_changed |= ui
                .add(egui::Slider::new(&mut self.steps, 100..=2000).text("Resolution (steps)"))
                .changed();

            if sweep_changed && !self.r_pinned {
                self.selected_r = self.selected_r.clamp(self.r_min, self.r_max);
                self.probe_r = self.selected_r;
            }

            ui.separator();
            self.draw_parameter_controls(ui, sweep_changed);

            ui.separator();
            ui.label(format!(
                "r = {:.4} ({}) | x₀ = {:.3} | λ = {:.3}",
                self.selected_r,
                if self.r_pinned { "pinned" } else { "live" },
                self.x0,
                self.selected_lyapunov
            ));
            ui.small(format!(
                "Live orbit step {}: x_n = {:.4}",
                self.orbit_step_count, self.orbit_x
            ));
            ui.small("Hover/click diagram or press Arrow keys + Enter to inspect r.");
        });
    }

    fn draw_parameter_controls(&mut self, ui: &mut egui::Ui, mut full_recompute: bool) {
        let ctx = ui.ctx().clone();
        self.selected_r = self.selected_r.clamp(self.r_min, self.r_max);
        let r_changed = ui
            .add(
                AnimatedSlider::new(&mut self.selected_r, self.r_min..=self.r_max)
                    .text("Selected r")
                    .id_salt("bifurcation_probe_r"),
            )
            .changed();
        if r_changed {
            self.probe_r = self.selected_r;
        }
        let x0_range = match self.map_mode {
            BifurcationMapMode::Logistic => 0.001..=0.999,
            BifurcationMapMode::Custom => -2.0..=2.0,
        };
        let x0_changed = ui
            .add(egui::Slider::new(&mut self.x0, x0_range).text("Initial x₀"))
            .changed();
        let pin_changed = ui
            .checkbox(&mut self.r_pinned, "Pin r crosshair")
            .accessible_hover_text("Freeze selected r against pointer hover; Enter toggles on plot")
            .changed();
        if r_changed || (pin_changed && self.r_pinned) {
            self.r_from_hover = false;
        }

        full_recompute |= x0_changed;
        let recompute_clicked = ui
            .button("▶ Recompute")
            .accessible_hover_text("Regenerate bifurcation, Lyapunov, cobweb, and orbit plots")
            .clicked();
        if recompute_clicked || full_recompute {
            self.recompute();
            if recompute_clicked {
                self.dispatch_sweep_request(&ctx);
            }
        } else if r_changed {
            self.recompute_coupled_views();
        }

        self.draw_live_orbit_and_audio_controls(ui);
    }

    fn draw_live_orbit_and_audio_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("⏭ Step Orbit").clicked() {
                <Self as crate::framework::InteractiveTool>::step_simulation(self);
            }
            if ui.button("↺ Reset Orbit").clicked() {
                self.reset_live_orbit();
            }
        });
        ui.separator();
        let frame = crate::audio::mappings::bifurcation_frame(self.selected_r);
        crate::audio::render_sonification_panel(ui, "bifurcation", &frame, false);
    }

    fn draw_custom_map_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            if ui
                .radio_value(&mut self.map_mode, BifurcationMapMode::Logistic, "Logistic")
                .clicked()
            {
                self.r_min = self.r_min.clamp(0.0, 3.95);
                self.r_max = self.r_max.clamp(self.r_min + 0.05, 4.0);
                self.selected_r = self.selected_r.clamp(self.r_min, self.r_max);
                self.probe_r = self.selected_r;
                self.x0 = self.x0.clamp(0.001, 0.999);
                changed = true;
            }
            if ui
                .radio_value(
                    &mut self.map_mode,
                    BifurcationMapMode::Custom,
                    "Custom f(r, x)",
                )
                .clicked()
            {
                changed = true;
            }
        });
        if self.map_mode.is_custom() {
            ui.add_space(4.0);
            changed |= self.math_input.ui(ui);
            if let Some(ref err_msg) = self.custom_eval_error {
                let palette = SemanticPalette::from_visuals(ui.visuals());
                ui.label(
                    egui::RichText::new(format!("⚠️ Map Error: {err_msg}"))
                        .color(palette.error)
                        .small(),
                );
            }
        }
        changed
    }

    pub(super) fn draw_central_panel(&mut self, ui: &mut egui::Ui) {
        self.focused_canvas = None;
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if self.atlas.is_visible() {
                if ui.available_width() >= 720.0 {
                    ui.columns(2, |cols| {
                        self.draw_coupled_plots(&mut cols[0]);
                        self.render_bifurcation_atlas_panel(&mut cols[1]);
                    });
                } else {
                    self.render_bifurcation_atlas_panel(ui);
                    ui.separator();
                    self.draw_coupled_plots(ui);
                }
            } else {
                self.draw_coupled_plots(ui);
            }
        });
    }

    pub(super) fn export_bifurcation_wav(
        &self,
        sample_rate: u32,
        duration_secs: f32,
    ) -> Result<Vec<u8>, String> {
        let r0 = self.r_min;
        let span = (self.r_max - self.r_min).max(0.0);
        let dur = f64::from(duration_secs.clamp(1.0, 15.0));
        Ok(crate::audio::render_frames_wav(
            sample_rate,
            duration_secs,
            |_, t_sec| {
                let frac = (t_sec / dur).clamp(0.0, 1.0);
                crate::audio::mappings::bifurcation_frame(r0 + frac * span)
            },
        ))
    }

    pub(super) fn export_bifurcation_midi(&self, duration_secs: f32) -> Result<Vec<u8>, String> {
        let r0 = self.r_min;
        let span = (self.r_max - self.r_min).max(0.0);
        let dur = f64::from(duration_secs.clamp(1.0, 15.0));
        Ok(crate::audio::render_frames_midi(
            "Bifurcation Diagram",
            duration_secs,
            |_, t_sec| {
                let frac = (t_sec / dur).clamp(0.0, 1.0);
                crate::audio::mappings::bifurcation_frame(r0 + frac * span)
            },
        ))
    }
}
