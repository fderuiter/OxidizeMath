//! Accessible parameter controls, fairing stamp buttons, CFL clamp readout, and bounce-back
//! aerodynamic telemetry panel for [`super::AdvectiveMorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{AdvectiveKineticsPreset, AdvectiveViewMode};
use super::AdvectiveMorphogenesisTool;
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::app::{SemanticPalette, ThemeMode};
use crate::framework::{BrushMode, FairingStamp};
use eframe::egui;

impl AdvectiveMorphogenesisTool {
    pub(crate) fn render_studio(&mut self, ui: &mut egui::Ui) {
        self.poll_incoming_fluid_handoff(ui.ctx());
        self.high_contrast = ThemeMode::from_visuals(ui.visuals()) == ThemeMode::HighContrast;

        if self.cfg.playing {
            for _ in 0..2 {
                self.engine.step_coupled(&self.cfg);
            }
            self.refresh_pixels_in_place();
            ui.ctx().request_repaint();
        }

        self.render_transport_toolbar(ui);
        ui.separator();

        let wide = ui.available_width() >= 860.0;
        if wide {
            ui.columns(2, |cols| {
                let left_w = cols[0].available_width();
                self.draw_wind_tunnel_canvas(&mut cols[0], left_w);
                cols[0].add_space(6.0);
                self.draw_wake_profile_plot(&mut cols[0]);
                self.render_inspector_controls(&mut cols[1]);
            });
        } else {
            let w = ui.available_width();
            self.draw_wind_tunnel_canvas(ui, w);
            ui.add_space(6.0);
            self.draw_wake_profile_plot(ui);
            ui.separator();
            self.render_inspector_controls(ui);
        }
    }

    fn render_transport_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let play_label = if self.cfg.playing {
                "⏸ Pause"
            } else {
                "▶ Play"
            };
            if ui
                .button(play_label)
                .accessible_hover_text(
                    "Toggle continuous co-stepped D2Q9 and reaction-diffusion evolution",
                )
                .clicked()
            {
                self.cfg.playing = !self.cfg.playing;
                let msg = if self.cfg.playing {
                    "Advective morphogenesis running"
                } else {
                    "Advective morphogenesis paused"
                };
                announce_live_status(ui.ctx(), msg);
            }
            if ui
                .button("⏭ Step")
                .accessible_hover_text("Advance one coupled fluid + reaction-diffusion step")
                .clicked()
            {
                self.step_once();
            }
            if ui
                .button("↺ Reset Wake & Seed")
                .accessible_hover_text("Reset D2Q9 wind tunnel and reseed morphogen nuclei")
                .clicked()
            {
                self.engine.reset_fluid_wake(self.cfg.viscosity);
                self.engine.reseed_morphogens(&self.cfg);
                self.engine.refresh_metrics(&self.cfg);
                self.refresh_pixels_in_place();
                announce_live_status(ui.ctx(), "Reset wind tunnel and reseeded morphogens");
            }
            self.render_preset_and_view_combos(ui);
        });
    }

    fn render_preset_and_view_combos(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Preset")
            .selected_text(self.cfg.preset.label())
            .show_ui(ui, |ui| {
                for candidate in AdvectiveKineticsPreset::ALL {
                    if ui
                        .selectable_label(self.cfg.preset == candidate, candidate.label())
                        .clicked()
                    {
                        self.apply_kinetics_preset(candidate);
                    }
                }
            });

        egui::ComboBox::from_label("View")
            .selected_text(self.cfg.view_mode.label())
            .show_ui(ui, |ui| {
                for mode in AdvectiveViewMode::ALL {
                    if ui
                        .selectable_label(self.cfg.view_mode == mode, mode.label())
                        .clicked()
                    {
                        self.cfg.view_mode = mode;
                        self.refresh_pixels_in_place();
                    }
                }
            });
    }

    fn render_inspector_controls(&mut self, ui: &mut egui::Ui) {
        self.render_hydrodynamic_readout(ui);
        ui.add_space(6.0);
        self.render_flow_and_peclet_sliders(ui);
        ui.add_space(6.0);
        self.render_kinetics_sliders(ui);
        ui.add_space(6.0);
        self.render_obstacle_sculpt_controls(ui);
    }

    fn render_hydrodynamic_readout(&self, ui: &mut egui::Ui) {
        let m = &self.engine.metrics;
        let palette = SemanticPalette::for_theme(ThemeMode::from_visuals(ui.visuals()));
        ui.group(|ui| {
            ui.label(egui::RichText::new("Live Hydrodynamics & CFL Guard").strong());
            ui.label(format!(
                "Re ≈ {:.1}  |  F_D = {:.4}  |  F_L = {:+.4}  |  C_D = {:.3}",
                m.reynolds, m.f_drag, m.f_lift, m.c_d
            ));
            ui.label(format!(
                "max |ω_z| = {:.4}  |  Wake Elongation = {:.2}  |  Solid Cells = {}",
                m.max_vorticity, m.plume_elongation, m.solid_cells
            ));
            let cfl_text = if m.cfl_clamped {
                format!(
                    "CFL Clamp Active: dt_eff = {:.4} (limit {:.4})",
                    m.effective_dt, m.cfl_limit
                )
            } else {
                format!(
                    "CFL Stable: dt = {:.4} ≤ {:.4}",
                    m.effective_dt, m.cfl_limit
                )
            };
            let color = if m.cfl_clamped {
                palette.warning
            } else {
                palette.success
            };
            ui.colored_label(color, cfl_text);
        });
    }

    fn render_flow_and_peclet_sliders(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label(egui::RichText::new("Fluid Wake & Advective Coupling").strong());
            let mut changed = false;
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.cfg.viscosity, 0.018..=0.25)
                        .text("Kinematic Viscosity ν"),
                )
                .accessible_hover_text("Lower viscosity increases Reynolds number and triggers unsteady vortex shedding")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.cfg.peclet, 0.0..=30.0)
                        .text("Péclet Advection Gain Pe"),
                )
                .accessible_hover_text("Scales 2D upwind fluid advection relative to reaction-diffusion")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.cfg.dt, 0.001..=1.0)
                        .logarithmic(true)
                        .text("Requested Step dt"),
                )
                .accessible_hover_text("Reaction-diffusion time step automatically clamped to combined diffusive + advective CFL bound")
                .changed();
            if changed {
                self.engine.sync_parameters(&self.cfg);
                self.engine.refresh_metrics(&self.cfg);
            }
        });
    }

    fn render_kinetics_sliders(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label(egui::RichText::new("Reaction-Diffusion Kinetics").strong());
            let mut changed = false;
            if self.cfg.preset.is_gray_scott() {
                changed |= ui
                    .add(egui::Slider::new(&mut self.cfg.f, 0.01..=0.10).text("Feed Rate F"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.cfg.k, 0.03..=0.08).text("Kill Rate k"))
                    .changed();
            } else {
                changed |= ui
                    .add(egui::Slider::new(&mut self.cfg.a, 0.01..=0.60).text("Activator Source a"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.cfg.b, 0.10..=2.00).text("Substrate Source b"))
                    .changed();
            }
            changed |= ui
                .add(egui::Slider::new(&mut self.cfg.d_u, 0.02..=2.00).text("Diffusivity D_u"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.cfg.d_v, 0.02..=12.00).text("Diffusivity D_v"))
                .changed();
            if changed {
                self.engine.sync_parameters(&self.cfg);
                self.engine.refresh_metrics(&self.cfg);
            }
        });
    }

    fn render_obstacle_sculpt_controls(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label(
                egui::RichText::new("Zero-Flux Obstacle Sculpting (Mouse + Keyboard)").strong(),
            );
            ui.horizontal_wrapped(|ui| {
                for stamp in FairingStamp::ALL {
                    let active = self.cfg.stamp == stamp && self.cfg.mask_hex.is_none();
                    if ui
                        .selectable_label(active, stamp.label())
                        .accessible_hover_text(
                            "Stamp aerodynamic obstacle profile onto the 100×50 lattice",
                        )
                        .clicked()
                    {
                        self.apply_fairing_stamp(stamp);
                        announce_live_status(ui.ctx(), format!("Stamped {}", stamp.label()));
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.cfg.brush_mode, BrushMode::Solid, "🧱 Paint Solid");
                ui.selectable_value(&mut self.cfg.brush_mode, BrushMode::Erase, "🧽 Erase Solid");
                ui.add(egui::Slider::new(&mut self.cfg.brush_radius, 0.0..=8.0).text("Brush R"));
            });
            ui.small(
                "Focus canvas and use Arrow keys + Space/Enter to toggle solid obstacle cells.",
            );
        });
    }
}
