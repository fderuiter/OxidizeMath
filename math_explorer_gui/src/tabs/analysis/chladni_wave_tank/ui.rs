#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use math_explorer::pure_math::analysis::pde::wave_2d::{
    analytic_clamped_square_omega, young_double_slit_fringe_spacing, BoundaryMode2D, Wave2DMedium,
};

use super::*;
use crate::framework::bounded_drag_value;
use crate::reflective_ui::render_copyable_metric;

impl ChladniWaveTankTool {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        self.register_shortcuts(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "chladni_wave_tank_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Wave Tank & Plate");
                ui.separator();
                self.draw_preset_and_transport_controls(ui);
                ui.separator();
                self.draw_medium_and_clamp_controls(ui);
                ui.separator();
                self.draw_driver_controls(ui);
                ui.separator();
                self.draw_optics_and_barrier_controls(ui);
                ui.separator();
                self.draw_visual_and_metric_readouts(ui);
            });
        });
    }

    fn register_shortcuts(&self, ctx: &egui::Context) {
        CommandRegistryData::register(
            ctx,
            "Chladni Wave Tank: Inject Pulse",
            "Inject a localized Gaussian wave pulse at the active driver coordinates",
            CommandTrigger::Shortcut(egui::Modifiers::NONE, egui::Key::Enter),
            true,
            "Analysis & Calculus",
        );
    }

    fn draw_preset_and_transport_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Starter Preset:");
        let mut selected = self.state.preset;
        egui::ComboBox::from_id_salt("chladni_preset_selector")
            .selected_text(selected.label())
            .show_ui(ui, |ui| {
                for p in [
                    TankPreset::SquareChladniPlate,
                    TankPreset::CircularDrumhead,
                    TankPreset::DoubleSlitInterference,
                    TankPreset::AcousticLens,
                ] {
                    ui.selectable_value(&mut selected, p, p.label());
                }
            });
        if selected != self.state.preset {
            self.apply_preset(selected);
        }

        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            let play_label = if self.paused { "▶ Play" } else { "⏸ Pause" };
            if ui.button(play_label).clicked() {
                self.paused = !self.paused;
            }
            if ui.button("⏭ Step").clicked() {
                self.advance_physics_step();
                self.refresh_analysis_outputs();
            }
            if ui.button("💥 Pulse").clicked() {
                self.solver.inject_gaussian_pulse(
                    self.state.driver_x,
                    self.state.driver_y,
                    0.045,
                    1.0,
                );
                self.refresh_analysis_outputs();
            }
            if ui.button("↺ Reset").clicked() {
                self.rebuild_domain_and_warmup(12);
            }
        });
    }

    fn draw_medium_and_clamp_controls(&mut self, ui: &mut egui::Ui) {
        let mut mask_dirty = false;
        let mut domain_dirty = false;
        ui.label("Governing PDE:");
        let prev_med = self.state.medium;
        egui::ComboBox::from_id_salt("chladni_medium")
            .selected_text(match self.state.medium {
                Wave2DMedium::AcousticWave => "2D Hyperbolic Wave",
                Wave2DMedium::BiharmonicPlate => "Kirchhoff-Love Biharmonic Plate",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.state.medium,
                    Wave2DMedium::BiharmonicPlate,
                    "Kirchhoff-Love Biharmonic Plate",
                );
                ui.selectable_value(
                    &mut self.state.medium,
                    Wave2DMedium::AcousticWave,
                    "2D Hyperbolic Wave",
                );
            });
        mask_dirty |= self.state.medium != prev_med;

        ui.label("Perimeter Boundary:");
        let prev_bnd = self.state.boundary;
        egui::ComboBox::from_id_salt("chladni_boundary")
            .selected_text(format!("{:?}", self.state.boundary))
            .show_ui(ui, |ui| {
                for b in [
                    BoundaryMode2D::Dirichlet,
                    BoundaryMode2D::Neumann,
                    BoundaryMode2D::Clamped,
                    BoundaryMode2D::Sponge,
                ] {
                    ui.selectable_value(&mut self.state.boundary, b, format!("{b:?}"));
                }
            });
        domain_dirty |= self.state.boundary != prev_bnd;

        mask_dirty |= self.draw_clamp_selector(ui);
        let param_changed = self.draw_pde_parameter_drags(ui);
        if domain_dirty {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.rebuild_domain_and_warmup(8);
        } else if mask_dirty {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.rebuild_masks();
        } else if param_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn draw_clamp_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut mask_dirty = false;
        ui.label("Plate Nodal Clamps:");
        let prev_clamp = self.state.clamp_layout;
        egui::ComboBox::from_id_salt("chladni_clamps")
            .selected_text(self.state.clamp_layout.label())
            .show_ui(ui, |ui| {
                for c in [
                    ClampLayout::None,
                    ClampLayout::CenterPin,
                    ClampLayout::QuadSymmetry,
                    ClampLayout::EdgeMidpoints,
                ] {
                    ui.selectable_value(&mut self.state.clamp_layout, c, c.label());
                }
            });
        mask_dirty |= self.state.clamp_layout != prev_clamp;
        if !self.custom_clamps.is_empty() && ui.small_button("Clear Custom Pins").clicked() {
            self.custom_clamps.clear();
            mask_dirty = true;
        }
        mask_dirty
    }

    fn draw_pde_parameter_drags(&mut self, ui: &mut egui::Ui) -> bool {
        let mut param_changed = false;
        ui.horizontal(|ui| {
            ui.label("Wave Speed c:");
            param_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.wave_speed,
                    SPEED_MIN..=SPEED_MAX,
                    0.05,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Plate Stiffness D:");
            param_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.plate_stiffness,
                    STIFFNESS_MIN..=STIFFNESS_MAX,
                    1.0e-5,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Damping γ:");
            param_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.damping,
                    DAMPING_MIN..=DAMPING_MAX,
                    0.01,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Step Δt:");
            param_changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.time_step,
                    DT_MIN..=DT_MAX,
                    0.002,
                ))
                .changed();
        });
        param_changed
    }

    fn draw_driver_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= ui
                .checkbox(&mut self.state.driver_enabled, "Harmonic Driver")
                .changed();
            changed |= ui
                .checkbox(&mut self.state.line_source, "Plane-Wave Line")
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Freq (Hz):");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.driver_frequency,
                    FREQ_MIN..=FREQ_MAX,
                    0.1,
                ))
                .changed();
            ui.label("Amp:");
            changed |= ui
                .add(bounded_drag_value(
                    &mut self.state.driver_amplitude,
                    AMP_MIN..=AMP_MAX,
                    0.5,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Driver (x, y):");
            changed |= ui
                .add(bounded_drag_value(&mut self.state.driver_x, 0.0..=1.0, 0.01))
                .changed();
            changed |= ui
                .add(bounded_drag_value(&mut self.state.driver_y, 0.0..=1.0, 0.01))
                .changed();
        });
        if changed {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.solver.reset_detector();
        }
    }

    fn draw_optics_and_barrier_controls(&mut self, ui: &mut egui::Ui) {
        let mut mask_changed = false;
        let prev_bar = self.state.barrier_enabled;
        let prev_lens = self.state.lens_enabled;
        ui.horizontal(|ui| {
            mask_changed |= ui
                .checkbox(&mut self.state.barrier_enabled, "Multi-Slit Wall")
                .changed();
            mask_changed |= ui
                .checkbox(&mut self.state.lens_enabled, "Acoustic Lens")
                .changed();
        });
        if self.state.barrier_enabled {
            mask_changed |= ui
                .add(egui::Slider::new(&mut self.state.slit_count, 1..=5).text("Slits"))
                .changed();
            mask_changed |= ui
                .add(egui::Slider::new(&mut self.state.slit_width, 1..=8).text("Slit Width"))
                .changed();
            ui.horizontal(|ui| {
                ui.label("Slit Sep d:");
                mask_changed |= ui
                    .add(bounded_drag_value(
                        &mut self.state.slit_separation,
                        SLIT_SEP_MIN..=SLIT_SEP_MAX,
                        0.5,
                    ))
                    .changed();
            });
        }
        if self.state.lens_enabled {
            ui.horizontal(|ui| {
                ui.label("Refractive n:");
                mask_changed |= ui
                    .add(bounded_drag_value(
                        &mut self.state.refractive_index,
                        REFRACTIVE_MIN..=REFRACTIVE_MAX,
                        0.05,
                    ))
                    .changed();
            });
        }
        if ui
            .add(egui::Slider::new(&mut self.state.detector_x, 16..=118).text("Detector X"))
            .changed()
        {
            self.solver.reset_detector();
            self.state_rev = self.state_rev.wrapping_add(1);
        }
        if self.state.barrier_enabled != prev_bar || self.state.lens_enabled != prev_lens {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.rebuild_domain_and_warmup(16);
        } else if mask_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.rebuild_masks();
            self.solver.reset_detector();
        }
    }

    fn draw_visual_and_metric_readouts(&mut self, ui: &mut egui::Ui) {
        let mut vis_changed = false;
        egui::ComboBox::from_id_salt("chladni_colormap")
            .selected_text(self.state.colormap.label())
            .show_ui(ui, |ui| {
                for cm in [
                    WaveColormap::PerceptualThermal,
                    WaveColormap::BipolarPhase,
                    WaveColormap::WcagHighContrast,
                ] {
                    vis_changed |= ui
                        .selectable_value(&mut self.state.colormap, cm, cm.label())
                        .changed();
                }
            });
        ui.horizontal(|ui| {
            vis_changed |= ui
                .checkbox(&mut self.state.show_nodal_contours, "Nodal Contours")
                .changed();
            vis_changed |= ui
                .checkbox(&mut self.state.show_sand_particles, "Chladni Sand (4k)")
                .changed();
        });
        ui.horizontal(|ui| {
            if ui.small_button("Reseed Sand").clicked() {
                self.state.rng_seed = self.state.rng_seed.wrapping_add(1);
                self.sand.reseed(
                    self.state.rng_seed,
                    GRID_SIZE,
                    GRID_SIZE,
                    &self.solver.wall_mask,
                );
                vis_changed = true;
            }
            vis_changed |= self.draw_secondary_plot_selector(ui);
        });
        if vis_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
            self.refresh_analysis_outputs();
        }
        ui.separator();
        self.draw_metric_cards(ui);
    }

    fn draw_secondary_plot_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_id_salt("chladni_sec_plot")
            .selected_text(match self.state.secondary_plot {
                SecondaryPlotMode::DetectorCrossSection => "Detector I(y)",
                SecondaryPlotMode::RadialFftSpectrum => "2D FFT Spectrum",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(
                        &mut self.state.secondary_plot,
                        SecondaryPlotMode::DetectorCrossSection,
                        "Detector I(y)",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut self.state.secondary_plot,
                        SecondaryPlotMode::RadialFftSpectrum,
                        "2D FFT Spectrum",
                    )
                    .changed();
            });
        changed
    }

    fn draw_metric_cards(&self, ui: &mut egui::Ui) {
        let cfl = self
            .solver
            .effective_courant(&self.pde_params(), self.state.time_step);
        render_copyable_metric(
            ui,
            "Courant / Substeps",
            &format!("{cfl:.3} ({}x)", self.last_substeps),
        );
        render_copyable_metric(ui, "Step Budget", &format!("{:.2} ms", self.last_step_ms));
        render_copyable_metric(
            ui,
            "Dominant 2D FFT |k|",
            &format!("{:.2}", self.dominant_peak.k_radial),
        );
        if self.state.barrier_enabled && self.state.wave_speed > 0.0 {
            let lam = self.state.wave_speed / self.state.driver_frequency.max(1.0e-3);
            let screen_d =
                (self.state.detector_x.saturating_sub(self.state.barrier_x)).max(1) as f64;
            let dy = young_double_slit_fringe_spacing(lam, self.state.slit_separation, screen_d);
            render_copyable_metric(ui, "Young Fringe Δy", &format!("{dy:.2} cells"));
        } else if self.state.medium == Wave2DMedium::BiharmonicPlate {
            let omega11 =
                analytic_clamped_square_omega(1, 1, 1.0, self.state.plate_stiffness.max(1.0e-9));
            render_copyable_metric(
                ui,
                "Clamped f(1,1)",
                &format!("{:.2} Hz", omega11 / std::f64::consts::TAU),
            );
        }
    }
}
