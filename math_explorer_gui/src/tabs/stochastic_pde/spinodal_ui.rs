//! Side-panel controls, quench temperature & noise sliders, and radial $S(k, t)$ / $k_{\max}(t)$
//! `egui_plot` coarsening telemetry for [`SpinodalDecompositionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points};
use math_commons::math_kernel::colormap::ensure_dark_canvas_contrast;

use super::spinodal_tool::{
    PhaseFieldModelChoice, SpinodalBrushAction, SpinodalDecompositionTool, SpinodalPlotMode,
};
use crate::async_sim::SimWorkerState;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{bounded_drag_value, PlaybackAction, PlaybackTransport};
use crate::reflective_ui::AnimatedSlider;

impl SpinodalDecompositionTool {
    pub(crate) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let hc = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.high_contrast != hc {
            self.high_contrast = hc;
            self.refresh_pixels_in_place();
        }

        if self.playing && !FiniteStateGuard::diverged(self) {
            self.step_frame_cooperative();
            if self.playing {
                ctx.request_repaint();
            }
        }

        self.draw_spinodal_sidebar(ui);
        self.draw_spinodal_central(ui);
    }

    fn draw_spinodal_sidebar(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "spinodal_decomposition_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Spinodal Decomposition");
                ui.separator();
                self.draw_spinodal_recovery_banner(ui);
                self.draw_spinodal_transport_and_model(ui);
                ui.separator();
                self.draw_spinodal_thermodynamic_sliders(ui);
                ui.separator();
                self.draw_spinodal_grid_and_brush(ui);
            });
        });
    }

    fn draw_spinodal_recovery_banner(&mut self, ui: &mut egui::Ui) {
        if !FiniteStateGuard::diverged(self) {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        egui::Frame::group(ui.style())
            .stroke(egui::Stroke::new(1.5_f32, palette.error))
            .show(ui, |ui| {
                ui.colored_label(palette.error, "⚠ Phase-Field Guard Triggered");
                if let Some(diag) = &self.diagnostic {
                    ui.small(&diag.message);
                }
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Auto-Scale dt").clicked() {
                        FiniteStateGuard::auto_scale_step_size(self);
                    }
                    if ui.button("Clamp Params").clicked() {
                        FiniteStateGuard::clamp_parameters(self);
                    }
                    if ui.button("Reset Default").clicked() {
                        FiniteStateGuard::reset_parameters(self);
                    }
                });
            });
        ui.separator();
    }

    fn draw_spinodal_transport_and_model(&mut self, ui: &mut egui::Ui) {
        let state = if FiniteStateGuard::diverged(self) {
            SimWorkerState::Faulted
        } else if self.playing {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        if let Some(act) = PlaybackTransport::from_worker_state(
            "spinodal_transport",
            "Spinodal Decomposition",
            state,
        )
        .can_reset(true)
        .show(ui)
        {
            match act {
                PlaybackAction::Play => self.playing = true,
                PlaybackAction::Pause => self.playing = false,
                PlaybackAction::Step => {
                    self.playing = false;
                    self.step_once();
                }
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                PlaybackAction::Reset => self.reseed_spinodal_state(),
            }
        }

        ui.label("Dynamics Class:");
        ui.horizontal_wrapped(|ui| {
            for m in [
                PhaseFieldModelChoice::CahnHilliard,
                PhaseFieldModelChoice::AllenCahn,
            ] {
                if ui.selectable_label(self.model == m, m.label()).clicked() && self.model != m {
                    self.model = m;
                    self.sync_solver_params();
                }
            }
        });
    }

    fn draw_spinodal_thermodynamic_sliders(&mut self, ui: &mut egui::Ui) {
        let before = (
            self.temperature,
            self.critical_temperature,
            self.mobility,
            self.kappa,
            self.noise_sigma,
            self.dt,
        );
        ui.add(
            AnimatedSlider::new(&mut self.temperature, 0.0..=2.0)
                .text("Quench Temp T")
                .id_salt("spinodal_temp"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.critical_temperature, 0.2..=2.5)
                .text("Critical Temp Tc")
                .id_salt("spinodal_tc"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.mobility, 0.05..=5.0)
                .text("Mobility M")
                .id_salt("spinodal_mobility"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.kappa, 0.05..=4.0)
                .text("Interfacial κ")
                .id_salt("spinodal_kappa"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.noise_sigma, 0.0..=0.50)
                .text("Thermal Noise σ")
                .id_salt("spinodal_sigma"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.dt, 0.005..=0.25)
                .text("Time Step dt")
                .id_salt("spinodal_dt"),
        );
        if (
            self.temperature,
            self.critical_temperature,
            self.mobility,
            self.kappa,
            self.noise_sigma,
            self.dt,
        ) != before
        {
            self.sync_solver_params();
            self.refresh_pixels_in_place();
        }
    }

    fn draw_spinodal_grid_and_brush(&mut self, ui: &mut egui::Ui) {
        self.draw_spinodal_grid_controls(ui);
        self.draw_spinodal_brush_controls(ui);
    }

    fn draw_spinodal_grid_controls(&mut self, ui: &mut egui::Ui) {
        let prev_reseed = (self.grid_size, self.mean_phi.to_bits(), self.rng_seed);
        ui.horizontal(|ui| {
            ui.label("Mean <φ>:");
            ui.add(bounded_drag_value(&mut self.mean_phi, -0.65..=0.65, 0.02));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Grid:");
            for dim in [32usize, 64, 96, 128] {
                if ui
                    .selectable_label(self.grid_size == dim, format!("{dim}²"))
                    .clicked()
                {
                    self.grid_size = dim;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Steps / Frame:");
            let mut spf_f64 = self.steps_per_frame as f64;
            if ui
                .add(bounded_drag_value(&mut spf_f64, 1.0..=12.0, 1.0))
                .changed()
            {
                self.steps_per_frame = spf_f64.round() as usize;
            }
        });
        ui.horizontal(|ui| {
            ui.label("RNG Seed:");
            let mut seed_f64 = (self.rng_seed.min(1_000_000)) as f64;
            if ui
                .add(bounded_drag_value(&mut seed_f64, 0.0..=1_000_000.0, 1.0))
                .changed()
            {
                self.rng_seed = seed_f64.round() as u64;
            }
        });
        if (self.grid_size, self.mean_phi.to_bits(), self.rng_seed) != prev_reseed {
            self.reseed_spinodal_state();
        }
    }

    fn draw_spinodal_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.brush_action,
                SpinodalBrushAction::EnrichPositive,
                "+φ Droplet",
            );
            ui.selectable_value(
                &mut self.brush_action,
                SpinodalBrushAction::EnrichNegative,
                "-φ Droplet",
            );
            let mut rad_f64 = self.brush_radius as f64;
            if ui
                .add(bounded_drag_value(&mut rad_f64, 1.0..=16.0, 1.0))
                .changed()
            {
                self.brush_radius = rad_f64.round() as usize;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Plot:");
            for m in [
                SpinodalPlotMode::StructureFactor,
                SpinodalPlotMode::PeakWavenumber,
            ] {
                ui.selectable_value(&mut self.plot_mode, m, m.label());
            }
        });
    }

    fn draw_spinodal_central(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.draw_spinodal_telemetry_bar(ui);
            ui.separator();
            let avail = ui.available_size();
            if avail.x >= 680.0 {
                ui.columns(2, |cols| {
                    let side = cols[0].available_width().min(avail.y - 16.0);
                    self.draw_phase_field_canvas(&mut cols[0], side);
                    self.draw_spinodal_plot(&mut cols[1], avail.y - 16.0);
                });
            } else {
                let half_h = ((avail.y - 24.0) * 0.48).max(180.0);
                self.draw_phase_field_canvas(ui, half_h.min(avail.x));
                ui.add_space(6.0);
                self.draw_spinodal_plot(ui, half_h);
            }
        });
    }

    fn draw_spinodal_telemetry_bar(&self, ui: &mut egui::Ui) {
        let sum = self.solver.telemetry().latest_summary();
        ui.horizontal_wrapped(|ui| {
            ui.label(format!(
                "t = {:.2} | <φ> = {:.6} | F[φ] = {:.2}",
                self.solver.time(),
                self.solver.mean_concentration(),
                self.solver.free_energy(),
            ));
            ui.separator();
            ui.label(format!(
                "k_max(t) = {:.3} | k_1(t) = {:.3}",
                sum.k_max, sum.k_first_moment
            ));
            ui.separator();
            ui.small(format!("Step: {:.2} ms", self.last_step_ms));
        });
    }

    fn draw_spinodal_plot(&self, ui: &mut egui::Ui, height: f32) {
        let datasets = self.collect_export_datasets();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let c0 = ensure_dark_canvas_contrast([
            palette.accent.r(),
            palette.accent.g(),
            palette.accent.b(),
        ]);
        let c1 = ensure_dark_canvas_contrast([
            palette.warning.r(),
            palette.warning.g(),
            palette.warning.b(),
        ]);
        let primary_col = egui::Color32::from_rgb(c0[0], c0[1], c0[2]);
        let secondary_col = egui::Color32::from_rgb(c1[0], c1[1], c1[2]);

        let (x_lbl, y_lbl) = match self.plot_mode {
            SpinodalPlotMode::StructureFactor => {
                ("Radial Wavenumber k", "Structure Factor S(k, t)")
            }
            SpinodalPlotMode::PeakWavenumber => ("Time t", "Characteristic Wavenumber k(t)"),
        };

        let sum = self.solver.telemetry().latest_summary();
        Plot::new(crate::framework::scoped_panel_id(ui, "spinodal_sk_plot"))
            .height(height.max(180.0))
            .legend(Legend::default())
            .x_axis_label(x_lbl)
            .y_axis_label(y_lbl)
            .show(ui, |plot_ui| {
                for (idx, (name, pts)) in datasets.into_iter().enumerate() {
                    if idx == 0 {
                        plot_ui.line(
                            Line::new(name, PlotPoints::from(pts))
                                .color(primary_col)
                                .width(2.2_f32),
                        );
                    } else {
                        plot_ui.line(
                            Line::new(name, PlotPoints::from(pts))
                                .color(secondary_col)
                                .width(1.6_f32)
                                .style(LineStyle::dashed_dense()),
                        );
                    }
                }
                if self.plot_mode == SpinodalPlotMode::StructureFactor && sum.s_max > 0.0 {
                    plot_ui.points(
                        Points::new("Peak k_max(t)", vec![[sum.k_max, sum.s_max]])
                            .color(secondary_col)
                            .radius(4.5_f32),
                    );
                }
            });
    }

    pub(crate) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let tel = self.solver.telemetry();
        let mut out = Vec::new();
        match self.plot_mode {
            SpinodalPlotMode::StructureFactor => {
                let sk_pts: Vec<[f64; 2]> = tel
                    .k_bins()
                    .iter()
                    .zip(tel.s_k().iter())
                    .skip(1)
                    .map(|(&k, &s)| [k, s])
                    .collect();
                if !sk_pts.is_empty() {
                    out.push(("Radial Structure Factor S(k, t)".to_string(), sk_pts));
                }
                let sum = tel.latest_summary();
                if sum.s_max > 0.0 {
                    out.push((
                        "Peak Wavenumber Marker k_max(t)".to_string(),
                        vec![[sum.k_max, 0.0], [sum.k_max, sum.s_max]],
                    ));
                }
            }
            SpinodalPlotMode::PeakWavenumber => {
                let kmax_pts: Vec<[f64; 2]> = tel
                    .times()
                    .iter()
                    .zip(tel.k_max_history().iter())
                    .map(|(&t, &k)| [t, k])
                    .collect();
                let kmom_pts: Vec<[f64; 2]> = tel
                    .times()
                    .iter()
                    .zip(tel.k_moment_history().iter())
                    .map(|(&t, &k)| [t, k])
                    .collect();
                if !kmax_pts.is_empty() {
                    out.push(("Peak k_max(t)".to_string(), kmax_pts));
                }
                if !kmom_pts.is_empty() {
                    out.push(("First Spectral Moment k_1(t)".to_string(), kmom_pts));
                }
            }
        }
        out
    }
}
