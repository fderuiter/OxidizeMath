//! Side-panel controls, `bounded_drag_value` inputs, and live Family-Vicsek / $\beta$-exponent
//! `egui_plot` telemetry for [`KpzInterfaceTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints};
use math_commons::math_kernel::colormap::ensure_dark_canvas_contrast;
use math_explorer::physics::stat_mech::interface_growth::{EW_BETA_1D, KPZ_BETA_1D};

use super::kpz_tool::{KpzDimensionMode, KpzInterfaceTool, KpzPlotView};
use crate::async_sim::SimWorkerState;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{bounded_drag_value, PlaybackAction, PlaybackTransport};
use crate::reflective_ui::AnimatedSlider;

impl KpzInterfaceTool {
    pub(crate) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        self.focused_canvas = None;
        let ctx = ui.ctx().clone();
        let hc = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.high_contrast != hc {
            self.high_contrast = hc;
            self.refresh_2d_pixels();
        }

        if self.playing && !FiniteStateGuard::diverged(self) {
            self.step_frame_cooperative();
            if self.playing {
                ctx.request_repaint();
            }
        }

        self.draw_side_panel(ui);
        self.draw_central_split(ui);
    }

    fn draw_side_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "kpz_interface_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("KPZ Interface Growth");
                ui.separator();
                self.draw_kpz_recovery_banner(ui);
                self.draw_kpz_transport_and_modes(ui);
                ui.separator();
                self.draw_kpz_physics_sliders(ui);
                ui.separator();
                self.draw_kpz_lattice_controls(ui);
            });
        });
    }

    fn draw_kpz_recovery_banner(&mut self, ui: &mut egui::Ui) {
        if !FiniteStateGuard::diverged(self) {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        egui::Frame::group(ui.style())
            .stroke(egui::Stroke::new(1.5_f32, palette.error))
            .show(ui, |ui| {
                ui.colored_label(palette.error, "⚠ FiniteStateGuard Halted Step");
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

    fn draw_kpz_transport_and_modes(&mut self, ui: &mut egui::Ui) {
        let state = if FiniteStateGuard::diverged(self) {
            SimWorkerState::Faulted
        } else if self.playing {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        if let Some(act) =
            PlaybackTransport::from_worker_state("kpz_transport", "KPZ Interface Growth", state)
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
                PlaybackAction::Reset => self.rebuild_and_seed(),
            }
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("Lattice:");
            for mode in [KpzDimensionMode::Dim1D, KpzDimensionMode::Dim2D] {
                if ui
                    .selectable_label(self.dimension == mode, mode.label())
                    .clicked()
                    && self.dimension != mode
                {
                    self.dimension = mode;
                    if mode == KpzDimensionMode::Dim2D && self.system_size > 128 {
                        self.system_size = 64;
                    }
                    self.rebuild_and_seed();
                }
            }
        });

        ui.horizontal_wrapped(|ui| {
            if ui.button("EW Regime (λ = 0)").clicked() {
                self.lambda = 0.0;
                self.sync_solver_params();
            }
            if ui.button("KPZ Regime (λ = 3.5)").clicked() {
                self.lambda = 3.5;
                self.sync_solver_params();
            }
        });
    }

    fn draw_kpz_physics_sliders(&mut self, ui: &mut egui::Ui) {
        let before = (self.nu, self.lambda, self.noise_sigma, self.dx, self.dt);
        ui.add(
            AnimatedSlider::new(&mut self.nu, 0.05..=5.0)
                .text("Surface Tension ν")
                .id_salt("kpz_nu"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.lambda, -8.0..=8.0)
                .text("KPZ Nonlinearity λ")
                .id_salt("kpz_lambda"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.noise_sigma, 0.0..=4.0)
                .text("Noise Amplitude σ")
                .id_salt("kpz_sigma"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.dt, 0.002..=0.10)
                .text("Time Step dt")
                .id_salt("kpz_dt"),
        );
        ui.horizontal(|ui| {
            ui.label("Spacing dx:");
            ui.add(bounded_drag_value(&mut self.dx, 0.25..=4.0, 0.05));
        });
        if (self.nu, self.lambda, self.noise_sigma, self.dx, self.dt) != before {
            self.sync_solver_params();
        }
    }

    fn draw_kpz_lattice_controls(&mut self, ui: &mut egui::Ui) {
        let max_l = match self.dimension {
            KpzDimensionMode::Dim1D => 512.0,
            KpzDimensionMode::Dim2D => 128.0,
        };
        let prev_l = self.system_size;
        let prev_seed = self.rng_seed;
        ui.horizontal(|ui| {
            ui.label("System Size L:");
            let mut l_f64 = self.system_size as f64;
            if ui
                .add(bounded_drag_value(&mut l_f64, 32.0..=max_l, 4.0))
                .changed()
            {
                self.system_size = l_f64.round() as usize;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Steps / Frame:");
            let mut spf_f64 = self.steps_per_frame as f64;
            if ui
                .add(bounded_drag_value(&mut spf_f64, 1.0..=16.0, 1.0))
                .changed()
            {
                self.steps_per_frame = spf_f64.round() as usize;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Brush Radius:");
            let mut rad_f64 = self.brush_radius as f64;
            if ui
                .add(bounded_drag_value(&mut rad_f64, 1.0..=24.0, 1.0))
                .changed()
            {
                self.brush_radius = rad_f64.round() as usize;
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
        if self.system_size != prev_l || self.rng_seed != prev_seed {
            self.rebuild_and_seed();
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("Telemetry:");
            for view in [KpzPlotView::RawLogLog, KpzPlotView::FamilyVicsek] {
                ui.selectable_value(&mut self.plot_view, view, view.label());
            }
        });
    }

    fn draw_central_split(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.draw_kpz_telemetry_strip(ui);
            ui.separator();
            let avail = ui.available_size();
            if avail.x >= 680.0 {
                ui.columns(2, |cols| {
                    let canvas_sz = egui::vec2(cols[0].available_width(), avail.y - 16.0);
                    self.draw_interface_canvas(&mut cols[0], canvas_sz);
                    self.draw_scaling_plot(&mut cols[1], avail.y - 16.0);
                });
            } else {
                let half_h = ((avail.y - 24.0) * 0.48).max(180.0);
                self.draw_interface_canvas(ui, egui::vec2(avail.x, half_h));
                ui.add_space(6.0);
                self.draw_scaling_plot(ui, half_h);
            }
        });
    }

    fn draw_kpz_telemetry_strip(&self, ui: &mut egui::Ui) {
        let (t, w_rms) = match self.dimension {
            KpzDimensionMode::Dim1D => (self.solver_1d.time(), self.solver_1d.interface_width()),
            KpzDimensionMode::Dim2D => (self.solver_2d.time(), self.solver_2d.interface_width()),
        };
        let (_, target_beta, _) = self
            .current_growth_kind()
            .scaling_exponents(self.current_dim_num());
        let beta_str = match self.fitted_beta() {
            Some((b, _)) => format!("{b:.3} (theory {target_beta:.3})"),
            None => format!("-- (theory {target_beta:.3})"),
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("t = {t:.2} | W(L, t) = {w_rms:.4}"));
            ui.separator();
            ui.label(format!("Fitted slope β = {beta_str}"));
            ui.separator();
            ui.small(format!("Step: {:.2} ms", self.last_step_ms));
        });
    }

    fn draw_scaling_plot(&self, ui: &mut egui::Ui, height: f32) {
        let datasets = self.collect_export_datasets();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let c0 = ensure_dark_canvas_contrast([
            palette.accent.r(),
            palette.accent.g(),
            palette.accent.b(),
        ]);
        let c1 = ensure_dark_canvas_contrast([
            palette.success.r(),
            palette.success.g(),
            palette.success.b(),
        ]);
        let c2 = ensure_dark_canvas_contrast([
            palette.warning.r(),
            palette.warning.g(),
            palette.warning.b(),
        ]);
        let colors = [
            egui::Color32::from_rgb(c0[0], c0[1], c0[2]),
            egui::Color32::from_rgb(c1[0], c1[1], c1[2]),
            egui::Color32::from_rgb(c2[0], c2[1], c2[2]),
        ];

        let (x_lbl, y_lbl) = match self.plot_view {
            KpzPlotView::RawLogLog => ("ln(t)", "ln W(L, t)"),
            KpzPlotView::FamilyVicsek => ("ln(t / L^z)", "ln(W / L^alpha)"),
        };

        Plot::new(crate::framework::scoped_panel_id(ui, "kpz_scaling_plot"))
            .height(height.max(180.0))
            .legend(Legend::default())
            .x_axis_label(x_lbl)
            .y_axis_label(y_lbl)
            .show(ui, |plot_ui| {
                for (idx, (name, pts)) in datasets.into_iter().enumerate() {
                    let col = colors[idx % colors.len()];
                    let line = if idx == 0 {
                        Line::new(name, PlotPoints::from(pts))
                            .color(col)
                            .width(2.2_f32)
                    } else {
                        Line::new(name, PlotPoints::from(pts))
                            .color(col)
                            .width(1.5_f32)
                            .style(LineStyle::dashed_dense())
                    };
                    plot_ui.line(line);
                }
            });
    }

    pub(crate) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        let (xs, ys, series_label) = match self.plot_view {
            KpzPlotView::RawLogLog => (
                self.telemetry.times(),
                self.telemetry.widths(),
                "ln W(L, t)",
            ),
            KpzPlotView::FamilyVicsek => (
                self.telemetry.rescaled_times(),
                self.telemetry.rescaled_widths(),
                "Family-Vicsek ln(W / L^alpha)",
            ),
        };

        let log_pts: Vec<[f64; 2]> = xs
            .iter()
            .zip(ys.iter())
            .filter(|(&t, &w)| t > 0.0 && w > 1e-14 && t.is_finite() && w.is_finite())
            .map(|(&t, &w)| [t.ln(), w.ln()])
            .collect();

        if let (Some(&first), Some(&last)) = (log_pts.first(), log_pts.last()) {
            out.push((series_label.to_string(), log_pts.clone()));
            if (last[0] - first[0]).abs() > 1e-9 {
                let kpz_guide = vec![
                    first,
                    [last[0], first[1] + KPZ_BETA_1D * (last[0] - first[0])],
                ];
                let ew_guide = vec![
                    first,
                    [last[0], first[1] + EW_BETA_1D * (last[0] - first[0])],
                ];
                out.push(("KPZ Reference Slope (1/3)".to_string(), kpz_guide));
                out.push(("EW Reference Slope (1/4)".to_string(), ew_guide));
            }
        }
        out
    }
}
