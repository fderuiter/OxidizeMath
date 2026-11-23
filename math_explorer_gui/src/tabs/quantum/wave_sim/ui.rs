#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::persistence::truncate_utf8_bytes;
use super::presets::{
    wave_parameter_label, WavePotentialPreset, DT_MAX, DT_MIN, K0_MAX, K0_MIN,
    MAX_CUSTOM_EXPR_BYTES, MAX_STEPS_PER_FRAME, POTENTIAL_DISPLAY_SCALE, SIGMA_MAX, SIGMA_MIN,
    V0_MAX, V0_MIN, WIDTH_MAX, WIDTH_MIN, X0_MAX, X0_MIN,
};
use super::runner::WaveData;
use super::WaveSimulatorTool;
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{SimCommand, SimWorkerState, StateSnapshot};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints};

impl WaveSimulatorTool {
    pub(super) fn draw_left_panel(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "Wave Simulator_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.heading("Wave Simulator");
                    ui.separator();
                    self.draw_status_and_transport(ui, &ctx);
                    ui.separator();
                    self.draw_preset_and_sandbox(ui, &ctx);
                    ui.separator();
                    self.draw_wavepacket_and_barrier_sliders(ui, &ctx);
                    ui.separator();
                    self.draw_tunneling_readouts(ui);
                });
        });
    }

    fn draw_status_and_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let state = self.worker.state();
        if let Some(fault_action) =
            render_worker_fault_banner(ui, state, self.worker.diagnostic_report(), true)
        {
            match fault_action {
                WorkerFaultAction::Reset => self.sync_config_and_reset(Some(ctx)),
                WorkerFaultAction::ResetParameters => self.reset_to_defaults(Some(ctx)),
                WorkerFaultAction::Recovery(action) => self.handle_recovery_action(action, ctx),
            }
            ui.separator();
        }

        let state = self.worker.state();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal(|ui| {
            ui.label("Status:");
            let (status_text, color) = match state {
                SimWorkerState::Uninitialized => ("Uninitialized", palette.muted_text),
                SimWorkerState::Ready => ("Ready", palette.success),
                SimWorkerState::Running => ("Running", palette.info),
                SimWorkerState::Stepping => ("Stepping", palette.warning),
                SimWorkerState::Faulted => ("Faulted", palette.error),
                SimWorkerState::Terminated => ("Terminated", palette.muted_text),
            };
            ui.colored_label(color, status_text);
        });

        if let Some(action) =
            PlaybackTransport::from_worker_state("wave_sim_transport", "Wave Simulator", state)
                .can_reset(true)
                .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    let _ = self.worker.send_command(SimCommand::Start);
                }
                PlaybackAction::Pause => {
                    let _ = self.worker.send_command(SimCommand::Pause);
                }
                PlaybackAction::Step => {
                    let _ = self.worker.send_command(SimCommand::Step(1));
                }
                PlaybackAction::Reset => {
                    self.sync_config_and_reset(Some(ctx));
                }
            }
        }

        let speed_slider = egui::Slider::new(&mut self.steps_per_frame, 1..=MAX_STEPS_PER_FRAME)
            .clamping(egui::SliderClamping::Always)
            .text(wave_parameter_label("steps_per_frame"));
        if ui.add(speed_slider).changed() {
            self.bump_revision();
            self.export_runner = None;
            self.export_data = None;
            if let Ok(mut cfg) = self.shared_config.write() {
                cfg.steps_per_frame = self.steps_per_frame;
            }
            let _ = self
                .worker
                .send_command(SimCommand::SetSpeed(self.steps_per_frame));
        }
    }

    fn draw_preset_and_sandbox(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut reset_needed = false;
        let combo_resp = egui::ComboBox::from_label(wave_parameter_label("preset"))
            .selected_text(self.preset.label())
            .show_ui(ui, |ui| {
                for candidate in WavePotentialPreset::ALL {
                    if ui
                        .selectable_value(&mut self.preset, candidate, candidate.label())
                        .changed()
                    {
                        reset_needed = true;
                    }
                }
            });
        let _ = combo_resp.response.accessible_hover_text(
            "Choose a built-in quantum potential profile or author a custom V(x) formula.",
        );

        ui.add_space(4.0);
        if ui
            .button("🎯 50% Tunneling Split")
            .accessible_hover_text(
                "Configure a sub-barrier wavepacket (E0 = 12.5, V0 = 11.35, width = 0.5) that splits 50/50 across a rectangular barrier.",
            )
            .clicked()
        {
            self.apply_tunneling_split_challenge(Some(ctx));
            let _ = self.worker.send_command(SimCommand::Start);
            return;
        }

        ui.add_space(4.0);
        if let Some((latex_eq, plain_eq)) = self.preset.formula_display() {
            crate::reflective_ui::render_formula_with_export(ui, latex_eq, Some(plain_eq));
        } else if self.math_input.ui(ui) {
            if self.math_input.raw_text.len() > MAX_CUSTOM_EXPR_BYTES {
                let bounded = truncate_utf8_bytes(&self.math_input.raw_text, MAX_CUSTOM_EXPR_BYTES)
                    .to_string();
                self.math_input.set_expression(&bounded);
            }
            self.bump_revision();
            if let Some(expr) = self.math_input.expr() {
                self.last_valid_expr = expr.clone();
                reset_needed = true;
            }
        }

        if reset_needed {
            self.sync_config_and_reset(Some(ctx));
        }
    }

    fn draw_wavepacket_and_barrier_sliders(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Wavepacket & Barrier Controls");
        let mut reset_needed = false;

        reset_needed |= draw_bounded_slider(
            ui,
            &mut self.x0,
            X0_MIN..=X0_MAX,
            0.05,
            wave_parameter_label("x0"),
            "Initial center x0 of the normalized Gaussian wavepacket.",
        );
        reset_needed |= draw_bounded_slider(
            ui,
            &mut self.k0,
            K0_MIN..=K0_MAX,
            0.1,
            wave_parameter_label("k0"),
            "Initial wave number k0 setting kinetic energy E0 = k0² / 2.",
        );
        reset_needed |= draw_bounded_slider(
            ui,
            &mut self.sigma,
            SIGMA_MIN..=SIGMA_MAX,
            0.05,
            wave_parameter_label("sigma"),
            "Spatial spread σ of the initial Gaussian wavepacket.",
        );

        if self.preset.uses_barrier_controls() {
            reset_needed |= draw_bounded_slider(
                ui,
                &mut self.v0,
                V0_MIN..=V0_MAX,
                0.05,
                wave_parameter_label("v0"),
                "Potential barrier height V0 compared against kinetic energy E0.",
            );
            reset_needed |= draw_bounded_slider(
                ui,
                &mut self.width,
                WIDTH_MIN..=WIDTH_MAX,
                0.05,
                wave_parameter_label("width"),
                "Characteristic barrier width parameter w.",
            );
        }

        self.draw_dt_and_energy_summary(ui);
        if reset_needed {
            self.sync_config_and_reset(Some(ctx));
        }
    }

    fn draw_dt_and_energy_summary(&mut self, ui: &mut egui::Ui) {
        if draw_bounded_slider(
            ui,
            &mut self.dt,
            DT_MIN..=DT_MAX,
            0.001,
            wave_parameter_label("dt"),
            "Integration time step Δt for unitary Schrödinger evolution.",
        ) {
            self.bump_revision();
            self.export_runner = None;
            self.export_data = None;
            if let Ok(mut cfg) = self.shared_config.write() {
                cfg.dt = self.dt;
            }
        }

        let e0 = 0.5 * self.k0 * self.k0;
        let regime = if !self.preset.uses_barrier_controls() {
            "Confining potential"
        } else if e0 < self.v0 {
            "Sub-barrier tunneling (E0 < V0)"
        } else {
            "Over-barrier scattering (E0 ≥ V0)"
        };
        ui.small(format!("Incident Energy E0 = k0²/2 = {e0:.2} ({regime})"));
    }

    fn draw_tunneling_readouts(&self, ui: &mut egui::Ui) {
        ui.label("Born Tunneling Observables");
        let Some(wave_data) = self.latest_wave_data() else {
            ui.small("Awaiting initial wavepacket snapshot...");
            return;
        };
        crate::reflective_ui::render_copyable_metric(
            ui,
            "Transmission T(t)",
            &format!(
                "{:.1}% ({:.4})",
                wave_data.transmission * 100.0,
                wave_data.transmission
            ),
        );
        crate::reflective_ui::render_copyable_metric(
            ui,
            "Reflection R(t)",
            &format!(
                "{:.1}% ({:.4})",
                wave_data.reflection * 100.0,
                wave_data.reflection
            ),
        );
        ui.small(format!(
            "Barrier Interior B(t): {:.1}% ([{:.2}, {:.2}])",
            wave_data.barrier_prob * 100.0,
            wave_data.x_left,
            wave_data.x_right
        ));
        let scatter_text = match wave_data.latched_transmission {
            Some(latched) if wave_data.right_wall_prob >= 0.02 => {
                format!("{:.1}% ({latched:.4}, pre-wall frozen)", latched * 100.0)
            }
            Some(latched) => format!("{:.1}% ({latched:.4})", latched * 100.0),
            None => "Pending transit...".to_string(),
        };
        crate::reflective_ui::render_copyable_metric(ui, "Latched Scatter T", &scatter_text);
    }

    pub(super) fn draw_central_panel(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let Some(wave_data) = self.latest_wave_data() else {
                render_empty_plot_guidance(
                    ui,
                    "Wavepacket ready — click ▶ Play in the side panel to evolve ψ(x, t).",
                );
                return;
            };
            render_wave_plot(ui, wave_data);
        });
    }

    fn latest_wave_data(&self) -> Option<&WaveData> {
        self.worker
            .latest_snapshot()
            .and_then(|snap: &StateSnapshot| snap.structured_data.as_ref())
            .and_then(|any_data| any_data.downcast_ref::<WaveData>())
    }
}

fn draw_bounded_slider(
    ui: &mut egui::Ui,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    step: f64,
    label: &str,
    hover_desc: &str,
) -> bool {
    let slider = egui::Slider::new(value, range)
        .step_by(step)
        .clamping(egui::SliderClamping::Always)
        .text(label);
    let resp = ui.add(slider);
    let _ = resp.clone().accessible_hover_text(hover_desc);
    resp.changed()
}

fn render_wave_plot(ui: &mut egui::Ui, wave_data: &WaveData) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let psi_color = palette.series_color(0);
    let potential_color = palette.series_color(1);
    let energy_color = palette.series_color(2);

    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Time: {:.3}", wave_data.time));
        ui.separator();
        ui.colored_label(psi_color, "━ |ψ(x, t)|²");
        ui.colored_label(potential_color, "━ V(x) (scaled)");
        ui.colored_label(
            energy_color,
            format!("┅ E0 = {:.2} (scaled)", wave_data.kinetic_energy),
        );
    });

    let len = wave_data
        .x_axis
        .len()
        .min(wave_data.prob_density.len())
        .min(wave_data.potential.len());
    let mut psi_points = Vec::with_capacity(len);
    let mut v_points = Vec::with_capacity(len);
    for i in 0..len {
        let x = wave_data.x_axis[i];
        psi_points.push([x, wave_data.prob_density[i]]);
        v_points.push([x, wave_data.potential[i] * POTENTIAL_DISPLAY_SCALE]);
    }
    let x_start = wave_data.x_axis.first().copied().unwrap_or(-5.0);
    let x_end = wave_data.x_axis.last().copied().unwrap_or(5.0);
    let e0_scaled = wave_data.kinetic_energy * POTENTIAL_DISPLAY_SCALE;

    Plot::new("quantum_plot")
        .legend(Legend::default())
        .x_axis_label("Position (x)")
        .y_axis_label("Probability Density |ψ|² / Scaled Energy")
        .view_aspect(2.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Probability |ψ(x, t)|²", PlotPoints::new(psi_points))
                    .color(psi_color)
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new("Potential V(x) (scaled)", PlotPoints::new(v_points))
                    .color(potential_color)
                    .width(1.75_f32),
            );
            plot_ui.line(
                Line::new(
                    format!(
                        "Kinetic Energy E0 = {:.2} (scaled)",
                        wave_data.kinetic_energy
                    ),
                    PlotPoints::new(vec![[x_start, e0_scaled], [x_end, e0_scaled]]),
                )
                .style(LineStyle::Dashed { length: 6.0_f32 })
                .color(energy_color),
            );
        });
}
