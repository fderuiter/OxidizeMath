#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::canvas::render_wave_central_panel;
use super::persistence::truncate_utf8_bytes;
use super::presets::{
    wave_parameter_label, WavePotentialPreset, DT_MAX, DT_MIN, K0_MAX, K0_MIN,
    MAX_CUSTOM_EXPR_BYTES, MAX_STEPS_PER_FRAME, SIGMA_MAX, SIGMA_MIN, V0_MAX, V0_MIN, WIDTH_MAX,
    WIDTH_MIN, X0_MAX, X0_MIN,
};
use super::runner::WaveData;
use super::WaveSimulatorTool;
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{SimCommand, SimWorkerState, StateSnapshot};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;

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
                    let is_paused = !matches!(
                        self.worker.state(),
                        SimWorkerState::Running | SimWorkerState::Stepping
                    );
                    crate::audio::render_sonification_panel(
                        ui,
                        "wave_sim",
                        &self.latest_sonification,
                        is_paused,
                    );
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
                        self.custom_potential = None;
                        reset_needed = true;
                    }
                }
            });
        let _ = combo_resp.response.accessible_hover_text(
            "Choose a built-in quantum potential profile or author a custom V(x) formula.",
        );

        ui.add_space(4.0);
        let mut launched_challenge = false;
        let mut clear_sculpt = false;
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🎯 50% Tunneling Split")
                .accessible_hover_text(
                    "Configure a sub-barrier wavepacket (E0 = 12.5, V0 = 11.35, width = 0.5) that splits 50/50 across a rectangular barrier.",
                )
                .clicked()
            {
                launched_challenge = true;
            }
            if ui
                .add_enabled(
                    self.custom_potential.is_some(),
                    egui::Button::new("Clear Sculpted V(x)"),
                )
                .accessible_hover_text(
                    "Erase freehand potential strokes and restore the selected preset or AST formula.",
                )
                .clicked()
            {
                clear_sculpt = true;
            }
        });
        if launched_challenge {
            self.apply_tunneling_split_challenge(Some(ctx));
            let _ = self.worker.send_command(SimCommand::Start);
            return;
        }
        if clear_sculpt {
            self.clear_custom_potential(Some(ctx));
            return;
        }

        ui.add_space(4.0);
        if self.draw_formula_or_custom_input(ui) || reset_needed {
            self.sync_config_and_reset(Some(ctx));
        }
    }

    fn draw_formula_or_custom_input(&mut self, ui: &mut egui::Ui) -> bool {
        if let Some((latex_eq, plain_eq)) = self.preset.formula_display() {
            crate::reflective_ui::render_formula_with_export(ui, latex_eq, Some(plain_eq));
            return false;
        }
        if self.math_input.ui(ui) {
            if self.math_input.raw_text.len() > MAX_CUSTOM_EXPR_BYTES {
                let bounded = truncate_utf8_bytes(&self.math_input.raw_text, MAX_CUSTOM_EXPR_BYTES)
                    .to_string();
                self.math_input.set_expression(&bounded);
            }
            self.bump_revision();
            if let Some(expr) = self.math_input.expr() {
                self.last_valid_expr = expr.clone();
                self.custom_potential = None;
                return true;
            }
        }
        false
    }

    fn draw_wavepacket_and_barrier_sliders(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Wavepacket & Barrier Controls");
        let mut packet_changed = false;
        let mut barrier_changed = false;

        packet_changed |= draw_bounded_slider(
            ui,
            &mut self.x0,
            X0_MIN..=X0_MAX,
            0.05,
            wave_parameter_label("x0"),
            "Initial center x0 of the normalized Gaussian wavepacket.",
        );
        packet_changed |= draw_bounded_slider(
            ui,
            &mut self.k0,
            K0_MIN..=K0_MAX,
            0.1,
            wave_parameter_label("k0"),
            "Initial wave number k0 setting kinetic energy E0 = k0² / 2.",
        );
        packet_changed |= draw_bounded_slider(
            ui,
            &mut self.sigma,
            SIGMA_MIN..=SIGMA_MAX,
            0.05,
            wave_parameter_label("sigma"),
            "Spatial spread σ of the initial Gaussian wavepacket.",
        );

        if self.preset.uses_barrier_controls() {
            barrier_changed |= draw_bounded_slider(
                ui,
                &mut self.v0,
                V0_MIN..=V0_MAX,
                0.05,
                wave_parameter_label("v0"),
                "Potential barrier height V0 compared against kinetic energy E0.",
            );
            barrier_changed |= draw_bounded_slider(
                ui,
                &mut self.width,
                WIDTH_MIN..=WIDTH_MAX,
                0.05,
                wave_parameter_label("width"),
                "Characteristic barrier width parameter w.",
            );
        }

        self.draw_dt_and_energy_summary(ui);
        if barrier_changed {
            self.custom_potential = None;
            self.sync_config_and_reset(Some(ctx));
        } else if packet_changed {
            self.update_wavepacket_live(self.x0, self.k0, self.sigma, Some(ctx));
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
        let regime = if self.custom_potential.is_some() {
            "Freehand sculpted V(x)"
        } else if !self.preset.uses_barrier_controls() {
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
        let xb = wave_data
            .x_axis
            .get(wave_data.barrier_idx)
            .copied()
            .unwrap_or(0.0);
        ui.small(format!(
            "Two-Region Split (x_b = {xb:.2}): P_T = {:.1}% | P_R = {:.1}%",
            wave_data.p_transmitted * 100.0,
            wave_data.p_reflected * 100.0
        ));
    }

    pub(super) fn draw_central_panel(&mut self, parent_ui: &mut egui::Ui) {
        render_wave_central_panel(self, parent_ui);
    }

    pub(super) fn latest_wave_data(&self) -> Option<&WaveData> {
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
