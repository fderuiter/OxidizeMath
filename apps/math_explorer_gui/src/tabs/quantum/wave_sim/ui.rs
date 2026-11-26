#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::challenges::ActiveWaveChallenge;
use super::persistence::truncate_utf8_bytes;
use super::plot_ui::render_wave_plot;
use super::presets::{
    wave_parameter_label, WavePotentialPreset, DT_MAX, DT_MIN, K0_MAX, K0_MIN,
    MAX_CUSTOM_EXPR_BYTES, MAX_STEPS_PER_FRAME, SIGMA_MAX, SIGMA_MIN, V0_MAX, V0_MIN, WIDTH_MAX,
    WIDTH_MIN, X0_MAX, X0_MIN,
};
use super::runner::WaveData;
use super::WaveSimulatorTool;
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{SimCommand, SimWorkerState, StateSnapshot};
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::reflective_ui::AnimatedSlider;
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
                    let locked = self.is_challenge_locked();
                    ui.add_enabled_ui(!locked, |ui| self.draw_sculpt_controls(ui, &ctx));
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

    fn apply_playback_action(&mut self, action: PlaybackAction, ctx: &egui::Context) {
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
            PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
            PlaybackAction::Reset => self.sync_config_and_reset(Some(ctx)),
        }
    }

    fn draw_worker_status_badge(ui: &mut egui::Ui, state: SimWorkerState) {
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
        Self::draw_worker_status_badge(ui, state);

        let canvas_sculpt_focused = !self.is_challenge_locked()
            && self.brush_kit.is_sculpting()
            && self
                .brush_kit
                .focused_plot_id()
                .is_some_and(|id| ctx.memory(|m| m.has_focus(id)));
        if let Some(action) =
            PlaybackTransport::from_worker_state("wave_sim_transport", "Wave Simulator", state)
                .can_reset(true)
                .allow_space_shortcut(!canvas_sculpt_focused)
                .show(ui)
        {
            self.apply_playback_action(action, ctx);
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

    fn challenge_lock_banner(&self) -> Option<&'static str> {
        match self.active_challenge? {
            ActiveWaveChallenge::Barrier50Split => {
                Some("🔒 50/50 Split Challenge active: x₀ = -2.0, k₀ = 5.0, σ = 0.5 locked.")
            }
            ActiveWaveChallenge::FabryPerotResonance => {
                Some("🔒 Fabry-Pérot Challenge active: V₀ = 20.0, w = 1.0, x₀ = -2.5 locked.")
            }
            ActiveWaveChallenge::Subbarrier50Split => {
                Some("🔒 Sub-Barrier 50/50 Challenge active: x₀ = -2.0, σ = 0.5 locked.")
            }
            ActiveWaveChallenge::DoubleBarrierResonance => {
                Some("🔒 Double-Barrier Resonance active: V₀ = 28.0, w = 0.8, x₀ = -2.0 locked.")
            }
        }
    }

    fn draw_preset_and_sandbox(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut reset_needed = false;
        let locked = self.is_challenge_locked();
        ui.add_enabled_ui(!locked, |ui| {
            let combo_resp = egui::ComboBox::from_label(wave_parameter_label("preset"))
                .selected_text(self.preset.label())
                .show_ui(ui, |ui| {
                    for candidate in WavePotentialPreset::ALL {
                        if ui
                            .selectable_value(&mut self.preset, candidate, candidate.label())
                            .changed()
                        {
                            self.sparse_potential.clear();
                            reset_needed = true;
                        }
                    }
                });
            let _ = combo_resp.response.accessible_hover_text(
                "Choose a built-in quantum potential profile or author a custom V(x) formula.",
            );
        });
        ui.add_space(4.0);
        self.draw_atlas_and_challenge_buttons(ui, ctx);
        if let Some(banner) = self.challenge_lock_banner() {
            ui.small(banner);
        }
        ui.add_space(4.0);
        if let Some((latex_eq, plain_eq)) = self.preset.formula_display() {
            crate::reflective_ui::render_formula_with_export(ui, latex_eq, Some(plain_eq));
        } else if !locked && self.math_input.ui(ui) {
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
        let (x0_locked, k0_locked, sigma_locked, barrier_locked) = (
            self.is_challenge_locked(),
            self.locks_k0(),
            self.locks_sigma(),
            self.locks_barrier_geometry(),
        );
        let mut reset_needed = false;
        ui.add_enabled_ui(!x0_locked, |ui| {
            reset_needed |= draw_bounded_slider(
                ui,
                &mut self.x0,
                X0_MIN..=X0_MAX,
                0.05,
                wave_parameter_label("x0"),
                "Initial center x0 of the normalized Gaussian wavepacket.",
            );
        });
        ui.add_enabled_ui(!k0_locked, |ui| {
            reset_needed |= draw_bounded_slider(
                ui,
                &mut self.k0,
                K0_MIN..=K0_MAX,
                0.05,
                wave_parameter_label("k0"),
                "Initial wave number k0 setting kinetic energy E0 = k0² / 2.",
            );
        });
        ui.add_enabled_ui(!sigma_locked, |ui| {
            reset_needed |= draw_bounded_slider(
                ui,
                &mut self.sigma,
                SIGMA_MIN..=SIGMA_MAX,
                0.05,
                wave_parameter_label("sigma"),
                "Spatial spread σ of the initial Gaussian wavepacket.",
            );
        });
        if self.preset.uses_barrier_controls() {
            reset_needed |= self.draw_barrier_sliders(ui, barrier_locked);
        }
        self.draw_dt_and_energy_summary(ui);
        self.flush_pending_slider_reset(ctx, reset_needed);
    }

    fn draw_barrier_sliders(&mut self, ui: &mut egui::Ui, barrier_locked: bool) -> bool {
        let mut changed = false;
        ui.add_enabled_ui(!barrier_locked, |ui| {
            changed |= draw_bounded_slider(
                ui,
                &mut self.v0,
                V0_MIN..=V0_MAX,
                0.05,
                wave_parameter_label("v0"),
                "Potential barrier height V0 compared against kinetic energy E0.",
            );
            changed |= draw_bounded_slider(
                ui,
                &mut self.width,
                WIDTH_MIN..=WIDTH_MAX,
                0.05,
                wave_parameter_label("width"),
                "Characteristic barrier width parameter w.",
            );
        });
        changed
    }

    fn flush_pending_slider_reset(&mut self, ctx: &egui::Context, reset_needed: bool) {
        if reset_needed {
            self.enforce_challenge_locks();
            self.bump_revision();
            self.export_runner = None;
            self.export_data = None;
            self.pending_slider_reset = true;
        }
        if self.pending_slider_reset {
            if ctx.input(|i| i.pointer.primary_down()) {
                ctx.request_repaint();
            } else {
                self.pending_slider_reset = false;
                self.sync_config_and_reset(Some(ctx));
            }
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
        let fmt_pct = |v: f64| format!("{:.1}% ({v:.4})", v * 100.0);
        let (t_stat, t_wp) = self.stationary_and_wavepacket_transmission();
        crate::reflective_ui::render_copyable_metric(ui, "Stationary T(k0)", &fmt_pct(t_stat));
        crate::reflective_ui::render_copyable_metric(ui, "Wavepacket <T>_σ", &fmt_pct(t_wp));
        let Some(wd) = self.active_wave_data() else {
            ui.small("Awaiting initial wavepacket snapshot...");
            return;
        };
        crate::reflective_ui::render_copyable_metric(
            ui,
            "Transmission T(t)",
            &fmt_pct(wd.transmission),
        );
        crate::reflective_ui::render_copyable_metric(
            ui,
            "Reflection R(t)",
            &fmt_pct(wd.reflection),
        );
        ui.small(format!(
            "Barrier Interior B(t): {:.1}% ([{:.2}, {:.2}])",
            wd.barrier_prob * 100.0,
            wd.x_left,
            wd.x_right
        ));
        let scatter_text = match wd.latched_transmission {
            Some(l) if wd.quasi_bound_latched => {
                format!("{:.1}% ({l:.4}, quasi-bound pre-wall)", l * 100.0)
            }
            Some(l) if wd.wall_arrived => format!("{:.1}% ({l:.4}, pre-wall frozen)", l * 100.0),
            Some(l) => fmt_pct(l),
            None => "Pending transit...".to_string(),
        };
        crate::reflective_ui::render_copyable_metric(ui, "Latched Scatter T", &scatter_text);
        crate::reflective_ui::render_copyable_metric(
            ui,
            "Steady Lattice T",
            &fmt_pct(wd.steady_transmission),
        );
        ui.small(format!("Unitary Norm ||ψ||²: {:.6}", wd.norm_sq));
    }

    pub(super) fn draw_central_panel(&mut self, parent_ui: &mut egui::Ui) {
        let locked = self.is_challenge_locked();
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.atlas.is_visible() {
                self.render_wave_atlas_panel(ui);
                ui.separator();
            }
            let Some(wave_data) = self.active_wave_data().cloned() else {
                render_empty_plot_guidance(
                    ui,
                    "Wavepacket ready — click ▶ Play in the side panel to evolve ψ(x, t).",
                );
                return;
            };
            self.sync_wave_callouts(&wave_data);
            let (plot_resp, frame) = render_wave_plot(
                ui,
                &wave_data,
                &mut self.brush_kit,
                &mut self.callout_manager,
                locked,
            );
            if !locked {
                if let Some(brush_frame) = frame {
                    brush_frame.dispatch(ui, &plot_resp, self);
                }
            }
        });
    }

    pub(crate) fn active_wave_data(&self) -> Option<&WaveData> {
        self.export_data.as_ref().or_else(|| {
            self.worker
                .latest_snapshot()
                .and_then(|snap: &StateSnapshot| snap.structured_data.as_ref())
                .and_then(|any_data| any_data.downcast_ref::<WaveData>())
        })
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
    let prev = *value;
    let slider = AnimatedSlider::new(value, range.clone())
        .hard_bounds(range)
        .step_by(step)
        .text(label)
        .id_salt(("wave_sim_param", label));
    let resp = ui.add(slider);
    let _ = resp.clone().accessible_hover_text(hover_desc);
    resp.changed() && (*value - prev).abs() > 1.0e-9
}
