#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Side-panel controls for model selection, twin initial conditions, AST offsets,
//! opt-in `PlaybackTransport` scrubbing, and live Lyapunov / fractal metrics.

use super::twin_state::{AttractorSystemKind, ON_ATTRACTOR_LORENZ_STATE};
use super::{AttractorPlotter, SolverExecutionState, MAX_SIMULATION_SPEED};
use crate::accessibility::announce_status_with_priority;
use crate::app::SemanticPalette;
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::{bounded_drag_value, PlaybackAction, PlaybackTransport};
use eframe::egui;

impl AttractorPlotter {
    pub(super) fn draw_model_and_parameter_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("System:");
            let mut selected = self.system_kind;
            egui::ComboBox::from_id_salt(ui.id().with("chaos_system_kind"))
                .selected_text(selected.label())
                .show_ui(ui, |ui| {
                    for kind in AttractorSystemKind::ALL {
                        ui.selectable_value(&mut selected, kind, kind.label());
                    }
                });
            if selected != self.system_kind {
                self.switch_system_kind(selected);
            }
        });

        egui::CollapsingHeader::new("Parameters")
            .default_open(true)
            .show(ui, |ui| match self.system_kind {
                AttractorSystemKind::Lorenz => self.draw_lorenz_sliders(ui),
                AttractorSystemKind::Rossler => self.draw_rossler_sliders(ui),
                AttractorSystemKind::DoublePendulum => self.draw_pendulum_sliders(ui),
            });
    }

    fn draw_lorenz_sliders(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.system.sigma, 0.0..=50.0).text("Prandtl (σ)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.system.rho, 0.0..=100.0).text("Rayleigh (ρ)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.system.beta, 0.0..=10.0).text("Geometric (β)"))
            .changed();
        if changed {
            self.on_slider_changed();
        }
    }

    fn draw_rossler_sliders(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.extra_params.rossler_a, 0.0..=0.6).text("Spiral (a)"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.extra_params.rossler_b, 0.0..=2.0)
                    .text("Injection (b)"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.extra_params.rossler_c, 1.0..=15.0).text("Folding (c)"),
            )
            .changed();
        if changed {
            self.on_slider_changed();
        }
    }

    fn draw_pendulum_sliders(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.extra_params.pendulum_m1, 0.2..=5.0).text("Mass m₁"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.extra_params.pendulum_m2, 0.2..=5.0).text("Mass m₂"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.extra_params.pendulum_l1, 0.2..=3.0).text("Length l₁"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.extra_params.pendulum_l2, 0.2..=3.0).text("Length l₂"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.extra_params.pendulum_g, 1.0..=25.0)
                    .text("Gravity (g)"),
            )
            .changed();
        if changed {
            self.on_slider_changed();
        }
    }

    pub(super) fn draw_twin_initial_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::CollapsingHeader::new("Initial State & Chaos Twins")
            .default_open(true)
            .show(ui, |ui| {
                let labels = self.system_kind.coord_labels();
                let mut init_changed = false;
                ui.horizontal_wrapped(|ui| {
                    for (&lbl, coord) in labels.iter().zip(&mut self.initial_state) {
                        ui.label(format!("{lbl}:"));
                        init_changed |= ui
                            .add(bounded_drag_value(coord, -80.0..=80.0, 0.05))
                            .changed();
                    }
                });
                if self.system_kind == AttractorSystemKind::Lorenz
                    && ui.small_button("Snap to Attractor Wing").clicked()
                {
                    self.initial_state = ON_ATTRACTOR_LORENZ_STATE;
                    init_changed = true;
                }
                if ui
                    .checkbox(&mut self.shadow_enabled, "Evolve Perturbed Shadow Twin")
                    .changed()
                {
                    init_changed = true;
                }
                if self.shadow_enabled {
                    init_changed |= ui
                        .add(
                            egui::Slider::new(&mut self.epsilon, 1e-12..=1e-1)
                                .logarithmic(true)
                                .text("Perturbation (ε)"),
                        )
                        .changed();
                    if self.perturbation_bar.ui(ui) {
                        self.perturbation_expr = self.perturbation_bar.raw_text.clone();
                        init_changed = true;
                    }
                    self.render_eval_error_alert(ui);
                }
                if init_changed {
                    self.recompute_perturbation_and_sync();
                    self.reset();
                    ctx.request_repaint();
                }
            });
    }

    fn render_eval_error_alert(&mut self, ui: &mut egui::Ui) {
        if !self.perturbation_bar.is_valid() {
            self.last_eval_error_announced = None;
            return;
        }
        if let Some(ref eval_err) = self.perturbation_eval_error {
            if self.last_eval_error_announced.as_deref() != Some(eval_err.as_str()) {
                announce_status_with_priority(eval_err, "assertive");
                self.last_eval_error_announced = Some(eval_err.clone());
            }
            let palette = SemanticPalette::from_visuals(ui.visuals());
            ui.label(
                egui::RichText::new(format!("⚠️ {eval_err}"))
                    .color(palette.error)
                    .small(),
            );
        } else {
            self.last_eval_error_announced = None;
        }
    }

    pub(super) fn draw_simulation_and_scrubber_controls(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        worker_faulted: bool,
    ) {
        egui::CollapsingHeader::new("Simulation & Timeline")
            .default_open(true)
            .show(ui, |ui| {
                let transport_state = if self.diverged() || worker_faulted {
                    SimWorkerState::Faulted
                } else if self.worker.state().is_stepping() {
                    SimWorkerState::Stepping
                } else if self.is_paused() {
                    SimWorkerState::Ready
                } else {
                    self.worker.state()
                };

                let can_back = !self.diverged()
                    && !worker_faulted
                    && !self.worker.state().is_stepping()
                    && self.timeline.can_step_back();

                if let Some(action) = PlaybackTransport::from_worker_state(
                    "attractor_transport",
                    "Attractor Plotter",
                    transport_state,
                )
                .can_step_back(can_back)
                .timeline_scrubber(self.timeline.cursor(), self.timeline.len())
                .can_reset(true)
                .show(ui)
                {
                    self.handle_playback_action(action, ctx);
                }

                self.draw_stepping_sliders(ui);
            });
    }

    pub(super) fn handle_playback_action(&mut self, action: PlaybackAction, ctx: &egui::Context) {
        match action {
            PlaybackAction::Play => {
                let _ = self.try_resume();
            }
            PlaybackAction::Pause => {
                self.execution_state = SolverExecutionState::Paused;
                let _ = self.worker.send_command(SimCommand::Pause);
            }
            PlaybackAction::StepBack => {
                self.execution_state = SolverExecutionState::Paused;
                if let Some(frame) = self.timeline.step_back(1) {
                    self.apply_restored_frame(frame);
                }
                let _ = self.worker.send_command(SimCommand::StepBack(1));
                ctx.request_repaint();
            }
            PlaybackAction::Step => {
                self.execution_state = SolverExecutionState::Paused;
                let _ = self.worker.send_command(SimCommand::Step(1));
            }
            PlaybackAction::Scrub(frame_idx) => {
                self.seek_timeline_frame(frame_idx, ctx);
            }
            PlaybackAction::Reset => {
                self.reset();
                ctx.request_repaint();
            }
        }
    }

    fn draw_stepping_sliders(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.simulation_speed, 1..=MAX_SIMULATION_SPEED)
                    .text("Speed (steps/frame)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.dt, 0.001..=0.05).text("Time Step (dt)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.max_points, 100..=4096).text("Max Points"))
            .changed();
        if changed {
            self.on_slider_changed();
        }
    }

    pub(super) fn draw_chaos_metrics_panel(&mut self, ui: &mut egui::Ui) {
        self.ensure_lyapunov_cache();
        egui::CollapsingHeader::new("Lyapunov & Fractal Metrics")
            .default_open(true)
            .show(ui, |ui| {
                ui.label(format!(
                    "Wolf λ₁: {:.4} 1/t",
                    self.metrics_cache.wolf_lambda1
                ));
                if let Some(t_lyap) = self.metrics_cache.lyapunov_time() {
                    ui.label(format!("Lyapunov Time (1/λ₁): {t_lyap:.2} t"));
                } else {
                    ui.label("Lyapunov Time (1/λ₁): Non-chaotic (λ₁ ≤ 0)");
                }
                ui.label(format!(
                    "Windowed FTLE: {:.4} 1/t",
                    self.metrics_cache.windowed_ftle
                ));
                if let Some(curr) = self.timeline.current() {
                    ui.label(format!(
                        "Frame {} / {} (t = {:.2}, ||δx|| = {:.3e})",
                        self.timeline.cursor(),
                        self.timeline.len().saturating_sub(1),
                        curr.time,
                        curr.separation
                    ));
                }
                ui.horizontal(|ui| {
                    ui.label("Radius r:");
                    let _ = ui.add(bounded_drag_value(
                        &mut self.metrics_cache.correlation_radius,
                        0.05..=50.0,
                        0.1,
                    ));
                    if ui.button("Compute C(r)").clicked() {
                        self.compute_correlation_dimension_on_demand();
                    }
                });
                if let Some((r, c_r)) = self.metrics_cache.correlation_result {
                    ui.label(format!("Correlation Sum C({r:.2}) = {c_r:.4}"));
                }
            });
    }
}
