//! Scoped side-panel controls and side-by-side uncontrolled vs EKF + RL `egui_plot` views.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::canvas::show_phase_kick_canvas;
use super::pipeline::ClosedLoopPipeline;
use super::state::{ClosedLoopState, ControllerKind, PlantKind};
use super::ClosedLoopControlTool;
use crate::app::theme::SemanticPalette;
use crate::framework::{bounded_drag_value, PaneSidePanel, PlaybackAction, PlaybackTransport};
use eframe::egui::{self, Color32, RichText};
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints, Polygon};

/// Renders the complete `ClosedLoopControlTool` inside a root or split-pane `egui::Ui`.
pub fn show_tool_inside(tool: &mut ClosedLoopControlTool, ui: &mut egui::Ui) {
    let frame_start = crate::web_io::PlatformInstant::now();
    if tool.poll_training_worker() {
        ui.ctx().request_repaint();
    }
    if tool.playing {
        let _ = tool.step_closed_loop();
        ui.ctx().request_repaint();
    }

    PaneSidePanel::left("closed_loop_control_panel")
        .default_width(305.0)
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_side_controls(tool, ui);
            });
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        render_central_workspace(tool, ui);
    });
    tool.last_frame_ms = frame_start.elapsed().as_secs_f64() * 1.0e3;
}

fn render_side_controls(tool: &mut ClosedLoopControlTool, ui: &mut egui::Ui) {
    ui.heading("Closed-Loop EKF Control");
    render_transport_and_telemetry(tool, ui);
    ui.separator();

    let arch_changed = render_plant_and_controller_selectors(&mut tool.state, ui);
    ui.separator();
    let obs_changed = render_observer_and_budget_sliders(&mut tool.state, ui);
    ui.separator();
    let rl_changed = render_rl_hyperparameters(&mut tool.state, ui);
    ui.separator();
    render_training_and_kick_buttons(tool, ui);

    if arch_changed || rl_changed {
        tool.rebuild_simulation();
    } else if obs_changed {
        tool.pipeline.reset_live_traces_only(&tool.state);
        for _ in 0..64 {
            let _ = tool.pipeline.step_zero_alloc(&tool.state);
        }
    }
}

fn render_transport_and_telemetry(tool: &mut ClosedLoopControlTool, ui: &mut egui::Ui) {
    let canvas_focused = tool
        .focused_canvas
        .is_some_and(|id| ui.ctx().memory(|m| m.has_focus(id)));
    match PlaybackTransport::in_thread(
        "closed_loop_control_transport",
        "Closed-Loop EKF Control",
        tool.playing,
    )
    .allow_space_shortcut(!canvas_focused)
    .show(ui)
    {
        Some(PlaybackAction::Play) => tool.playing = true,
        Some(PlaybackAction::Pause) => tool.playing = false,
        Some(PlaybackAction::Step) => {
            let _ = tool.step_closed_loop();
        }
        Some(PlaybackAction::Reset) => {
            tool.rebuild_simulation();
        }
        Some(PlaybackAction::StepBack | PlaybackAction::ScrubTo(_)) | None => {}
    }

    let palette = SemanticPalette::from_visuals(ui.visuals());
    let out = tool.pipeline.last_output;
    ui.horizontal_wrapped(|ui| {
        ui.label("Plant:");
        ui.label(
            RichText::new(tool.state.plant.label())
                .strong()
                .color(palette.info),
        );
    });
    ui.label(format!(
        "EKF: {:.3} ± {:.3} | ν: {:+.3} | u: {:+.3}",
        out.ekf_primary, out.ekf_primary_std, out.innovation, out.actuation
    ));
    let rl_summary = match tool.state.controller {
        ControllerKind::GrpoPolicy => format!(
            "Episodes: {} | GRPO R̄: {:+.2} | J_clip: {:+.3}",
            tool.pipeline.episodes_trained,
            tool.pipeline.grpo.last_mean_return,
            tool.pipeline.grpo.last_surrogate_after
        ),
        ControllerKind::DiscreteQLearning => format!(
            "Episodes: {} | Q-Return: {:+.2} | Frame: {:.2} ms",
            tool.pipeline.episodes_trained,
            tool.pipeline.q_ctrl.last_episode_return,
            tool.last_frame_ms
        ),
        ControllerKind::MpcPolicy => format!(
            "MPC Iter: {} | J*: {:.2} | Dose: {:.1} mg",
            tool.pipeline.mpc_ctrl.last_stats.iterations,
            tool.pipeline.mpc_ctrl.last_stats.optimal_cost,
            tool.pipeline.mpc_ctrl.last_dose_mg
        ),
    };
    ui.label(rl_summary);
}

fn render_plant_and_controller_selectors(state: &mut ClosedLoopState, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.label(RichText::new("Plant & Controller Coupling").strong());
    egui::ComboBox::from_label("Plant")
        .selected_text(state.plant.label())
        .show_ui(ui, |ui| {
            for candidate in [
                PlantKind::LorenzChaos,
                PlantKind::SeirEpidemic,
                PlantKind::Pharmacokinetics,
            ] {
                if ui
                    .selectable_value(&mut state.plant, candidate, candidate.label())
                    .changed()
                {
                    changed = true;
                }
            }
        });
    egui::ComboBox::from_label("Controller")
        .selected_text(state.controller.label())
        .show_ui(ui, |ui| {
            for candidate in [
                ControllerKind::GrpoPolicy,
                ControllerKind::DiscreteQLearning,
                ControllerKind::MpcPolicy,
            ] {
                if ui
                    .selectable_value(&mut state.controller, candidate, candidate.label())
                    .changed()
                {
                    changed = true;
                }
            }
        });
    changed
}

fn render_observer_and_budget_sliders(state: &mut ClosedLoopState, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.label(RichText::new("EKF Sensor Noise & Actuation Budget").strong());
    changed |= ui
        .add(egui::Slider::new(&mut state.sensor_noise, 0.0..=5.0).text("Sensor Noise σ"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut state.actuation_budget, 0.0..=6.0).text("Actuation Budget"))
        .changed();
    changed |= drag_f64_row(
        ui,
        "Process Noise Q:",
        &mut state.process_noise,
        1.0e-6..=2.0,
        0.005,
    );
    changed |= drag_f64_row(
        ui,
        "Kick Impulse:",
        &mut state.kick_magnitude,
        0.05..=3.0,
        0.05,
    );
    if state.plant == PlantKind::Pharmacokinetics {
        changed |= drag_f64_row(ui, "MEC (mg/L):", &mut state.pk_mec, 0.2..=4.5, 0.05);
        let min_mtc = (state.pk_mec + 0.5).max(2.0);
        changed |= drag_f64_row(ui, "MTC (mg/L):", &mut state.pk_mtc, min_mtc..=15.0, 0.1);
    }
    changed
}

fn render_rl_hyperparameters(state: &mut ClosedLoopState, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.label(RichText::new("Policy Learning & Seed").strong());
    match state.controller {
        ControllerKind::GrpoPolicy => {
            changed |= drag_f64_row(ui, "GRPO Step η:", &mut state.grpo_lr, 0.005..=0.80, 0.01);
            changed |= drag_f64_row(ui, "Clip ε:", &mut state.grpo_clip_eps, 0.05..=0.50, 0.01);
            changed |= drag_f64_row(
                ui,
                "KL Penalty β:",
                &mut state.grpo_kl_beta,
                0.0..=0.25,
                0.005,
            );
        }
        ControllerKind::DiscreteQLearning => {
            changed |= drag_f64_row(ui, "Q Alpha α:", &mut state.q_alpha, 0.01..=0.95, 0.01);
            changed |= drag_f64_row(ui, "Discount γ:", &mut state.q_gamma, 0.50..=0.995, 0.005);
            changed |= drag_f64_row(ui, "Explore ε:", &mut state.q_epsilon, 0.0..=0.80, 0.01);
        }
        ControllerKind::MpcPolicy => {}
    }
    let mut seed_f = state.rng_seed as f64;
    if drag_f64_row(ui, "OxidizeRng Seed:", &mut seed_f, 1.0..=1_000_000.0, 1.0) {
        state.rng_seed = seed_f.round().max(1.0) as u64;
        changed = true;
    }
    changed
}

fn render_training_and_kick_buttons(tool: &mut ClosedLoopControlTool, ui: &mut egui::Ui) {
    ui.label(RichText::new("Rollout Training & Disturbances").strong());
    ui.horizontal_wrapped(|ui| {
        if ui.button("Train +8 Episodes").clicked() {
            tool.train_episodes_sync(8);
        }
        if ui.button("Train +24 Async").clicked() {
            tool.start_async_training(24);
        }
        if ui.button("Reset Untrained").clicked() {
            tool.reset_untrained_policy();
        }
    });
    ui.horizontal_wrapped(|ui| {
        if ui.button("Inject +State Kick").clicked() {
            let _ = tool.apply_kick(0.75, 0.50);
        }
        if ui.button("Inject -State Kick").clicked() {
            let _ = tool.apply_kick(-0.75, -0.50);
        }
    });
}

fn render_central_workspace(tool: &mut ClosedLoopControlTool, ui: &mut egui::Ui) {
    let total_h = ui.available_height().max(390.0);
    let canvas_h = (total_h * 0.32).clamp(145.0, 215.0);
    let canvas_out = show_phase_kick_canvas(ui, &tool.state, &tool.pipeline, canvas_h);
    tool.focused_canvas = canvas_out.focused_id;
    let mut pending_kick = canvas_out.kick_request;

    ui.add_space(4.0);
    let mid_h = (total_h * 0.40).max(145.0);
    ui.columns(2, |cols| {
        render_uncontrolled_plot(&mut cols[0], &tool.state, &tool.pipeline, mid_h);
        if let Some(plot_kick) =
            render_closed_loop_ekf_plot(&mut cols[1], &tool.state, &tool.pipeline, mid_h)
        {
            pending_kick = Some(plot_kick);
        }
    });

    ui.add_space(4.0);
    let bot_h = ui.available_height().max(105.0);
    render_actuation_and_innovation_plot(ui, &tool.state, &tool.pipeline, bot_h);

    if let Some([dx, dy]) = pending_kick {
        let _ = tool.apply_kick(dx, dy);
    }
}

fn render_uncontrolled_plot(
    ui: &mut egui::Ui,
    state: &ClosedLoopState,
    pipeline: &ClosedLoopPipeline,
    height: f32,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.label(RichText::new("Uncontrolled Open-Loop Twin").strong());
    Plot::new("closed_loop_uncontrolled_plot")
        .height((height - 20.0).max(105.0))
        .legend(Legend::default())
        .x_axis_label(state.plant.x_axis_label())
        .y_axis_label(state.plant.primary_y_label())
        .show(ui, |plot_ui| {
            draw_pk_bounds_if_active(plot_ui, state, &palette);
            plot_ui.line(
                Line::new(
                    "Uncontrolled Twin",
                    PlotPoints::from(pipeline.uncontrolled_trace.clone()),
                )
                .color(palette.error)
                .width(2.0_f32),
            );
        });
}

fn render_closed_loop_ekf_plot(
    ui: &mut egui::Ui,
    state: &ClosedLoopState,
    pipeline: &ClosedLoopPipeline,
    height: f32,
) -> Option<[f64; 2]> {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.label(
        RichText::new(format!(
            "EKF + {} Closed-Loop (±1σ Band)",
            state.controller.label()
        ))
        .strong(),
    );
    let band_fill = Color32::from_rgba_unmultiplied(
        palette.success.r(),
        palette.success.g(),
        palette.success.b(),
        46,
    );
    let mut clicked_kick = None;
    let plot_resp = Plot::new("closed_loop_ekf_controlled_plot")
        .height((height - 20.0).max(105.0))
        .legend(Legend::default())
        .x_axis_label(state.plant.x_axis_label())
        .y_axis_label(state.plant.primary_y_label())
        .show(ui, |plot_ui| {
            draw_closed_loop_traces(plot_ui, state, pipeline, &palette, band_fill);
            if let Some(coord) = plot_ui.pointer_coordinate() {
                let delta = coord.y - pipeline.last_output.controlled_primary;
                clicked_kick = Some([delta.signum() * 0.75, 0.35]);
            }
        });
    if plot_resp.response.clicked() {
        clicked_kick
    } else {
        None
    }
}

fn draw_closed_loop_traces(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    state: &ClosedLoopState,
    pipeline: &ClosedLoopPipeline,
    palette: &SemanticPalette,
    band_fill: Color32,
) {
    draw_pk_bounds_if_active(plot_ui, state, palette);
    if pipeline.ekf_band_polygon.len() >= 4 {
        plot_ui.polygon(
            Polygon::new(
                "EKF ±1σ Band",
                PlotPoints::from(pipeline.ekf_band_polygon.clone()),
            )
            .fill_color(band_fill),
        );
    }
    plot_ui.line(
        Line::new(
            "Noisy Sensor z(t)",
            PlotPoints::from(pipeline.noisy_meas_trace.clone()),
        )
        .color(palette.muted_text)
        .style(LineStyle::Dotted { spacing: 4.0 }),
    );
    plot_ui.line(
        Line::new(
            "True Controlled",
            PlotPoints::from(pipeline.controlled_trace.clone()),
        )
        .color(palette.info)
        .width(2.2_f32),
    );
    plot_ui.line(
        Line::new(
            "EKF Posterior Estimate",
            PlotPoints::from(pipeline.ekf_primary_trace.clone()),
        )
        .color(palette.success)
        .width(1.9_f32),
    );
    plot_ui.line(
        Line::new(
            state.plant.latent_label(),
            PlotPoints::from(pipeline.ekf_latent_trace.clone()),
        )
        .color(palette.warning)
        .style(LineStyle::Dashed { length: 6.0 }),
    );
}

fn render_actuation_and_innovation_plot(
    ui: &mut egui::Ui,
    state: &ClosedLoopState,
    pipeline: &ClosedLoopPipeline,
    height: f32,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    Plot::new("closed_loop_actuation_innovation_plot")
        .height(height)
        .legend(Legend::default())
        .x_axis_label(state.plant.x_axis_label())
        .y_axis_label("Actuation & Innovation")
        .show(ui, |plot_ui| {
            plot_ui.hline(HLine::new("Zero", 0.0).color(palette.graph_edge));
            plot_ui.line(
                Line::new(
                    state.plant.actuation_label(),
                    PlotPoints::from(pipeline.actuation_trace.clone()),
                )
                .color(palette.accent)
                .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(
                    "EKF Innovation ν(t)",
                    PlotPoints::from(pipeline.innovation_trace.clone()),
                )
                .color(palette.warning)
                .width(1.6_f32),
            );
        });
}

fn draw_pk_bounds_if_active(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    state: &ClosedLoopState,
    palette: &SemanticPalette,
) {
    if state.plant != PlantKind::Pharmacokinetics {
        return;
    }
    plot_ui.hline(
        HLine::new("MEC", state.pk_mec)
            .color(palette.success)
            .style(LineStyle::Dashed { length: 7.0 }),
    );
    plot_ui.hline(
        HLine::new("MTC", state.pk_mtc)
            .color(palette.error)
            .style(LineStyle::Dashed { length: 7.0 }),
    );
}

fn drag_f64_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    speed: f64,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(bounded_drag_value(value, range, speed)).changed()
    })
    .inner
}

// [cite:grpo]
