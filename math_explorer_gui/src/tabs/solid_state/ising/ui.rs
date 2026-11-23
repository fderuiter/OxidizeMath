#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::{FermiGamePreset, IsingInteractionMode};
use super::{ising_spin_color, IsingModelTool};
use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::SimCommand;
use crate::framework::{
    InputMode, InteractionContext, InteractiveTool, PlaybackAction, PlaybackTransport,
};
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints};

impl IsingModelTool {
    fn draw_mode_and_preset_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Dynamics Kernel:");
        ui.horizontal(|ui| {
            if ui
                .selectable_value(
                    &mut self.mode,
                    IsingInteractionMode::Ferromagnetic,
                    "🧲 Ferromagnetic",
                )
                .accessible_hover_text("Switch to scalar ferromagnetic Metropolis-Hastings model")
                .clicked()
            {
                self.sync_shared_params();
                self.request_metric_refresh(ctx);
            }
            if ui
                .selectable_value(
                    &mut self.mode,
                    IsingInteractionMode::FermiGame,
                    "🎲 Fermi Game",
                )
                .accessible_hover_text("Switch to spatial 2x2 evolutionary Fermi imitation game")
                .clicked()
            {
                self.apply_game_preset(self.game_preset, ctx);
            }
        });

        if self.mode == IsingInteractionMode::FermiGame {
            ui.add_space(4.0);
            ui.label("Evolutionary Game Presets:");
            ui.horizontal_wrapped(|ui| {
                for preset in [
                    FermiGamePreset::PrisonersDilemma,
                    FermiGamePreset::StagHunt,
                    FermiGamePreset::HawkDove,
                ] {
                    let selected = self.game_preset == preset;
                    if ui
                        .selectable_label(selected, preset.label())
                        .accessible_hover_text(format!("Load {} payoff preset", preset.label()))
                        .clicked()
                    {
                        self.apply_game_preset(preset, ctx);
                    }
                }
            });
        }
    }

    fn draw_ferromagnetic_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if ui
            .add(
                egui::Slider::new(&mut self.temperature, 0.1..=10.0)
                    .text("Temperature (T) - K (relative)"),
            )
            .changed()
        {
            self.sync_shared_params();
        }
        ui.small("Critical Temp Tc ≈ 2.269 (for J=1)");

        if ui
            .add(
                egui::Slider::new(&mut self.j_coupling, -2.0..=2.0)
                    .text("Coupling Constant (J) - J"),
            )
            .changed()
        {
            self.sync_shared_params();
            self.request_metric_refresh(ctx);
        }
        ui.small("Positive: Ferromagnetic\nNegative: Antiferromagnetic");

        if ui
            .add(egui::Slider::new(&mut self.h_field, -2.0..=2.0).text("External Field (H) - H"))
            .changed()
        {
            self.sync_shared_params();
            self.request_metric_refresh(ctx);
        }
    }

    fn draw_payoff_matrix_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("ising_fermi_payoff_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.label("");
                ui.label("vs C (+1)");
                ui.label("vs D (-1)");
                ui.end_row();

                ui.label("C (+1)");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.game_payoffs[0])
                            .prefix("R: ")
                            .speed(0.05)
                            .range(-20.0..=20.0),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.game_payoffs[1])
                            .prefix("S: ")
                            .speed(0.05)
                            .range(-20.0..=20.0),
                    )
                    .changed();
                ui.end_row();

                ui.label("D (-1)");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.game_payoffs[2])
                            .prefix("T: ")
                            .speed(0.05)
                            .range(-20.0..=20.0),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.game_payoffs[3])
                            .prefix("P: ")
                            .speed(0.05)
                            .range(-20.0..=20.0),
                    )
                    .changed();
                ui.end_row();
            });
        changed
    }

    fn draw_fermi_game_controls(&mut self, ui: &mut egui::Ui) {
        if ui
            .add(
                egui::Slider::new(&mut self.temperature, 1e-3..=2.5)
                    .logarithmic(true)
                    .text("Rationality Noise (T)"),
            )
            .changed()
        {
            self.temperature = self.temperature.max(1e-3);
            self.sync_shared_params();
        }
        ui.small("Low T: deterministic imitation • High T: thermal drift");

        ui.add_space(4.0);
        ui.label("2×2 Payoff Matrix (C = +1, D = -1):");
        if self.draw_payoff_matrix_grid(ui) {
            self.game_preset = FermiGamePreset::Custom;
            self.sync_shared_params();
            self.recompute_replicator_curve();
        }
    }

    fn draw_brush_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Droplet Brush:");
        ui.add(egui::Slider::new(&mut self.brush_radius, 0..=25).text("Radius (r)"));
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.brush_cooperator, true, "+1 (C / ↑)");
            ui.selectable_value(&mut self.brush_cooperator, false, "-1 (D / ↓)");
        });
        if ui
            .button("🧹 Fill -1 Defector Sea")
            .accessible_hover_text(
                "Set all 100x100 lattice sites to -1 so brushed +1 droplets test critical nucleation",
            )
            .clicked()
            && self.worker.send_command(SimCommand::ClearObstacles).is_ok()
        {
            self.pending_ode_resync = true;
            ctx.request_repaint();
        }
    }

    fn draw_transport_and_metrics(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) = PlaybackTransport::from_worker_state(
            "ising_transport",
            "Ising Model",
            self.worker.state(),
        )
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
                    self.reset();
                    ctx.request_repaint();
                }
                PlaybackAction::StepBack | PlaybackAction::Scrub(_) => {}
            }
        }

        ui.separator();
        ui.label(format!("Magnetization: {:.3}", self.magnetization));
        ui.label(format!("Energy: {:.3}", self.energy));
        ui.label(format!(
            "Cooperator (+1) Density: {:.1}%",
            self.cooperator_fraction() * 100.0
        ));
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "ising_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .resizable(false)
            .default_width(245.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Ising Model Controls");
                    ui.separator();

                    if let Some(WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters) =
                        render_worker_fault_banner(
                            ui,
                            self.worker.state(),
                            self.worker.diagnostic_report(),
                            false,
                        )
                    {
                        self.reset();
                        ctx.request_repaint();
                        ui.separator();
                    }

                    self.draw_mode_and_preset_controls(ui, ctx);
                    ui.separator();

                    match self.mode {
                        IsingInteractionMode::Ferromagnetic => {
                            self.draw_ferromagnetic_controls(ui, ctx);
                        }
                        IsingInteractionMode::FermiGame => {
                            self.draw_fermi_game_controls(ui);
                        }
                    }

                    ui.separator();
                    self.draw_brush_controls(ui, ctx);

                    ui.separator();
                    if ui
                        .add(
                            egui::Slider::new(&mut self.steps_per_frame, 100..=50000)
                                .text("Simulation Speed (Steps/Frame)")
                                .logarithmic(true),
                        )
                        .changed()
                    {
                        self.sync_shared_params();
                    }

                    ui.separator();
                    self.draw_transport_and_metrics(ui, ctx);
                });
            });
    }

    fn dispatch_canvas_brush(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let active_stroke = response.dragged()
            || response.clicked()
            || response.secondary_clicked()
            || response.is_pointer_button_down_on();
        if !active_stroke {
            return;
        }
        let pointer_pos = response
            .interact_pointer_pos()
            .or_else(|| ui.input(|i| i.pointer.interact_pos()))
            .filter(|pos| response.rect.contains(*pos) || response.dragged());
        if pointer_pos.is_none() {
            return;
        }
        let input_mode = ui.ctx().data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        let interaction_ctx = InteractionContext {
            pointer_pos,
            delta: response.drag_delta(),
            is_dragging: response.dragged(),
            is_clicked: response.clicked(),
            response,
            input_mode,
            multi_touch: ui.input(|i| i.multi_touch()),
            keys_down: ui.input(|i| i.keys_down.clone()),
            modifiers: ui.input(|i| i.modifiers),
        };
        self.on_brush(&interaction_ctx);
    }

    fn draw_lattice_canvas(&mut self, ui: &mut egui::Ui, side: f32) {
        let Some(texture) = self.texture.clone() else {
            return;
        };
        let alt_label = format!(
            "2D Ising Model spin lattice visualization (T = {:.3}, M = {:.3}, E = {:.3}, C = {:.1}%)",
            self.temperature,
            self.magnetization,
            self.energy,
            self.cooperator_fraction() * 100.0
        );
        let response = ui
            .add(
                egui::Image::new(&texture)
                    .fit_to_exact_size(egui::vec2(side, side))
                    .alt_text(alt_label)
                    .sense(egui::Sense::click_and_drag()),
            )
            .accessible_theory_hover(self);
        ui.painter().rect_stroke(
            response.rect,
            0.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        self.last_canvas_rect = Some(response.rect);
        self.dispatch_canvas_brush(ui, &response);

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal(|ui| {
            ui.colored_label(ising_spin_color(1, &palette), "■ +1 (C / ↑)");
            ui.colored_label(ising_spin_color(-1, &palette), "■ -1 (D / ↓)");
            ui.small("Click/drag to paint droplets (Shift/Right-click: invert)");
        });
    }

    fn draw_replicator_overlay_plot(&self, ui: &mut egui::Ui, height: f32) {
        ui.strong(format!(
            "Spatial Reciprocity (100×100 Fermi) vs 0D Replicator ODE — {}",
            self.game_preset.label()
        ));
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let spatial_pts: PlotPoints = self
            .spatial_history
            .iter()
            .copied()
            .collect::<Vec<[f64; 2]>>()
            .into();
        let ode_pts: PlotPoints = self.replicator_curve.clone().into();

        Plot::new("ising_fermi_replicator_overlay")
            .height(height.max(140.0))
            .legend(Legend::default())
            .include_y(0.0)
            .include_y(1.0)
            .x_axis_label("Monte Carlo Sweeps / Replicator Time t")
            .y_axis_label("Cooperator Fraction x_C")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("2D Spatial Lattice (100×100 Fermi)", spatial_pts)
                        .color(palette.axis_x)
                        .width(2.5_f32),
                );
                plot_ui.line(
                    Line::new("0D Replicator Dynamics (Well-Mixed ODE)", ode_pts)
                        .color(palette.axis_z)
                        .style(LineStyle::Dashed { length: 7.0_f32 })
                        .width(2.0_f32),
                );
            });
    }

    pub(super) fn draw_central_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.sync_lattice_texture(ctx);
            if self.texture.is_none() {
                render_empty_plot_guidance(
                    ui,
                    "Spin lattice ready — click ▶ Play or ⏭ Step to run spin flips.",
                );
                return;
            }

            let avail = ui.available_size();
            match self.mode {
                IsingInteractionMode::Ferromagnetic => {
                    let side = avail.x.min((avail.y - 28.0).max(100.0)).clamp(100.0, 620.0);
                    self.draw_lattice_canvas(ui, side);
                }
                IsingInteractionMode::FermiGame if avail.x >= 520.0 => {
                    let side = (avail.x * 0.46)
                        .min((avail.y - 32.0).max(120.0))
                        .clamp(120.0, 520.0);
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            self.draw_lattice_canvas(ui, side);
                        });
                        ui.separator();
                        ui.vertical(|ui| {
                            let plot_h = (avail.y - 32.0).max(160.0);
                            self.draw_replicator_overlay_plot(ui, plot_h);
                        });
                    });
                }
                IsingInteractionMode::FermiGame => {
                    let side = avail.x.min((avail.y * 0.5).max(100.0)).clamp(100.0, 360.0);
                    self.draw_lattice_canvas(ui, side);
                    ui.separator();
                    let plot_h = (ui.available_height() - 24.0).max(140.0);
                    self.draw_replicator_overlay_plot(ui, plot_h);
                }
            }
        });
    }
}
