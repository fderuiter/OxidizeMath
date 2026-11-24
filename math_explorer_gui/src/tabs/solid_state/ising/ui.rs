#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::onsager_critical_temp;
use super::preset::{FermiGamePreset, IsingInteractionMode};
use super::{ising_spin_color, IsingBrushMode, IsingModelTool, DEFAULT_ISING_SIZE};
use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::SimCommand;
use crate::framework::{
    draw_brush_palette_controls, PlaybackAction, PlaybackTransport, RasterBrushCanvas,
};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;

impl IsingModelTool {
    fn draw_mode_and_preset_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let locked = self.active_challenge.is_some();
        ui.label("Dynamics Kernel:");
        ui.add_enabled_ui(!locked, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .selectable_value(
                        &mut self.mode,
                        IsingInteractionMode::Ferromagnetic,
                        "🧲 Ferromagnetic",
                    )
                    .accessible_hover_text(
                        "Switch to scalar ferromagnetic/antiferromagnetic Metropolis-Hastings model",
                    )
                    .clicked()
                {
                    if self.atlas.axis_keys() == ("T", "S") {
                        let _ = self.atlas.set_axis_keys("temperature", "h_field");
                    }
                    self.sync_shared_params();
                    self.request_metric_refresh(ctx);
                }
                if ui
                    .selectable_value(
                        &mut self.mode,
                        IsingInteractionMode::FermiGame,
                        "🎲 Fermi Game",
                    )
                    .accessible_hover_text(
                        "Switch to spatial 2x2 evolutionary Fermi imitation game",
                    )
                    .clicked()
                {
                    if self.atlas.axis_keys() != ("T", "S") {
                        let _ = self.atlas.set_axis_keys("T", "S");
                    }
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
        });
    }

    fn draw_ferromagnetic_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let locked = self.active_challenge.is_some();
        ui.add_enabled_ui(!locked, |ui| {
            if ui
                .add(
                    AnimatedSlider::new(&mut self.temperature, 0.1..=10.0)
                        .hard_bounds(0.1..=10.0)
                        .text("Temperature (T) - K (relative)")
                        .id_salt("ising_temperature"),
                )
                .changed()
            {
                self.sync_shared_params();
            }
            ui.small(format!(
                "Onsager Tc(J) = 2|J|/ln(1+√2) ≈ {:.3}",
                onsager_critical_temp(self.j_coupling)
            ));

            if ui
                .add(
                    AnimatedSlider::new(&mut self.j_coupling, -2.0..=2.0)
                        .hard_bounds(-2.0..=2.0)
                        .text("Coupling Constant (J)")
                        .id_salt("ising_j_coupling"),
                )
                .changed()
            {
                self.sync_shared_params();
                self.request_metric_refresh(ctx);
            }
            ui.small("Positive: Ferromagnetic • Negative: Antiferromagnetic Checkerboard");

            if ui
                .add(
                    AnimatedSlider::new(&mut self.h_field, -2.0..=2.0)
                        .hard_bounds(-2.0..=2.0)
                        .text("External Field (h)")
                        .id_salt("ising_h_field"),
                )
                .changed()
            {
                self.sync_shared_params();
                self.request_metric_refresh(ctx);
            }
        });
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
        let locked = self.active_challenge.is_some();
        ui.add_enabled_ui(!locked, |ui| {
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
            if ui
                .add(
                    egui::Slider::new(&mut self.mutation_rate, 0.0..=0.05)
                        .text("Strategy Mutation (μ)"),
                )
                .accessible_hover_text(
                    "Spontaneous strategy exploration probability μ ∈ [0, 0.05] escaping uniform absorbing states",
                )
                .changed()
            {
                self.mutation_rate = self.mutation_rate.clamp(0.0, 0.05);
                self.sync_shared_params();
            }
            ui.small("Low T: deterministic imitation • μ > 0: unfreezes absorbed lattices");

            ui.add_space(4.0);
            ui.label("2×2 Payoff Matrix (C = +1, D = -1):");
            if self.draw_payoff_matrix_grid(ui) {
                self.game_preset = FermiGamePreset::Custom;
                self.sync_shared_params();
                self.recompute_replicator_curve();
            }
        });
    }

    fn draw_brush_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let in_challenge = self.active_challenge.is_some();
        let max_r = if in_challenge { 10 } else { 25 };
        ui.label("Spin & Boundary Brush");
        if in_challenge {
            self.brush_state.stencil = crate::framework::BrushStencil::Circle;
            self.brush_state.symmetry = crate::framework::SymmetryMode::None;
            if !matches!(
                self.brush_mode,
                IsingBrushMode::SpinUp | IsingBrushMode::SpinDown
            ) {
                self.brush_mode = IsingBrushMode::SpinUp;
            }
        }
        egui::ComboBox::from_id_salt("ising_brush_mode")
            .selected_text(self.brush_mode.label())
            .show_ui(ui, |ui| {
                for mode in IsingBrushMode::ALL {
                    if in_challenge
                        && !matches!(mode, IsingBrushMode::SpinUp | IsingBrushMode::SpinDown)
                    {
                        continue;
                    }
                    ui.selectable_value(&mut self.brush_mode, mode, mode.label());
                }
            });
        draw_brush_palette_controls(ui, "ising", &mut self.brush_state, max_r);
        self.sync_brush_fields_from_palette();
        ui.horizontal_wrapped(|ui| {
            if !in_challenge && ui.button("Clear Pinned Sites").clicked() {
                self.clear_pins(Some(ctx));
            }
            if ui
                .button("🧹 Fill -1 Defector Sea")
                .accessible_hover_text(
                    "Set unpinned 100x100 lattice sites to -1 so brushed +1 droplets test critical nucleation",
                )
                .clicked()
            {
                self.defector_sea = true;
                self.droplet_seeds.clear();
                self.sync_shared_params();
                if self.worker.send_command(SimCommand::ClearObstacles).is_ok() {
                    self.refresh_challenge_workspace();
                    self.pending_ode_resync = true;
                    ctx.request_repaint();
                }
            }
        });
    }

    fn draw_transport_and_metrics(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let canvas_focused = self
            .brush_state
            .focused_id
            .is_some_and(|id| ctx.memory(|m| m.has_focus(id)));
        if let Some(action) = PlaybackTransport::from_worker_state(
            "ising_transport",
            "Ising Model",
            self.worker.state(),
        )
        .can_reset(true)
        .allow_space_shortcut(!canvas_focused)
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
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                PlaybackAction::Reset => {
                    self.reset();
                    ctx.request_repaint();
                }
            }
        }

        ui.separator();
        if self.mode == IsingInteractionMode::Ferromagnetic && self.j_coupling < 0.0 {
            ui.label(format!(
                "Staggered Magnetization (M_stag): {:.3}",
                self.staggered_magnetization
            ));
            ui.label(format!(
                "Uniform Magnetization: {:.3}",
                self.uniform_magnetization
            ));
        } else {
            ui.label(format!("Magnetization: {:.3}", self.magnetization));
        }
        ui.label(format!("Energy: {:.3}", self.energy));
        ui.label(format!(
            "Cooperator (+1) Density: {:.1}%",
            self.cooperator_fraction() * 100.0
        ));
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "ising_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .resizable(true)
            .default_width(260.0)
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
                    let _ = self.atlas.show_toggle_button(ui);
                    self.render_ising_atlas_panel(ui, ctx);

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

    fn draw_lattice_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, side: f32) {
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
        let pinned_count = self.pin_cells.iter().filter(|&&c| c != 0).count();
        let summary = format!(
            "M = {:.3}, E = {:.1}, C = {:.1}%, {pinned_count} pinned sites",
            self.magnetization,
            self.energy,
            self.cooperator_fraction() * 100.0
        );
        let (response, stroke) = RasterBrushCanvas::new(
            "ising_lattice_canvas",
            DEFAULT_ISING_SIZE,
            DEFAULT_ISING_SIZE,
            &alt_label,
        )
        .value_summary(&summary)
        .max_canvas_size(egui::vec2(side, side))
        .show_texture(ui, &texture, &mut self.brush_state);
        let response = response.accessible_theory_hover(self);
        self.last_canvas_rect = Some(response.rect);
        self.sync_brush_fields_from_palette();
        if let Some(stroke) = stroke {
            self.apply_brush_stroke(stroke, Some(ctx));
        }

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(ising_spin_color(1, &palette), "■ +1 (C / ↑)");
            ui.colored_label(ising_spin_color(-1, &palette), "■ -1 (D / ↓)");
            ui.colored_label(ising_spin_color(2, &palette), "■ Pin +1");
            ui.colored_label(ising_spin_color(-2, &palette), "■ Pin -1");
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
                    self.draw_lattice_canvas(ui, ctx, side);
                }
                IsingInteractionMode::FermiGame if avail.x >= 520.0 => {
                    let side = (avail.x * 0.46)
                        .min((avail.y - 32.0).max(120.0))
                        .clamp(120.0, 520.0);
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            self.draw_lattice_canvas(ui, ctx, side);
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
                    self.draw_lattice_canvas(ui, ctx, side);
                    ui.separator();
                    let plot_h = (ui.available_height() - 24.0).max(140.0);
                    self.draw_replicator_overlay_plot(ui, plot_h);
                }
            }
        });
    }
}
