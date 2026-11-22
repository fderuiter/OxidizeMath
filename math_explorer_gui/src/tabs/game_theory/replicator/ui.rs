#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ReplicatorDynamicsTool;
use crate::accessibility::AccessibleHoverText;
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, LineStyle, Plot, PlotPoints, VLine};

impl ReplicatorDynamicsTool {
    fn show_payoff_grid(&mut self, ui: &mut egui::Ui) {
        let rows = self.payoff_matrix.nrows();
        let cols = self.payoff_matrix.ncols();
        egui::Grid::new("payoff_grid").striped(true).show(ui, |ui| {
            ui.label("");
            for j in 0..cols {
                ui.label(
                    self.strategy_names
                        .get(j)
                        .map_or("?", std::string::String::as_str),
                );
            }
            ui.end_row();

            for i in 0..rows {
                ui.label(
                    self.strategy_names
                        .get(i)
                        .map_or("?", std::string::String::as_str),
                );
                for j in 0..cols {
                    if self.symbolic_mode {
                        ui.label(format!("{:.2}", self.payoff_matrix[(i, j)]));
                    } else if ui
                        .add(egui::DragValue::new(&mut self.payoff_matrix[(i, j)]).speed(0.1))
                        .changed()
                    {
                        self.recompute_trajectory_preserving_playback();
                    }
                }
                ui.end_row();
            }
        });
    }

    fn show_symbolic_payoff_controls(&mut self, ui: &mut egui::Ui) {
        let mut formula_changed = false;
        let rows = self.symbolic_bars.len();
        for i in 0..rows {
            let row_name = self
                .strategy_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("S{i}"));
            for j in 0..self.symbolic_bars[i].len() {
                let col_name = self
                    .strategy_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("S{j}"));
                ui.label(
                    egui::RichText::new(format!("A[{row_name}, {col_name}]:"))
                        .small()
                        .strong(),
                );
                ui.push_id(("symbolic_payoff_cell", i, j), |ui| {
                    if self.symbolic_bars[i][j].ui(ui) {
                        formula_changed = true;
                    }
                });
            }
        }
        if formula_changed {
            self.sync_and_eval_symbolic_payoffs();
            self.recompute_trajectory_preserving_playback();
        }
        if let Some(ref err) = self.symbolic_error {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            ui.colored_label(palette.error, format!("⚠️ {err}"));
        }
        if self.symbolic_sliders.ui(ui) {
            self.evaluate_symbolic_matrix();
            self.recompute_trajectory_preserving_playback();
        }
    }

    fn show_preset_buttons(&mut self, ui: &mut egui::Ui) {
        ui.label("Presets:");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🦅 Hawk-Dove")
                .accessible_hover_text("Load Hawk-Dove preset payoff matrix")
                .clicked()
            {
                self.load_preset_hawk_dove();
            }
            if ui
                .button("🧬 Symbolic Hawk-Dove")
                .accessible_hover_text("Load symbolic Hawk-Dove payoff formulas with v, c sliders")
                .clicked()
            {
                self.load_preset_symbolic_hawk_dove();
            }
            if ui
                .button("🔒 Prisoner's Dilemma")
                .accessible_hover_text("Load Prisoner's Dilemma where Defection lowers mean fitness")
                .clicked()
            {
                self.load_preset_prisoners_dilemma();
            }
            if ui
                .button("🦌 Stag Hunt")
                .accessible_hover_text("Load Stag Hunt bistable coordination game")
                .clicked()
            {
                self.load_preset_stag_hunt();
            }
            if ui
                .button("✂️ RPS")
                .accessible_hover_text("Load neutral Rock-Paper-Scissors payoff matrix")
                .clicked()
            {
                self.load_preset_rps();
            }
            if ui
                .button("🌀 Spiral RPS")
                .accessible_hover_text("Load damped spiral Rock-Paper-Scissors attractor")
                .clicked()
            {
                self.load_preset_spiral_rps();
            }
        });
    }

    pub(super) fn show_settings_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.separator();
        self.show_preset_buttons(ui);

        ui.separator();
        ui.heading("Payoff Matrix");
        if ui
            .checkbox(&mut self.symbolic_mode, "Symbolic Formula Payoffs")
            .accessible_hover_text(
                "Evaluate payoff matrix cells from custom mathematical formulas and parameter sliders",
            )
            .changed()
            && self.symbolic_mode
        {
            self.sync_and_eval_symbolic_payoffs();
            self.recompute_trajectory_preserving_playback();
        }
        self.show_payoff_grid(ui);
        if self.symbolic_mode {
            ui.add_space(4.0);
            self.show_symbolic_payoff_controls(ui);
        }

        ui.separator();
        ui.heading("Initial Population");
        for i in 0..self.initial_population.len() {
            ui.horizontal(|ui| {
                ui.label(
                    self.strategy_names
                        .get(i)
                        .map_or("?", std::string::String::as_str),
                );
                ui.add(egui::Slider::new(
                    &mut self.initial_population[i],
                    0.0..=1.0,
                ));
            });
        }

        ui.separator();
        ui.heading("Simulation");
        ui.add(egui::Slider::new(&mut self.time_horizon, 10.0..=200.0).text("Duration"));
        ui.add(egui::Slider::new(&mut self.dt, 0.01..=0.5).text("Time Step"));

        if ui
            .button("▶ Run Simulation")
            .accessible_hover_text(
                "Execute the simulation with the current payoff matrix and initial population",
            )
            .clicked()
        {
            self.run_simulation();
        }
    }

    pub(super) fn show_trajectory_plot(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let text_color = ui.visuals().text_color();
        Plot::new("replicator_plot")
            .legend(Legend::default())
            .allow_drag(true)
            .allow_zoom(true)
            .show(ui, |plot_ui| {
                if self.trajectory.is_empty() {
                    return;
                }
                let n_strategies = self.trajectory[0].1.len();
                for i in 0..n_strategies {
                    let points: PlotPoints = self
                        .trajectory
                        .iter()
                        .map(|(t, state)| [*t, state[i]])
                        .collect();
                    let name = self
                        .strategy_names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("Strategy {i}"));
                    plot_ui.line(Line::new(name, points).color(palette.series_color(i)));
                }
                if !self.mean_fitness_series.is_empty() {
                    let fit_pts: PlotPoints = self.mean_fitness_series.iter().copied().collect();
                    plot_ui.line(
                        Line::new("Mean Fitness (xᵀAx)", fit_pts)
                            .color(palette.series_color(n_strategies + 1))
                            .style(LineStyle::Dashed { length: 8.0 })
                            .width(2.0_f32),
                    );
                }
                plot_ui.vline(VLine::new("Current Time", self.playback_time).color(text_color));
                self.callout_manager.render_on_plot(plot_ui);
            });
    }

    pub(super) fn show_bottom_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "replicator_bottom_panel");
        egui::TopBottomPanel::bottom(panel_id)
            .resizable(true)
            .min_height(200.0)
            .show_inside(parent_ui, |ui| {
                ui.heading("Population Distribution");
                ui.horizontal(|ui| {
                    if let Some(action) = PlaybackTransport::in_thread(
                        "replicator_transport",
                        "Replicator Dynamics",
                        self.is_playing,
                    )
                    .show(ui)
                    {
                        match action {
                            PlaybackAction::Play => self.is_playing = true,
                            PlaybackAction::Pause => self.is_playing = false,
                            PlaybackAction::Step => self.step_once(),
                            PlaybackAction::Reset => self.reset_playback(),
                        }
                    }

                    ui.add(
                        egui::Slider::new(&mut self.playback_time, 0.0..=self.time_horizon)
                            .text("Time"),
                    );
                    ui.label("Speed:");
                    ui.add(
                        egui::DragValue::new(&mut self.playback_speed)
                            .speed(0.1)
                            .range(0.1..=10.0),
                    );
                });

                self.show_population_bar_chart(ui);
            });
    }

    fn show_population_bar_chart(&self, ui: &mut egui::Ui) {
        if self.trajectory.is_empty() {
            ui.label("No data available.");
            return;
        }

        let idx = ((self.playback_time / self.dt).round() as usize)
            .min(self.trajectory.len().saturating_sub(1));
        let (current_time, current_state) = &self.trajectory[idx];

        let bars: Vec<Bar> = current_state
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                Bar::new(i as f64, val)
                    .name(
                        self.strategy_names
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("Strategy {i}")),
                    )
                    .width(0.5)
            })
            .collect();

        Plot::new("population_bar_chart")
            .show_axes([false, true])
            .show_x(false)
            .legend(Legend::default())
            .allow_zoom(false)
            .allow_drag(false)
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new("Distribution", bars));
            });

        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("Time: {current_time:.2}"));
            if let Some(&[_, fit]) = self.mean_fitness_series.get(idx) {
                ui.separator();
                ui.label(format!("xᵀAx: {fit:.3}"));
            }
            ui.separator();
            for (i, val) in current_state.iter().enumerate() {
                let name = self
                    .strategy_names
                    .get(i)
                    .map_or("?", std::string::String::as_str);
                ui.label(format!("{name}: {val:.2}"));
            }
        });
    }
}
