use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use crate::presets::{InteractivePreset, PresetDialogState, ReplicatorPresetConfig};
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints, VLine};
use math_explorer::applied::game_theory::evolutionary::ReplicatorDynamics;
use nalgebra::{DMatrix, DVector};

pub struct ReplicatorDynamicsTool {
    payoff_matrix: DMatrix<f64>,
    initial_population: DVector<f64>,
    trajectory: Vec<(f64, DVector<f64>)>,
    time_horizon: f64,
    dt: f64,
    strategy_names: Vec<String>,
    // New fields for playback and visualization
    playback_time: f64,
    is_playing: bool,
    playback_speed: f64,
    dialog_state: PresetDialogState,
}

impl Default for ReplicatorDynamicsTool {
    fn default() -> Self {
        // Default to Hawk-Dove: V=2, C=4.
        // H vs H: (V-C)/2 = -1
        // H vs D: V = 2
        // D vs H: 0
        // D vs D: V/2 = 1
        let payoff = DMatrix::from_row_slice(
            2,
            2,
            &[
                -1.0, 2.0, // Hawk row
                0.0, 1.0, // Dove row
            ],
        );
        let init = DVector::from_vec(vec![0.1, 0.9]); // Mostly Doves initially
        let mut tool = Self {
            payoff_matrix: payoff,
            initial_population: init,
            trajectory: Vec::new(),
            time_horizon: 20.0,
            dt: 0.05,
            strategy_names: vec!["Hawk".to_string(), "Dove".to_string()],
            playback_time: 0.0,
            is_playing: false,
            playback_speed: 1.0,
            dialog_state: PresetDialogState::default(),
        };
        tool.run_simulation();
        tool
    }
}

impl ReplicatorDynamicsTool {
    fn run_simulation(&mut self) {
        // Normalize initial population
        let sum = self.initial_population.sum();
        if sum > 0.0 {
            self.initial_population /= sum;
        } else {
            // Fallback to uniform
            let n = self.initial_population.len();
            self.initial_population = DVector::from_element(n, 1.0 / n as f64);
        }

        if let Ok(system) = ReplicatorDynamics::new(self.payoff_matrix.clone()) {
            self.trajectory =
                system.simulate(self.initial_population.clone(), self.time_horizon, self.dt);

            // Reset playback
            self.playback_time = 0.0;
            self.is_playing = false;
        }
    }

    fn load_preset_hawk_dove(&mut self) {
        self.payoff_matrix = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, 0.0, 1.0]);
        self.initial_population = DVector::from_vec(vec![0.1, 0.9]);
        self.strategy_names = vec!["Hawk".to_string(), "Dove".to_string()];
        self.time_horizon = 20.0;
        self.run_simulation();
    }

    fn load_preset_rps(&mut self) {
        self.payoff_matrix =
            DMatrix::from_row_slice(3, 3, &[0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0]);
        self.initial_population = DVector::from_vec(vec![0.33, 0.33, 0.34]);
        self.strategy_names = vec![
            "Rock".to_string(),
            "Paper".to_string(),
            "Scissors".to_string(),
        ];
        self.time_horizon = 50.0;
        self.run_simulation();
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_settings_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.separator();

        ui.label("Presets:");
        ui.horizontal(|ui| {
            if ui
                .button("🦅 Hawk-Dove")
                .accessible_hover_text("Load Hawk-Dove preset payoff matrix")
                .clicked()
            {
                self.load_preset_hawk_dove();
            }
            if ui
                .button("✂️ RPS")
                .accessible_hover_text("Load Rock-Paper-Scissors preset payoff matrix")
                .clicked()
            {
                self.load_preset_rps();
            }
        });

        ui.separator();
        ui.heading("Payoff Matrix");
        let rows = self.payoff_matrix.nrows();
        let cols = self.payoff_matrix.ncols();

        egui::Grid::new("payoff_grid").striped(true).show(ui, |ui| {
            // Header
            ui.label("");
            for j in 0..cols {
                ui.label(
                    self.strategy_names
                        .get(j)
                        .map(|s| s.as_str())
                        .unwrap_or("?"),
                );
            }
            ui.end_row();

            for i in 0..rows {
                ui.label(
                    self.strategy_names
                        .get(i)
                        .map(|s| s.as_str())
                        .unwrap_or("?"),
                );
                for j in 0..cols {
                    ui.add(egui::DragValue::new(&mut self.payoff_matrix[(i, j)]).speed(0.1));
                }
                ui.end_row();
            }
        });

        ui.separator();
        ui.heading("Initial Population");
        let mut changed = false;
        for i in 0..self.initial_population.len() {
            ui.horizontal(|ui| {
                ui.label(
                    self.strategy_names
                        .get(i)
                        .map(|s| s.as_str())
                        .unwrap_or("?"),
                );
                if ui
                    .add(egui::Slider::new(
                        &mut self.initial_population[i],
                        0.0..=1.0,
                    ))
                    .changed()
                {
                    changed = true;
                }
            });
        }

        if changed {
            // Optional: Normalize strictly?
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

        let mut dialog_state = std::mem::take(&mut self.dialog_state);
        if crate::presets::render_preset_buttons(&mut dialog_state, ui, self) {
            self.run_simulation();
        }
        self.dialog_state = dialog_state;
    }

    fn show_trajectory_plot(&self, ui: &mut egui::Ui) {
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
                        .unwrap_or(format!("Strategy {}", i));
                    plot_ui.line(Line::new(name, points));
                }

                // Add vertical line for current playback time
                plot_ui.vline(
                    VLine::new("Current Time", self.playback_time).color(egui::Color32::WHITE),
                );
            });
    }

    fn show_population_bar_chart(&self, ui: &mut egui::Ui) {
        if self.trajectory.is_empty() {
            ui.label("No data available.");
            return;
        }

        // Find state at current time
        // Clamp index to valid range
        let idx = ((self.playback_time / self.dt).round() as usize)
            .min(self.trajectory.len().saturating_sub(1));

        let (current_time, current_state) = &self.trajectory[idx];

        // Create bars
        let bars: Vec<Bar> = current_state
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                Bar::new(i as f64, val)
                    .name(
                        self.strategy_names
                            .get(i)
                            .cloned()
                            .unwrap_or(format!("Strategy {}", i)),
                    )
                    .width(0.5)
            })
            .collect();

        Plot::new("population_bar_chart")
            .show_axes([false, true]) // Show Y axis but hide X axis numbers
            .show_x(false)
            .legend(Legend::default())
            .allow_zoom(false)
            .allow_drag(false)
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new("Distribution", bars));
            });

        // Numerical readout
        ui.horizontal(|ui| {
            ui.strong(format!("Time: {:.2}", current_time));
            ui.separator();
            for (i, val) in current_state.iter().enumerate() {
                let name = self
                    .strategy_names
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                ui.label(format!("{}: {:.2}", name, val));
            }
        });
    }
}

impl InteractiveTool for ReplicatorDynamicsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Replicator Dynamics"
    }

    fn show(&mut self, ctx: &egui::Context) {
        // Animation Logic
        if self.is_playing && !self.trajectory.is_empty() {
            let dt = ctx.input(|i| i.stable_dt) as f64;
            self.playback_time += dt * self.playback_speed;

            if self.playback_time > self.time_horizon {
                self.playback_time = 0.0; // Loop
            }
            ctx.request_repaint();
        }

        egui::SidePanel::left("replicator_control_panel").show(ctx, |ui| {
            self.show_settings_panel(ui);
        });

        egui::TopBottomPanel::bottom("replicator_bottom_panel")
            .resizable(true)
            .min_height(200.0)
            .show(ctx, |ui| {
                ui.heading("Population Distribution");

                // Playback Controls
                ui.horizontal(|ui| {
                    if ui
                        .button(if self.is_playing {
                            "⏸ Pause"
                        } else {
                            "▶ Play"
                        })
                        .clicked()
                    {
                        self.is_playing = !self.is_playing;
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

        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_trajectory_plot(ui);
        });
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "ReplicatorDynamicsTool",
        domain: "game_theory",
        tags: &[],
        build: || Box::new(ReplicatorDynamicsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for ReplicatorDynamicsTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

impl InteractivePreset for ReplicatorDynamicsTool {
    type Config = ReplicatorPresetConfig;

    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn title(&self) -> String {
        "Replicator Dynamics Simulation".to_string()
    }

    fn export_config(&self) -> Self::Config {
        let rows = self.payoff_matrix.nrows();
        let cols = self.payoff_matrix.ncols();
        let mut matrix = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(self.payoff_matrix[(i, j)]);
            }
            matrix.push(row);
        }
        let initial_pop: Vec<f64> = self.initial_population.iter().copied().collect();

        ReplicatorPresetConfig {
            payoff_matrix: matrix,
            initial_population: initial_pop,
            time_horizon: self.time_horizon,
            dt: self.dt,
            strategy_names: self.strategy_names.clone(),
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        config.validate()?;

        let rows = config.payoff_matrix.len();
        let mut flat = Vec::with_capacity(rows * rows);
        for row in &config.payoff_matrix {
            flat.extend_from_slice(row);
        }

        self.payoff_matrix = DMatrix::from_row_slice(rows, rows, &flat);
        self.initial_population = DVector::from_vec(config.initial_population);
        self.time_horizon = config.time_horizon;
        self.dt = config.dt;
        self.strategy_names = config.strategy_names;
        self.run_simulation();
        Ok(())
    }
}
