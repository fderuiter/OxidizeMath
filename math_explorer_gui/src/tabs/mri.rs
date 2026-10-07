// @explorer_feature = "physics"
use crate::accessibility::AccessibleHoverText;
use crate::tabs::ExplorerTab;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
use nalgebra::Vector3;
use std::collections::VecDeque;

pub struct MriTab {
    simulator: BlochSimulator,
    solver: RungeKutta4<Vector3<f64>>,

    // Simulation Control
    paused: bool,
    time_scale: f64,
    current_time: f64,

    // Visualization
    history: VecDeque<(f64, Vector3<f64>)>,
    max_history: usize,
}

impl Default for MriTab {
    fn default() -> Self {
        let initial_m = Vector3::new(1.0, 0.0, 0.0); // Start tipped in x (90 degree pulse applied)
        let m0 = 1.0;
        let mut simulator = BlochSimulator::new(initial_m, m0);

        // Typical MRI values (approximate)
        // T1 ~ 1000ms, T2 ~ 100ms
        simulator.set_relaxation(1.0, 0.1);
        // B0 field along Z (main magnet) + some off-resonance or gradient
        simulator.set_b_field(Vector3::new(0.0, 0.0, 1.0));

        // Initialize solver with the initial state shape
        let solver = RungeKutta4::new(&initial_m);

        Self {
            simulator,
            solver,
            paused: true,
            time_scale: 1.0,
            current_time: 0.0,
            history: VecDeque::new(),
            max_history: math_commons::registry::MAX_ITERATIONS,
        }
    }
}

impl MriTab {
    /// Resets the simulation to the initial state defined by current parameters
    fn reset(&mut self) {
        // Reset to M0 aligned with Z, or some initial excitation state?
        // Let's reset to "Just excited 90 degrees to X"
        self.simulator.magnetization = Vector3::new(self.simulator.m0, 0.0, 0.0);
        self.current_time = 0.0;
        self.history.clear();
    }
}

impl ExplorerTab for MriTab {
    fn name(&self) -> &'static str {
        "MRI Physics"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct MriState {
            paused: bool,
            time_scale: f64,
        }

        serde_json::to_string(&MriState {
            paused: self.paused,
            time_scale: self.time_scale,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct MriState {
            paused: bool,
            time_scale: f64,
        }

        if let Ok(s) = serde_json::from_str::<MriState>(state) {
            self.paused = s.paused;
            self.time_scale = s.time_scale;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Simulation Step ---
        if !self.paused {
            // Use a fixed internal time step for numerical stability
            // But run multiple steps per frame if time_scale is high
            let dt = 0.001;
            let steps = (self.time_scale).ceil() as usize;

            for _ in 0..steps {
                // Step the physics using the RK4 solver
                TimeStepper::step_with(&mut self.simulator, &mut self.solver, dt);
                self.current_time += dt;

                // Record history (decimated if needed, but here every step for smoothness)
                if self.history.len() >= self.max_history {
                    self.history.pop_front();
                }
                self.history
                    .push_back((self.current_time, self.simulator.magnetization));
            }

            // Request a repaint to keep the animation loop running
            ctx.request_repaint();
        }

        // --- GUI Layout ---
        egui::SidePanel::left("controls_panel").show(ctx, |ui| {
            ui.heading("MRI Physics Control");
            ui.separator();

            ui.collapsing("Relaxation Parameters", |ui| {
                ui.add(egui::Slider::new(&mut self.simulator.t1, 0.1..=5.0).logarithmic(true).text("Longitudinal (T1) [s]"));

                ui.add(egui::Slider::new(&mut self.simulator.t2, 0.01..=2.0).logarithmic(true).text("Transverse (T2) [s]"));

                if self.simulator.t2 > self.simulator.t1 {
                    ui.colored_label(egui::Color32::RED, "Warning: T2 > T1 is unphysical!");
                }
            });

            ui.collapsing("Magnetic Fields", |ui| {
                ui.label("Main Field (B0) [T]");
                ui.horizontal(|ui| {
                    ui.label("Bz");
                    ui.add(egui::DragValue::new(&mut self.simulator.b_field.z).speed(0.01));
                });

                ui.label("Gradients / RF (B1) [T]");
                ui.horizontal(|ui| {
                    ui.label("Bx");
                    ui.add(egui::DragValue::new(&mut self.simulator.b_field.x).speed(0.001));
                    ui.label("By");
                    ui.add(egui::DragValue::new(&mut self.simulator.b_field.y).speed(0.001));
                });
            });

            ui.separator();

            ui.heading("Simulation");
            ui.horizontal(|ui| {
                if ui
                    .button(if self.paused { "▶ Play" } else { "⏸ Pause" })
                    .accessible_hover_text(if self.paused {
                        "Start the simulation"
                    } else {
                        "Pause the simulation"
                    })
                    .clicked()
                {
                    self.paused = !self.paused;
                }
                if ui
                    .button("↻ Reset (90° Pulse)")
                    .accessible_hover_text("Reset to the initial state with a 90° tip pulse")
                    .clicked()
                {
                    self.reset();
                }
            });

            ui.add(egui::Slider::new(&mut self.time_scale, 1.0..=100.0).logarithmic(true).text("Time Scale (Speed)"));

            ui.separator();
            ui.label(format!("Time: {:.3} s", self.current_time));
            ui.label(format!("|M|: {:.3}", self.simulator.magnetization.norm()));
            ui.label(format!("Mx: {:.3}", self.simulator.magnetization.x));
            ui.label(format!("My: {:.3}", self.simulator.magnetization.y));
            ui.label(format!("Mz: {:.3}", self.simulator.magnetization.z));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Magnetization Vector Evolution");

            let mx: Vec<[f64; 2]> = self.history.iter().map(|(t, m)| [*t, m.x]).collect();
            let my: Vec<[f64; 2]> = self.history.iter().map(|(t, m)| [*t, m.y]).collect();
            let mz: Vec<[f64; 2]> = self.history.iter().map(|(t, m)| [*t, m.z]).collect();

            Plot::new("bloch_plot")
                .legend(Legend::default())
                .x_axis_label("Time (s)")
                .y_axis_label("Magnetization (M/M0)")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Mx (Transverse)", PlotPoints::new(mx)).color(egui::Color32::RED));
                    plot_ui.line(Line::new("My (Transverse)", PlotPoints::new(my)).color(egui::Color32::GREEN));
                    plot_ui.line(Line::new("Mz (Longitudinal)", PlotPoints::new(mz)).color(egui::Color32::BLUE));
                });

            ui.label("Observe how Mz recovers to M0 (T1 relaxation) and Mx/My decay to 0 (T2 relaxation).");
            ui.label("Adjust B-fields to see precession.");
        });
    }
}

// [cite:graph_parameters_rust]
