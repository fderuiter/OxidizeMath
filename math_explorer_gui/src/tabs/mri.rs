// @explorer_feature = "physics"
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{BackgroundWorker, SimCommand, SimulationRunner, StateSnapshot};
use crate::tabs::ExplorerTab;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

#[allow(missing_docs)]
pub struct MriSharedParams {
    pub t1: f64,
    pub t2: f64,
    pub b_field: Vector3<f64>,
    pub time_scale: f64,
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct MriSnapshot {
    pub current_time: f64,
    pub magnetization: Vector3<f64>,
    pub history: VecDeque<(f64, Vector3<f64>)>,
}

#[allow(missing_docs)]
pub struct MriRunner {
    pub simulator: BlochSimulator,
    pub solver: RungeKutta4<Vector3<f64>>,
    pub params: Arc<std::sync::RwLock<MriSharedParams>>,
    pub current_time: f64,
    pub history: VecDeque<(f64, Vector3<f64>)>,
    pub max_history: usize,
}

impl SimulationRunner for MriRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        if let SimCommand::Reset = cmd {
            self.simulator.magnetization = Vector3::new(self.simulator.m0, 0.0, 0.0);
            self.current_time = 0.0;
            self.history.clear();
        }
    }

    fn step(&mut self) {
        let (t1, t2, b_field, time_scale) = {
            let p = self.params.read().unwrap();
            (p.t1, p.t2, p.b_field, p.time_scale)
        };
        self.simulator.t1 = t1;
        self.simulator.t2 = t2;
        self.simulator.b_field = b_field;

        let dt = 0.001;
        let steps = time_scale.ceil() as usize;

        for _ in 0..steps {
            TimeStepper::step_with(&mut self.simulator, &mut self.solver, dt);
            self.current_time += dt;

            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history
                .push_back((self.current_time, self.simulator.magnetization));
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let snapshot = MriSnapshot {
            current_time: self.current_time,
            magnetization: self.simulator.magnetization,
            history: self.history.clone(),
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![
                self.current_time,
                self.simulator.magnetization.x,
                self.simulator.magnetization.y,
                self.simulator.magnetization.z,
            ],
            structured_data: Some(Box::new(snapshot)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }
}

pub struct MriTab {
    simulator: BlochSimulator,
    _solver: RungeKutta4<Vector3<f64>>,

    // Simulation Control
    paused: bool,
    time_scale: f64,
    current_time: f64,

    // Visualization
    history: VecDeque<(f64, Vector3<f64>)>,
    _max_history: usize,

    params: Arc<std::sync::RwLock<MriSharedParams>>,
    worker: BackgroundWorker<MriRunner>,
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

        let params = Arc::new(std::sync::RwLock::new(MriSharedParams {
            t1: 1.0,
            t2: 0.1,
            b_field: Vector3::new(0.0, 0.0, 1.0),
            time_scale: 1.0,
        }));

        let runner_simulator = BlochSimulator::new(initial_m, m0);
        let runner_solver = RungeKutta4::new(&initial_m);
        let runner = MriRunner {
            simulator: runner_simulator,
            solver: runner_solver,
            params: Arc::clone(&params),
            current_time: 0.0,
            history: VecDeque::new(),
            max_history: math_commons::registry::MAX_ITERATIONS,
        };
        let worker = BackgroundWorker::new(runner);

        Self {
            simulator,
            _solver: solver,
            paused: true,
            time_scale: 1.0,
            current_time: 0.0,
            history: VecDeque::new(),
            _max_history: math_commons::registry::MAX_ITERATIONS,
            params,
            worker,
        }
    }
}

impl MriTab {
    /// Resets the simulation to the initial state defined by current parameters
    fn reset(&mut self) {
        self.simulator.magnetization = Vector3::new(self.simulator.m0, 0.0, 0.0);
        self.current_time = 0.0;
        self.history.clear();
        let _ = self.worker.send_command(SimCommand::Reset);
    }
}

impl ExplorerTab for MriTab {
    fn name(&self) -> &'static str {
        "MRI Physics"
    }

    fn domain(&self) -> &'static str {
        "physics"
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
            self.params.write().unwrap().time_scale = self.time_scale;
            if self.paused {
                let _ = self.worker.send_command(SimCommand::Pause);
            } else {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Simulation Step ---
        if !self.paused {
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
            ctx.request_repaint();
        } else if self.worker.is_running() {
            let _ = self.worker.send_command(SimCommand::Pause);
        }

        if let Some(snapshot) = self.worker.update() {
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MriSnapshot>())
            {
                self.current_time = data.current_time;
                self.simulator.magnetization = data.magnetization;
                self.history = data.history.clone();
            }
        }

        // --- GUI Layout ---
        egui::SidePanel::left("controls_panel").show(ctx, |ui| {
            ui.heading("MRI Physics Control");
            ui.separator();

            ui.collapsing("Relaxation Parameters", |ui| {
                if ui.add(egui::Slider::new(&mut self.simulator.t1, 0.1..=5.0).logarithmic(true).text("Longitudinal (T1) [s]")).changed() {
                    self.params.write().unwrap().t1 = self.simulator.t1;
                }

                if ui.add(egui::Slider::new(&mut self.simulator.t2, 0.01..=2.0).logarithmic(true).text("Transverse (T2) [s]")).changed() {
                    self.params.write().unwrap().t2 = self.simulator.t2;
                }

                if self.simulator.t2 > self.simulator.t1 {
                    ui.colored_label(egui::Color32::RED, "Warning: T2 > T1 is unphysical!");
                }
            });

            ui.collapsing("Magnetic Fields", |ui| {
                ui.label("Main Field (B0) [T]");
                ui.horizontal(|ui| {
                    ui.label("Bz");
                    if ui.add(egui::DragValue::new(&mut self.simulator.b_field.z).speed(0.01)).changed() {
                        self.params.write().unwrap().b_field.z = self.simulator.b_field.z;
                    }
                });

                ui.label("Gradients / RF (B1) [T]");
                ui.horizontal(|ui| {
                    ui.label("Bx");
                    if ui.add(egui::DragValue::new(&mut self.simulator.b_field.x).speed(0.001)).changed() {
                        self.params.write().unwrap().b_field.x = self.simulator.b_field.x;
                    }
                    ui.label("By");
                    if ui.add(egui::DragValue::new(&mut self.simulator.b_field.y).speed(0.001)).changed() {
                        self.params.write().unwrap().b_field.y = self.simulator.b_field.y;
                    }
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
                    if self.paused {
                        let _ = self.worker.send_command(SimCommand::Pause);
                    } else {
                        let _ = self.worker.send_command(SimCommand::Start);
                    }
                }
                if ui
                    .button("↻ Reset (90° Pulse)")
                    .accessible_hover_text("Reset to the initial state with a 90° tip pulse")
                    .clicked()
                {
                    self.reset();
                }
            });

            if ui.add(egui::Slider::new(&mut self.time_scale, 1.0..=100.0).logarithmic(true).text("Time Scale (Speed)")).changed() {
                self.params.write().unwrap().time_scale = self.time_scale;
            }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mri_tab_initial_state() {
        let tab = MriTab::default();
        assert!(tab.paused);
        assert_eq!(tab.time_scale, 1.0);
        assert_eq!(tab.current_time, 0.0);
    }

    #[test]
    fn test_mri_worker_simulation_step() {
        let mut tab = MriTab::default();
        let _ = tab.worker.send_command(SimCommand::Start);
        let mut stepped = false;
        for _ in 0..50 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            if let Some(snap) = tab.worker.update() {
                if snap.custom_data.first().copied().unwrap_or(0.0) > 0.0 {
                    stepped = true;
                    break;
                }
            }
        }
        assert!(stepped);
    }
}

