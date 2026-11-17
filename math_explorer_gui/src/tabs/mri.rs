// @explorer_feature = "physics"
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{
    BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner, StateSnapshot,
};
use crate::tabs::ExplorerTab;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint};
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(serde::Serialize, serde::Deserialize)]
struct MriState {
    #[serde(default)]
    state: Option<SimWorkerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    paused: Option<bool>,
    time_scale: f64,
}

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
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
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
    time_scale: f64,
    current_time: f64,

    // Visualization
    history: VecDeque<(f64, Vector3<f64>)>,
    _max_history: usize,
    pub mx_points: Vec<PlotPoint>,
    pub my_points: Vec<PlotPoint>,
    pub mz_points: Vec<PlotPoint>,

    pub params: Arc<std::sync::RwLock<MriSharedParams>>,
    pub worker: BackgroundWorker<MriRunner>,
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
            time_scale: 1.0,
            current_time: 0.0,
            history: VecDeque::new(),
            _max_history: math_commons::registry::MAX_ITERATIONS,
            mx_points: Vec::new(),
            my_points: Vec::new(),
            mz_points: Vec::new(),
            params,
            worker,
        }
    }
}

impl MriTab {
    #[allow(missing_docs)]
    pub fn restart_worker(&mut self) {
        let (t1, t2, b_field) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.t1, p.t2, p.b_field)
        };
        self.simulator.t1 = t1;
        self.simulator.t2 = t2;
        self.simulator.b_field = b_field;
        let mut runner_simulator =
            BlochSimulator::new(self.simulator.magnetization, self.simulator.m0);
        runner_simulator.set_relaxation(t1, t2);
        runner_simulator.set_b_field(b_field);
        let runner = MriRunner {
            simulator: runner_simulator,
            solver: RungeKutta4::new(&self.simulator.magnetization),
            params: Arc::clone(&self.params),
            current_time: self.current_time,
            history: self.history.clone(),
            max_history: math_commons::registry::MAX_ITERATIONS,
        };
        self.worker = BackgroundWorker::new(runner);
    }

    #[allow(missing_docs)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    /// Repopulates cached plot point buffers from history while preserving allocated capacity
    pub fn update_plot_buffers(&mut self) {
        self.mx_points.clear();
        self.my_points.clear();
        self.mz_points.clear();
        self.mx_points.reserve(self.history.len());
        self.my_points.reserve(self.history.len());
        self.mz_points.reserve(self.history.len());
        for (t, m) in &self.history {
            self.mx_points.push(PlotPoint::new(*t, m.x));
            self.my_points.push(PlotPoint::new(*t, m.y));
            self.mz_points.push(PlotPoint::new(*t, m.z));
        }
    }

    /// Resets the simulation to the initial state defined by current parameters
    pub fn reset(&mut self) {
        self.simulator.magnetization = Vector3::new(self.simulator.m0, 0.0, 0.0);
        self.current_time = 0.0;
        self.history.clear();
        self.mx_points.clear();
        self.my_points.clear();
        self.mz_points.clear();
        if self.worker.state() == SimWorkerState::Faulted
            || self.worker.state() == SimWorkerState::Terminated
        {
            self.retry();
        } else {
            let _ = self.worker.send_command(SimCommand::Reset);
        }
    }
}

impl ExplorerTab for MriTab {
    fn name(&self) -> &'static str {
        "MRI Physics"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&MriState {
            state: Some(self.worker.state()),
            paused: None,
            time_scale: self.time_scale,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<MriState>(state) {
            self.time_scale = s.time_scale;
            self.params.write().unwrap_or_else(|e| e.into_inner()).time_scale = self.time_scale;
            let target_state = s.state.unwrap_or_else(|| {
                if s.paused.unwrap_or(false) {
                    SimWorkerState::Ready
                } else {
                    SimWorkerState::Running
                }
            });
            if target_state == SimWorkerState::Running {
                if !self.worker.is_running() {
                    let _ = self.worker.send_command(SimCommand::Start);
                }
            } else if self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(snapshot) = self.worker.update() {
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MriSnapshot>())
            {
                self.current_time = data.current_time;
                self.simulator.magnetization = data.magnetization;
                self.history = data.history.clone();
                self.update_plot_buffers();
            }
        }

        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        // --- GUI Layout ---
        egui::SidePanel::left("controls_panel").show(ctx, |ui| {
            ui.heading("MRI Physics Control");
            ui.separator();

            if self.worker.state() == SimWorkerState::Faulted {
                ui.group(|ui| {
                    ui.colored_label(egui::Color32::RED, "⚠ Worker Faulted: Simulation thread disconnected.");
                    if ui.button("🔄 Retry").accessible_hover_text("Restart background worker").clicked() {
                        self.retry();
                    }
                });
                ui.separator();
            }

            ui.collapsing("Relaxation Parameters", |ui| {
                if ui.add(egui::Slider::new(&mut self.simulator.t1, 0.1..=5.0).logarithmic(true).text("Longitudinal (T1) [s]")).changed() {
                    self.params.write().unwrap_or_else(|e| e.into_inner()).t1 = self.simulator.t1;
                }

                if ui.add(egui::Slider::new(&mut self.simulator.t2, 0.01..=2.0).logarithmic(true).text("Transverse (T2) [s]")).changed() {
                    self.params.write().unwrap_or_else(|e| e.into_inner()).t2 = self.simulator.t2;
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
                        self.params.write().unwrap_or_else(|e| e.into_inner()).b_field.z = self.simulator.b_field.z;
                    }
                });

                ui.label("Gradients / RF (B1) [T]");
                ui.horizontal(|ui| {
                    ui.label("Bx");
                    if ui.add(egui::DragValue::new(&mut self.simulator.b_field.x).speed(0.001)).changed() {
                        self.params.write().unwrap_or_else(|e| e.into_inner()).b_field.x = self.simulator.b_field.x;
                    }
                    ui.label("By");
                    if ui.add(egui::DragValue::new(&mut self.simulator.b_field.y).speed(0.001)).changed() {
                        self.params.write().unwrap_or_else(|e| e.into_inner()).b_field.y = self.simulator.b_field.y;
                    }
                });
            });

            ui.separator();

            ui.heading("Simulation");
            ui.horizontal(|ui| {
                let is_running = self.worker.state().is_running();
                let can_toggle = matches!(
                    self.worker.state(),
                    SimWorkerState::Ready | SimWorkerState::Running
                );
                if ui
                    .add_enabled(
                        can_toggle,
                        egui::Button::new(if is_running { "⏸ Pause" } else { "▶ Play" }),
                    )
                    .accessible_hover_text(if is_running {
                        "Pause the simulation"
                    } else {
                        "Start the simulation"
                    })
                    .clicked()
                {
                    if is_running {
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
                self.params.write().unwrap_or_else(|e| e.into_inner()).time_scale = self.time_scale;
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

            Plot::new("bloch_plot")
                .legend(Legend::default())
                .x_axis_label("Time (s)")
                .y_axis_label("Magnetization (M/M0)")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Mx (Transverse)", &self.mx_points[..]).color(egui::Color32::RED));
                    plot_ui.line(Line::new("My (Transverse)", &self.my_points[..]).color(egui::Color32::GREEN));
                    plot_ui.line(Line::new("Mz (Longitudinal)", &self.mz_points[..]).color(egui::Color32::BLUE));
                });

            ui.label("Observe how Mz recovers to M0 (T1 relaxation) and Mx/My decay to 0 (T2 relaxation).");
            ui.label("Adjust B-fields to see precession.");
        });
    }
}

// [cite:graph_parameters_rust]

#[cfg(test)]
#[path = "mri/tests.rs"]
mod tests;

