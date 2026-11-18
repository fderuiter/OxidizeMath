// @explorer_feature = "physics"
use crate::accessibility::AccessibleHoverText;
#[cfg(test)]
use crate::async_sim::SimulationRunner;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::tabs::ExplorerTab;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint};
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

#[path = "mri/runner.rs"]
mod runner;
pub use runner::*;

#[derive(serde::Serialize, serde::Deserialize)]
struct MriState {
    #[serde(default)]
    state: Option<SimWorkerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    paused: Option<bool>,
    time_scale: f64,
}

pub struct MriTab {
    simulator: BlochSimulator,
    _solver: RungeKutta4<Vector3<f64>>,
    time_scale: f64,
    current_time: f64,
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
        let initial_m = Vector3::new(1.0, 0.0, 0.0);
        let m0 = 1.0;
        let mut simulator = BlochSimulator::new(initial_m, m0);
        simulator.set_relaxation(1.0, 0.1);
        simulator.set_b_field(Vector3::new(0.0, 0.0, 1.0));
        let solver = RungeKutta4::new(&initial_m);

        let params = Arc::new(std::sync::RwLock::new(MriSharedParams {
            t1: 1.0,
            t2: 0.1,
            b_field: Vector3::new(0.0, 0.0, 1.0),
            time_scale: 1.0,
        }));

        let runner = MriRunner {
            simulator: BlochSimulator::new(initial_m, m0),
            solver: RungeKutta4::new(&initial_m),
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
        if matches!(
            self.worker.state(),
            SimWorkerState::Faulted | SimWorkerState::Terminated
        ) {
            self.retry();
        } else {
            let _ = self.worker.send_command(SimCommand::Reset);
        }
    }

    fn draw_relaxation_section(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Relaxation Parameters", |ui| {
            if ui
                .add(
                    egui::Slider::new(&mut self.simulator.t1, 0.1..=5.0)
                        .logarithmic(true)
                        .text("Longitudinal (T1) [s]"),
                )
                .changed()
            {
                self.params.write().unwrap_or_else(|e| e.into_inner()).t1 = self.simulator.t1;
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.simulator.t2, 0.01..=2.0)
                        .logarithmic(true)
                        .text("Transverse (T2) [s]"),
                )
                .changed()
            {
                self.params.write().unwrap_or_else(|e| e.into_inner()).t2 = self.simulator.t2;
            }
            if self.simulator.t2 > self.simulator.t1 {
                ui.colored_label(egui::Color32::RED, "Warning: T2 > T1 is unphysical!");
            }
        });
    }

    fn draw_magnetic_field_section(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Magnetic Fields", |ui| {
            ui.label("Main Field (B0) [T]");
            ui.horizontal(|ui| {
                ui.label("Bz");
                if ui
                    .add(egui::DragValue::new(&mut self.simulator.b_field.z).speed(0.01))
                    .changed()
                {
                    self.params
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .b_field
                        .z = self.simulator.b_field.z;
                }
            });

            ui.label("Gradients / RF (B1) [T]");
            ui.horizontal(|ui| {
                ui.label("Bx");
                if ui
                    .add(egui::DragValue::new(&mut self.simulator.b_field.x).speed(0.001))
                    .changed()
                {
                    self.params
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .b_field
                        .x = self.simulator.b_field.x;
                }
                ui.label("By");
                if ui
                    .add(egui::DragValue::new(&mut self.simulator.b_field.y).speed(0.001))
                    .changed()
                {
                    self.params
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .b_field
                        .y = self.simulator.b_field.y;
                }
            });
        });
    }

    fn draw_magnetization_readout(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label(format!("Time: {:.3} s", self.current_time));
        ui.label(format!("|M|: {:.3}", self.simulator.magnetization.norm()));
        ui.label(format!("Mx: {:.3}", self.simulator.magnetization.x));
        ui.label(format!("My: {:.3}", self.simulator.magnetization.y));
        ui.label(format!("Mz: {:.3}", self.simulator.magnetization.z));
    }

    fn draw_fault_banner(&mut self, ui: &mut egui::Ui) {
        if self.worker.state() == SimWorkerState::Faulted {
            ui.group(|ui| {
                ui.colored_label(
                    egui::Color32::RED,
                    "⚠ Worker Faulted: Simulation thread disconnected.",
                );
                if ui
                    .button("🔄 Retry")
                    .accessible_hover_text("Restart background worker")
                    .clicked()
                {
                    self.retry();
                }
            });
            ui.separator();
        }
    }

    fn draw_controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls_panel").show(ctx, |ui| {
            ui.heading("MRI Physics Control");
            ui.separator();

            self.draw_fault_banner(ui);
            self.draw_relaxation_section(ui);
            self.draw_magnetic_field_section(ui);
            ui.separator();
            ui.heading("Simulation");

            if let Some(action) = PlaybackTransport::from_worker_state(
                "mri_transport",
                "MRI Physics",
                self.worker.state(),
            )
            .reset_label("↻ Reset (90° Pulse)")
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
                    }
                }
            }

            if ui
                .add(
                    egui::Slider::new(&mut self.time_scale, 1.0..=100.0)
                        .logarithmic(true)
                        .text("Time Scale (Speed)"),
                )
                .changed()
            {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .time_scale = self.time_scale;
            }

            self.draw_magnetization_readout(ui);
        });
    }

    fn draw_plot(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Magnetization Vector Evolution");

            Plot::new("bloch_plot")
                .legend(Legend::default())
                .x_axis_label("Time (s)")
                .y_axis_label("Magnetization (M/M0)")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Mx (Transverse)", &self.mx_points[..]).color(egui::Color32::RED),
                    );
                    plot_ui.line(
                        Line::new("My (Transverse)", &self.my_points[..])
                            .color(egui::Color32::GREEN),
                    );
                    plot_ui.line(
                        Line::new("Mz (Longitudinal)", &self.mz_points[..])
                            .color(egui::Color32::BLUE),
                    );
                });

            ui.label(
                "Observe how Mz recovers to M0 (T1 relaxation) and Mx/My decay to 0 (T2 relaxation).",
            );
            ui.label("Adjust B-fields to see precession.");
        });
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
            self.params
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .time_scale = self.time_scale;
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

        self.draw_controls(ctx);
        self.draw_plot(ctx);
    }
}

// [cite:graph_parameters_rust]

#[cfg(test)]
#[path = "mri/tests.rs"]
mod tests;
