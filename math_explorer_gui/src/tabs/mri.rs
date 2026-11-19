// @explorer_feature = "physics"
use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::tabs::ExplorerTab;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoint, Points};
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

        let runner = MriRunner::new(Arc::clone(&params), initial_m, m0);
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
    #[cfg(test)]
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
        let mut history = self.history.clone();
        if history.is_empty() {
            history.push_back((self.current_time, self.simulator.magnetization));
        }
        let runner = MriRunner {
            simulator: runner_simulator,
            solver: RungeKutta4::new(&self.simulator.magnetization),
            params: Arc::clone(&self.params),
            current_time: self.current_time,
            history,
            max_history: math_commons::registry::MAX_ITERATIONS,
        };
        self.worker = BackgroundWorker::new(runner);
    }

    #[cfg(test)]
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
        let initial_m = Vector3::new(self.simulator.m0, 0.0, 0.0);
        self.simulator.magnetization = initial_m;
        self.current_time = 0.0;
        self.history.clear();
        self.mx_points.clear();
        self.my_points.clear();
        self.mz_points.clear();
        let params = Arc::clone(&self.params);
        let m0 = self.simulator.m0;
        self.worker
            .reset_or_respawn(|| MriRunner::new(params, initial_m, m0));
    }

    fn draw_relaxation_controls(&mut self, ui: &mut egui::Ui) {
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
                let err = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
                ui.colored_label(err, "Warning: T2 > T1 is unphysical!");
            }
        });
    }

    fn draw_magnetic_field_controls(&mut self, ui: &mut egui::Ui) {
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

    fn draw_playback_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
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
                let cmd = if is_running {
                    SimCommand::Pause
                } else {
                    SimCommand::Start
                };
                let _ = self.worker.send_command(cmd);
            }
            if ui
                .button("↻ Reset (90° Pulse)")
                .accessible_hover_text("Reset to the initial state with a 90° tip pulse")
                .clicked()
            {
                self.reset();
                ctx.request_repaint();
            }
        });

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
    }

    fn draw_side_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls_panel").show(ctx, |ui| {
            ui.heading("MRI Physics Control");
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

            self.draw_relaxation_controls(ui);
            self.draw_magnetic_field_controls(ui);
            ui.separator();
            self.draw_playback_controls(ui, ctx);

            ui.separator();
            ui.label(format!("Time: {:.3} s", self.current_time));
            ui.label(format!("|M|: {:.3}", self.simulator.magnetization.norm()));
            ui.label(format!("Mx: {:.3}", self.simulator.magnetization.x));
            ui.label(format!("My: {:.3}", self.simulator.magnetization.y));
            ui.label(format!("Mz: {:.3}", self.simulator.magnetization.z));
        });
    }

    fn draw_central_panel(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Magnetization Vector Evolution");

            if self.mx_points.len() <= 1 {
                render_empty_plot_guidance(
                    ui,
                    "Initial 90° tipped magnetization M = (1, 0, 0) ready — click ▶ Play to simulate Bloch relaxation.",
                );
            }

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            Plot::new("bloch_plot")
                .legend(Legend::default())
                .x_axis_label("Time (s)")
                .y_axis_label("Magnetization (M/M0)")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("Mx (Transverse)", &self.mx_points[..]).color(palette.axis_x),
                    );
                    plot_ui.line(
                        Line::new("My (Transverse)", &self.my_points[..]).color(palette.axis_y),
                    );
                    plot_ui.line(
                        Line::new("Mz (Longitudinal)", &self.mz_points[..]).color(palette.axis_z),
                    );
                    if self.mx_points.len() == 1 {
                        let mx0 = [self.mx_points[0].x, self.mx_points[0].y];
                        let my0 = [self.my_points[0].x, self.my_points[0].y];
                        let mz0 = [self.mz_points[0].x, self.mz_points[0].y];
                        plot_ui.points(
                            Points::new("Mx (t=0)", vec![mx0])
                                .radius(4.5_f32)
                                .color(palette.axis_x),
                        );
                        plot_ui.points(
                            Points::new("My (t=0)", vec![my0])
                                .radius(4.5_f32)
                                .color(palette.axis_y),
                        );
                        plot_ui.points(
                            Points::new("Mz (t=0)", vec![mz0])
                                .radius(4.5_f32)
                                .color(palette.axis_z),
                        );
                    }
                });

            ui.label(
                "Observe how Mz recovers to M0 (T1 relaxation) and Mx/My decay to 0 (T2 relaxation).",
            );
            ui.label("Adjust B-fields to see precession.");
        });
    }

    /// Updates from the background worker and renders the MRI tab UI.
    pub fn show_ui(&mut self, ctx: &egui::Context) {
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

        self.draw_side_panel(ctx);
        self.draw_central_panel(ctx);
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
        self.show_ui(ctx);
    }
}

// [cite:graph_parameters_rust]

#[cfg(test)]
use crate::async_sim::SimulationRunner;
#[cfg(test)]
#[path = "mri/tests.rs"]
mod tests;
