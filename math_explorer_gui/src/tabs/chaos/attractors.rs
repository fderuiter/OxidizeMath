#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{BackgroundWorker, SimCommand};
use crate::framework::{
    Camera3D, CoupledExpressionBank, ExportCapabilities, InteractiveTool, Scene3DBuilder,
};
use eframe::egui;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use nalgebra::{Point3, Vector3};
use std::collections::VecDeque;
use std::sync::Arc;

#[path = "attractors/flow.rs"]
pub mod flow;
pub use flow::{default_flow_bank, AttractorMode, CompiledFlowJob, DEFAULT_FLOW_EXPRS};

#[path = "attractors/invariants.rs"]
mod invariants;

#[path = "attractors/runner.rs"]
mod runner;
#[path = "attractors/serde_vec3.rs"]
mod serde_vec3;
pub use runner::*;

#[path = "attractors/regime_map.rs"]
mod regime_map;
pub use regime_map::*;

#[path = "attractors/regime_ui.rs"]
mod regime_ui;

#[path = "attractors/ui.rs"]
mod ui;

#[path = "attractors/export.rs"]
pub mod export;

pub struct AttractorPlotter {
    pub(crate) mode: AttractorMode,
    pub(crate) flow_bank: CoupledExpressionBank,
    pub(crate) system: LorenzSystem,
    pub(crate) execution_state: SolverExecutionState,
    pub(crate) step_count: usize,
    pub(crate) simulation_speed: usize,
    pub(crate) dt: f64,
    pub(crate) tube_radius: f64,
    pub(crate) history: VecDeque<Vector3<f64>>,
    pub(crate) max_points: usize,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    pub(crate) trajectory_scratch: Vec<Point3<f64>>,
    pub(crate) params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    pub(crate) worker: BackgroundWorker<AttractorRunner>,
    pub(crate) latest_lyapunov: Option<f64>,
    pub(crate) latest_correlation_sum: Option<f64>,
    pub(crate) regime_atlas: LorenzRegimeAtlasState,
    pub show_annotations: bool,
    pub(crate) fixed_callout_offsets: [egui::Vec2; 2],
    pub(crate) dragged_callout: Option<usize>,
    pub(crate) invariants_cache: invariants::CachedAttractorInvariants,
}

impl Default for AttractorPlotter {
    fn default() -> Self {
        let initial_state = LorenzState::new(1.0, 1.0, 1.0);
        let system = LorenzBuilder::new().build(initial_state);
        let flow_bank = default_flow_bank();
        let custom_flow = Some(Arc::new(CompiledFlowJob::from_bank(&flow_bank)));
        let params = Arc::new(std::sync::RwLock::new(AttractorSharedParams {
            sigma: system.sigma,
            rho: system.rho,
            beta: system.beta,
            dt: 0.01,
            simulation_speed: 5,
            max_points: 2000,
            mode: AttractorMode::Lorenz,
            custom_flow,
        }));
        let runner = AttractorRunner::from_params(Arc::clone(&params));
        let worker = BackgroundWorker::new(runner);
        let empty_hist = VecDeque::new();
        let (initial_lyap, _) = compute_cached_attractor_metrics(&system, 0.01, &empty_hist);
        let initial_beta = system.beta;

        let camera = Camera3D::new(0.0, 0.0, 1.0).with_perspective_distance(75.0);
        Self {
            mode: AttractorMode::Lorenz,
            flow_bank,
            system,
            execution_state: SolverExecutionState::Ready,
            step_count: 0,
            simulation_speed: 5,
            dt: 0.01,
            tube_radius: export::DEFAULT_TUBE_RADIUS,
            history: VecDeque::with_capacity(2000),
            max_points: 2000,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            trajectory_scratch: Vec::with_capacity(2000),
            params,
            worker,
            latest_lyapunov: initial_lyap,
            latest_correlation_sum: None,
            regime_atlas: LorenzRegimeAtlasState::new(initial_beta),
            show_annotations: true,
            fixed_callout_offsets: [egui::Vec2::new(35.0, -25.0), egui::Vec2::new(-35.0, 25.0)],
            dragged_callout: None,
            invariants_cache: invariants::CachedAttractorInvariants::default(),
        }
    }
}

impl AttractorPlotter {
    #[must_use]
    pub fn diverged(&self) -> bool {
        matches!(self.execution_state, SolverExecutionState::Diverged { .. })
    }

    #[must_use]
    pub fn is_paused(&self) -> bool {
        matches!(
            self.execution_state,
            SolverExecutionState::Paused | SolverExecutionState::Diverged { .. }
        )
    }

    #[must_use]
    pub fn is_finite_state(&self) -> bool {
        let v = self.system.state.vec;
        let lorenz_ok = self.mode != AttractorMode::Lorenz
            || (self.system.sigma.is_finite()
                && self.system.rho.is_finite()
                && self.system.beta.is_finite());
        v.x.is_finite()
            && v.y.is_finite()
            && v.z.is_finite()
            && lorenz_ok
            && self.dt.is_finite()
            && self.tube_radius.is_finite()
            && self.flow_bank.param_slice().iter().all(|x| x.is_finite())
    }

    pub fn try_resume(&mut self) -> Result<(), &'static str> {
        let v = self.system.state.vec;
        if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
            return Err("Cannot resume simulation: state coordinates are non-finite.");
        }
        if !self.is_finite_state() {
            return Err("Cannot resume simulation: system parameters are non-finite.");
        }
        self.execution_state = SolverExecutionState::Running;
        let _ = self.worker.send_command(SimCommand::Start);
        Ok(())
    }

    pub fn on_slider_changed(&mut self) {
        let job = Arc::new(CompiledFlowJob::from_bank(&self.flow_bank));
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.sigma = self.system.sigma;
            p.rho = self.system.rho;
            p.beta = self.system.beta;
            p.dt = self.dt;
            p.simulation_speed = self.simulation_speed;
            p.max_points = self.max_points;
            p.mode = self.mode;
            p.custom_flow = Some(job);
        }
        if self.diverged() && self.is_finite_state() {
            self.execution_state = SolverExecutionState::Ready;
        }
        self.refresh_local_metrics();
    }

    fn refresh_local_metrics(&mut self) {
        if self.diverged() || !self.is_finite_state() {
            self.latest_lyapunov = None;
            self.latest_correlation_sum = None;
        } else if self.mode == AttractorMode::Custom {
            self.latest_lyapunov = None;
            self.latest_correlation_sum = compute_cached_correlation_only(&self.history);
        } else {
            let (lyap, corr) =
                compute_cached_attractor_metrics(&self.system, self.dt, &self.history);
            self.latest_lyapunov = lyap;
            self.latest_correlation_sum = corr;
        }
    }

    pub fn reset(&mut self) {
        let initial_state = LorenzState::new(1.0, 1.0, 1.0);
        let sigma = if self.system.sigma.is_finite() {
            self.system.sigma
        } else {
            10.0
        };
        let rho = if self.system.rho.is_finite() {
            self.system.rho
        } else {
            28.0
        };
        let beta = if self.system.beta.is_finite() {
            self.system.beta
        } else {
            8.0 / 3.0
        };
        if !self.dt.is_finite() || self.dt <= 0.0 {
            self.dt = 0.01;
        }

        self.system = LorenzBuilder::new()
            .sigma(sigma)
            .rho(rho)
            .beta(beta)
            .build(initial_state);
        self.history.clear();
        self.step_count = 0;
        self.execution_state = SolverExecutionState::Ready;
        let job = Arc::new(CompiledFlowJob::from_bank(&self.flow_bank));
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.sigma = sigma;
            p.rho = rho;
            p.beta = beta;
            p.dt = self.dt;
            p.simulation_speed = self.simulation_speed;
            p.max_points = self.max_points;
            p.mode = self.mode;
            p.custom_flow = Some(job);
        }
        let params = Arc::clone(&self.params);
        self.worker
            .reset_or_respawn(|| AttractorRunner::from_params(params));
        self.refresh_local_metrics();
    }

    pub fn reset_parameters(&mut self) {
        self.system.sigma = 10.0;
        self.system.rho = 28.0;
        self.system.beta = 8.0 / 3.0;
        self.flow_bank = default_flow_bank();
        self.dt = 0.01;
        self.tube_radius = export::DEFAULT_TUBE_RADIUS;
        self.reset();
    }

    /// Projects 3D point to 2D screen space based on rotation, pan, zoom, and projection mode.
    pub(crate) fn project(&self, p: Vector3<f64>) -> [f64; 2] {
        let center_offset = match self.mode {
            AttractorMode::Lorenz => Vector3::new(0.0, 0.0, 25.0),
            AttractorMode::Custom => Vector3::zeros(),
        };
        let p_centered = p - center_offset;
        self.camera
            .project(&[p_centered.x, p_centered.y, p_centered.z])
    }

    fn sync_worker_snapshot(&mut self) {
        let current_vec = self.system.state.vec;
        if (!current_vec.x.is_finite() || !current_vec.y.is_finite() || !current_vec.z.is_finite())
            && !self.diverged()
        {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: current_vec,
                diverged_at_step: self.step_count,
            };
        }
        let ui_diverged = self.diverged();
        let was_paused = self.is_paused();
        if let Some(snapshot) = self.worker.update() {
            if !ui_diverged {
                if let Some(data) = snapshot
                    .structured_data
                    .as_ref()
                    .and_then(|d| d.downcast_ref::<AttractorSnapshot>())
                {
                    self.history = data.history.clone();
                    self.system.state.vec = data.current_vec;
                    self.step_count = data.step_count;
                    self.latest_lyapunov = data.lyapunov_exponent;
                    self.latest_correlation_sum = data.correlation_sum;
                    if matches!(data.execution_state, SolverExecutionState::Diverged { .. })
                        || !was_paused
                    {
                        self.execution_state = data.execution_state;
                    }
                }
            }
        }
        if self.diverged() {
            self.latest_lyapunov = None;
            self.latest_correlation_sum = None;
        }
    }
}

impl InteractiveTool for AttractorPlotter {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Attractor Plotter"
    }
    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.mode as u8).hash(&mut h);
        for bar in self.flow_bank.bars() {
            bar.raw_text.hash(&mut h);
        }
        for &pv in self.flow_bank.param_slice() {
            pv.to_bits().hash(&mut h);
        }
        self.system.sigma.to_bits().hash(&mut h);
        self.system.rho.to_bits().hash(&mut h);
        self.system.beta.to_bits().hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.tube_radius.to_bits().hash(&mut h);
        self.simulation_speed.hash(&mut h);
        self.max_points.hash(&mut h);
        self.is_paused().hash(&mut h);
        self.diverged().hash(&mut h);
        self.show_annotations.hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        !self.diverged() && self.is_finite_state()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_state(state)
    }

    fn step_simulation(&mut self) {
        if self.diverged() {
            self.latest_lyapunov = None;
            self.latest_correlation_sum = None;
            return;
        }
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        let custom_job = (self.mode == AttractorMode::Custom)
            .then(|| CompiledFlowJob::from_bank(&self.flow_bank));
        for _ in 0..self.simulation_speed.clamp(1, MAX_SIMULATION_SPEED) {
            let current_vec = self.system.state.vec;
            if !current_vec.x.is_finite()
                || !current_vec.y.is_finite()
                || !current_vec.z.is_finite()
            {
                self.execution_state = SolverExecutionState::Diverged {
                    invalid_vec: current_vec,
                    diverged_at_step: self.step_count,
                };
                break;
            }

            if let Some(ref job) = custom_job {
                self.system.state.vec = job.rk4_step(current_vec, self.dt);
            } else {
                self.system.step(self.dt);
            }
            self.step_count += 1;

            let vec = self.system.state.vec;
            if !vec.x.is_finite() || !vec.y.is_finite() || !vec.z.is_finite() {
                self.execution_state = SolverExecutionState::Diverged {
                    invalid_vec: vec,
                    diverged_at_step: self.step_count,
                };
                break;
            }

            if self.history.len() >= self.max_points {
                self.history.pop_front();
            }
            self.history.push_back(vec);
        }
        self.refresh_local_metrics();
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        Some(self.build_trajectory_mesh_3d())
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::ColoredMesh, String>> {
        Some(self.build_trajectory_colored_mesh_3d())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut points: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();
        if points.is_empty() && self.is_finite_state() {
            points.push(self.project(self.system.state.vec));
        }
        vec![("Trajectory".to_string(), points)]
    }

    fn export_datasets_3d(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        let mut points: Vec<[f64; 3]> = self.history.iter().map(|p| [p.x, p.y, p.z]).collect();
        if points.is_empty() && self.is_finite_state() {
            let v = self.system.state.vec;
            points.push([v.x, v.y, v.z]);
        }
        vec![("Trajectory".to_string(), points)]
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_worker_snapshot();
        let high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        self.sync_regime_atlas(&ctx, high_contrast);

        let worker_faulted = self.worker.state().is_faulted();
        if !self.is_paused() && !worker_faulted {
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
            ctx.request_repaint();
        } else if self.worker.is_running() {
            let _ = self.worker.send_command(SimCommand::Pause);
        }

        if self.worker.state().is_stepping() {
            ctx.request_repaint();
        }

        self.draw_side_panel(ui, &ctx);
        self.draw_central_panel(ui, &ctx);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "AttractorPlotter",
        display_name: "Attractor Plotter",
        domain: "chaos",
        tags: &["lorenz", "attractor", "chaos"],
        build: || Box::new(AttractorPlotter::default()),
    }
}

#[path = "attractors/preset.rs"]
mod preset;

#[cfg(test)]
#[path = "attractors/attractor_tests.rs"]
mod attractor_tests;

#[cfg(test)]
#[path = "attractors/lyapunov_bridge_tests.rs"]
mod lyapunov_bridge_tests;

#[cfg(test)]
#[path = "attractors/custom_flow_tests.rs"]
mod custom_flow_tests;

#[cfg(test)]
#[path = "attractors/export_tests.rs"]
mod export_tests;
