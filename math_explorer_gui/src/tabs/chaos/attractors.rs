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

#[path = "attractors/counterfactual.rs"]
mod counterfactual;
pub use counterfactual::*;

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

#[path = "attractors/ui.rs"]
mod ui;

#[path = "attractors/export.rs"]
pub mod export;

pub(crate) const DEFAULT_FIXED_CALLOUT_OFFSETS: [[f32; 2]; 2] = [[35.0, -25.0], [-35.0, 25.0]];

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
    trajectory_scratch: Vec<Point3<f64>>,
    pub(crate) params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    pub(crate) worker: BackgroundWorker<AttractorRunner>,
    pub show_annotations: bool,
    pub(crate) fixed_callout_offsets: [egui::Vec2; 2],
    pub(crate) dragged_callout: Option<usize>,
    pub(crate) twin: Option<AttractorTwinState>,
    pub branch_engine: crate::framework::CounterfactualBranchEngine,
    pub(super) latest_sonification: crate::audio::SonificationFrame,
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
            history: VecDeque::with_capacity(ATTRACTOR_RING_CAPACITY),
            max_points: 2000,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            trajectory_scratch: Vec::with_capacity(2000),
            params,
            worker,
            show_annotations: true,
            fixed_callout_offsets: DEFAULT_FIXED_CALLOUT_OFFSETS.map(egui::Vec2::from),
            dragged_callout: None,
            twin: None,
            branch_engine: crate::framework::CounterfactualBranchEngine::default(),
            latest_sonification: crate::audio::mappings::attractor_frame(
                [1.0, 1.0, 1.0],
                [0.0, 26.0, -1.67],
                false,
            ),
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
        self.twin = None;
        self.branch_engine.clear();
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
    }

    pub fn reset_parameters(&mut self) {
        self.system.sigma = 10.0;
        self.system.rho = 28.0;
        self.system.beta = 8.0 / 3.0;
        self.flow_bank = default_flow_bank();
        self.dt = 0.01;
        self.tube_radius = export::DEFAULT_TUBE_RADIUS;
        self.fixed_callout_offsets = DEFAULT_FIXED_CALLOUT_OFFSETS.map(egui::Vec2::from);
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

    fn step_single_tick(&mut self, cap: usize, custom_job: Option<&CompiledFlowJob>) -> bool {
        let current_vec = self.system.state.vec;
        if !current_vec.x.is_finite() || !current_vec.y.is_finite() || !current_vec.z.is_finite() {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: current_vec,
                diverged_at_step: self.step_count,
            };
            return false;
        }
        if let Some(ref mut twin) = self.twin {
            if !twin.step_branch_a(self.dt, cap) {
                self.execution_state = SolverExecutionState::Diverged {
                    invalid_vec: twin.system_a.state.vec,
                    diverged_at_step: self.step_count,
                };
                return false;
            }
        }
        if let Some(job) = custom_job {
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
            return false;
        }
        if self.history.len() >= cap {
            self.history.pop_front();
        }
        self.history.push_back(vec);
        if let Some(ref mut twin) = self.twin {
            twin.observe_lockstep(self.step_count, self.dt, vec);
        }
        true
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
        for offset in &self.fixed_callout_offsets {
            let x_bits = if offset.x == 0.0 {
                0
            } else {
                offset.x.to_bits()
            };
            let y_bits = if offset.y == 0.0 {
                0
            } else {
                offset.y.to_bits()
            };
            x_bits.hash(&mut h);
            y_bits.hash(&mut h);
        }
        self.camera.hash_state(&mut h);
        if let Some(ref twin) = self.twin {
            true.hash(&mut h);
            twin.fork_step.hash(&mut h);
            twin.delta_x.to_bits().hash(&mut h);
            twin.system_a.sigma.to_bits().hash(&mut h);
            twin.system_a.rho.to_bits().hash(&mut h);
            twin.system_a.beta.to_bits().hash(&mut h);
        } else {
            false.hash(&mut h);
        }
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
            return;
        }
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        let cap = self.max_points.clamp(1, ATTRACTOR_RING_CAPACITY);
        let custom_job = (self.mode == AttractorMode::Custom)
            .then(|| CompiledFlowJob::from_bank(&self.flow_bank));
        for _ in 0..self.simulation_speed.clamp(1, MAX_SIMULATION_SPEED) {
            if !self.step_single_tick(cap, custom_job.as_ref()) {
                break;
            }
        }
        self.sync_local_branch_engine();
    }

    fn counterfactual_snapshot(&self) -> Option<crate::framework::DivergenceSnapshot> {
        self.twin
            .as_ref()
            .map(|t| t.snapshot(self.step_count, self.dt))
    }

    fn apply_counterfactual_action(
        &mut self,
        action: crate::framework::BranchControlAction,
    ) -> bool {
        match action {
            crate::framework::BranchControlAction::ForkHere => self.fork_here(),
            crate::framework::BranchControlAction::SwapBranches => self.swap_branches(),
            crate::framework::BranchControlAction::ClearFork => self.clear_fork(),
            crate::framework::BranchControlAction::SetPerturbation(delta) => {
                self.branch_engine.set_effective_delta(delta);
                true
            }
        }
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
        self.collect_export_datasets_2d()
    }

    fn export_datasets_3d(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        self.collect_export_datasets_3d()
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
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
            self.latest_sonification = snapshot.sonification;
            if !ui_diverged {
                if let Some(data) = snapshot
                    .structured_data
                    .as_ref()
                    .and_then(|d| d.downcast_ref::<AttractorSnapshot>())
                {
                    self.history = data.history.clone();
                    self.system.state.vec = data.current_vec;
                    self.step_count = data.step_count;
                    self.twin = data.twin.clone();
                    if matches!(data.execution_state, SolverExecutionState::Diverged { .. })
                        || !was_paused
                    {
                        self.execution_state = data.execution_state;
                    }
                }
                self.sync_local_branch_engine();
            }
        }

        let worker_faulted = self.worker.state().is_faulted();
        if self.diverged() || worker_faulted {
            self.latest_sonification = crate::audio::SonificationFrame::silent();
            crate::audio::notify_solver_diverged();
        }
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

#[path = "attractors/preset.rs"]
mod preset;

#[cfg(test)]
#[path = "attractors/attractor_tests.rs"]
mod attractor_tests;

#[cfg(test)]
#[path = "attractors/counterfactual_tests.rs"]
mod counterfactual_tests;

#[cfg(test)]
#[path = "attractors/custom_flow_tests.rs"]
mod custom_flow_tests;

#[cfg(test)]
#[path = "attractors/export_tests.rs"]
mod export_tests;
