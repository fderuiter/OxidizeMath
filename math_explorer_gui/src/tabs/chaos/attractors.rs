#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{BackgroundWorker, SimCommand};
use crate::framework::{
    Camera3D, CoupledExpressionBank, MathExpressionInputBar, Scene3DBuilder, TimelineRingBuffer,
    DEFAULT_TIMELINE_RING_CAPACITY,
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

#[path = "attractors/runner_types.rs"]
mod runner_types;
pub use runner_types::*;

#[path = "attractors/runner.rs"]
mod runner;
#[path = "attractors/serde_vec3.rs"]
mod serde_vec3;
pub use runner::*;

#[path = "attractors/twin_state.rs"]
pub mod twin_state;
pub use twin_state::*;

#[path = "attractors/ui.rs"]
mod ui;

#[path = "attractors/scrubber_ui.rs"]
mod scrubber_ui;

#[path = "attractors/divergence_plot.rs"]
mod divergence_plot;

#[path = "attractors/tool_impl.rs"]
mod tool_impl;

#[path = "attractors/preset.rs"]
mod preset;

#[path = "attractors/export.rs"]
pub mod export;

/// Interactive 3D strange attractor and twin-trajectory divergence explorer.
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
    pub(crate) shadow_history: VecDeque<Vector3<f64>>,
    pub(crate) timeline: TimelineRingBuffer<TwinFrame>,
    pub(crate) max_points: usize,
    pub(crate) camera: Camera3D,
    pub(crate) scene_builder: Scene3DBuilder,
    pub(super) trajectory_scratch: Vec<Point3<f64>>,
    pub(crate) params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    pub(crate) worker: BackgroundWorker<AttractorRunner>,
    /// Whether Lorenz equilibrium callouts ($C_+$, $C_-$) are displayed.
    pub show_annotations: bool,
    pub(crate) fixed_callout_offsets: [egui::Vec2; 2],
    pub(crate) dragged_callout: Option<usize>,
    pub(crate) system_kind: AttractorSystemKind,
    pub(crate) initial_state: [f64; 3],
    pub(crate) shadow_enabled: bool,
    pub(crate) epsilon: f64,
    pub(crate) perturbation_expr: String,
    pub(crate) perturbation_bar: MathExpressionInputBar,
    pub(crate) perturbation_offset: [f64; 3],
    pub(crate) perturbation_eval_error: Option<String>,
    pub(crate) last_eval_error_announced: Option<String>,
    pub(crate) extra_params: ExtraChaosParams,
    pub(crate) metrics_cache: ChaosMetricsCache,
    pub(crate) invariants_cache: invariants::CachedAttractorInvariants,
}

impl Default for AttractorPlotter {
    fn default() -> Self {
        let initial = [1.0, 1.0, 1.0];
        let system =
            LorenzBuilder::new().build(LorenzState::new(initial[0], initial[1], initial[2]));
        let flow_bank = default_flow_bank();
        let custom_flow = Some(Arc::new(CompiledFlowJob::from_bank(&flow_bank)));
        let perturbation_bar = MathExpressionInputBar::new(
            "δx₀",
            DEFAULT_PERTURBATION_EXPR,
            PERTURBATION_PRESETS.to_vec(),
        );
        let offset = [DEFAULT_EPSILON, 0.0, 0.0];
        let params = Self::initial_shared_params(&system, custom_flow, initial, offset);
        let runner = AttractorRunner::from_params(Arc::clone(&params));
        let timeline = runner.timeline.clone();
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
            history: VecDeque::with_capacity(2000),
            shadow_history: VecDeque::with_capacity(2000),
            timeline,
            max_points: 2000,
            camera,
            scene_builder: Scene3DBuilder::new(camera),
            trajectory_scratch: Vec::with_capacity(2000),
            params,
            worker,
            show_annotations: true,
            fixed_callout_offsets: [egui::Vec2::new(35.0, -25.0), egui::Vec2::new(-35.0, 25.0)],
            dragged_callout: None,
            system_kind: AttractorSystemKind::Lorenz,
            initial_state: initial,
            shadow_enabled: true,
            epsilon: DEFAULT_EPSILON,
            perturbation_expr: DEFAULT_PERTURBATION_EXPR.to_string(),
            perturbation_bar,
            perturbation_offset: offset,
            perturbation_eval_error: None,
            last_eval_error_announced: None,
            extra_params: ExtraChaosParams::default(),
            metrics_cache: ChaosMetricsCache::default(),
            invariants_cache: invariants::CachedAttractorInvariants::default(),
        }
    }
}

impl AttractorPlotter {
    fn initial_shared_params(
        system: &LorenzSystem,
        custom_flow: Option<Arc<CompiledFlowJob>>,
        initial: [f64; 3],
        offset: [f64; 3],
    ) -> Arc<std::sync::RwLock<AttractorSharedParams>> {
        Arc::new(std::sync::RwLock::new(AttractorSharedParams {
            sigma: system.sigma,
            rho: system.rho,
            beta: system.beta,
            dt: 0.01,
            simulation_speed: 5,
            max_points: 2000,
            mode: AttractorMode::Lorenz,
            custom_flow,
            system_kind: AttractorSystemKind::Lorenz,
            initial_state: initial,
            shadow_enabled: true,
            perturbation_offset: offset,
            extra: ExtraChaosParams::default(),
        }))
    }

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
            && self.dt > 0.0
            && self.tube_radius.is_finite()
            && self.epsilon.is_finite()
            && self.initial_state.iter().all(|c| c.is_finite())
            && self.extra_params.is_valid()
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

    pub(crate) fn switch_system_kind(&mut self, kind: AttractorSystemKind) {
        self.system_kind = kind;
        self.initial_state = kind.default_initial_state();
        self.dt = kind.default_dt();
        self.metrics_cache.correlation_radius = kind.default_correlation_radius();
        self.metrics_cache.correlation_result = None;
        self.recompute_perturbation_and_sync();
        self.reset();
    }

    pub(crate) fn recompute_perturbation_and_sync(&mut self) {
        match evaluate_perturbation_bar(&self.perturbation_bar, self.initial_state, self.epsilon) {
            Ok(offset) => {
                self.perturbation_offset = offset;
                self.perturbation_eval_error = None;
            }
            Err(err) => {
                self.perturbation_offset = [self.epsilon, 0.0, 0.0];
                if self.perturbation_bar.is_valid() {
                    self.perturbation_eval_error = Some(err);
                } else {
                    self.perturbation_eval_error = None;
                }
            }
        }
    }

    fn write_shared_params(&self) {
        let job = Arc::new(CompiledFlowJob::from_bank(&self.flow_bank));
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.sigma = self.system.sigma;
        p.rho = self.system.rho;
        p.beta = self.system.beta;
        p.dt = self.dt;
        p.simulation_speed = self.simulation_speed;
        p.max_points = self.max_points;
        p.mode = self.mode;
        p.custom_flow = Some(job);
        p.system_kind = self.system_kind;
        p.initial_state = self.initial_state;
        p.shadow_enabled = self.shadow_enabled;
        p.perturbation_offset = self.perturbation_offset;
        p.extra = self.extra_params;
    }

    pub(crate) fn ensure_lyapunov_cache(&mut self) {
        if self.metrics_cache.wolf_dirty {
            let p = self
                .params
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            self.metrics_cache.refresh_wolf_lyapunov(&p);
        }
    }

    pub fn on_slider_changed(&mut self) {
        self.write_shared_params();
        self.metrics_cache.wolf_dirty = true;
        if self.diverged() && self.is_finite_state() {
            self.execution_state = SolverExecutionState::Ready;
        }
    }

    fn sanitize_reset_inputs(&mut self) -> (f64, f64, f64) {
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
            self.dt = self.system_kind.default_dt();
        }
        if !self.tube_radius.is_finite() {
            self.tube_radius = export::DEFAULT_TUBE_RADIUS;
        }
        if !self.epsilon.is_finite() {
            self.epsilon = DEFAULT_EPSILON;
        }
        if self.initial_state.iter().any(|c| !c.is_finite()) {
            self.initial_state = self.system_kind.default_initial_state();
        }
        self.extra_params.sanitize();
        self.recompute_perturbation_and_sync();
        (sigma, rho, beta)
    }

    pub fn reset(&mut self) {
        let (sigma, rho, beta) = self.sanitize_reset_inputs();
        let init = self.initial_state;
        self.system = LorenzBuilder::new()
            .sigma(sigma)
            .rho(rho)
            .beta(beta)
            .build(LorenzState::new(init[0], init[1], init[2]));
        self.history.clear();
        self.shadow_history.clear();
        self.timeline.clear();
        let init_p = [init[0], init[1], init[2], 0.0];
        let off = if self.shadow_enabled {
            self.perturbation_offset
        } else {
            [0.0; 3]
        };
        let init_s = [init[0] + off[0], init[1] + off[1], init[2] + off[2], 0.0];
        self.timeline.push(TwinFrame::from_states(
            0,
            0.0,
            init_p,
            init_s,
            self.shadow_enabled,
        ));
        self.step_count = 0;
        self.execution_state = SolverExecutionState::Ready;
        self.metrics_cache.wolf_dirty = true;
        self.metrics_cache.windowed_ftle = 0.0;
        self.metrics_cache.correlation_result = None;
        self.write_shared_params();

        let params = Arc::clone(&self.params);
        self.worker
            .reset_or_respawn(|| AttractorRunner::from_params(params));
    }

    pub fn reset_parameters(&mut self) {
        self.system.sigma = 10.0;
        self.system.rho = 28.0;
        self.system.beta = 8.0 / 3.0;
        self.flow_bank = default_flow_bank();
        self.extra_params = ExtraChaosParams::default();
        self.dt = if self.mode == AttractorMode::Custom {
            0.01
        } else {
            self.system_kind.default_dt()
        };
        self.tube_radius = export::DEFAULT_TUBE_RADIUS;
        self.epsilon = DEFAULT_EPSILON;
        self.perturbation_expr = DEFAULT_PERTURBATION_EXPR.to_string();
        self.perturbation_bar
            .set_expression(DEFAULT_PERTURBATION_EXPR);
        self.reset();
    }

    pub(crate) fn apply_restored_frame(&mut self, frame: TwinFrame) {
        self.step_count = frame.step;
        self.system.state.vec = frame.primary_vec3();
        self.history.clear();
        self.shadow_history.clear();
        let cap = self.max_points.clamp(1, DEFAULT_TIMELINE_RING_CAPACITY);
        for item in self.timeline.iter_up_to_cursor().skip(1) {
            if self.history.len() >= cap {
                self.history.pop_front();
            }
            self.history.push_back(item.primary_vec3());
            if self.shadow_history.len() >= cap {
                self.shadow_history.pop_front();
            }
            self.shadow_history.push_back(item.shadow_vec3());
        }
        self.metrics_cache.update_windowed_ftle(&self.timeline);
    }

    pub(crate) fn seek_timeline_frame(&mut self, frame_idx: usize, ctx: &egui::Context) {
        self.execution_state = SolverExecutionState::Paused;
        if let Some(frame) = self.timeline.seek(frame_idx) {
            self.apply_restored_frame(frame);
        }
        let _ = self.worker.send_command(SimCommand::SeekFrame(frame_idx));
        ctx.request_repaint();
    }

    pub(crate) fn compute_correlation_dimension_on_demand(&mut self) {
        let pts: Vec<Vector3<f64>> = self
            .timeline
            .iter_up_to_cursor()
            .map(|f| f.primary_vec3())
            .collect();
        let _ = self.metrics_cache.compute_correlation_on_demand(&pts);
    }

    /// Projects 3D point to 2D screen space based on rotation, pan, zoom, and projection mode.
    pub(crate) fn project(&self, p: Vector3<f64>) -> [f64; 2] {
        let center_offset = match self.mode {
            AttractorMode::Lorenz => self.system_kind.center_offset(),
            AttractorMode::Custom => Vector3::zeros(),
        };
        let p_centered = p - center_offset;
        self.camera
            .project(&[p_centered.x, p_centered.y, p_centered.z])
    }
}

#[cfg(test)]
use crate::framework::InteractiveTool;

#[cfg(test)]
#[path = "attractors/attractor_tests.rs"]
mod attractor_tests;

#[cfg(test)]
#[path = "attractors/twin_tests.rs"]
mod twin_tests;

#[cfg(test)]
#[path = "attractors/custom_flow_tests.rs"]
mod custom_flow_tests;

#[cfg(test)]
#[path = "attractors/export_tests.rs"]
mod export_tests;
