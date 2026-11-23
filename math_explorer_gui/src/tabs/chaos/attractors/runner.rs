#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::flow::{AttractorMode, CompiledFlowJob};
use super::serde_vec3;
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use crate::tabs::geometry_topology::vietoris_rips::{
    handoff::{publish_live_attractor_landmarks, ChaosLandmarkHandoff},
    MAX_CHAOS_LANDMARKS,
};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use math_explorer::pure_math::statistics::tda::{farthest_point_landmarks, Point3D};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

/// Maximum allowed simulation sub-steps per frame for [`super::AttractorPlotter`].
pub const MAX_SIMULATION_SPEED: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SolverExecutionState {
    Ready,
    Running,
    Paused,
    Diverged {
        #[serde(with = "serde_vec3")]
        invalid_vec: Vector3<f64>,
        diverged_at_step: usize,
    },
}

#[allow(missing_docs)]
pub struct AttractorSharedParams {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
    pub dt: f64,
    pub simulation_speed: usize,
    pub max_points: usize,
    pub mode: AttractorMode,
    pub custom_flow: Option<Arc<CompiledFlowJob>>,
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct AttractorSnapshot {
    pub history: VecDeque<Vector3<f64>>,
    pub current_vec: Vector3<f64>,
    pub step_count: usize,
    pub execution_state: SolverExecutionState,
}

#[allow(missing_docs)]
pub struct AttractorRunner {
    pub system: LorenzSystem,
    pub params: Arc<std::sync::RwLock<AttractorSharedParams>>,
    pub history: VecDeque<Vector3<f64>>,
    pub step_count: usize,
    pub execution_state: SolverExecutionState,
}

impl AttractorRunner {
    /// Builds a fresh runner initialized from shared parameters.
    pub fn from_params(params: Arc<std::sync::RwLock<AttractorSharedParams>>) -> Self {
        let (sigma, rho, beta, max_points) = {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            (p.sigma, p.rho, p.beta, p.max_points)
        };
        Self {
            system: LorenzBuilder::new()
                .sigma(sigma)
                .rho(rho)
                .beta(beta)
                .build(LorenzState::new(1.0, 1.0, 1.0)),
            params,
            history: VecDeque::with_capacity(max_points),
            step_count: 0,
            execution_state: SolverExecutionState::Ready,
        }
    }

    fn sync_params(
        &mut self,
    ) -> (
        f64,
        usize,
        usize,
        AttractorMode,
        Option<Arc<CompiledFlowJob>>,
    ) {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        self.system.sigma = p.sigma;
        self.system.rho = p.rho;
        self.system.beta = p.beta;
        (
            p.dt,
            p.simulation_speed.clamp(1, MAX_SIMULATION_SPEED),
            p.max_points,
            p.mode,
            p.custom_flow.clone(),
        )
    }

    fn advance_tick(
        &mut self,
        dt: f64,
        max_points: usize,
        mode: AttractorMode,
        custom_flow: Option<&CompiledFlowJob>,
    ) -> bool {
        let current_vec = self.system.state.vec;
        let lorenz_ok = mode != AttractorMode::Lorenz
            || (self.system.sigma.is_finite()
                && self.system.rho.is_finite()
                && self.system.beta.is_finite());
        if !current_vec.x.is_finite()
            || !current_vec.y.is_finite()
            || !current_vec.z.is_finite()
            || !lorenz_ok
            || !dt.is_finite()
        {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: current_vec,
                diverged_at_step: self.step_count,
            };
            return false;
        }

        match mode {
            AttractorMode::Lorenz => self.system.step(dt),
            AttractorMode::Custom => {
                if let Some(job) = custom_flow {
                    self.system.state.vec = job.rk4_step(current_vec, dt);
                }
            }
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

        if self.history.len() >= max_points {
            self.history.pop_front();
        }
        self.history.push_back(vec);
        true
    }
}

impl SimulationRunner for AttractorRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Start => {
                if !matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
                    self.execution_state = SolverExecutionState::Running;
                }
            }
            SimCommand::Pause => {
                if !matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
                    self.execution_state = SolverExecutionState::Paused;
                }
            }
            SimCommand::Reset => {
                let (sigma, rho, beta) = {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                    (p.sigma, p.rho, p.beta)
                };
                self.system = LorenzBuilder::new()
                    .sigma(sigma)
                    .rho(rho)
                    .beta(beta)
                    .build(LorenzState::new(1.0, 1.0, 1.0));
                self.history.clear();
                self.step_count = 0;
                self.execution_state = SolverExecutionState::Ready;
            }
            SimCommand::SetSpeed(speed) => {
                self.params
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .simulation_speed = speed.clamp(1, MAX_SIMULATION_SPEED);
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, simulation_speed, max_points, mode, custom_flow) = self.sync_params();
        if self.execution_state == SolverExecutionState::Ready {
            self.execution_state = SolverExecutionState::Running;
        }
        for _ in 0..simulation_speed {
            if !self.advance_tick(dt, max_points, mode, custom_flow.as_deref()) {
                break;
            }
        }
        if self.history.len() >= 16 && self.step_count % 15 < simulation_speed {
            let pts: Vec<Point3D> = self
                .history
                .iter()
                .map(|v| Point3D::new(v.x, v.y, v.z))
                .collect();
            if let Ok(landmarks) = farthest_point_landmarks(&pts, MAX_CHAOS_LANDMARKS) {
                publish_live_attractor_landmarks(ChaosLandmarkHandoff {
                    points: landmarks,
                    sigma: self.system.sigma,
                    rho: self.system.rho,
                    beta: self.system.beta,
                    dt,
                });
            }
        }
    }

    fn step_once(&mut self) {
        if matches!(self.execution_state, SolverExecutionState::Diverged { .. }) {
            return;
        }
        let (dt, _, max_points, mode, custom_flow) = self.sync_params();
        if self.advance_tick(dt, max_points, mode, custom_flow.as_deref()) {
            self.execution_state = SolverExecutionState::Paused;
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let v = self.system.state.vec;
        let (dx, dy, dz) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            match (p.mode, p.custom_flow.as_deref()) {
                (AttractorMode::Custom, Some(job)) => {
                    let rhs = job.eval_rhs(v);
                    (rhs.x, rhs.y, rhs.z)
                }
                _ => (
                    self.system.sigma * (v.y - v.x),
                    v.x * (self.system.rho - v.z) - v.y,
                    v.x * v.y - self.system.beta * v.z,
                ),
            }
        };
        let diverged = matches!(self.execution_state, SolverExecutionState::Diverged { .. });
        let sonification =
            crate::audio::mappings::attractor_frame([v.x, v.y, v.z], [dx, dy, dz], diverged);
        let snapshot = AttractorSnapshot {
            history: self.history.clone(),
            current_vec: v,
            step_count: self.step_count,
            execution_state: self.execution_state,
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![v.x, v.y, v.z, self.step_count as f64],
            structured_data: Some(Box::new(snapshot)),
            sonification,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        let v = self.system.state.vec;
        !matches!(self.execution_state, SolverExecutionState::Diverged { .. })
            && v.x.is_finite()
            && v.y.is_finite()
            && v.z.is_finite()
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
        let step_dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.01 };
        if let SolverExecutionState::Diverged {
            invalid_vec,
            diverged_at_step,
        } = self.execution_state
        {
            let var = if !invalid_vec.x.is_finite() {
                "x"
            } else if !invalid_vec.y.is_finite() {
                "y"
            } else if !invalid_vec.z.is_finite() {
                "z"
            } else {
                "lorenz_state"
            };
            Some(DiagnosticReport::new(
                diverged_at_step as f64 * step_dt,
                diverged_at_step,
                var,
                format!(
                    "Non-finite attractor state [{:.2}, {:.2}, {:.2}] at step {}",
                    invalid_vec.x, invalid_vec.y, invalid_vec.z, diverged_at_step
                ),
            ))
        } else if !self.is_finite_state() {
            Some(DiagnosticReport::new(
                self.step_count as f64 * step_dt,
                self.step_count,
                "lorenz_state",
                "Non-finite attractor trajectory coordinates detected",
            ))
        } else {
            None
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for super::AttractorPlotter {
    fn theory_description(&self) -> String {
        match self.mode {
            AttractorMode::Lorenz => format!(
                "Lorenz attractor dynamical system: $\\dot{{x}} = \\sigma(y - x)$, $\\dot{{y}} = x(\\rho - z) - y$, $\\dot{{z}} = xy - \\beta z$ with parameters $\\sigma = {:.2}$, $\\rho = {:.2}$, $\\beta = {:.2}$.",
                self.system.sigma, self.system.rho, self.system.beta
            ),
            AttractorMode::Custom => {
                let exprs = self.flow_bank.expressions();
                format!(
                    "Custom 3D dynamical flow: dx/dt = {}, dy/dt = {}, dz/dt = {}.",
                    exprs[0], exprs[1], exprs[2]
                )
            }
        }
    }
    fn phonetic_description(&self) -> String {
        "3D strange attractor dynamical system theory".into()
    }
    fn theory_citation(&self) -> String {
        "Deterministic Nonperiodic Flow (Lorenz, 1963); Nonlinear Dynamics and Chaos (Strogatz, 2014)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        self.system.available_descriptions()
    }
    fn theory_parameters(
        &self,
    ) -> std::collections::HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        use scientific_metadata::theory::ParameterConstraint;
        std::collections::HashMap::from([
            (
                "sigma".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 50.0,
                    step: 0.1,
                },
            ),
            (
                "rho".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 100.0,
                    step: 0.5,
                },
            ),
            (
                "beta".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 10.0,
                    step: 0.1,
                },
            ),
            (
                "dt".to_string(),
                ParameterConstraint {
                    min: 0.001,
                    max: 0.05,
                    step: 0.001,
                },
            ),
        ])
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "3d_attractor",
        title: "3D Attractor",
        description: "ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz: Canonical 3D chaotic strange attractor trajectory.",
        category: "Chaos Theory",
        icon: "🦋",
        domain: "chaos",
        tool_name: Some("AttractorPlotter"),
        preset_generator: || crate::starter_gallery::PRESET_3D_ATTRACTOR,
    }
}
