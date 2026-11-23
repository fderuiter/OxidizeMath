#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Shared parameter, state, and snapshot types for [`super::AttractorRunner`].

use super::flow::{AttractorMode, CompiledFlowJob};
use super::serde_vec3;
use super::twin_state::{AttractorSystemKind, ExtraChaosParams, TwinFrame, DEFAULT_EPSILON};
use crate::async_sim::StateSnapshot;
use crate::framework::recovery::DiagnosticReport;
use crate::framework::TimelineRingBuffer;
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::Arc;

/// Maximum allowed simulation sub-steps per frame for [`super::AttractorPlotter`].
pub const MAX_SIMULATION_SPEED: usize = 50;

/// Execution lifecycle state of [`super::AttractorPlotter`] and [`super::AttractorRunner`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SolverExecutionState {
    /// Initialized and ready to run.
    Ready,
    /// Actively integrating frames.
    Running,
    /// Paused by user interaction, single-stepping, or timeline scrubbing.
    Paused,
    /// Halted due to non-finite coordinates or parameters.
    Diverged {
        /// Coordinates observed when divergence occurred.
        #[serde(with = "serde_vec3")]
        invalid_vec: Vector3<f64>,
        /// Step index at which non-finite state was detected.
        diverged_at_step: usize,
    },
}

/// Shared parameter block read by [`super::AttractorRunner`] without dynamic heap allocation.
#[derive(Debug, Clone)]
pub struct AttractorSharedParams {
    /// Lorenz Prandtl number $\sigma$.
    pub sigma: f64,
    /// Lorenz Rayleigh number $\rho$.
    pub rho: f64,
    /// Lorenz geometric factor $\beta$.
    pub beta: f64,
    /// Integration step size `dt`.
    pub dt: f64,
    /// Sub-steps executed per worker frame.
    pub simulation_speed: usize,
    /// Maximum rendered trail length (`<= 4,096`).
    pub max_points: usize,
    /// Active dynamical flow mode (`Lorenz` built-in systems vs `Custom` 3D flow).
    pub mode: AttractorMode,
    /// `Arc`-shared compiled user-authored 3D flow bytecode for `AttractorMode::Custom`.
    pub custom_flow: Option<Arc<CompiledFlowJob>>,
    /// Active built-in dynamical system kind.
    pub system_kind: AttractorSystemKind,
    /// Primary initial coordinates `[c0, c1, c2]`.
    pub initial_state: [f64; 3],
    /// Whether the perturbed shadow trajectory is enabled.
    pub shadow_enabled: bool,
    /// Pre-evaluated AST perturbation offset `[dx0, dy0, dz0]`.
    pub perturbation_offset: [f64; 3],
    /// Auxiliary Rössler and double-pendulum constants.
    pub extra: ExtraChaosParams,
}

impl Default for AttractorSharedParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            dt: 0.01,
            simulation_speed: 5,
            max_points: 2000,
            mode: AttractorMode::Lorenz,
            custom_flow: None,
            system_kind: AttractorSystemKind::Lorenz,
            initial_state: [1.0, 1.0, 1.0],
            shadow_enabled: true,
            perturbation_offset: [DEFAULT_EPSILON, 0.0, 0.0],
            extra: ExtraChaosParams::default(),
        }
    }
}

/// Validated per-frame stepping snapshot extracted from [`AttractorSharedParams`].
pub(super) struct SyncedStepParams {
    pub mode: AttractorMode,
    pub custom_flow: Option<Arc<CompiledFlowJob>>,
    pub kind: AttractorSystemKind,
    pub dt: f64,
    pub speed: usize,
    pub max_points: usize,
    pub shadow_enabled: bool,
}

/// Worker snapshot published to the UI thread.
#[derive(Clone)]
pub struct AttractorSnapshot {
    /// Primary 3D trajectory points up to the active timeline cursor.
    pub history: VecDeque<Vector3<f64>>,
    /// Shadow 3D trajectory points up to the active timeline cursor.
    pub shadow_history: VecDeque<Vector3<f64>>,
    /// Full ring buffer of recorded `TwinFrame`s for UI scrubbing.
    pub timeline: TimelineRingBuffer<TwinFrame>,
    /// Current primary 3D phase-space vector.
    pub current_vec: Vector3<f64>,
    /// Current step index at the active timeline cursor.
    pub step_count: usize,
    /// Worker solver state.
    pub execution_state: SolverExecutionState,
}

impl AttractorSnapshot {
    pub(super) fn into_state_snapshot(
        self,
        sonification: crate::audio::SonificationFrame,
    ) -> StateSnapshot {
        let v = self.current_vec;
        let steps = self.step_count as f64;
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![v.x, v.y, v.z, steps],
            structured_data: Some(Box::new(self)),
            sonification,
        }
    }
}

pub(super) fn build_runner_sonification(
    runner: &super::AttractorRunner,
    p: &AttractorSharedParams,
    v: Vector3<f64>,
) -> crate::audio::SonificationFrame {
    let deriv = match (p.mode, p.custom_flow.as_deref(), p.system_kind) {
        (AttractorMode::Custom, Some(job), _) => {
            let rhs = job.eval_rhs(v);
            [rhs.x, rhs.y, rhs.z]
        }
        (_, _, AttractorSystemKind::Rossler) => [
            -v.y - v.z,
            v.x + runner.rossler_primary.a * v.y,
            runner.rossler_primary.b + v.z * (v.x - runner.rossler_primary.c),
        ],
        (_, _, AttractorSystemKind::DoublePendulum) => {
            let s = runner.pendulum_primary.state.state.0;
            [s[2], s[3], 0.0]
        }
        _ => [
            runner.system.sigma * (v.y - v.x),
            v.x * (runner.system.rho - v.z) - v.y,
            v.x * v.y - runner.system.beta * v.z,
        ],
    };
    let diverged = matches!(
        runner.execution_state,
        SolverExecutionState::Diverged { .. }
    );
    crate::audio::mappings::attractor_frame([v.x, v.y, v.z], deriv, diverged)
}

pub(super) fn build_runner_diagnostic(
    execution_state: SolverExecutionState,
    step_count: usize,
    dt: f64,
    is_finite: bool,
) -> Option<DiagnosticReport> {
    if let SolverExecutionState::Diverged {
        invalid_vec,
        diverged_at_step,
    } = execution_state
    {
        Some(DiagnosticReport::new(
            diverged_at_step as f64,
            diverged_at_step,
            "attractor_state",
            format!(
                "Non-finite attractor state [{:.2}, {:.2}, {:.2}] at step {}",
                invalid_vec.x, invalid_vec.y, invalid_vec.z, diverged_at_step
            ),
        ))
    } else if !is_finite {
        Some(DiagnosticReport::new(
            step_count as f64 * if dt.is_finite() { dt } else { 0.01 },
            step_count,
            "attractor_state",
            "Non-finite chaotic trajectory coordinates detected",
        ))
    } else {
        None
    }
}
