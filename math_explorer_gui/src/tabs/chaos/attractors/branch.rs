//! Timeline ring-buffer checkpointing and 3D lockstep shadow-twin integration
//! for [`super::AttractorRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    AttractorMode, AttractorRunner, AttractorSharedParams, CompiledFlowJob, SolverExecutionState,
};
use crate::async_sim::SimCommand;
use crate::framework::{
    compute_slice_divergence, BranchDivergenceAccumulator, BranchDivergenceMetrics,
    TimelineRingBuffer,
};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

/// Maximum checkpoints stored in `AttractorRunner`'s ring buffer (`4096 * 40 B ≈ 160 KiB << 2 MiB`).
pub const ATTRACTOR_RING_SLOTS: usize = 4096;
/// Preallocated capacity for shadow-twin 3D history so inner RK4 ticks never allocate.
pub const MAX_SHADOW_HISTORY_CAP: usize = 10_000;

/// Worker-to-UI snapshot of timeline bounds, divergence metrics, and shadow 3D history.
#[derive(Clone, Default)]
pub struct AttractorBranchSnapshot {
    /// Earliest retained checkpoint step index in the ring buffer.
    pub min_step: usize,
    /// Latest retained checkpoint step index in the ring buffer.
    pub max_step: usize,
    /// Current integrated simulation time $t$.
    pub current_time: f64,
    /// Whether an unperturbed 3D shadow-twin branch is currently active.
    pub fork_active: bool,
    /// Step index $k$ at which the shadow twin was forked.
    pub fork_step: Option<usize>,
    /// Simulation time $t_k$ at which the shadow twin was forked.
    pub fork_time: Option<f64>,
    /// World-space 3D state $(x_k, y_k, z_k)$ at the fork point.
    pub fork_vec: Option<Vector3<f64>>,
    /// Compact summary of the baseline parameters frozen on the shadow twin.
    pub baseline_summary: String,
    /// Live divergence metrics between counterfactual and baseline trajectories.
    pub divergence: BranchDivergenceMetrics,
    /// 3D world-space trajectory points accumulated by the shadow twin.
    pub shadow_history: VecDeque<Vector3<f64>>,
}

/// Unperturbed strange-attractor twin forked at $(k, t_k)$.
pub struct AttractorShadowTwin {
    pub(crate) system: LorenzSystem,
    pub(crate) mode: AttractorMode,
    pub(crate) custom_flow: Option<Arc<CompiledFlowJob>>,
    pub(crate) baseline_dt: f64,
    pub(crate) baseline_summary: String,
    pub(crate) fork_step: usize,
    pub(crate) fork_time: f64,
    pub(crate) fork_vec: Vector3<f64>,
    pub(crate) history: VecDeque<Vector3<f64>>,
    pub(crate) divergence: BranchDivergenceAccumulator,
}

/// Timeline ring-buffer and lockstep 3D shadow-twin state.
pub struct AttractorBranchRuntime {
    pub(crate) current_time: f64,
    pub(crate) timeline: TimelineRingBuffer<[f64; 3]>,
    pub(crate) shadow: Option<AttractorShadowTwin>,
    pub(crate) worker_view: Option<AttractorBranchSnapshot>,
}

impl Default for AttractorBranchRuntime {
    fn default() -> Self {
        let mut timeline = TimelineRingBuffer::new_default(
            std::mem::size_of::<[f64; 3]>(),
            ATTRACTOR_RING_SLOTS,
            1,
        );
        timeline.push_checkpoint(0, 0.0, [1.0, 1.0, 1.0]);
        Self {
            current_time: 0.0,
            timeline,
            shadow: None,
            worker_view: None,
        }
    }
}

impl AttractorBranchRuntime {
    pub(crate) fn reset(&mut self) {
        self.current_time = 0.0;
        self.shadow = None;
        self.worker_view = None;
        self.timeline.clear();
        self.timeline.push_checkpoint(0, 0.0, [1.0, 1.0, 1.0]);
    }

    /// Zero-allocation inner lockstep RK4 step for `AttractorRunner` and `VerifiedEngine::run_verified`.
    pub fn on_step_in_place(
        &mut self,
        step_count: usize,
        dt: f64,
        max_points: usize,
        primary_vec: Vector3<f64>,
    ) -> (f64, f64) {
        self.current_time += dt;
        let mut div_pair = (0.0, 0.0);
        if let Some(ref mut twin) = self.shadow {
            let cur = twin.system.state.vec;
            match twin.mode {
                AttractorMode::Lorenz => twin.system.step(twin.baseline_dt),
                AttractorMode::Custom => {
                    if let Some(ref job) = twin.custom_flow {
                        twin.system.state.vec = job.rk4_step(cur, twin.baseline_dt);
                    }
                }
            }
            let sv = twin.system.state.vec;
            let cap = max_points.clamp(1, MAX_SHADOW_HISTORY_CAP);
            while twin.history.len() >= cap {
                twin.history.pop_front();
            }
            if twin.history.len() < twin.history.capacity() {
                twin.history.push_back(sv);
            }
            let (l2, peak) = compute_slice_divergence(
                &[primary_vec.x, primary_vec.y, primary_vec.z],
                &[sv.x, sv.y, sv.z],
            );
            twin.divergence.record_step(dt, l2, peak);
            div_pair = (l2, peak);
        }
        if self.timeline.should_record_step(step_count) {
            self.timeline.push_checkpoint(
                step_count,
                self.current_time,
                [primary_vec.x, primary_vec.y, primary_vec.z],
            );
        }
        div_pair
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fork_at(
        &mut self,
        step_count: usize,
        system: &LorenzSystem,
        mode: AttractorMode,
        custom_flow: Option<Arc<CompiledFlowJob>>,
        dt: f64,
        primary_history: &VecDeque<Vector3<f64>>,
    ) {
        let fv = system.state.vec;
        let mut shadow_hist = VecDeque::with_capacity(MAX_SHADOW_HISTORY_CAP);
        for &pt in primary_history.iter().take(MAX_SHADOW_HISTORY_CAP) {
            shadow_hist.push_back(pt);
        }
        let summary = format!(
            "sigma={:.2}, rho={:.2}, beta={:.2}, x0=({:.5},{:.5},{:.5})",
            system.sigma, system.rho, system.beta, fv.x, fv.y, fv.z
        );
        let twin_sys = LorenzBuilder::new()
            .sigma(system.sigma)
            .rho(system.rho)
            .beta(system.beta)
            .build(LorenzState::new(fv.x, fv.y, fv.z));
        self.shadow = Some(AttractorShadowTwin {
            system: twin_sys,
            mode,
            custom_flow,
            baseline_dt: dt,
            baseline_summary: summary,
            fork_step: step_count,
            fork_time: self.current_time,
            fork_vec: fv,
            history: shadow_hist,
            divergence: BranchDivergenceAccumulator::default(),
        });
        self.worker_view = None;
    }

    pub(crate) fn clear_branch(&mut self) {
        self.shadow = None;
        self.worker_view = None;
    }

    pub(crate) fn to_snapshot(&self, step_count: usize) -> AttractorBranchSnapshot {
        let (min_s, max_s) = self
            .timeline
            .step_bounds()
            .unwrap_or((step_count, step_count));
        if let Some(ref twin) = self.shadow {
            AttractorBranchSnapshot {
                min_step: min_s,
                max_step: max_s.max(step_count),
                current_time: self.current_time,
                fork_active: true,
                fork_step: Some(twin.fork_step),
                fork_time: Some(twin.fork_time),
                fork_vec: Some(twin.fork_vec),
                baseline_summary: twin.baseline_summary.clone(),
                divergence: twin.divergence.metrics(),
                shadow_history: twin.history.clone(),
            }
        } else {
            AttractorBranchSnapshot {
                min_step: min_s,
                max_step: max_s.max(step_count),
                current_time: self.current_time,
                ..AttractorBranchSnapshot::default()
            }
        }
    }

    pub(crate) fn effective_view(&self, step_count: usize) -> AttractorBranchSnapshot {
        if let Some(ref wv) = self.worker_view {
            wv.clone()
        } else {
            self.to_snapshot(step_count)
        }
    }
}

struct RunnerSlot {
    key: usize,
    weak: Weak<RwLock<AttractorSharedParams>>,
    branch: AttractorBranchRuntime,
}

static RUNNER_SLOTS: OnceLock<Mutex<Vec<RunnerSlot>>> = OnceLock::new();

fn slots_mutex() -> &'static Mutex<Vec<RunnerSlot>> {
    RUNNER_SLOTS.get_or_init(|| Mutex::new(Vec::with_capacity(32)))
}

pub(crate) fn register_runner_branch(params: &Arc<RwLock<AttractorSharedParams>>) {
    let key = Arc::as_ptr(params) as usize;
    let mut guard = slots_mutex().lock().unwrap_or_else(|e| e.into_inner());
    guard.retain(|s| s.weak.strong_count() > 0 && s.key != key);
    guard.push(RunnerSlot {
        key,
        weak: Arc::downgrade(params),
        branch: AttractorBranchRuntime::default(),
    });
}

pub(crate) fn with_existing_runner_branch<R>(
    params: &Arc<RwLock<AttractorSharedParams>>,
    f: impl FnOnce(&mut AttractorBranchRuntime) -> R,
) -> Option<R> {
    let lock = RUNNER_SLOTS.get()?;
    let key = Arc::as_ptr(params) as usize;
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    let slot = guard.iter_mut().find(|s| s.key == key)?;
    Some(f(&mut slot.branch))
}

fn with_or_insert_runner_branch<R>(
    params: &Arc<RwLock<AttractorSharedParams>>,
    f: impl FnOnce(&mut AttractorBranchRuntime) -> R,
) -> R {
    let key = Arc::as_ptr(params) as usize;
    let mut guard = slots_mutex().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = guard.iter().position(|s| s.key == key) {
        return f(&mut guard[pos].branch);
    }
    guard.retain(|s| s.weak.strong_count() > 0);
    guard.push(RunnerSlot {
        key,
        weak: Arc::downgrade(params),
        branch: AttractorBranchRuntime::default(),
    });
    let last = guard.len() - 1;
    f(&mut guard[last].branch)
}

impl AttractorRunner {
    pub(crate) fn snapshot_branch(&self) -> AttractorBranchSnapshot {
        with_existing_runner_branch(&self.params, |b| b.to_snapshot(self.step_count)).unwrap_or(
            AttractorBranchSnapshot {
                min_step: 0,
                max_step: self.step_count,
                current_time: 0.0,
                ..AttractorBranchSnapshot::default()
            },
        )
    }

    pub(crate) fn reset_runner_branch(&mut self) {
        let _ = with_existing_runner_branch(&self.params, AttractorBranchRuntime::reset);
    }

    pub(crate) fn handle_branch_command(&mut self, cmd: &SimCommand) -> bool {
        match cmd {
            SimCommand::ForkBranch => {
                let (dt, _, _, mode, custom_flow) = self.sync_params_pub();
                let sys = self.system.clone();
                let step = self.step_count;
                let hist = self.history.clone();
                with_or_insert_runner_branch(&self.params, |b| {
                    b.fork_at(step, &sys, mode, custom_flow, dt, &hist);
                });
                true
            }
            SimCommand::ClearBranch => {
                with_or_insert_runner_branch(&self.params, AttractorBranchRuntime::clear_branch);
                true
            }
            SimCommand::StepBack(count) => {
                let cur = self.step_count;
                let prev = if *count <= 1 {
                    with_or_insert_runner_branch(&self.params, |b| {
                        b.timeline.previous_step_before(cur)
                    })
                    .unwrap_or_else(|| cur.saturating_sub(1))
                } else {
                    cur.saturating_sub(*count)
                };
                self.scrub_runner_to_step(prev);
                true
            }
            SimCommand::ScrubToStep(target) => {
                self.scrub_runner_to_step(*target);
                true
            }
            SimCommand::Custom(ref action) if action == "PerturbDeltaX:1e-5" => {
                self.system.state.vec.x += 1.0e-5;
                true
            }
            _ => false,
        }
    }

    pub(crate) fn scrub_runner_to_step(&mut self, target_step: usize) {
        let cp = with_or_insert_runner_branch(&self.params, |b| {
            let found = b
                .timeline
                .checkpoint_at_or_before(target_step)
                .map(|c| (c.step, c.time, c.state))?;
            b.timeline.truncate_after(target_step);
            b.current_time = found.1;
            Some(found)
        });
        let Some((cp_step, _, cp_xyz)) = cp else {
            return;
        };
        if target_step <= self.step_count {
            let drop_n = self.step_count.saturating_sub(cp_step);
            let keep = self.history.len().saturating_sub(drop_n);
            self.history.truncate(keep);
        }
        self.system.state.vec = Vector3::new(cp_xyz[0], cp_xyz[1], cp_xyz[2]);
        self.step_count = cp_step;
        let (dt, _, max_points, mode, custom_flow) = self.sync_params_pub();
        while self.step_count < target_step {
            if !self.advance_tick_pub(dt, max_points, mode, custom_flow.as_deref()) {
                break;
            }
        }
        let step_after = self.step_count;
        let _ = with_existing_runner_branch(&self.params, |b| {
            if let Some(ref twin) = b.shadow {
                if step_after < twin.fork_step {
                    b.clear_branch();
                }
            }
        });
        self.execution_state = SolverExecutionState::Paused;
    }
}
