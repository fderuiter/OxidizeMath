//! Timeline ring-buffer checkpointing and zero-allocation Bloch shadow-twin integration
//! for [`super::runner::MriRunner`] and [`super::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::{MriRunner, MriSharedParams};
use crate::async_sim::SimCommand;
use crate::framework::{
    compute_slice_divergence, BranchDivergenceAccumulator, BranchDivergenceMetrics,
    TimelineRingBuffer,
};
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

/// Maximum checkpoints stored in `MriTool` / `MriRunner` (`4096 * 40 B ≈ 160 KiB << 2 MiB`).
pub const MRI_RING_SLOTS: usize = 4096;
/// Preallocated capacity for shadow-twin magnetization history so inner RK4 ticks never allocate.
pub const MAX_MRI_SHADOW_HISTORY: usize = 10_000;

/// Worker-to-UI snapshot of timeline bounds, divergence metrics, and shadow Bloch history.
#[derive(Clone, Default)]
pub struct MriBranchSnapshot {
    /// Current Bloch integration tick count $k$.
    pub step_count: usize,
    /// Earliest retained checkpoint step index in the ring buffer.
    pub min_step: usize,
    /// Latest retained checkpoint step index in the ring buffer.
    pub max_step: usize,
    /// Current simulated relaxation time $t$ in seconds.
    pub current_time: f64,
    /// Whether an unperturbed Bloch shadow twin is currently active.
    pub fork_active: bool,
    /// Step index $k$ at which the shadow twin was forked.
    pub fork_step: Option<usize>,
    /// Simulated relaxation time $t_k$ (seconds) at which the shadow twin was forked.
    pub fork_time: Option<f64>,
    /// Summary of baseline $(T_1, T_2, \mathbf{B})$ parameters frozen on the shadow twin.
    pub baseline_summary: String,
    /// Live divergence metrics between counterfactual and baseline magnetization vectors.
    pub divergence: BranchDivergenceMetrics,
    /// Time-stamped $(t, \mathbf{M}_{\text{shadow}}(t))$ magnetization trajectory.
    pub shadow_history: VecDeque<(f64, Vector3<f64>)>,
}

/// Unperturbed Bloch relaxation shadow twin forked at $(k, t_k)$.
pub struct MriShadowTwin {
    pub(crate) simulator: BlochSimulator,
    pub(crate) solver: RungeKutta4<Vector3<f64>>,
    pub(crate) current_time: f64,
    pub(crate) baseline_summary: String,
    pub(crate) fork_step: usize,
    pub(crate) fork_time: f64,
    pub(crate) history: VecDeque<(f64, Vector3<f64>)>,
    pub(crate) divergence: BranchDivergenceAccumulator,
}

impl MriShadowTwin {
    fn step_rotating_frame_in_place(&mut self, dt: f64) -> Vector3<f64> {
        super::counterfactual::step_bloch_rotating_frame(
            &mut self.simulator,
            &mut self.solver,
            dt,
            0.0,
        );
        self.current_time += dt;
        self.simulator.magnetization
    }
}

/// Timeline ring-buffer and lockstep Bloch shadow-twin runtime.
pub struct MriBranchRuntime {
    pub(crate) step_count: usize,
    pub(crate) current_time: f64,
    pub(crate) timeline: TimelineRingBuffer<[f64; 3]>,
    pub(crate) shadow: Option<MriShadowTwin>,
    pub(crate) worker_view: Option<MriBranchSnapshot>,
}

impl Default for MriBranchRuntime {
    fn default() -> Self {
        Self::with_initial_state(Vector3::new(1.0, 0.0, 0.0))
    }
}

impl MriBranchRuntime {
    /// Creates a preallocated MRI timeline runtime seeded at $t = 0$.
    #[must_use]
    pub fn with_initial_state(initial_m: Vector3<f64>) -> Self {
        let mut timeline =
            TimelineRingBuffer::new_default(std::mem::size_of::<[f64; 3]>(), MRI_RING_SLOTS, 1);
        timeline.push_checkpoint(0, 0.0, [initial_m.x, initial_m.y, initial_m.z]);
        Self {
            step_count: 0,
            current_time: 0.0,
            timeline,
            shadow: None,
            worker_view: None,
        }
    }

    pub(crate) fn reset(&mut self, initial_m: Vector3<f64>) {
        self.step_count = 0;
        self.current_time = 0.0;
        self.shadow = None;
        self.worker_view = None;
        self.timeline.clear();
        self.timeline
            .push_checkpoint(0, 0.0, [initial_m.x, initial_m.y, initial_m.z]);
    }

    /// Zero-allocation inner lockstep Bloch step for `MriRunner` and `VerifiedEngine::run_verified`.
    pub fn on_step_in_place(
        &mut self,
        dt: f64,
        max_history: usize,
        primary_time: f64,
        primary_m: Vector3<f64>,
    ) -> (f64, f64) {
        self.step_count += 1;
        self.current_time = primary_time;
        let mut div_pair = (0.0, 0.0);
        if let Some(ref mut twin) = self.shadow {
            let sm = twin.step_rotating_frame_in_place(dt);
            let cap = max_history.clamp(1, MAX_MRI_SHADOW_HISTORY);
            while twin.history.len() >= cap {
                twin.history.pop_front();
            }
            if twin.history.len() < twin.history.capacity() {
                twin.history.push_back((twin.current_time, sm));
            }
            let (l2, peak) = compute_slice_divergence(
                &[primary_m.x, primary_m.y, primary_m.z],
                &[sm.x, sm.y, sm.z],
            );
            twin.divergence.record_step(dt, l2, peak);
            div_pair = (l2, peak);
        }
        if self.timeline.should_record_step(self.step_count) {
            self.timeline.push_checkpoint(
                self.step_count,
                primary_time,
                [primary_m.x, primary_m.y, primary_m.z],
            );
        }
        div_pair
    }

    pub(crate) fn fork_at(
        &mut self,
        simulator: &BlochSimulator,
        current_time: f64,
        primary_history: &VecDeque<(f64, Vector3<f64>)>,
    ) {
        let m = simulator.magnetization;
        let b_lab = simulator.b_field;
        let mut twin_sim = BlochSimulator::new(m, simulator.m0);
        twin_sim.set_relaxation(simulator.t1, simulator.t2);
        twin_sim.set_b_field(b_lab);
        let mut shadow_hist = VecDeque::with_capacity(MAX_MRI_SHADOW_HISTORY);
        if primary_history.is_empty() {
            shadow_hist.push_back((current_time, m));
        } else {
            for &pt in primary_history.iter().take(MAX_MRI_SHADOW_HISTORY) {
                shadow_hist.push_back(pt);
            }
        }
        let summary = format!(
            "T1={:.3}s, T2={:.3}s, B=({:.3},{:.3},{:.3})T",
            simulator.t1, simulator.t2, b_lab.x, b_lab.y, b_lab.z
        );
        self.shadow = Some(MriShadowTwin {
            simulator: twin_sim,
            solver: RungeKutta4::new(&m),
            current_time,
            baseline_summary: summary,
            fork_step: self.step_count,
            fork_time: current_time,
            history: shadow_hist,
            divergence: BranchDivergenceAccumulator::default(),
        });
        self.worker_view = None;
    }

    pub(crate) fn clear_branch(&mut self) {
        self.shadow = None;
        self.worker_view = None;
    }

    pub(crate) fn to_snapshot(&self) -> MriBranchSnapshot {
        let (min_s, max_s) = self
            .timeline
            .step_bounds()
            .unwrap_or((self.step_count, self.step_count));
        if let Some(ref twin) = self.shadow {
            MriBranchSnapshot {
                step_count: self.step_count,
                min_step: min_s,
                max_step: max_s.max(self.step_count),
                current_time: self.current_time,
                fork_active: true,
                fork_step: Some(twin.fork_step),
                fork_time: Some(twin.fork_time),
                baseline_summary: twin.baseline_summary.clone(),
                divergence: twin.divergence.metrics(),
                shadow_history: twin.history.clone(),
            }
        } else {
            MriBranchSnapshot {
                step_count: self.step_count,
                min_step: min_s,
                max_step: max_s.max(self.step_count),
                current_time: self.current_time,
                ..MriBranchSnapshot::default()
            }
        }
    }

    pub(crate) fn effective_view(&self) -> MriBranchSnapshot {
        if let Some(ref wv) = self.worker_view {
            wv.clone()
        } else {
            self.to_snapshot()
        }
    }
}

struct MriRunnerSlot {
    key: usize,
    weak: Weak<RwLock<MriSharedParams>>,
    branch: MriBranchRuntime,
}

static MRI_RUNNER_SLOTS: OnceLock<Mutex<Vec<MriRunnerSlot>>> = OnceLock::new();

fn mri_slots_mutex() -> &'static Mutex<Vec<MriRunnerSlot>> {
    MRI_RUNNER_SLOTS.get_or_init(|| Mutex::new(Vec::with_capacity(32)))
}

pub(crate) fn register_mri_runner_branch(
    params: &Arc<RwLock<MriSharedParams>>,
    initial_m: Vector3<f64>,
) {
    let key = Arc::as_ptr(params) as usize;
    let mut guard = mri_slots_mutex().lock().unwrap_or_else(|e| e.into_inner());
    guard.retain(|s| s.weak.strong_count() > 0 && s.key != key);
    guard.push(MriRunnerSlot {
        key,
        weak: Arc::downgrade(params),
        branch: MriBranchRuntime::with_initial_state(initial_m),
    });
}

pub(crate) fn with_existing_mri_runner_branch<R>(
    params: &Arc<RwLock<MriSharedParams>>,
    f: impl FnOnce(&mut MriBranchRuntime) -> R,
) -> Option<R> {
    let lock = MRI_RUNNER_SLOTS.get()?;
    let key = Arc::as_ptr(params) as usize;
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    let slot = guard.iter_mut().find(|s| s.key == key)?;
    Some(f(&mut slot.branch))
}

fn with_or_insert_mri_runner_branch<R>(
    params: &Arc<RwLock<MriSharedParams>>,
    initial_m: Vector3<f64>,
    f: impl FnOnce(&mut MriBranchRuntime) -> R,
) -> R {
    let key = Arc::as_ptr(params) as usize;
    let mut guard = mri_slots_mutex().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = guard.iter().position(|s| s.key == key) {
        return f(&mut guard[pos].branch);
    }
    guard.retain(|s| s.weak.strong_count() > 0);
    guard.push(MriRunnerSlot {
        key,
        weak: Arc::downgrade(params),
        branch: MriBranchRuntime::with_initial_state(initial_m),
    });
    let last = guard.len() - 1;
    f(&mut guard[last].branch)
}

impl MriRunner {
    pub(crate) fn snapshot_branch(&self) -> MriBranchSnapshot {
        with_existing_mri_runner_branch(&self.params, |b| b.to_snapshot()).unwrap_or(
            MriBranchSnapshot {
                step_count: self.history.len().saturating_sub(1),
                min_step: 0,
                max_step: self.history.len().saturating_sub(1),
                current_time: self.current_time,
                ..MriBranchSnapshot::default()
            },
        )
    }

    pub(crate) fn handle_branch_command(&mut self, cmd: &SimCommand) -> bool {
        let init_m = Vector3::new(self.simulator.m0, 0.0, 0.0);
        match cmd {
            SimCommand::ForkBranch => {
                let _ = self.sync_params_pub();
                let sim = &self.simulator;
                let t = self.current_time;
                let hist = &self.history;
                with_or_insert_mri_runner_branch(&self.params, init_m, |b| {
                    b.fork_at(sim, t, hist);
                });
                true
            }
            SimCommand::ClearBranch => {
                with_or_insert_mri_runner_branch(&self.params, init_m, |b| b.clear_branch());
                true
            }
            SimCommand::StepBack(count) => {
                let prev = with_or_insert_mri_runner_branch(&self.params, init_m, |b| {
                    if *count <= 1 {
                        b.timeline
                            .previous_step_before(b.step_count)
                            .unwrap_or_else(|| b.step_count.saturating_sub(1))
                    } else {
                        b.step_count.saturating_sub(*count)
                    }
                });
                self.scrub_runner_to_step(prev);
                true
            }
            SimCommand::ScrubToStep(target) => {
                self.scrub_runner_to_step(*target);
                true
            }
            _ => false,
        }
    }

    pub(crate) fn scrub_runner_to_step(&mut self, target_step: usize) {
        let init_m = Vector3::new(self.simulator.m0, 0.0, 0.0);
        let cp = with_or_insert_mri_runner_branch(&self.params, init_m, |b| {
            let found = b
                .timeline
                .checkpoint_at_or_before(target_step)
                .map(|c| (c.step, c.time, c.state))?;
            let prev_step = b.step_count;
            b.timeline.truncate_after(target_step);
            b.step_count = found.0;
            b.current_time = found.1;
            Some((prev_step, found.0, found.1, found.2))
        });
        let Some((prev_step, cp_step, cp_time, cp_m)) = cp else {
            return;
        };
        if target_step <= prev_step {
            while self
                .history
                .back()
                .is_some_and(|(t, _)| *t > cp_time + 1.0e-12)
            {
                self.history.pop_back();
            }
        }
        self.simulator.magnetization = Vector3::new(cp_m[0], cp_m[1], cp_m[2]);
        self.current_time = cp_time;
        self.step_count = cp_step;
        if cp_step < target_step {
            let _ = self.sync_params_pub();
            self.advance_bloch_ticks_pub(target_step - cp_step);
        }
        let _ = with_existing_mri_runner_branch(&self.params, |b| {
            if let Some(ref twin) = b.shadow {
                if b.step_count < twin.fork_step {
                    b.clear_branch();
                }
            }
        });
    }
}
