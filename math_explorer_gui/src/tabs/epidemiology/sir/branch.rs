//! Timeline ring-buffer checkpointing and lockstep counterfactual branching for [`super::SirTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "branch_ui.rs"]
mod branch_ui;

use super::{SirTool, MAX_SIR_STEPS};
use crate::framework::{
    compute_slice_divergence, decimate_plot_points_minmax, BranchDivergenceAccumulator,
    TimelineRingBuffer, MAX_DISPLAY_POLYLINE_POINTS,
};
use egui_plot::PlotPoint;
use math_explorer::epidemiology::compartmental::{SIRModel, SIRState};
use math_explorer::pure_math::analysis::ode::TimeStepper;
use std::sync::Arc;

/// Fixed integration step size (days) used by [`SirTool`].
pub const SIR_DT: f64 = 0.1;
/// Maximum checkpoints retained in `SirTool`'s ring buffer (`4096 * 40 B ≈ 160 KiB << 2 MiB`).
const SIR_RING_MAX_SLOTS: usize = 4096;

/// Copyable `(S, I, R)` compartment state stored in [`TimelineRingBuffer`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SirCheckpoint {
    /// Susceptible count $S(t_k)$.
    pub s: f64,
    /// Infected count $I(t_k)$.
    pub i: f64,
    /// Recovered count $R(t_k)$.
    pub r: f64,
}

impl From<SIRState> for SirCheckpoint {
    fn from(st: SIRState) -> Self {
        Self {
            s: st.s,
            i: st.i,
            r: st.r,
        }
    }
}

/// Unperturbed baseline SIR twin forked at $(k, t_k)$.
pub struct SirShadowTwin {
    pub(crate) model: SIRModel,
    pub(crate) baseline_summary: String,
    pub(crate) fork_step: usize,
    pub(crate) fork_time: f64,
    pub(crate) fork_state: SirCheckpoint,
    pub(crate) divergence: BranchDivergenceAccumulator,
    pub(crate) s_points: Vec<PlotPoint>,
    pub(crate) i_points: Vec<PlotPoint>,
    pub(crate) r_points: Vec<PlotPoint>,
    pub(crate) display_s: Arc<Vec<PlotPoint>>,
    pub(crate) display_i: Arc<Vec<PlotPoint>>,
    pub(crate) display_r: Arc<Vec<PlotPoint>>,
}

/// In-memory timeline ring buffer and optional shadow twin owned by [`SirTool`].
pub struct SirBranchRuntime {
    pub(crate) live_model: SIRModel,
    pub(crate) current_step: usize,
    pub(crate) is_playing: bool,
    pub(crate) timeline: TimelineRingBuffer<SirCheckpoint>,
    pub(crate) shadow: Option<SirShadowTwin>,
}

impl Default for SirBranchRuntime {
    fn default() -> Self {
        let model = SIRModel::new(1000.0, 10.0, 0.5, 0.1)
            .expect("Default SIRModel parameters must be valid");
        Self {
            live_model: model,
            current_step: 0,
            is_playing: false,
            timeline: TimelineRingBuffer::new_default(
                std::mem::size_of::<SirCheckpoint>(),
                SIR_RING_MAX_SLOTS,
                1,
            ),
            shadow: None,
        }
    }
}

impl SirBranchRuntime {
    /// Zero-allocation inner lockstep step for `VerifiedEngine::run_verified` benchmarks and tests.
    pub fn step_lockstep_in_place(&mut self, dynamics: (f64, f64, f64), dt: f64) -> (f64, f64) {
        self.live_model.dynamics.n = dynamics.0;
        self.live_model.dynamics.beta = dynamics.1;
        self.live_model.dynamics.gamma = dynamics.2;
        self.live_model.step(dt);
        self.current_step = self.current_step.saturating_add(1);
        let t = self.current_step as f64 * dt;
        let p = *self.live_model.get_state();

        let mut div_pair = (0.0, 0.0);
        if let Some(ref mut twin) = self.shadow {
            twin.model.step(dt);
            let s = *twin.model.get_state();
            let (l2, peak) = compute_slice_divergence(&[p.s, p.i, p.r], &[s.s, s.i, s.r]);
            twin.divergence.record_step(dt, l2, peak);
            div_pair = (l2, peak);
        }

        if self.timeline.should_record_step(self.current_step) {
            self.timeline
                .push_checkpoint(self.current_step, t, SirCheckpoint::from(p));
        }
        div_pair
    }
}

impl SirTool {
    pub(super) fn current_dynamics(&self) -> (f64, f64, f64) {
        (self.n.max(1.0), self.beta.max(0.0), self.gamma.max(0.0))
    }

    fn apply_current_dynamics(&mut self) {
        let (n, beta, gamma) = self.current_dynamics();
        self.branch.live_model.dynamics.n = n;
        self.branch.live_model.dynamics.beta = beta;
        self.branch.live_model.dynamics.gamma = gamma;
    }

    pub(super) fn rebuild_timeline_from_recalculate(
        &mut self,
        final_model: SIRModel,
        steps: usize,
    ) {
        let stride = (steps / (SIR_RING_MAX_SLOTS - 1)).max(1);
        self.branch.timeline = TimelineRingBuffer::new_default(
            std::mem::size_of::<SirCheckpoint>(),
            SIR_RING_MAX_SLOTS,
            stride,
        );
        for (k, ((sp, ip), rp)) in self
            .s_points
            .iter()
            .zip(self.i_points.iter())
            .zip(self.r_points.iter())
            .enumerate()
        {
            if self.branch.timeline.should_record_step(k) || k == steps {
                self.branch.timeline.push_checkpoint(
                    k,
                    sp.x,
                    SirCheckpoint {
                        s: sp.y,
                        i: ip.y,
                        r: rp.y,
                    },
                );
            }
        }
        self.branch.live_model = final_model;
        self.branch.current_step = steps;
        self.branch.is_playing = false;
        self.branch.shadow = None;
    }

    fn refresh_all_display_buffers(&mut self) {
        self.series_key = (0, 0, 0, 0, 0, 0);
        self.sync_series_cache();
        let cap = MAX_DISPLAY_POLYLINE_POINTS;
        if let Some(ref mut twin) = self.branch.shadow {
            twin.display_s = decimate_plot_points_minmax(&twin.s_points, cap);
            twin.display_i = decimate_plot_points_minmax(&twin.i_points, cap);
            twin.display_r = decimate_plot_points_minmax(&twin.r_points, cap);
        }
    }

    /// Scrubs `SirTool` to `target_step` ($t_k = k \cdot \Delta t$), restoring exact $(S, I, R)$
    /// and truncating future trajectory history without resetting to $t = 0$.
    pub fn scrub_to_step(&mut self, target_step: usize) {
        let Some((cp_step, cp_state)) = self
            .branch
            .timeline
            .checkpoint_at_or_before(target_step)
            .map(|cp| (cp.step, cp.state))
        else {
            return;
        };
        self.branch.is_playing = false;
        self.apply_current_dynamics();
        *self.branch.live_model.get_state_mut() = SIRState {
            s: cp_state.s,
            i: cp_state.i,
            r: cp_state.r,
        };
        let keep_cp = (cp_step + 1).min(self.s_points.len());
        self.s_points.truncate(keep_cp);
        self.i_points.truncate(keep_cp);
        self.r_points.truncate(keep_cp);
        self.branch.timeline.truncate_after(target_step);
        self.branch.current_step = cp_step;

        while self.branch.current_step < target_step {
            self.branch.live_model.step(SIR_DT);
            self.branch.current_step += 1;
            let t = self.branch.current_step as f64 * SIR_DT;
            let st = *self.branch.live_model.get_state();
            self.s_points.push(PlotPoint::new(t, st.s));
            self.i_points.push(PlotPoint::new(t, st.i));
            self.r_points.push(PlotPoint::new(t, st.r));
            if self
                .branch
                .timeline
                .should_record_step(self.branch.current_step)
            {
                self.branch.timeline.push_checkpoint(
                    self.branch.current_step,
                    t,
                    SirCheckpoint::from(st),
                );
            }
        }
        let exact_len = target_step + 1;
        self.s_points.truncate(exact_len);
        self.i_points.truncate(exact_len);
        self.r_points.truncate(exact_len);
        self.resync_shadow_after_scrub();
        self.refresh_all_display_buffers();
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    fn resync_shadow_after_scrub(&mut self) {
        let cur = self.branch.current_step;
        let Some(ref mut twin) = self.branch.shadow else {
            return;
        };
        if cur < twin.fork_step {
            self.branch.shadow = None;
            return;
        }
        let prefix = (twin.fork_step + 1).min(twin.s_points.len());
        twin.s_points.truncate(prefix);
        twin.i_points.truncate(prefix);
        twin.r_points.truncate(prefix);
        *twin.model.get_state_mut() = SIRState {
            s: twin.fork_state.s,
            i: twin.fork_state.i,
            r: twin.fork_state.r,
        };
        twin.divergence.reset();
        for step_idx in (twin.fork_step + 1)..=cur {
            twin.model.step(SIR_DT);
            let t = step_idx as f64 * SIR_DT;
            let s = *twin.model.get_state();
            twin.s_points.push(PlotPoint::new(t, s.s));
            twin.i_points.push(PlotPoint::new(t, s.i));
            twin.r_points.push(PlotPoint::new(t, s.r));
            if step_idx < self.s_points.len() {
                let p = [
                    self.s_points[step_idx].y,
                    self.i_points[step_idx].y,
                    self.r_points[step_idx].y,
                ];
                let (l2, pk) = compute_slice_divergence(&p, &[s.s, s.i, s.r]);
                twin.divergence.record_step(SIR_DT, l2, pk);
            }
        }
    }

    /// Forks an unperturbed shadow twin at the current step $t_k$.
    pub fn fork_branch(&mut self) {
        let k = self.branch.current_step;
        let t_k = k as f64 * SIR_DT;
        self.apply_current_dynamics();
        let st = *self.branch.live_model.get_state();
        let cap = MAX_DISPLAY_POLYLINE_POINTS;
        let shadow = SirShadowTwin {
            model: self.branch.live_model.clone(),
            baseline_summary: self.param_summary(),
            fork_step: k,
            fork_time: t_k,
            fork_state: SirCheckpoint::from(st),
            divergence: BranchDivergenceAccumulator::default(),
            s_points: self.s_points.clone(),
            i_points: self.i_points.clone(),
            r_points: self.r_points.clone(),
            display_s: decimate_plot_points_minmax(&self.s_points, cap),
            display_i: decimate_plot_points_minmax(&self.i_points, cap),
            display_r: decimate_plot_points_minmax(&self.r_points, cap),
        };
        self.branch.shadow = Some(shadow);
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Clears any active shadow-twin branch overlay.
    pub fn clear_branch(&mut self) {
        self.branch.shadow = None;
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Advances primary and shadow SIR models in lockstep by `count` steps (`dt = 0.1`).
    pub(super) fn step_timeline_branch_lockstep(&mut self, count: usize) {
        let dyn_now = self.current_dynamics();
        for _ in 0..count {
            if self.branch.current_step >= MAX_SIR_STEPS {
                self.branch.is_playing = false;
                break;
            }
            self.branch.step_lockstep_in_place(dyn_now, SIR_DT);
            let t = self.branch.current_step as f64 * SIR_DT;
            let p = *self.branch.live_model.get_state();
            self.s_points.push(PlotPoint::new(t, p.s));
            self.i_points.push(PlotPoint::new(t, p.i));
            self.r_points.push(PlotPoint::new(t, p.r));
            if let Some(ref mut twin) = self.branch.shadow {
                let s = *twin.model.get_state();
                twin.s_points.push(PlotPoint::new(t, s.s));
                twin.i_points.push(PlotPoint::new(t, s.i));
                twin.r_points.push(PlotPoint::new(t, s.r));
            }
        }
        self.refresh_all_display_buffers();
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Re-integrates counterfactual and baseline curves from $t_k$ to `duration` when parameters change while forked.
    pub(super) fn recompute_counterfactual_from_fork(&mut self, sweep_active: bool) {
        let Some((fork_step, fork_state)) = self
            .branch
            .shadow
            .as_ref()
            .map(|tw| (tw.fork_step, tw.fork_state))
        else {
            self.record_history_and_recalculate(sweep_active);
            return;
        };
        let horizon_steps = ((self.duration.max(0.0) / SIR_DT) as usize)
            .min(MAX_SIR_STEPS)
            .max(fork_step);
        let prefix = (fork_step + 1).min(self.s_points.len());
        self.s_points.truncate(prefix);
        self.i_points.truncate(prefix);
        self.r_points.truncate(prefix);
        self.branch.timeline.truncate_after(fork_step);
        self.branch.current_step = fork_step;
        *self.branch.live_model.get_state_mut() = SIRState {
            s: fork_state.s,
            i: fork_state.i,
            r: fork_state.r,
        };
        if let Some(ref mut twin) = self.branch.shadow {
            twin.s_points.truncate(prefix);
            twin.i_points.truncate(prefix);
            twin.r_points.truncate(prefix);
            *twin.model.get_state_mut() = SIRState {
                s: fork_state.s,
                i: fork_state.i,
                r: fork_state.r,
            };
            twin.divergence.reset();
        }
        let delta_steps = horizon_steps.saturating_sub(fork_step);
        self.step_timeline_branch_lockstep(delta_steps);
        self.last_param_summary = self.param_summary();
    }
}
