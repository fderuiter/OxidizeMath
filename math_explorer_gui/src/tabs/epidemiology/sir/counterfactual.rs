//! Counterfactual shadow twin engine integration for [`super::SirTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{SirTool, MAX_SIR_STEPS};
use crate::framework::{
    l2_distance_3d, sanitize_perturbation_delta, sir_peak_infections_averted, BranchControlAction,
    CounterfactualBranchEngine, CounterfactualTransport, DivergenceSnapshot,
    DomainInterventionDelta, PlaybackAction, PlaybackTransport, SeparationTracker,
};
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::epidemiology::compartmental::{SIRModel, SIRState};
use math_explorer::pure_math::analysis::ode::TimeStepper;
use std::sync::Arc;

/// Fixed RK4 integration step (`0.1` days) used by `SirTool`.
pub const SIR_STEP_DT: f64 = 0.1;

/// Compact serializable fork descriptor for `SirTool` (`< 250` bytes JSON).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SirForkDescriptor {
    /// Step index `k > 0` (`t_k = k * 0.1` days) where the intervention fork occurred.
    pub fork_step: usize,
    /// Frozen baseline population `N` on `Branch A`.
    pub baseline_n: f64,
    /// Frozen baseline initial infected `I0` on `Branch A`.
    pub baseline_i0: f64,
    /// Frozen baseline transmission rate `β` on `Branch A`.
    pub baseline_beta: f64,
    /// Frozen baseline recovery rate `γ` on `Branch A`.
    pub baseline_gamma: f64,
    /// Optional state perturbation `δx` transferred from `S(t_k)` to `I(t_k)` on `Branch B`.
    pub delta_x: f64,
}

/// Zero-allocation lockstep `SIRModel` twin stepper safe under `VerifiedAllocator`.
#[derive(Clone, Debug)]
pub struct SirTwinStepper {
    /// Unperturbed baseline `SIRModel` (`Branch A`).
    pub model_a: SIRModel,
    /// Perturbed active `SIRModel` (`Branch B`).
    pub model_b: SIRModel,
    /// Current discrete step index `>= fork_step`.
    pub step_index: usize,
    /// Integration step `dt`.
    pub dt: f64,
    /// Live Euclidean separation and logarithmic slope tracker.
    pub tracker: SeparationTracker,
}

impl SirTwinStepper {
    /// Clones `base_model` at `fork_step` into `Branch A` and configures `Branch B` with `(n_b, beta_b, gamma_b, delta_x)`.
    pub fn fork_at(
        base_model: &SIRModel,
        fork_step: usize,
        dt: f64,
        n_b: f64,
        beta_b: f64,
        gamma_b: f64,
        delta_x: f64,
    ) -> Self {
        let model_a = base_model.clone();
        let mut model_b = base_model.clone();
        model_b.dynamics.n = n_b.max(1.0);
        model_b.dynamics.beta = beta_b.max(0.0);
        model_b.dynamics.gamma = gamma_b.max(0.0);

        let clean_delta = sanitize_perturbation_delta(delta_x);
        if clean_delta > 0.0 {
            let st: &mut SIRState = model_b.get_state_mut();
            let shift = clean_delta.min(st.s.max(0.0));
            st.s -= shift;
            st.i += shift;
        }

        let sa = model_a.state();
        let sb = model_b.state();
        let init_l2 = l2_distance_3d([sa.s, sa.i, sa.r], [sb.s, sb.i, sb.r]);
        let fork_time = fork_step as f64 * dt;

        Self {
            model_a,
            model_b,
            step_index: fork_step,
            dt,
            tracker: SeparationTracker::new(fork_time, init_l2),
        }
    }

    /// Advances `Branch A` and `Branch B` in lockstep by one `dt` increment with zero heap allocations.
    pub fn step_lockstep(&mut self) {
        self.model_a.step(self.dt);
        self.model_b.step(self.dt);
        self.step_index += 1;
        let sa = self.model_a.state();
        let sb = self.model_b.state();
        let l2 = l2_distance_3d([sa.s, sa.i, sa.r], [sb.s, sb.i, sb.r]);
        self.tracker.observe(self.step_index as f64 * self.dt, l2);
    }
}

/// Runtime counterfactual branching and playback state owned by `SirTool`.
pub struct SirCounterfactualState {
    /// Active fork descriptor when a shadow twin is active.
    pub fork: Option<SirForkDescriptor>,
    /// Selected intervention day `t_k` for new forks (`days`).
    pub target_fork_day: f64,
    /// Optional interactive playhead step (`None` evaluates full horizon `0..=steps`).
    pub playhead_step: Option<usize>,
    /// Whether in-thread playback is actively advancing `playhead_step`.
    pub is_playing: bool,
    /// Cached ghost curve points for `Branch A` (`S_A, I_A, R_A`).
    pub s_points_a: Vec<PlotPoint>,
    /// Cached ghost infected curve points for `Branch A`.
    pub i_points_a: Vec<PlotPoint>,
    /// Cached ghost recovered curve points for `Branch A`.
    pub r_points_a: Vec<PlotPoint>,
    /// Decimated ghost display buffers for `Branch A` (`S_A, I_A, R_A`).
    pub(crate) display_s_a: Arc<Vec<PlotPoint>>,
    pub(crate) display_i_a: Arc<Vec<PlotPoint>>,
    pub(crate) display_r_a: Arc<Vec<PlotPoint>>,
    /// Infected value `I(t_k)` at the fork point for anchoring the `"Fork @ t_k"` callout.
    pub fork_infected_y: f64,
    /// Shared framework `CounterfactualBranchEngine` state.
    pub branch_engine: CounterfactualBranchEngine,
}

impl Default for SirCounterfactualState {
    fn default() -> Self {
        Self {
            fork: None,
            target_fork_day: 10.0,
            playhead_step: None,
            is_playing: false,
            s_points_a: Vec::new(),
            i_points_a: Vec::new(),
            r_points_a: Vec::new(),
            display_s_a: Arc::new(Vec::new()),
            display_i_a: Arc::new(Vec::new()),
            display_r_a: Arc::new(Vec::new()),
            fork_infected_y: 0.0,
            branch_engine: CounterfactualBranchEngine::with_delta(0.0),
        }
    }
}

impl SirCounterfactualState {
    pub(super) fn reset_branch_a_points(&mut self) {
        self.s_points_a.clear();
        self.i_points_a.clear();
        self.r_points_a.clear();
        self.display_s_a = Arc::new(Vec::new());
        self.display_i_a = Arc::new(Vec::new());
        self.display_r_a = Arc::new(Vec::new());
    }
}

impl SirTool {
    pub(super) fn max_horizon_steps(&self) -> usize {
        if !self.duration.is_finite() || self.duration <= 0.0 {
            return 0;
        }
        ((self.duration / SIR_STEP_DT).round() as usize).min(MAX_SIR_STEPS)
    }

    pub(super) fn effective_end_step(&self) -> usize {
        let total = self.max_horizon_steps();
        self.cf.playhead_step.unwrap_or(total).min(total)
    }

    /// Forks a counterfactual shadow twin at `fork_day > 0` (`t_k` in days).
    pub fn fork_at_day(&mut self, fork_day: f64) -> bool {
        let total = self.max_horizon_steps();
        if total == 0 || !fork_day.is_finite() || fork_day <= 0.0 {
            return false;
        }
        let step = ((fork_day / SIR_STEP_DT).round() as usize).clamp(1, total);
        self.cf.target_fork_day = step as f64 * SIR_STEP_DT;
        self.cf.fork = Some(SirForkDescriptor {
            fork_step: step,
            baseline_n: self.n.max(1.0),
            baseline_i0: self.i0.clamp(1.0, self.n.max(1.0)),
            baseline_beta: self.beta.max(0.0),
            baseline_gamma: self.gamma.max(0.0),
            delta_x: self.cf.branch_engine.effective_delta(),
        });
        self.recalculate();
        true
    }

    /// Forks at the current interactive playhead (if `> 0`) or `target_fork_day`.
    pub fn fork_here(&mut self) -> bool {
        let day = self
            .cf
            .playhead_step
            .filter(|&s| s > 0)
            .map(|s| s as f64 * SIR_STEP_DT)
            .unwrap_or(self.cf.target_fork_day);
        self.fork_at_day(day)
    }

    /// Swaps baseline `Branch A` parameters with active `Branch B` parameters.
    pub fn swap_branches(&mut self) -> bool {
        let Some(ref mut desc) = self.cf.fork else {
            return false;
        };
        std::mem::swap(&mut self.n, &mut desc.baseline_n);
        std::mem::swap(&mut self.i0, &mut desc.baseline_i0);
        std::mem::swap(&mut self.beta, &mut desc.baseline_beta);
        std::mem::swap(&mut self.gamma, &mut desc.baseline_gamma);
        self.recalculate();
        true
    }

    /// Clears the active counterfactual fork while retaining active `Branch B` parameters.
    pub fn clear_fork(&mut self) -> bool {
        let had = self.cf.fork.take().is_some();
        self.cf.reset_branch_a_points();
        self.cf.branch_engine.clear();
        if had {
            self.recalculate();
        }
        had
    }

    /// Resets playback and clears any active counterfactual fork.
    pub fn reset_playback_and_fork(&mut self) {
        self.cf.is_playing = false;
        self.cf.playhead_step = None;
        let _ = self.clear_fork();
        self.recalculate();
    }

    /// Advances interactive lockstep playback by `count` steps (`Δt = 0.1` days each).
    pub fn step_lockstep(&mut self, count: usize) {
        let total = self.max_horizon_steps();
        if total == 0 {
            return;
        }
        let start = self
            .cf
            .playhead_step
            .unwrap_or_else(|| self.cf.fork.map_or(0, |f| f.fork_step));
        let next = start.saturating_add(count).min(total);
        self.cf.playhead_step = Some(next);
        if next >= total {
            self.cf.is_playing = false;
        }
        self.recalculate();
    }

    pub(super) fn refresh_display_polylines(&mut self) {
        let cap = crate::framework::MAX_DISPLAY_POLYLINE_POINTS;
        self.display_s = crate::framework::decimate_plot_points_minmax(&self.s_points, cap);
        self.display_i = crate::framework::decimate_plot_points_minmax(&self.i_points, cap);
        self.display_r = crate::framework::decimate_plot_points_minmax(&self.r_points, cap);
        if self.cf.fork.is_some() {
            self.cf.display_s_a =
                crate::framework::decimate_plot_points_minmax(&self.cf.s_points_a, cap);
            self.cf.display_i_a =
                crate::framework::decimate_plot_points_minmax(&self.cf.i_points_a, cap);
            self.cf.display_r_a =
                crate::framework::decimate_plot_points_minmax(&self.cf.r_points_a, cap);
        }
    }

    pub(super) fn recalculate_forked(&mut self, desc: SirForkDescriptor, end_step: usize) {
        self.cf.reset_branch_a_points();

        let base_n = desc.baseline_n.max(1.0);
        let base_i0 = desc.baseline_i0.clamp(0.0, base_n);
        let Ok(mut pre_model) = SIRModel::new(
            base_n,
            base_i0,
            desc.baseline_beta.max(0.0),
            desc.baseline_gamma.max(0.0),
        ) else {
            self.cf.branch_engine.clear();
            return;
        };

        let fork_step = desc.fork_step.clamp(1, self.max_horizon_steps().max(1));
        self.trace_sir_pre_fork(&mut pre_model, fork_step, end_step);
        self.cf.fork_infected_y = pre_model.state().i;

        let mut twin = SirTwinStepper::fork_at(
            &pre_model,
            fork_step,
            SIR_STEP_DT,
            self.n,
            self.beta,
            self.gamma,
            desc.delta_x,
        );
        self.trace_sir_post_fork(&mut twin, fork_step, end_step);
        self.refresh_display_polylines();
        self.sync_sir_divergence_snapshot(&twin, desc, fork_step, end_step);
    }

    fn trace_sir_pre_fork(&mut self, pre_model: &mut SIRModel, fork_step: usize, end_step: usize) {
        let pre_end = end_step.min(fork_step);
        for idx in 0..=pre_end {
            let t = idx as f64 * SIR_STEP_DT;
            let st = pre_model.state();
            self.s_points.push(PlotPoint::new(t, st.s));
            self.i_points.push(PlotPoint::new(t, st.i));
            self.r_points.push(PlotPoint::new(t, st.r));
            self.cf.s_points_a.push(PlotPoint::new(t, st.s));
            self.cf.i_points_a.push(PlotPoint::new(t, st.i));
            self.cf.r_points_a.push(PlotPoint::new(t, st.r));
            if idx < fork_step {
                pre_model.step(SIR_STEP_DT);
            }
        }
    }

    fn trace_sir_post_fork(&mut self, twin: &mut SirTwinStepper, fork_step: usize, end_step: usize) {
        if end_step >= fork_step {
            let sb0 = twin.model_b.state();
            if let Some(last_s) = self.s_points.last_mut() {
                last_s.y = sb0.s;
            }
            if let Some(last_i) = self.i_points.last_mut() {
                last_i.y = sb0.i;
            }
            if let Some(last_r) = self.r_points.last_mut() {
                last_r.y = sb0.r;
            }
        }
        for _ in (fork_step + 1)..=end_step {
            twin.step_lockstep();
            let t = twin.step_index as f64 * SIR_STEP_DT;
            let sa = twin.model_a.state();
            let sb = twin.model_b.state();
            self.cf.s_points_a.push(PlotPoint::new(t, sa.s));
            self.cf.i_points_a.push(PlotPoint::new(t, sa.i));
            self.cf.r_points_a.push(PlotPoint::new(t, sa.r));
            self.s_points.push(PlotPoint::new(t, sb.s));
            self.i_points.push(PlotPoint::new(t, sb.i));
            self.r_points.push(PlotPoint::new(t, sb.r));
        }
    }

    fn sync_sir_divergence_snapshot(
        &mut self,
        twin: &SirTwinStepper,
        desc: SirForkDescriptor,
        fork_step: usize,
        end_step: usize,
    ) {
        let ia: Vec<[f64; 2]> = self.cf.i_points_a.iter().map(|p| [p.x, p.y]).collect();
        let ib: Vec<[f64; 2]> = self.i_points.iter().map(|p| [p.x, p.y]).collect();
        let delta_i_peak = sir_peak_infections_averted(&ia, &ib);
        let snap = DivergenceSnapshot {
            fork_step,
            fork_time: fork_step as f64 * SIR_STEP_DT,
            current_time: end_step as f64 * SIR_STEP_DT,
            delta_x: desc.delta_x,
            l2_separation: twin.tracker.l2_separation(),
            log_separation_slope: twin.tracker.log_separation_slope(),
            peak_separation: twin.tracker.peak_separation(),
            primary_delta: Some(DomainInterventionDelta::new(
                "ΔI_peak",
                format!("{delta_i_peak:+.2} averted"),
                delta_i_peak,
                "Peak infections averted (max I_A - max I_B) by the Branch B intervention",
            )),
            secondary_delta: None,
        };
        self.cf.branch_engine.sync_snapshot(Some(snap));
    }

    pub(super) fn render_counterfactual_bar(&mut self, ui: &mut egui::Ui) {
        let total = self.max_horizon_steps();
        let at_end = self.cf.playhead_step.is_some_and(|s| s >= total);
        if self.cf.is_playing && !at_end {
            self.step_lockstep(5);
            ui.ctx().request_repaint();
        }

        ui.horizontal_wrapped(|ui| {
            let can_reset = self.cf.playhead_step.is_some() || self.cf.fork.is_some();
            if let Some(action) = PlaybackTransport::in_thread(
                "sir_playback_transport",
                "SIR Model",
                self.cf.is_playing,
            )
            .can_step(!at_end)
            .can_reset(can_reset)
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => {
                        if self.cf.playhead_step.is_none() || at_end {
                            self.cf.playhead_step = Some(self.cf.fork.map_or(0, |f| f.fork_step));
                        }
                        self.cf.is_playing = true;
                    }
                    PlaybackAction::Pause => self.cf.is_playing = false,
                    PlaybackAction::Step => {
                        self.cf.is_playing = false;
                        self.step_lockstep(1);
                    }
                    PlaybackAction::Reset => self.reset_playback_and_fork(),
                }
            }

            let max_day = self.duration.clamp(1.0, 365.0);
            self.cf.target_fork_day = self.cf.target_fork_day.clamp(0.1, max_day);
            if ui
                .add(
                    egui::Slider::new(&mut self.cf.target_fork_day, 0.1..=max_day)
                        .text("Fork Day (t_k)"),
                )
                .changed()
                && self.cf.fork.is_some()
            {
                let day = self.cf.target_fork_day;
                let _ = self.fork_at_day(day);
            }
        });

        let is_forked = self.cf.fork.is_some();
        if let Some(cf_action) =
            CounterfactualTransport::new(&mut self.cf.branch_engine, total > 0, is_forked).show(ui)
        {
            let _ = self.apply_sir_branch_action(cf_action);
        }
    }

    pub(super) fn apply_sir_branch_action(&mut self, action: BranchControlAction) -> bool {
        match action {
            BranchControlAction::ForkHere => self.fork_here(),
            BranchControlAction::SwapBranches => self.swap_branches(),
            BranchControlAction::ClearFork => self.clear_fork(),
            BranchControlAction::SetPerturbation(delta) => {
                self.cf.branch_engine.set_effective_delta(delta);
                if let Some(ref mut desc) = self.cf.fork {
                    desc.delta_x = self.cf.branch_engine.effective_delta();
                    self.recalculate();
                }
                true
            }
        }
    }
}
