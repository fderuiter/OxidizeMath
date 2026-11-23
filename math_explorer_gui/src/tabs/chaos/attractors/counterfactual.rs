//! Counterfactual shadow twin engine integration for [`super::AttractorPlotter`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    AttractorMode, AttractorPlotter, AttractorRunner, CompiledFlowJob, SolverExecutionState,
};
use crate::app::SemanticPalette;
use crate::async_sim::SimCommand;
use crate::framework::{
    l2_distance_3d, sanitize_perturbation_delta, DivergenceSnapshot, Scene3DBuilder,
    SeparationTracker,
};
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use nalgebra::{Point3, Vector3};
use std::collections::VecDeque;
use std::sync::Arc;

/// Pre-allocated trajectory ring-buffer capacity so locked `step()` loops never reallocate.
pub const ATTRACTOR_RING_CAPACITY: usize = 10_000;

/// Compact serializable fork descriptor fitting inside `4 KiB` URL state fragments.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttractorForkDescriptor {
    /// Step index `k > 0` where `(t_k, x(t_k))` was cloned.
    pub fork_step: usize,
    /// Number of lockstep post-fork steps evolved since `fork_step`.
    #[serde(default)]
    pub post_fork_steps: usize,
    /// Frozen Prandtl number `σ` on baseline `Branch A`.
    pub baseline_sigma: f64,
    /// Frozen Rayleigh number `ρ` on baseline `Branch A`.
    pub baseline_rho: f64,
    /// Frozen geometric factor `β` on baseline `Branch A`.
    pub baseline_beta: f64,
    /// State perturbation `δx` applied to `x(t_k)` on `Branch B`.
    pub delta_x: f64,
}

/// Pre-allocated baseline shadow twin (`Branch A`) evolved in lockstep with `AttractorPlotter`.
#[derive(Clone, Debug)]
pub struct AttractorTwinState {
    /// Frozen-parameter Lorenz integrator for baseline `Branch A`.
    pub system_a: LorenzSystem,
    /// Frozen dynamical mode (`Lorenz` or `Custom`) at fork time.
    pub mode_a: AttractorMode,
    /// Frozen compiled 3D custom vector field for `Branch A` when `mode_a == Custom`.
    pub custom_flow_a: Option<Arc<CompiledFlowJob>>,
    /// Pre-allocated trajectory history for `Branch A`.
    pub history_a: VecDeque<Vector3<f64>>,
    /// 3D state coordinate where the fork occurred (`Fork @ t_k`).
    pub fork_point: Vector3<f64>,
    /// Step index `k` at fork time.
    pub fork_step: usize,
    /// Perturbation `δx` applied on `Branch B` at fork time.
    pub delta_x: f64,
    /// Live Euclidean separation and finite-time Lyapunov slope tracker.
    pub tracker: SeparationTracker,
}

impl AttractorTwinState {
    /// Clones `(step_count, state_vec, params)` into a pre-allocated shadow twin and nudges `state_b`.
    pub fn fork_from(
        step_count: usize,
        dt: f64,
        system_b: &mut LorenzSystem,
        history_b: &VecDeque<Vector3<f64>>,
        delta_x: f64,
    ) -> Self {
        let clean_delta = sanitize_perturbation_delta(delta_x);
        let fork_point = system_b.state.vec;
        let system_a = LorenzBuilder::new()
            .sigma(system_b.sigma)
            .rho(system_b.rho)
            .beta(system_b.beta)
            .build(LorenzState::new(fork_point.x, fork_point.y, fork_point.z));

        let mut history_a = VecDeque::with_capacity(ATTRACTOR_RING_CAPACITY);
        for &pt in history_b {
            if history_a.len() >= ATTRACTOR_RING_CAPACITY {
                history_a.pop_front();
            }
            history_a.push_back(pt);
        }

        system_b.state.vec.x += clean_delta;
        let fork_time = step_count as f64 * if dt.is_finite() && dt > 0.0 { dt } else { 0.01 };
        let init_l2 = l2_distance_3d(
            [fork_point.x, fork_point.y, fork_point.z],
            [
                system_b.state.vec.x,
                system_b.state.vec.y,
                system_b.state.vec.z,
            ],
        );

        Self {
            system_a,
            mode_a: AttractorMode::Lorenz,
            custom_flow_a: None,
            history_a,
            fork_point,
            fork_step: step_count,
            delta_x: clean_delta,
            tracker: SeparationTracker::new(fork_time, init_l2),
        }
    }

    /// Attaches custom 3D flow configuration when forking in [`AttractorMode::Custom`].
    #[must_use]
    pub fn with_flow(
        mut self,
        mode_a: AttractorMode,
        custom_flow_a: Option<Arc<CompiledFlowJob>>,
    ) -> Self {
        self.mode_a = mode_a;
        self.custom_flow_a = custom_flow_a;
        self
    }

    /// Advances `system_a` by one `dt` step without heap allocation.
    pub fn step_branch_a(&mut self, dt: f64, max_points: usize) -> bool {
        let v = self.system_a.state.vec;
        let lorenz_ok = self.mode_a != AttractorMode::Lorenz
            || (self.system_a.sigma.is_finite()
                && self.system_a.rho.is_finite()
                && self.system_a.beta.is_finite());
        if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() || !lorenz_ok || !dt.is_finite()
        {
            return false;
        }
        match self.mode_a {
            AttractorMode::Lorenz => self.system_a.step(dt),
            AttractorMode::Custom => {
                if let Some(ref job) = self.custom_flow_a {
                    self.system_a.state.vec = job.rk4_step(v, dt);
                }
            }
        }
        let next = self.system_a.state.vec;
        if !next.x.is_finite() || !next.y.is_finite() || !next.z.is_finite() {
            return false;
        }
        let cap = max_points.clamp(1, ATTRACTOR_RING_CAPACITY);
        if self.history_a.len() >= cap {
            self.history_a.pop_front();
        }
        self.history_a.push_back(next);
        true
    }

    /// Updates separation metrics after both branches have stepped to `step_count`.
    pub fn observe_lockstep(&mut self, step_count: usize, dt: f64, vec_b: Vector3<f64>) {
        let vec_a = self.system_a.state.vec;
        let l2 = l2_distance_3d([vec_a.x, vec_a.y, vec_a.z], [vec_b.x, vec_b.y, vec_b.z]);
        let current_time = step_count as f64 * if dt.is_finite() && dt > 0.0 { dt } else { 0.01 };
        self.tracker.observe(current_time, l2);
    }

    /// Builds a [`DivergenceSnapshot`] for `SimulationFramework` and `DivergenceStrip`.
    #[must_use]
    pub fn snapshot(&self, step_count: usize, dt: f64) -> DivergenceSnapshot {
        let current_time = step_count as f64 * if dt.is_finite() && dt > 0.0 { dt } else { 0.01 };
        DivergenceSnapshot {
            fork_step: self.fork_step,
            fork_time: self.tracker.fork_time(),
            current_time,
            delta_x: self.delta_x,
            l2_separation: self.tracker.l2_separation(),
            log_separation_slope: self.tracker.log_separation_slope(),
            peak_separation: self.tracker.peak_separation(),
            primary_delta: None,
            secondary_delta: None,
        }
    }

    /// Builds a compact descriptor for preset/URL serialization.
    #[must_use]
    pub fn to_descriptor(&self, current_step: usize) -> AttractorForkDescriptor {
        AttractorForkDescriptor {
            fork_step: self.fork_step,
            post_fork_steps: current_step.saturating_sub(self.fork_step),
            baseline_sigma: self.system_a.sigma,
            baseline_rho: self.system_a.rho,
            baseline_beta: self.system_a.beta,
            delta_x: self.delta_x,
        }
    }
}

impl AttractorRunner {
    pub(super) fn handle_custom_command(&mut self, cmd: &str) {
        if let Some(rest) = cmd.strip_prefix("fork:") {
            if self.step_count > 0
                && !matches!(self.execution_state, SolverExecutionState::Diverged { .. })
            {
                let delta = rest.parse::<f64>().unwrap_or(0.0);
                let (dt, mode, flow) = {
                    let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                    (p.dt, p.mode, p.custom_flow.clone())
                };
                self.twin = Some(
                    AttractorTwinState::fork_from(
                        self.step_count,
                        dt,
                        &mut self.system,
                        &self.history,
                        delta,
                    )
                    .with_flow(mode, flow),
                );
            }
        } else if cmd == "swap" {
            self.swap_twin_branches();
        } else if cmd == "clear_fork" {
            self.twin = None;
        }
    }

    pub(super) fn swap_twin_branches(&mut self) {
        if let Some(ref mut twin) = self.twin {
            std::mem::swap(&mut self.system, &mut twin.system_a);
            std::mem::swap(&mut self.history, &mut twin.history_a);
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.sigma = self.system.sigma;
            p.rho = self.system.rho;
            p.beta = self.system.beta;
        }
    }
}

impl AttractorPlotter {
    /// Forks a counterfactual shadow twin at the current `step_count > 0`.
    pub fn fork_here(&mut self) -> bool {
        if self.step_count == 0 || self.diverged() {
            return false;
        }
        let delta = self.branch_engine.effective_delta();
        let custom_flow = (self.mode == AttractorMode::Custom)
            .then(|| Arc::new(CompiledFlowJob::from_bank(&self.flow_bank)));
        self.twin = Some(
            AttractorTwinState::fork_from(
                self.step_count,
                self.dt,
                &mut self.system,
                &self.history,
                delta,
            )
            .with_flow(self.mode, custom_flow),
        );
        let _ = self
            .worker
            .send_command(SimCommand::Custom(format!("fork:{delta}")));
        self.sync_local_branch_engine();
        true
    }

    /// Swaps baseline `Branch A` and active `Branch B` parameters and trajectories.
    pub fn swap_branches(&mut self) -> bool {
        let Some(ref mut twin) = self.twin else {
            return false;
        };
        std::mem::swap(&mut self.system, &mut twin.system_a);
        std::mem::swap(&mut self.history, &mut twin.history_a);
        self.on_slider_changed();
        let _ = self.worker.send_command(SimCommand::Custom("swap".into()));
        self.sync_local_branch_engine();
        true
    }

    /// Clears the active counterfactual fork while retaining `Branch B`.
    pub fn clear_fork(&mut self) -> bool {
        let had_fork = self.twin.take().is_some();
        self.branch_engine.clear();
        let _ = self
            .worker
            .send_command(SimCommand::Custom("clear_fork".into()));
        had_fork
    }

    pub(super) fn sync_local_branch_engine(&mut self) {
        let snap = self
            .twin
            .as_ref()
            .map(|t| t.snapshot(self.step_count, self.dt));
        self.branch_engine.sync_snapshot(snap);
    }

    fn validate_fork_descriptor(desc: &AttractorForkDescriptor) -> Result<(), String> {
        if desc.fork_step == 0
            || desc.fork_step > 50_000
            || desc.post_fork_steps > 50_000
            || !desc.baseline_sigma.is_finite()
            || !desc.baseline_rho.is_finite()
            || !desc.baseline_beta.is_finite()
            || !desc.delta_x.is_finite()
        {
            return Err("Invalid AttractorPlotter fork descriptor".to_string());
        }
        Ok(())
    }

    fn advance_replay_step(&mut self, cap: usize, custom_job: Option<&CompiledFlowJob>) {
        if let Some(job) = custom_job {
            self.system.state.vec = job.rk4_step(self.system.state.vec, self.dt);
        } else {
            self.system.step(self.dt);
        }
        self.step_count += 1;
        if self.history.len() >= cap {
            self.history.pop_front();
        }
        self.history.push_back(self.system.state.vec);
    }

    /// Deterministically reconstructs a forked twin state from a compact preset/URL descriptor.
    pub(super) fn apply_fork_descriptor(
        &mut self,
        desc: AttractorForkDescriptor,
    ) -> Result<(), String> {
        Self::validate_fork_descriptor(&desc)?;
        let (active_sigma, active_rho, active_beta) =
            (self.system.sigma, self.system.rho, self.system.beta);
        let cap = self.max_points.clamp(1, ATTRACTOR_RING_CAPACITY);
        let custom_flow = (self.mode == AttractorMode::Custom)
            .then(|| Arc::new(CompiledFlowJob::from_bank(&self.flow_bank)));

        self.system = LorenzBuilder::new()
            .sigma(desc.baseline_sigma)
            .rho(desc.baseline_rho)
            .beta(desc.baseline_beta)
            .build(LorenzState::new(1.0, 1.0, 1.0));
        self.history.clear();
        self.step_count = 0;

        for _ in 0..desc.fork_step {
            self.advance_replay_step(cap, custom_flow.as_deref());
        }

        let mut twin = AttractorTwinState::fork_from(
            self.step_count,
            self.dt,
            &mut self.system,
            &self.history,
            desc.delta_x,
        )
        .with_flow(self.mode, custom_flow.clone());
        self.system.sigma = active_sigma;
        self.system.rho = active_rho;
        self.system.beta = active_beta;
        self.branch_engine.set_effective_delta(desc.delta_x);

        for _ in 0..desc.post_fork_steps {
            let _ = twin.step_branch_a(self.dt, cap);
            self.advance_replay_step(cap, custom_flow.as_deref());
            twin.observe_lockstep(self.step_count, self.dt, self.system.state.vec);
        }

        self.twin = Some(twin);
        self.on_slider_changed();
        self.respawn_worker_with_current_state();
        self.sync_local_branch_engine();
        Ok(())
    }

    pub(super) fn respawn_worker_with_current_state(&mut self) {
        let params = std::sync::Arc::clone(&self.params);
        let mut runner = AttractorRunner::from_params(params);
        runner.system = self.system.clone();
        runner.history = self.history.clone();
        runner.step_count = self.step_count;
        runner.execution_state = self.execution_state;
        runner.twin = self.twin.clone();
        self.worker = crate::async_sim::BackgroundWorker::new(runner);
    }

    /// Populates `self.scene_builder` with world `Point3<f64>` attractor trajectory segments.
    #[cfg(test)]
    pub fn populate_scene(&mut self, series_color: egui::Color32) {
        self.populate_scene_colors(series_color, egui::Color32::GRAY, series_color);
    }

    /// Populates `self.scene_builder` using theme-aware semantic colors for active and shadow branches.
    pub(super) fn populate_scene_with_palette(&mut self, palette: &SemanticPalette) {
        self.populate_scene_colors(palette.series_color(0), palette.muted_text, palette.accent);
    }

    fn push_trajectory_series(
        builder: &mut Scene3DBuilder,
        name: &str,
        pts: &[Point3<f64>],
        stroke: egui::Stroke,
        style: LineStyle,
    ) {
        if pts.len() == 1 {
            builder.add_points(name, pts, 4.0_f32, stroke.color, MarkerShape::Circle, true);
        } else if pts.len() > 1 {
            builder.add_polyline(name, pts, stroke, style, true);
        }
    }

    fn populate_twin_scene(
        &mut self,
        series_color: egui::Color32,
        muted_color: egui::Color32,
        accent_color: egui::Color32,
    ) {
        let Some(ref twin) = self.twin else {
            return;
        };
        let mut pts_a: Vec<Point3<f64>> = twin
            .history_a
            .iter()
            .map(|v| Point3::new(v.x, v.y, v.z))
            .collect();
        if pts_a.is_empty() {
            let va = twin.system_a.state.vec;
            pts_a.push(Point3::new(va.x, va.y, va.z));
        }
        Self::push_trajectory_series(
            &mut self.scene_builder,
            "Branch A (Baseline)",
            &pts_a,
            egui::Stroke::new(1.5_f32, muted_color),
            LineStyle::dashed_loose(),
        );
        if self.trajectory_scratch.is_empty() && self.is_finite_state() {
            let vb = self.system.state.vec;
            self.trajectory_scratch.push(Point3::new(vb.x, vb.y, vb.z));
        }
        Self::push_trajectory_series(
            &mut self.scene_builder,
            "Branch B (Active)",
            &self.trajectory_scratch,
            egui::Stroke::new(2.25_f32, series_color),
            LineStyle::Solid,
        );
        let fp = twin.fork_point;
        self.scene_builder.add_points(
            "Fork @ t_k",
            &[Point3::new(fp.x, fp.y, fp.z)],
            5.0_f32,
            accent_color,
            MarkerShape::Circle,
            false,
        );
    }

    fn populate_scene_colors(
        &mut self,
        series_color: egui::Color32,
        muted_color: egui::Color32,
        accent_color: egui::Color32,
    ) {
        let origin_offset = match self.mode {
            AttractorMode::Lorenz => Point3::new(0.0, 0.0, 25.0),
            AttractorMode::Custom => Point3::origin(),
        };
        self.scene_builder.begin(self.camera);
        self.scene_builder.set_origin_offset(origin_offset);
        self.trajectory_scratch.clear();
        self.trajectory_scratch
            .extend(self.history.iter().map(|v| Point3::new(v.x, v.y, v.z)));
        if self.twin.is_some() {
            self.populate_twin_scene(series_color, muted_color, accent_color);
        } else if !self.trajectory_scratch.is_empty() {
            Self::push_trajectory_series(
                &mut self.scene_builder,
                "Trajectory",
                &self.trajectory_scratch,
                egui::Stroke::new(1.5_f32, series_color),
                LineStyle::Solid,
            );
        } else if self.is_finite_state() {
            let v = self.system.state.vec;
            self.scene_builder.add_points(
                "Initial State",
                &[Point3::new(v.x, v.y, v.z)],
                4.0_f32,
                series_color,
                MarkerShape::Circle,
                true,
            );
        }
    }
}
