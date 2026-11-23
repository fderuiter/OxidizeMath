//! UI transport, 3D shadow-twin scene rendering, and timeline hooks for [`super::AttractorPlotter`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    AttractorMode, AttractorPlotter, CompiledFlowJob, SolverExecutionState,
    ATTRACTOR_RING_CAPACITY, MAX_SIMULATION_SPEED,
};
use crate::async_sim::SimCommand;
use crate::framework::{
    announce_branch_cleared, announce_branch_forked, BranchDivergenceHud, BranchMetadata,
    TimelineBranchAction, TimelineTransportState,
};
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape};
use nalgebra::{Point3, Vector3};
use std::sync::Arc;

impl AttractorPlotter {
    /// Scrubs local state & worker to `target_step` without resetting to $(1, 1, 1)$.
    pub fn scrub_to_step(&mut self, target_step: usize) {
        if let Some((cp_step, cp_time, cp_xyz)) = self
            .sync_branch
            .timeline
            .checkpoint_at_or_before(target_step)
            .map(|cp| (cp.step, cp.time, cp.state))
        {
            let drop_n = self.step_count.saturating_sub(cp_step);
            let keep = self.history.len().saturating_sub(drop_n);
            self.history.truncate(keep);
            self.system.state.vec = Vector3::new(cp_xyz[0], cp_xyz[1], cp_xyz[2]);
            self.sync_branch.timeline.truncate_after(target_step);
            self.step_count = cp_step;
            self.sync_branch.current_time = cp_time;
            let custom_job = (self.mode == AttractorMode::Custom)
                .then(|| CompiledFlowJob::from_bank(&self.flow_bank));
            while self.step_count < target_step {
                self.step_single_local_tick(custom_job.as_ref());
            }
            if let Some(ref twin) = self.sync_branch.shadow {
                if self.step_count < twin.fork_step {
                    self.sync_branch.clear_branch();
                }
            }
        }
        self.sync_branch.worker_view = None;
        self.execution_state = SolverExecutionState::Paused;
        let _ = self
            .worker
            .send_command(SimCommand::ScrubToStep(target_step));
    }

    /// Forks an unperturbed 3D shadow twin at the current step $t_k$.
    pub fn fork_branch(&mut self) {
        let custom_flow = (self.mode == AttractorMode::Custom)
            .then(|| Arc::new(CompiledFlowJob::from_bank(&self.flow_bank)));
        self.sync_branch.fork_at(
            self.step_count,
            &self.system,
            self.mode,
            custom_flow,
            self.dt,
            &self.history,
        );
        let _ = self.worker.send_command(SimCommand::ForkBranch);
    }

    /// Clears the active 3D shadow-twin branch overlay.
    pub fn clear_branch(&mut self) {
        self.sync_branch.clear_branch();
        let _ = self.worker.send_command(SimCommand::ClearBranch);
    }

    /// Applies a `1e-5` state perturbation along $x$ to observe butterfly-effect separation.
    pub fn apply_butterfly_perturbation(&mut self) {
        self.system.state.vec.x += 1.0e-5;
        let _ = self
            .worker
            .send_command(SimCommand::Custom("PerturbDeltaX:1e-5".to_string()));
    }

    pub(super) fn step_single_tick(
        &mut self,
        cap: usize,
        custom_job: Option<&CompiledFlowJob>,
    ) -> bool {
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
        if !self.step_single_local_tick(custom_job) {
            return false;
        }
        let vec = self.system.state.vec;
        if let Some(ref mut twin) = self.twin {
            twin.observe_lockstep(self.step_count, self.dt, vec);
        }
        true
    }

    pub(super) fn step_single_local_tick(&mut self, custom_job: Option<&CompiledFlowJob>) -> bool {
        let cap = self.max_points.clamp(1, ATTRACTOR_RING_CAPACITY);
        let current_vec = self.system.state.vec;
        if !current_vec.x.is_finite() || !current_vec.y.is_finite() || !current_vec.z.is_finite() {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: current_vec,
                diverged_at_step: self.step_count,
            };
            return false;
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
        self.sync_branch
            .on_step_in_place(self.step_count, self.dt, self.max_points, vec);
        true
    }

    pub(super) fn step_simulation_lockstep(&mut self) {
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

    pub(super) fn append_shadow_to_scene_colors(
        &mut self,
        shadow_color: egui::Color32,
        accent_color: egui::Color32,
    ) {
        let view = self.sync_branch.effective_view(self.step_count);
        if !view.fork_active || view.shadow_history.len() < 2 {
            return;
        }
        let shadow_pts: Vec<Point3<f64>> = view
            .shadow_history
            .iter()
            .map(|v| Point3::new(v.x, v.y, v.z))
            .collect();
        self.scene_builder.add_polyline(
            "Shadow Baseline",
            &shadow_pts,
            egui::Stroke::new(1.5_f32, shadow_color),
            LineStyle::Dashed { length: 6.0_f32 },
            true,
        );
        if let Some(fv) = view.fork_vec {
            self.scene_builder.add_points(
                "Fork @ t_k",
                &[Point3::new(fv.x, fv.y, fv.z)],
                5.5_f32,
                accent_color,
                MarkerShape::Diamond,
                true,
            );
        }
    }

    pub(super) fn attractor_timeline_state(&self) -> TimelineTransportState {
        let view = self.sync_branch.effective_view(self.step_count);
        TimelineTransportState {
            current_step: self.step_count,
            min_step: view.min_step,
            max_step: view.max_step.max(self.step_count),
            current_time: view.current_time,
            time_unit: "s",
            fork_active: view.fork_active,
            fork_step: view.fork_step,
            fork_time: view.fork_time,
        }
    }

    pub(super) fn apply_attractor_timeline_action(
        &mut self,
        action: TimelineBranchAction,
        ctx: Option<&egui::Context>,
        announce: bool,
    ) {
        let view = self.sync_branch.effective_view(self.step_count);
        let fallback_ctx;
        let actx = if let Some(c) = ctx {
            c
        } else {
            fallback_ctx = egui::Context::default();
            &fallback_ctx
        };
        match action {
            TimelineBranchAction::StepBack => {
                let prev = self
                    .sync_branch
                    .timeline
                    .previous_step_before(self.step_count)
                    .unwrap_or_else(|| self.step_count.saturating_sub(1));
                self.scrub_to_step(prev);
            }
            TimelineBranchAction::ScrubTo(step) => self.scrub_to_step(step),
            TimelineBranchAction::ToggleForkBranch => {
                if view.fork_active {
                    if announce {
                        announce_branch_cleared(actx, "Attractor Plotter");
                    }
                    self.clear_branch();
                } else {
                    if announce {
                        announce_branch_forked(
                            actx,
                            "Attractor Plotter",
                            self.step_count,
                            view.current_time,
                            "s",
                        );
                    }
                    self.fork_branch();
                }
            }
            TimelineBranchAction::ClearBranch => {
                if announce {
                    announce_branch_cleared(actx, "Attractor Plotter");
                }
                self.clear_branch();
            }
        }
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(super) fn draw_timeline_branch_hud_if_active(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
    ) {
        let view = self.sync_branch.effective_view(self.step_count);
        if !view.fork_active {
            return;
        }
        ui.horizontal(|ui| {
            if ui
                .button("🦋 Perturb Δx (+1e-5)")
                .on_hover_text(
                    "Inject a 1e-5 state offset into the active branch to observe exponential chaotic separation",
                )
                .clicked()
            {
                self.apply_butterfly_perturbation();
                ctx.request_repaint();
            }
        });
        let clear = BranchDivergenceHud::new(
            "attractor_branch_hud",
            "Attractor Plotter",
            view.fork_step.unwrap_or(self.step_count),
            view.fork_time.unwrap_or(view.current_time),
            "s",
            view.divergence,
        )
        .show(ui);
        if clear {
            announce_branch_cleared(ctx, "Attractor Plotter");
            self.clear_branch();
            ctx.request_repaint();
        }
    }

    pub(crate) fn attractor_branch_metadata(&self) -> Option<BranchMetadata> {
        let view = self.sync_branch.effective_view(self.step_count);
        if !view.fork_active {
            return None;
        }
        let v = self.system.state.vec;
        Some(BranchMetadata {
            fork_step: view.fork_step.unwrap_or(self.step_count),
            fork_time: view.fork_time.unwrap_or(view.current_time),
            time_unit: "s",
            baseline_summary: view.baseline_summary,
            branch_summary: format!(
                "sigma={:.2}, rho={:.2}, beta={:.2}, state=({:.5},{:.5},{:.5})",
                self.system.sigma, self.system.rho, self.system.beta, v.x, v.y, v.z
            ),
            divergence: view.divergence,
        })
    }
}
