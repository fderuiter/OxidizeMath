#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! `InteractiveTool` implementation, in-thread fallback stepping, and `ToolMetadata` registration.

use super::{
    AttractorMode, AttractorPlotter, AttractorSnapshot, AttractorSystemKind, CompiledFlowJob,
    SolverExecutionState, TwinFrame, DEFAULT_EPSILON, MAX_SIMULATION_SPEED,
};
use crate::async_sim::SimCommand;
use crate::framework::{ExportCapabilities, InteractiveTool, DEFAULT_TIMELINE_RING_CAPACITY};
use eframe::egui;
use math_explorer::physics::chaos::double_pendulum::{DoublePendulumBuilder, DoublePendulumState};
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use math_explorer::physics::chaos::rossler::{RosslerBuilder, RosslerState};
use nalgebra::Vector3;

impl AttractorPlotter {
    fn step_local_lorenz(&mut self, prev: TwinFrame) -> ([f64; 4], [f64; 4]) {
        self.system.step(self.dt);
        let pv = self.system.state.vec;
        let sv = if self.shadow_enabled {
            let mut sh = LorenzBuilder::new()
                .sigma(self.system.sigma)
                .rho(self.system.rho)
                .beta(self.system.beta)
                .build(LorenzState::new(
                    prev.shadow[0],
                    prev.shadow[1],
                    prev.shadow[2],
                ));
            sh.step(self.dt);
            sh.state.vec
        } else {
            pv
        };
        ([pv.x, pv.y, pv.z, 0.0], [sv.x, sv.y, sv.z, 0.0])
    }

    fn step_local_rossler(&mut self, prev: TwinFrame) -> ([f64; 4], [f64; 4]) {
        let mut pr = RosslerBuilder::new()
            .a(self.extra_params.rossler_a)
            .b(self.extra_params.rossler_b)
            .c(self.extra_params.rossler_c)
            .build(RosslerState::new(
                prev.primary[0],
                prev.primary[1],
                prev.primary[2],
            ));
        pr.step(self.dt);
        let pv = pr.state.vec;
        self.system.state.vec = pv;
        let sv = if self.shadow_enabled {
            let mut sh = RosslerBuilder::new()
                .a(self.extra_params.rossler_a)
                .b(self.extra_params.rossler_b)
                .c(self.extra_params.rossler_c)
                .build(RosslerState::new(
                    prev.shadow[0],
                    prev.shadow[1],
                    prev.shadow[2],
                ));
            sh.step(self.dt);
            sh.state.vec
        } else {
            pv
        };
        ([pv.x, pv.y, pv.z, 0.0], [sv.x, sv.y, sv.z, 0.0])
    }

    fn step_local_pendulum(&mut self, prev: TwinFrame) -> ([f64; 4], [f64; 4]) {
        let mut pr = DoublePendulumBuilder::new()
            .m1(self.extra_params.pendulum_m1)
            .m2(self.extra_params.pendulum_m2)
            .l1(self.extra_params.pendulum_l1)
            .l2(self.extra_params.pendulum_l2)
            .g(self.extra_params.pendulum_g)
            .build(DoublePendulumState::new(
                prev.primary[0],
                prev.primary[1],
                prev.primary[2],
                prev.primary[3],
            ));
        pr.step(self.dt);
        self.system.state.vec = pr.state.phase_point_3d();
        let s4 = if self.shadow_enabled {
            let mut sh = DoublePendulumBuilder::new()
                .m1(self.extra_params.pendulum_m1)
                .m2(self.extra_params.pendulum_m2)
                .l1(self.extra_params.pendulum_l1)
                .l2(self.extra_params.pendulum_l2)
                .g(self.extra_params.pendulum_g)
                .build(DoublePendulumState::new(
                    prev.shadow[0],
                    prev.shadow[1],
                    prev.shadow[2],
                    prev.shadow[3],
                ));
            sh.step(self.dt);
            sh.state.state.0
        } else {
            pr.state.state.0
        };
        (pr.state.state.0, s4)
    }

    fn step_local_pair(
        &mut self,
        prev: TwinFrame,
        custom_job: Option<&CompiledFlowJob>,
    ) -> ([f64; 4], [f64; 4]) {
        if let Some(job) = custom_job {
            let pv = job.rk4_step(prev.primary_vec3(), self.dt);
            self.system.state.vec = pv;
            let sv = if self.shadow_enabled {
                job.rk4_step(prev.shadow_vec3(), self.dt)
            } else {
                pv
            };
            return ([pv.x, pv.y, pv.z, 0.0], [sv.x, sv.y, sv.z, 0.0]);
        }
        match self.system_kind {
            AttractorSystemKind::Lorenz => self.step_local_lorenz(prev),
            AttractorSystemKind::Rossler => self.step_local_rossler(prev),
            AttractorSystemKind::DoublePendulum => self.step_local_pendulum(prev),
        }
    }

    fn record_local_frame(&mut self, p4: [f64; 4], s4: [f64; 4], cap: usize) -> bool {
        self.step_count += 1;
        if p4.iter().any(|v| !v.is_finite()) || s4.iter().any(|v| !v.is_finite()) {
            self.execution_state = SolverExecutionState::Diverged {
                invalid_vec: Vector3::new(p4[0], p4[1], p4[2]),
                diverged_at_step: self.step_count,
            };
            return false;
        }
        let frame = TwinFrame::from_states(
            self.step_count,
            self.step_count as f64 * self.dt,
            p4,
            s4,
            self.shadow_enabled,
        );
        self.timeline.push(frame);
        if self.history.len() >= cap {
            self.history.pop_front();
        }
        self.history.push_back(frame.primary_vec3());
        if self.shadow_history.len() >= cap {
            self.shadow_history.pop_front();
        }
        self.shadow_history.push_back(frame.shadow_vec3());
        true
    }

    fn ingest_worker_snapshot(&mut self, was_paused: bool) {
        let ui_diverged = self.diverged();
        let Some(data) = self
            .worker
            .update()
            .and_then(|s| s.structured_data.as_ref())
            .and_then(|d| d.downcast_ref::<AttractorSnapshot>())
            .cloned()
        else {
            return;
        };
        if ui_diverged || (was_paused && data.step_count == 0 && self.step_count > 0) {
            return;
        }
        self.history = data.history;
        self.shadow_history = data.shadow_history;
        self.timeline = data.timeline;
        self.system.state.vec = data.current_vec;
        self.step_count = data.step_count;
        self.metrics_cache.update_windowed_ftle(&self.timeline);
        if matches!(data.execution_state, SolverExecutionState::Diverged { .. }) || !was_paused {
            self.execution_state = data.execution_state;
        }
    }

    pub(crate) fn clamp_local_parameters(&mut self) {
        self.system.sigma = sanitize_scalar(self.system.sigma, 0.0, 50.0, 10.0);
        self.system.rho = sanitize_scalar(self.system.rho, 0.0, 100.0, 28.0);
        self.system.beta = sanitize_scalar(self.system.beta, 0.0, 10.0, 8.0 / 3.0);
        let def_dt = self.system_kind.default_dt();
        self.dt = if self.dt.is_finite() && self.dt > 0.0 {
            self.dt.clamp(0.001, 0.05)
        } else {
            def_dt
        };
        self.tube_radius = sanitize_scalar(
            self.tube_radius,
            super::export::MIN_TUBE_RADIUS,
            super::export::MAX_TUBE_RADIUS,
            super::export::DEFAULT_TUBE_RADIUS,
        );
        self.epsilon = sanitize_scalar(self.epsilon, 1e-12, 1e-1, DEFAULT_EPSILON);
        for c in &mut self.initial_state {
            *c = sanitize_scalar(*c, -80.0, 80.0, 1.0);
        }
        self.extra_params.sanitize();
        self.reset();
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
        self.system_kind.hash(&mut h);
        self.system.sigma.to_bits().hash(&mut h);
        self.system.rho.to_bits().hash(&mut h);
        self.system.beta.to_bits().hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.tube_radius.to_bits().hash(&mut h);
        self.simulation_speed.hash(&mut h);
        self.max_points.hash(&mut h);
        self.shadow_enabled.hash(&mut h);
        self.epsilon.to_bits().hash(&mut h);
        self.perturbation_expr.hash(&mut h);
        for c in self.initial_state {
            c.to_bits().hash(&mut h);
        }
        self.is_paused().hash(&mut h);
        self.diverged().hash(&mut h);
        self.show_annotations.hash(&mut h);
        self.camera.hash_state(&mut h);
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
        let custom_job = (self.mode == AttractorMode::Custom)
            .then(|| CompiledFlowJob::from_bank(&self.flow_bank));
        let cap = self.max_points.clamp(1, DEFAULT_TIMELINE_RING_CAPACITY);
        for _ in 0..self.simulation_speed.clamp(1, MAX_SIMULATION_SPEED) {
            let current_vec = self.system.state.vec;
            if !current_vec.x.is_finite()
                || !current_vec.y.is_finite()
                || !current_vec.z.is_finite()
            {
                self.execution_state = SolverExecutionState::Diverged {
                    invalid_vec: current_vec,
                    diverged_at_step: self.step_count,
                };
                break;
            }
            let cursor_len = self.timeline.cursor();
            while self.history.len() > cursor_len {
                self.history.pop_back();
            }
            while self.shadow_history.len() > cursor_len {
                self.shadow_history.pop_back();
            }
            let prev = self.timeline.current_copied().unwrap_or_default();
            let (p4, s4) = self.step_local_pair(prev, custom_job.as_ref());
            if !self.record_local_frame(p4, s4, cap) {
                break;
            }
        }
        self.metrics_cache.update_windowed_ftle(&self.timeline);
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
        let mut points: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();
        if points.is_empty() && self.is_finite_state() {
            points.push(self.project(self.system.state.vec));
        }
        let mut datasets = vec![("Trajectory".to_string(), points)];
        if self.shadow_enabled && self.is_finite_state() {
            let mut sh_pts: Vec<[f64; 2]> = self
                .shadow_history
                .iter()
                .map(|p| self.project(*p))
                .collect();
            if sh_pts.is_empty() {
                if let Some(curr) = self.timeline.current() {
                    sh_pts.push(self.project(curr.shadow_vec3()));
                }
            }
            datasets.push(("Shadow Trajectory".to_string(), sh_pts));
            datasets.push((
                "Log10 Divergence".to_string(),
                self.build_log10_divergence_series(),
            ));
        }
        datasets
    }

    fn export_datasets_3d(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        let mut points: Vec<[f64; 3]> = self.history.iter().map(|p| [p.x, p.y, p.z]).collect();
        if points.is_empty() && self.is_finite_state() {
            let v = self.system.state.vec;
            points.push([v.x, v.y, v.z]);
        }
        let mut datasets = vec![("Trajectory".to_string(), points)];
        if self.shadow_enabled && self.is_finite_state() {
            let mut sh_pts: Vec<[f64; 3]> = self
                .shadow_history
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect();
            if sh_pts.is_empty() {
                if let Some(curr) = self.timeline.current() {
                    let s = curr.shadow_vec3();
                    sh_pts.push([s.x, s.y, s.z]);
                }
            }
            datasets.push(("Shadow Trajectory".to_string(), sh_pts));
        }
        datasets
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

        let was_paused = self.is_paused();
        self.ingest_worker_snapshot(was_paused);

        let worker_faulted = self.worker.state().is_faulted();
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

fn sanitize_scalar(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "AttractorPlotter",
        display_name: "Attractor Plotter",
        domain: "chaos",
        tags: &["lorenz", "rossler", "double-pendulum", "lyapunov", "attractor", "chaos"],
        build: || Box::new(AttractorPlotter::default()),
    }
}
