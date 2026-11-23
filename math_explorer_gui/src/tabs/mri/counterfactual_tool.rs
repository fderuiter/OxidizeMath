//! Counterfactual branch actions and preset reconstruction for [`super::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::{
    MRI_STEP_DT, MriForkDescriptor, MriTwinState, step_bloch_rotating_frame,
};
use super::runner::MriRunner;
use super::tool::MriTool;
use crate::accessibility::{AccessibleHoverText, announce_status};
use crate::app::SemanticPalette;
use crate::async_sim::SimCommand;
use crate::framework::{
    BranchControlAction, branch_a_ghost_line, branch_b_solid_line, fork_marker_callout,
    fork_marker_vline,
};
use eframe::egui;
use egui_plot::{Line, PlotPoint};
use math_explorer::physics::mri::bloch::BlochSimulator;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use nalgebra::Vector3;

impl MriTool {
    pub(super) fn current_step_index(&self) -> usize {
        (self.current_time / MRI_STEP_DT).round().max(0.0) as usize
    }

    /// Forks a counterfactual shadow twin mid-decay (`current_time > 0`).
    pub fn fork_here(&mut self) -> bool {
        let step = self.current_step_index();
        if step == 0 {
            return false;
        }
        let delta = self.branch_engine.effective_delta();
        self.twin = Some(MriTwinState::fork_from(
            step,
            self.current_time,
            &mut self.simulator,
            &self.history,
            self.max_history,
            delta,
            false,
        ));
        let _ = self
            .worker
            .send_command(SimCommand::Custom(format!("fork:{delta}")));
        self.sync_mri_branch_engine();
        true
    }

    /// Applies a `180°` Hahn spin-echo refocusing pulse on `Branch B` (auto-forking if `t_k > 0`).
    pub fn apply_refocus_pulse_180(&mut self) -> bool {
        let step = self.current_step_index();
        if step == 0 && self.twin.is_none() {
            return false;
        }
        if let Some(ref mut twin) = self.twin {
            twin.apply_refocus_180(self.current_time, &mut self.simulator);
        } else {
            self.twin = Some(MriTwinState::fork_from(
                step,
                self.current_time,
                &mut self.simulator,
                &self.history,
                self.max_history,
                0.0,
                true,
            ));
        }
        let _ = self
            .worker
            .send_command(SimCommand::Custom("refocus_180".into()));
        self.sync_mri_branch_engine();
        true
    }

    /// Swaps baseline `Branch A` and active `Branch B` relaxation parameters and histories.
    pub fn swap_branches(&mut self) -> bool {
        let Some(ref mut twin) = self.twin else {
            return false;
        };
        std::mem::swap(
            &mut self.simulator.magnetization,
            &mut twin.simulator_a.magnetization,
        );
        std::mem::swap(&mut self.simulator.t1, &mut twin.simulator_a.t1);
        std::mem::swap(&mut self.simulator.t2, &mut twin.simulator_a.t2);
        std::mem::swap(&mut self.simulator.b_field, &mut twin.simulator_a.b_field);
        std::mem::swap(&mut self.history, &mut twin.history_a);
        if let Some(ref mut ens) = twin.spin_echo {
            std::mem::swap(&mut ens.spins_a, &mut ens.spins_b);
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.t1 = self.simulator.t1;
            p.t2 = self.simulator.t2;
            p.b_field = self.simulator.b_field;
        }
        let _ = self.worker.send_command(SimCommand::Custom("swap".into()));
        self.update_plot_buffers();
        self.sync_mri_branch_engine();
        true
    }

    /// Clears the active counterfactual fork while preserving `Branch B`.
    pub fn clear_fork(&mut self) -> bool {
        let had = self.twin.take().is_some();
        self.branch_engine.clear();
        let _ = self
            .worker
            .send_command(SimCommand::Custom("clear_fork".into()));
        had
    }

    pub(super) fn sync_mri_branch_engine(&mut self) {
        let snap = self
            .twin
            .as_ref()
            .map(|t| t.snapshot(self.current_time, self.simulator.magnetization));
        self.branch_engine.sync_snapshot(snap);
    }

    /// Reconstructs a forked MRI twin trajectory from a compact descriptor.
    pub(super) fn apply_mri_fork_descriptor(
        &mut self,
        desc: MriForkDescriptor,
    ) -> Result<(), String> {
        Self::validate_mri_fork_descriptor(&desc)?;
        let (active_t1, active_t2, active_b) =
            (self.simulator.t1, self.simulator.t2, self.simulator.b_field);
        self.trace_mri_pre_fork(&desc);

        let mut twin = MriTwinState::fork_from(
            desc.fork_step,
            self.current_time,
            &mut self.simulator,
            &self.history,
            self.max_history,
            desc.delta_x,
            desc.refocus_180,
        );
        self.simulator.set_relaxation(active_t1, active_t2);
        self.simulator.set_b_field(active_b);
        self.branch_engine.set_effective_delta(desc.delta_x);
        self.trace_mri_post_fork(&mut twin, desc.post_fork_steps);

        self.twin = Some(twin);
        self.update_plot_buffers();
        self.sync_mri_branch_engine();
        self.sync_mri_worker_with_local_state();
        Ok(())
    }

    fn validate_mri_fork_descriptor(desc: &MriForkDescriptor) -> Result<(), String> {
        if desc.fork_step == 0
            || desc.fork_step > 50_000
            || desc.post_fork_steps > 50_000
            || ![
                desc.baseline_t1,
                desc.baseline_t2,
                desc.baseline_bx,
                desc.baseline_by,
                desc.baseline_bz,
                desc.delta_x,
            ]
            .iter()
            .all(|v| v.is_finite())
            || !(0.1..=5.0).contains(&desc.baseline_t1)
            || !(0.01..=2.0).contains(&desc.baseline_t2)
        {
            return Err("Invalid MRI fork descriptor".to_string());
        }
        Ok(())
    }

    fn trace_mri_pre_fork(&mut self, desc: &MriForkDescriptor) {
        let initial_m = Vector3::new(self.simulator.m0, 0.0, 0.0);
        self.simulator.magnetization = initial_m;
        self.simulator
            .set_relaxation(desc.baseline_t1, desc.baseline_t2);
        self.simulator.set_b_field(Vector3::new(
            desc.baseline_bx,
            desc.baseline_by,
            desc.baseline_bz,
        ));
        self.current_time = 0.0;
        self.history.clear();
        self.history.push_back((0.0, initial_m));
        for _ in 0..desc.fork_step {
            step_bloch_rotating_frame(&mut self.simulator, &mut self.solver, MRI_STEP_DT, 0.0);
            self.current_time += MRI_STEP_DT;
            self.history
                .push_back((self.current_time, self.simulator.magnetization));
        }
    }

    fn trace_mri_post_fork(&mut self, twin: &mut MriTwinState, post_fork_steps: usize) {
        for _ in 0..post_fork_steps {
            let next_t = self.current_time + MRI_STEP_DT;
            twin.step_branch_a(next_t, MRI_STEP_DT, self.max_history);
            if !twin.step_branch_b_spin_echo(&mut self.simulator, &mut self.solver, MRI_STEP_DT) {
                step_bloch_rotating_frame(&mut self.simulator, &mut self.solver, MRI_STEP_DT, 0.0);
            }
            self.current_time = next_t;
            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history
                .push_back((self.current_time, self.simulator.magnetization));
            twin.observe_lockstep(self.current_time, self.simulator.magnetization);
        }
    }

    pub(super) fn sync_mri_worker_with_local_state(&mut self) {
        let mut runner_sim = BlochSimulator::new(self.simulator.magnetization, self.simulator.m0);
        runner_sim.set_relaxation(self.simulator.t1, self.simulator.t2);
        runner_sim.set_b_field(self.simulator.b_field);
        let mut history = self.history.clone();
        if history.is_empty() {
            history.push_back((self.current_time, self.simulator.magnetization));
        }
        let runner = MriRunner {
            simulator: runner_sim,
            solver: RungeKutta4::new(&self.simulator.magnetization),
            params: std::sync::Arc::clone(&self.params),
            current_time: self.current_time,
            history,
            max_history: self.max_history,
            twin: self.twin.clone(),
            step_count: self.current_step_index(),
            rng: oxidize_core::rng::OxidizeRng::new(0x4D52_495F_424C_4F43),
        };
        self.worker = crate::async_sim::BackgroundWorker::new(runner);
    }

    pub(super) fn apply_mri_branch_action(&mut self, action: BranchControlAction) -> bool {
        match action {
            BranchControlAction::ForkHere => self.fork_here(),
            BranchControlAction::SwapBranches => self.swap_branches(),
            BranchControlAction::ClearFork => self.clear_fork(),
            BranchControlAction::SetPerturbation(delta) => {
                self.branch_engine.set_effective_delta(delta);
                true
            }
        }
    }

    /// Restarts the background Bloch simulation worker with current shared parameters.
    #[cfg(test)]
    pub fn restart_worker(&mut self) {
        let (t1, t2, b_field) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.t1, p.t2, p.b_field)
        };
        self.simulator.t1 = t1;
        self.simulator.t2 = t2;
        self.simulator.b_field = b_field;
        self.sync_mri_worker_with_local_state();
    }

    /// Alias for [`Self::restart_worker`] used by worker fault recovery tests.
    #[cfg(test)]
    pub fn retry(&mut self) {
        self.restart_worker();
    }

    pub(super) fn render_refocus_pulse_button(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let can_pulse = self.current_step_index() > 0 || self.twin.is_some();
        let resp = ui
            .add_enabled(can_pulse, egui::Button::new("🧲 180° Refocusing Pulse"))
            .accessible_hover_text(
                "Apply a 180° RF spin-echo refocusing pulse to Branch B at the current decay step",
            )
            .on_disabled_hover_text(
                "Step or Play past t = 0 to apply a mid-decay 180° refocusing pulse",
            );
        if resp.clicked() && self.apply_refocus_pulse_180() {
            announce_status("Applied 180 degree spin-echo refocusing pulse on Branch B");
            ctx.request_repaint();
        }
    }
}

pub(super) fn render_mri_plot_curves<'a>(
    twin: Option<&MriTwinState>,
    mx_points: &'a [PlotPoint],
    my_points: &'a [PlotPoint],
    mz_points: &'a [PlotPoint],
    plot_ui: &mut egui_plot::PlotUi<'a>,
    palette: &SemanticPalette,
) {
    let to_vec = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
    if let Some(twin) = twin {
        let mxa: Vec<[f64; 2]> = twin.history_a.iter().map(|(t, m)| [*t, m.x]).collect();
        let mya: Vec<[f64; 2]> = twin.history_a.iter().map(|(t, m)| [*t, m.y]).collect();
        let mza: Vec<[f64; 2]> = twin.history_a.iter().map(|(t, m)| [*t, m.z]).collect();
        plot_ui.line(branch_a_ghost_line("Mx (Branch A)", mxa, palette.axis_x));
        plot_ui.line(branch_a_ghost_line("My (Branch A)", mya, palette.axis_y));
        plot_ui.line(branch_a_ghost_line("Mz (Branch A)", mza, palette.axis_z));
        plot_ui.line(branch_b_solid_line(
            "Mx (Branch B)",
            to_vec(mx_points),
            palette.axis_x,
        ));
        plot_ui.line(branch_b_solid_line(
            "My (Branch B)",
            to_vec(my_points),
            palette.axis_y,
        ));
        plot_ui.line(branch_b_solid_line(
            "Mz (Branch B)",
            to_vec(mz_points),
            palette.axis_z,
        ));
        let fork_t = twin.tracker.fork_time();
        plot_ui.vline(fork_marker_vline(palette, fork_t));
        plot_ui.callout(fork_marker_callout(
            [fork_t, twin.fork_m.x],
            fork_t,
            palette,
        ));
    } else {
        plot_ui.line(Line::new("Mx (Transverse)", mx_points).color(palette.axis_x));
        plot_ui.line(Line::new("My (Transverse)", my_points).color(palette.axis_y));
        plot_ui.line(Line::new("Mz (Longitudinal)", mz_points).color(palette.axis_z));
    }
}
