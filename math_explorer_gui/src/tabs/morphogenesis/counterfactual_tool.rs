//! Counterfactual shadow twin actions, recovery, and twin exports for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::{
    render_morphogenesis_heatmap, MorphogenesisForkDescriptor, MorphogenesisHeatmapMode,
    MorphogenesisSnapshotData,
};
use super::runner::MorphogenesisRunner;
use super::MorphogenesisTool;
use crate::async_sim::{BackgroundWorker, SimCommand, SimulationRunner};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::BranchControlAction;
use eframe::egui::ColorImage;

impl MorphogenesisTool {
    /// Returns `true` when a counterfactual shadow twin is currently active.
    #[must_use]
    pub fn is_forked(&self) -> bool {
        self.local_runner.twin.is_some()
    }

    /// Current active spatial heatmap comparison mode.
    #[must_use]
    pub fn heatmap_mode(&self) -> MorphogenesisHeatmapMode {
        self.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .heatmap_mode
    }

    /// Switches the spatial heatmap mode (`Active u_B`, `Split u_A | u_B`, `Difference |u_B - u_A|`).
    pub fn set_heatmap_mode(&mut self, mode: MorphogenesisHeatmapMode) {
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .heatmap_mode = mode;
        self.stepped_pixels = None;
        self.dirty = true;
    }

    /// Advances `local_runner` by `steps` lockstep ticks and syncs the background worker.
    pub fn step_lockstep(&mut self, steps: usize) {
        for _ in 0..steps {
            self.local_runner.step();
        }
        self.sync_branch_engine_from_local();
        self.respawn_worker_from_local();
        self.stepped_pixels = None;
        self.dirty = true;
    }

    /// Forks a counterfactual shadow twin at `step_count > 0`.
    pub fn fork_here(&mut self) -> bool {
        self.pull_latest_worker_snapshot();
        let delta = self.branch_engine.effective_delta();
        if !self.local_runner.fork_here(delta) {
            return false;
        }
        let _ = self
            .worker
            .send_command(SimCommand::Custom(format!("fork:{delta}")));
        self.sync_branch_engine_from_local();
        self.stepped_pixels = None;
        self.dirty = true;
        true
    }

    /// Swaps baseline `Branch A` and active `Branch B` reaction-diffusion fields and parameters.
    pub fn swap_branches(&mut self) -> bool {
        self.pull_latest_worker_snapshot();
        if !self.local_runner.swap_branches() {
            return false;
        }
        let _ = self.worker.send_command(SimCommand::Custom("swap".into()));
        self.sync_branch_engine_from_local();
        self.stepped_pixels = None;
        self.dirty = true;
        true
    }

    /// Clears the counterfactual fork while keeping `Branch B` active.
    pub fn clear_fork(&mut self) -> bool {
        let had = self.local_runner.clear_fork();
        self.branch_engine.clear();
        let _ = self
            .worker
            .send_command(SimCommand::Custom("clear_fork".into()));
        self.stepped_pixels = None;
        self.dirty = true;
        had
    }

    pub(crate) fn apply_morpho_branch_action(&mut self, action: BranchControlAction) -> bool {
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

    pub(crate) fn sync_branch_engine_from_local(&mut self) {
        let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
        let snap = self
            .local_runner
            .twin
            .as_ref()
            .map(|t| t.snapshot(self.local_runner.step_count, dt));
        self.branch_engine.sync_snapshot(snap);
    }

    pub(crate) fn pull_latest_worker_snapshot(&mut self) {
        let Some(snapshot) = self.worker.latest_snapshot() else {
            return;
        };
        let Some(data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MorphogenesisSnapshotData>())
        else {
            return;
        };
        if data.step_count > self.local_runner.step_count
            && data.u_b.len() == self.width * self.height
        {
            self.local_runner.step_count = data.step_count;
            self.local_runner.system.u_mut().copy_from_slice(&data.u_b);
            if let (Some(ref mut twin), Some(ref ua)) = (&mut self.local_runner.twin, &data.u_a) {
                if ua.len() == self.width * self.height {
                    twin.system_a.u_mut().copy_from_slice(ua);
                    let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
                    twin.observe_lockstep(data.step_count, dt, &data.u_b);
                }
            }
            if let Some(ref div) = data.divergence {
                self.branch_engine.sync_snapshot(Some(div.clone()));
            }
        }
    }

    pub(crate) fn respawn_worker_from_local(&mut self) {
        let cloned = self.local_runner.clone_outside_step();
        self.worker = BackgroundWorker::new(cloned);
    }

    /// Reconstructs a forked Turing morphogenesis simulation from a compact descriptor.
    pub(crate) fn apply_morpho_fork_descriptor(
        &mut self,
        desc: MorphogenesisForkDescriptor,
    ) -> Result<(), String> {
        desc.validate()?;
        let (active_a, active_b, active_du, active_dv, dt) = {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.heatmap_mode = desc.heatmap_mode;
            (p.a, p.b, p.d_u, p.d_v, p.dt)
        };
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = desc.baseline_a;
            p.b = desc.baseline_b;
            p.d_u = desc.baseline_du;
            p.d_v = desc.baseline_dv;
        }
        let mut runner = MorphogenesisRunner::new_seeded(&self.params, self.width, self.height);
        for _ in 0..desc.fork_step {
            runner.step();
        }
        let _ = runner.fork_here(desc.delta_u);
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = active_a;
            p.b = active_b;
            p.d_u = active_du;
            p.d_v = active_dv;
            p.dt = dt;
        }
        for _ in 0..desc.post_fork_steps {
            runner.step();
        }
        self.branch_engine.set_effective_delta(desc.delta_u);
        self.local_runner = runner;
        self.sync_branch_engine_from_local();
        self.respawn_worker_from_local();
        self.stepped_pixels = None;
        self.dirty = true;
        Ok(())
    }

    /// Renders the current twin or baseline heatmap image for figure export (`PNG`/`SVG`/`GIF`).
    pub(crate) fn render_current_heatmap_image(&self) -> ColorImage {
        let (mode, theme) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.heatmap_mode, p.theme_mode)
        };
        if let Some(snap) = self.worker.latest_snapshot() {
            if let Some(data) = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MorphogenesisSnapshotData>())
                .filter(|d| d.step_count >= self.local_runner.step_count)
            {
                return render_morphogenesis_heatmap(
                    &data.u_b,
                    data.u_a.as_deref(),
                    self.width,
                    self.height,
                    mode,
                    theme,
                );
            }
        }
        let ua = self.local_runner.twin.as_ref().map(|t| t.system_a.u());
        render_morphogenesis_heatmap(
            self.local_runner.system.u(),
            ua,
            self.width,
            self.height,
            mode,
            theme,
        )
    }

    /// Exports midline (`y = height / 2`) spatial profiles of `u_B`, `u_A`, and `|u_B - u_A|` when forked.
    pub(crate) fn export_morpho_twin_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let Some(ref twin) = self.local_runner.twin else {
            return Vec::new();
        };
        let row_y = self.height / 2;
        let start = row_y * self.width;
        let end = (start + self.width).min(self.local_runner.system.u().len());
        let ub = &self.local_runner.system.u()[start..end];
        let ua = &twin.system_a.u()[start..end];
        let mut series_b = Vec::with_capacity(ub.len());
        let mut series_a = Vec::with_capacity(ua.len());
        let mut series_diff = Vec::with_capacity(ub.len());
        for (x, (&a, &b)) in ua.iter().zip(ub.iter()).enumerate() {
            let xf = x as f64;
            series_b.push([xf, b]);
            series_a.push([xf, a]);
            series_diff.push([xf, (b - a).abs()]);
        }
        vec![
            ("Activator u_B (Perturbed)".to_string(), series_b),
            ("Activator u_A (Baseline)".to_string(), series_a),
            ("Difference |u_B - u_A|".to_string(), series_diff),
        ]
    }
}

impl FiniteStateGuard for MorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != crate::async_sim::SimWorkerState::Faulted
            && self.worker.diagnostic_report().is_none()
            && FiniteStateGuard::is_finite_state(&self.local_runner)
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.dt = (p.dt * 0.5).clamp(1e-4, 0.1);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        p.d_u = p.d_u.clamp(0.1, 5.0);
        p.d_v = p.d_v.clamp(10.0, 200.0);
        p.dt = p.dt.clamp(0.001, 0.1);
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = 0.1;
        p.b = 0.9;
        p.d_u = 1.0;
        p.d_v = 100.0;
        p.dt = 0.05;
        drop(p);
        self.reset_system();
    }
}
