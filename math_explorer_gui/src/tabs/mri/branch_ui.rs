//! Timeline scrubbing, counterfactual fork UI, plot overlays, and dataset/figure exports
//! for [`super::tool::MriTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::step_bloch_rotating_frame;
use super::pulse::MRI_PULSE_SAMPLES;
use super::runner::MriSnapshot;
use super::tool::MriTool;
use crate::app::{SemanticPalette, ThemeMode};
use crate::async_sim::SimCommand;
use crate::framework::{
    announce_branch_cleared, announce_branch_forked, build_branch_metadata_dataset_2d,
    build_dashed_export_primitives, compute_slice_divergence, BranchDivergenceAccumulator,
    BranchDivergenceHud, BranchDivergenceMetrics, BranchMetadata, PlaybackAction,
    PlaybackTransport, TimelineBranchAction, TimelineTransportState,
};
use eframe::egui;
use egui_plot::{Line, LineStyle, PlotPoints, PlotUi, VLine};
use nalgebra::Vector3;
use std::collections::VecDeque;

impl MriTool {
    /// Scrubs `MriTool` to `target_step` without resetting magnetization to $(M_0, 0, 0)$.
    pub fn scrub_to_step(&mut self, target_step: usize) {
        if let Some((cp_step, cp_time, cp_m)) = self
            .sync_branch
            .timeline
            .checkpoint_at_or_before(target_step)
            .map(|c| (c.step, c.time, c.state))
        {
            while self
                .history
                .back()
                .is_some_and(|(t, _)| *t > cp_time + 1.0e-12)
            {
                self.history.pop_back();
            }
            self.simulator.magnetization = Vector3::new(cp_m[0], cp_m[1], cp_m[2]);
            self.current_time = cp_time;
            self.sync_branch.timeline.truncate_after(target_step);
            self.sync_branch.step_count = cp_step;
            self.sync_branch.current_time = cp_time;
            while self.sync_branch.step_count < target_step {
                self.step_continuous_replay(0.01);
            }
            if let Some(ref twin) = self.sync_branch.shadow {
                if self.sync_branch.step_count < twin.fork_step {
                    self.sync_branch.clear_branch();
                }
            }
            self.update_plot_buffers();
        }
        self.sync_branch.worker_view = None;
        let _ = self.worker.send_command(SimCommand::Pause);
        let _ = self
            .worker
            .send_command(SimCommand::ScrubToStep(target_step));
    }

    /// Forks an unperturbed Bloch shadow twin at the current time $t_k$.
    pub fn fork_branch(&mut self) {
        self.sync_branch
            .fork_at(&self.simulator, self.current_time, &self.history);
        let _ = self.worker.send_command(SimCommand::ForkBranch);
    }

    /// Clears the active Bloch shadow-twin branch overlay.
    pub fn clear_branch(&mut self) {
        self.sync_branch.clear_branch();
        let _ = self.worker.send_command(SimCommand::ClearBranch);
    }

    fn step_continuous_replay(&mut self, dt: f64) {
        if self.history.is_empty() {
            self.history
                .push_back((self.current_time, self.simulator.magnetization));
        }
        step_bloch_rotating_frame(&mut self.simulator, &mut self.solver, dt, 0.0);
        self.current_time += dt;
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        let m = self.simulator.magnetization;
        self.history.push_back((self.current_time, m));
        self.sync_branch
            .on_step_in_place(dt, self.max_history, self.current_time, m);
    }

    pub(super) fn step_continuous_sync(&mut self, dt: f64) {
        if self.history.is_empty() {
            self.history
                .push_back((self.current_time, self.simulator.magnetization));
        }
        let next_t = self.current_time + dt;
        if let Some(ref mut twin) = self.twin {
            twin.step_branch_a(next_t, dt, self.max_history);
        }
        let used_echo = self.twin.as_mut().is_some_and(|twin| {
            twin.step_branch_b_spin_echo(&mut self.simulator, &mut self.solver, dt)
        });
        if !used_echo {
            step_bloch_rotating_frame(&mut self.simulator, &mut self.solver, dt, 0.0);
        }
        self.current_time = next_t;
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        let m = self.simulator.magnetization;
        self.history.push_back((self.current_time, m));
        if let Some(ref mut twin) = self.twin {
            twin.observe_lockstep(self.current_time, m);
        }
        self.sync_branch
            .on_step_in_place(dt, self.max_history, self.current_time, m);
    }

    pub(super) fn sync_pulse_sequence_timeline(&mut self) {
        self.sync_branch.timeline.clear();
        for idx in 0..MRI_PULSE_SAMPLES {
            let t = self.pulse_workspace.times[idx];
            let m = self.pulse_workspace.trajectory[idx];
            self.sync_branch
                .timeline
                .push_checkpoint(idx, t, [m.x, m.y, m.z]);
        }
        self.sync_branch.step_count = MRI_PULSE_SAMPLES - 1;
        self.sync_branch.current_time = self.current_time;
        if let Some(ref mut twin) = self.sync_branch.shadow {
            let mut acc = BranchDivergenceAccumulator::default();
            let n = twin.history.len().min(self.history.len());
            for idx in 1..n {
                let (t_prev, _) = self.history[idx - 1];
                let (t_cur, pm) = self.history[idx];
                let (_, sm) = twin.history[idx];
                let (l2, peak) = compute_slice_divergence(&[pm.x, pm.y, pm.z], &[sm.x, sm.y, sm.z]);
                acc.record_step((t_cur - t_prev).max(1.0e-6), l2, peak);
            }
            twin.divergence = acc;
        }
    }

    pub(super) fn mri_timeline_state(&self) -> TimelineTransportState {
        let view = self.sync_branch.effective_view();
        TimelineTransportState {
            current_step: view.step_count,
            min_step: view.min_step,
            max_step: view.max_step.max(view.step_count),
            current_time: view.current_time.max(self.current_time),
            time_unit: "s",
            fork_active: view.fork_active,
            fork_step: view.fork_step,
            fork_time: view.fork_time,
        }
    }

    pub(super) fn apply_mri_timeline_action(
        &mut self,
        action: TimelineBranchAction,
        ctx: Option<&egui::Context>,
        announce: bool,
    ) {
        let view = self.sync_branch.effective_view();
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
                    .previous_step_before(view.step_count)
                    .unwrap_or_else(|| view.step_count.saturating_sub(1));
                self.scrub_to_step(prev);
            }
            TimelineBranchAction::ScrubTo(step) => self.scrub_to_step(step),
            TimelineBranchAction::ToggleForkBranch => {
                if view.fork_active {
                    if announce {
                        announce_branch_cleared(actx, "MRI Bloch Relaxation");
                    }
                    self.clear_branch();
                } else {
                    if announce {
                        announce_branch_forked(
                            actx,
                            "MRI Bloch Relaxation",
                            view.step_count,
                            self.current_time,
                            "s",
                        );
                    }
                    self.fork_branch();
                }
            }
            TimelineBranchAction::ClearBranch => {
                if announce {
                    announce_branch_cleared(actx, "MRI Bloch Relaxation");
                }
                self.clear_branch();
            }
        }
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(super) fn draw_mri_transport_and_hud(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
    ) -> Option<PlaybackAction> {
        let ts = self.mri_timeline_state();
        let (play_act, branch_act) = PlaybackTransport::from_worker_state(
            "mri_transport",
            "MRI Bloch Relaxation",
            self.worker.state(),
        )
        .reset_label("↻ Reset (90° Pulse)")
        .can_reset(true)
        .show_with_timeline(ui, ts);

        if let Some(ba) = branch_act {
            self.apply_mri_timeline_action(ba, Some(ctx), false);
        }

        let view = self.sync_branch.effective_view();
        if view.fork_active {
            let clear = BranchDivergenceHud::new(
                "mri_branch_hud",
                "MRI Bloch Relaxation",
                view.fork_step.unwrap_or(view.step_count),
                view.fork_time.unwrap_or(self.current_time),
                "s",
                view.divergence,
            )
            .show(ui);
            if clear {
                announce_branch_cleared(ctx, "MRI Bloch Relaxation");
                self.clear_branch();
                ctx.request_repaint();
            }
        }
        play_act
    }

    pub(super) fn draw_mri_pulse_branch_hud(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let view = self.sync_branch.effective_view();
        let btn_label = if view.fork_active {
            "⑂ Clear Twin Branch"
        } else {
            "⑂ Fork Branch"
        };
        if ui.button(btn_label).clicked() {
            self.apply_mri_timeline_action(TimelineBranchAction::ToggleForkBranch, Some(ctx), true);
        }
        if view.fork_active {
            let clear = BranchDivergenceHud::new(
                "mri_pulse_branch_hud",
                "MRI Bloch Relaxation",
                view.fork_step.unwrap_or(view.step_count),
                view.fork_time.unwrap_or(self.current_time),
                "s",
                view.divergence,
            )
            .show(ui);
            if clear {
                announce_branch_cleared(ctx, "MRI Bloch Relaxation");
                self.clear_branch();
                ctx.request_repaint();
            }
        }
    }

    pub(super) fn render_shadow_on_plot(&self, plot_ui: &mut PlotUi<'_>, palette: SemanticPalette) {
        let view = self.sync_branch.effective_view();
        if !view.fork_active || view.shadow_history.is_empty() {
            return;
        }
        let sx: Vec<[f64; 2]> = view.shadow_history.iter().map(|(t, m)| [*t, m.x]).collect();
        let sy: Vec<[f64; 2]> = view.shadow_history.iter().map(|(t, m)| [*t, m.y]).collect();
        let sz: Vec<[f64; 2]> = view.shadow_history.iter().map(|(t, m)| [*t, m.z]).collect();
        let dash = LineStyle::Dashed { length: 6.0 };
        plot_ui.line(
            Line::new("Shadow Mx (Baseline)", PlotPoints::from(sx))
                .color(palette.axis_x)
                .style(dash),
        );
        plot_ui.line(
            Line::new("Shadow My (Baseline)", PlotPoints::from(sy))
                .color(palette.axis_y)
                .style(dash),
        );
        plot_ui.line(
            Line::new("Shadow Mz (Baseline)", PlotPoints::from(sz))
                .color(palette.axis_z)
                .style(dash),
        );
        if let Some(t_k) = view.fork_time {
            plot_ui.vline(
                VLine::new("Fork @ t_k", t_k)
                    .color(palette.accent)
                    .style(LineStyle::Dashed { length: 4.0 }),
            );
        }
    }

    pub(crate) fn mri_branch_divergence(&self) -> Option<BranchDivergenceMetrics> {
        let view = self.sync_branch.effective_view();
        view.fork_active.then_some(view.divergence)
    }

    pub(crate) fn mri_branch_metadata(&self) -> Option<BranchMetadata> {
        let view = self.sync_branch.effective_view();
        if !view.fork_active {
            return None;
        }
        let b = self.simulator.b_field;
        Some(BranchMetadata {
            fork_step: view.fork_step.unwrap_or(view.step_count),
            fork_time: view.fork_time.unwrap_or(self.current_time),
            time_unit: "s",
            baseline_summary: view.baseline_summary,
            branch_summary: format!(
                "T1={:.3}s, T2={:.3}s, B=({:.3},{:.3},{:.3})T",
                self.simulator.t1, self.simulator.t2, b.x, b.y, b.z
            ),
            divergence: view.divergence,
        })
    }

    fn effective_primary_history(&self) -> VecDeque<(f64, Vector3<f64>)> {
        if !self.history.is_empty() {
            return self.history.clone();
        }
        if let Some(snap) = self.worker.latest_snapshot() {
            if let Some(data) = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MriSnapshot>())
            {
                if !data.history.is_empty() {
                    return data.history.clone();
                }
            }
        }
        VecDeque::from([(self.current_time, self.simulator.magnetization)])
    }

    pub(super) fn build_mri_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if let Some(ref twin) = self.twin {
            if !self.history.is_empty() {
                let map_xyz = |h: &VecDeque<(f64, Vector3<f64>)>, labels: [&str; 3]| {
                    vec![
                        (
                            labels[0].to_string(),
                            h.iter().map(|(t, m)| [*t, m.x]).collect(),
                        ),
                        (
                            labels[1].to_string(),
                            h.iter().map(|(t, m)| [*t, m.y]).collect(),
                        ),
                        (
                            labels[2].to_string(),
                            h.iter().map(|(t, m)| [*t, m.z]).collect(),
                        ),
                    ]
                };
                let mut out = map_xyz(
                    &self.history,
                    ["Mx (Branch B)", "My (Branch B)", "Mz (Branch B)"],
                );
                out.extend(map_xyz(
                    &twin.history_a,
                    ["Mx (Branch A)", "My (Branch A)", "Mz (Branch A)"],
                ));
                return out;
            }
        }
        let hist = self.effective_primary_history();
        let mx: Vec<[f64; 2]> = hist.iter().map(|(t, m)| [*t, m.x]).collect();
        let my: Vec<[f64; 2]> = hist.iter().map(|(t, m)| [*t, m.y]).collect();
        let mz: Vec<[f64; 2]> = hist.iter().map(|(t, m)| [*t, m.z]).collect();
        let mut out = vec![
            ("Mx (Transverse)".to_string(), mx),
            ("My (Transverse)".to_string(), my),
            ("Mz (Longitudinal)".to_string(), mz),
        ];
        let view = self.sync_branch.effective_view();
        if view.fork_active {
            let t_k = view.fork_time.unwrap_or(0.0);
            out.push((
                format!("Shadow Mx @ t_k={t_k:.3}s"),
                view.shadow_history.iter().map(|(t, m)| [*t, m.x]).collect(),
            ));
            out.push((
                format!("Shadow My @ t_k={t_k:.3}s"),
                view.shadow_history.iter().map(|(t, m)| [*t, m.y]).collect(),
            ));
            out.push((
                format!("Shadow Mz @ t_k={t_k:.3}s"),
                view.shadow_history.iter().map(|(t, m)| [*t, m.z]).collect(),
            ));
            if let Some(meta) = self.mri_branch_metadata() {
                out.push(build_branch_metadata_dataset_2d(&meta));
            }
        }
        out
    }

    pub(super) fn build_mri_export_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        let view = self.sync_branch.effective_view();
        if self.twin.is_some() || !view.fork_active {
            return egui_plot::get_last_plot_primitives();
        }
        let palette = SemanticPalette::for_theme(ThemeMode::Dark);
        let mut prims = Vec::new();
        for (name, color, comp) in [
            ("Shadow Mx", palette.axis_x, 0_usize),
            ("Shadow My", palette.axis_y, 1_usize),
            ("Shadow Mz", palette.axis_z, 2_usize),
        ] {
            let pts: Vec<[f64; 2]> = view
                .shadow_history
                .iter()
                .map(|(t, m)| [*t, m[comp]])
                .collect();
            prims.extend(build_dashed_export_primitives(name, &pts, color, 1.5));
        }
        let hist = self.effective_primary_history();
        for (name, color, comp) in [
            ("Mx (Transverse)", palette.axis_x, 0_usize),
            ("My (Transverse)", palette.axis_y, 1_usize),
            ("Mz (Longitudinal)", palette.axis_z, 2_usize),
        ] {
            let pts: Vec<[f64; 2]> = hist.iter().map(|(t, m)| [*t, m[comp]]).collect();
            if pts.len() >= 2 {
                prims.push(egui_plot::ExportPrimitive::Polyline {
                    name: name.to_string(),
                    points: pts.into(),
                    color,
                    stroke_width: 1.8,
                    closed: false,
                    fill: None,
                });
            }
        }
        prims
    }
}
