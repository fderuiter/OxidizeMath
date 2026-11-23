//! UI controls, curve/raster overlays, and dataset/figure export helpers for [`super::UnifiedSimTool`] shadow branches.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::app::SemanticPalette;
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::{
    announce_branch_cleared, announce_branch_forked, build_branch_metadata_dataset_2d,
    build_dashed_export_primitives, build_fork_vline_primitive, BranchDivergenceHud,
    BranchDivergenceMetrics, BranchMetadata, PlaybackAction, PlaybackTransport,
    RasterBranchViewMode, TimelineBranchAction, TimelineTransportState,
};
use eframe::egui;
use egui_plot::{ExportPrimitive, Line, LineStyle, PlotPoints, VLine};

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn draw_unified_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let state = self.controller.state();
        let transport = PlaybackTransport::from_worker_state("unified_transport", M::name(), state)
            .can_reset(true);

        if !M::supports_timeline_branch() {
            if let Some(play_act) = transport.show(ui) {
                self.handle_unified_play_action(play_act, ctx);
            }
            return;
        }

        let Some(ts) = self.build_timeline_transport_state() else {
            return;
        };
        let (play_act, branch_act) = transport.show_with_timeline(ui, ts);
        if let Some(act) = play_act {
            self.handle_unified_play_action(act, ctx);
        }
        if let Some(b_act) = branch_act {
            self.dispatch_unified_timeline_cmd(b_act, Some(ctx));
        }

        self.draw_unified_branch_hud(ui, ctx);
    }

    fn handle_unified_play_action(&mut self, action: PlaybackAction, ctx: &egui::Context) {
        match action {
            PlaybackAction::Play => {
                let _ = self.controller.send_command(SimCommand::Start);
            }
            PlaybackAction::Pause => {
                let _ = self.controller.send_command(SimCommand::Pause);
            }
            PlaybackAction::Step => {
                let _ = self.controller.send_command(SimCommand::Step(1));
            }
            PlaybackAction::Reset => {
                self.reset_simulation(Some(ctx));
            }
        }
    }

    fn draw_unified_branch_hud(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let snap = self
            .branch_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if !snap.fork_active {
            return;
        }
        ui.add_space(4.0);
        if M::is_raster_model() {
            let mut mode = snap.view_mode;
            ui.horizontal_wrapped(|ui| {
                ui.label("Branch View:");
                for candidate in [
                    RasterBranchViewMode::DifferenceHeatmap,
                    RasterBranchViewMode::CounterfactualBranch,
                    RasterBranchViewMode::ShadowBaseline,
                ] {
                    if ui
                        .selectable_value(&mut mode, candidate, candidate.label())
                        .changed()
                    {
                        let _ = self
                            .controller
                            .send_command(SimCommand::SetRasterViewMode(mode));
                        self.dirty = true;
                        ctx.request_repaint();
                    }
                }
            });
        }
        let clear = BranchDivergenceHud::new(
            "unified_branch_hud",
            M::name(),
            snap.fork_step.unwrap_or(snap.current_step),
            snap.fork_time.unwrap_or(snap.current_time),
            M::time_unit(),
            snap.divergence,
        )
        .show(ui);
        if clear {
            announce_branch_cleared(ctx, M::name());
            let _ = self.controller.send_command(SimCommand::ClearBranch);
            self.dirty = true;
            ctx.request_repaint();
        }
    }

    pub(super) fn render_shadow_line_overlay(
        &self,
        plot_ui: &mut egui_plot::PlotUi,
        palette: SemanticPalette,
    ) {
        if !M::supports_timeline_branch() {
            return;
        }
        let guard = self.branch_state.read().unwrap_or_else(|e| e.into_inner());
        if !guard.fork_active {
            return;
        }
        let shadow_pts = pairs_from_interleaved(&guard.shadow_custom_data);
        if shadow_pts.len() >= 2 {
            let line = Line::new(
                format!("{} (Shadow Baseline)", M::name()),
                PlotPoints::from_iter(shadow_pts),
            )
            .color(palette.series_color(1))
            .style(LineStyle::Dashed { length: 7.0_f32 });
            plot_ui.line(line);
        }
        if let Some(fork_t) = guard.fork_time {
            plot_ui.vline(
                VLine::new(format!("Fork @ t_k={fork_t:.3}"), fork_t)
                    .color(palette.accent)
                    .style(LineStyle::Dashed { length: 4.0_f32 }),
            );
        }
    }

    pub(super) fn build_timeline_transport_state(&self) -> Option<TimelineTransportState> {
        if !M::supports_timeline_branch() {
            return None;
        }
        let g = self.branch_state.read().unwrap_or_else(|e| e.into_inner());
        Some(TimelineTransportState {
            current_step: g.current_step,
            min_step: g.min_step,
            max_step: g.max_step,
            current_time: g.current_time,
            time_unit: M::time_unit(),
            fork_active: g.fork_active,
            fork_step: g.fork_step,
            fork_time: g.fork_time,
        })
    }

    pub(super) fn apply_unified_timeline_action(
        &mut self,
        action: TimelineBranchAction,
        ctx: Option<&egui::Context>,
    ) {
        if !M::supports_timeline_branch() {
            return;
        }
        let (fork_active, step, time) = {
            let g = self.branch_state.read().unwrap_or_else(|e| e.into_inner());
            (g.fork_active, g.current_step, g.current_time)
        };
        let fallback_ctx;
        let announce_ctx = if let Some(c) = ctx {
            c
        } else {
            fallback_ctx = egui::Context::default();
            &fallback_ctx
        };
        match action {
            TimelineBranchAction::ToggleForkBranch => {
                if fork_active {
                    announce_branch_cleared(announce_ctx, M::name());
                } else {
                    announce_branch_forked(announce_ctx, M::name(), step, time, M::time_unit());
                }
            }
            TimelineBranchAction::ClearBranch => {
                announce_branch_cleared(announce_ctx, M::name());
            }
            TimelineBranchAction::StepBack | TimelineBranchAction::ScrubTo(_) => {}
        }
        self.dispatch_unified_timeline_cmd(action, ctx);
    }

    fn dispatch_unified_timeline_cmd(
        &mut self,
        action: TimelineBranchAction,
        ctx: Option<&egui::Context>,
    ) {
        if !M::supports_timeline_branch() {
            return;
        }
        let fork_active = self
            .branch_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .fork_active;
        let cmd = match action {
            TimelineBranchAction::StepBack => {
                if self.controller.state() == SimWorkerState::Running {
                    let _ = self.controller.send_command(SimCommand::Pause);
                }
                SimCommand::StepBack
            }
            TimelineBranchAction::ScrubTo(step) => SimCommand::ScrubToStep(step),
            TimelineBranchAction::ToggleForkBranch => {
                if fork_active {
                    SimCommand::ClearBranch
                } else {
                    SimCommand::ForkBranch
                }
            }
            TimelineBranchAction::ClearBranch => SimCommand::ClearBranch,
        };
        let _ = self.controller.send_command(cmd);
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(super) fn unified_branch_divergence(&self) -> Option<BranchDivergenceMetrics> {
        let g = self.branch_state.read().unwrap_or_else(|e| e.into_inner());
        g.fork_active.then_some(g.divergence)
    }

    pub(super) fn unified_branch_metadata(&self) -> Option<BranchMetadata> {
        let g = self.branch_state.read().unwrap_or_else(|e| e.into_inner());
        metadata_from_guard::<M>(&g)
    }

    fn build_raster_branch_datasets(
        g: &super::branching::UnifiedBranchSharedState,
    ) -> Vec<(String, Vec<[f64; 2]>)> {
        if !g.fork_active {
            return Vec::new();
        }
        let fork_t = g.fork_time.unwrap_or(0.0);
        let mut out = Vec::new();
        for (tag, profile) in [
            ("Counterfactual Branch", &g.primary_profile),
            ("Shadow Baseline", &g.shadow_profile),
            ("|Delta| Difference", &g.diff_profile),
        ] {
            if !profile.is_empty() {
                out.push((
                    format!("{} ({tag} @ t_k={fork_t:.3})", M::name()),
                    profile.clone(),
                ));
            }
        }
        if let Some(meta) = metadata_from_guard::<M>(g) {
            out.push(build_branch_metadata_dataset_2d(&meta));
        }
        out
    }

    pub(super) fn build_unified_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let g = self.branch_state.read().unwrap_or_else(|e| e.into_inner());
        if M::is_raster_model() {
            return Self::build_raster_branch_datasets(&g);
        }

        let mut out = Vec::new();
        if let Some(snap) = self
            .stepped_snapshot
            .as_ref()
            .or_else(|| self.controller.latest_snapshot())
        {
            let pts = pairs_from_interleaved(&snap.custom_data);
            if !pts.is_empty() {
                let label = if g.fork_active {
                    format!(
                        "{} (Counterfactual Branch @ t_k={:.3})",
                        M::name(),
                        g.fork_time.unwrap_or(0.0)
                    )
                } else {
                    M::name().to_string()
                };
                out.push((label, pts));
            }
        }
        if g.fork_active {
            let shadow_pts = pairs_from_interleaved(&g.shadow_custom_data);
            if !shadow_pts.is_empty() {
                out.push((
                    format!(
                        "{} (Shadow Baseline @ t_k={:.3})",
                        M::name(),
                        g.fork_time.unwrap_or(0.0)
                    ),
                    shadow_pts,
                ));
            }
            if let Some(meta) = metadata_from_guard::<M>(&g) {
                out.push(build_branch_metadata_dataset_2d(&meta));
            }
        }
        out
    }

    pub(super) fn build_unified_export_primitives(&self) -> Vec<ExportPrimitive> {
        if M::is_raster_model() {
            return Vec::new();
        }
        let g = self.branch_state.read().unwrap_or_else(|e| e.into_inner());
        if !g.fork_active {
            return egui_plot::get_last_plot_primitives();
        }
        let palette = SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        let mut prims = Vec::new();
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        if let Some(snap) = self
            .stepped_snapshot
            .as_ref()
            .or_else(|| self.controller.latest_snapshot())
        {
            let pts = pairs_from_interleaved(&snap.custom_data);
            for &[_, y] in &pts {
                if y.is_finite() {
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
            if pts.len() >= 2 {
                prims.push(ExportPrimitive::Polyline {
                    name: format!("{} (Counterfactual)", M::name()),
                    points: pts,
                    color: palette.series_color(0),
                    stroke_width: 2.0,
                    closed: false,
                    fill: None,
                });
            }
        }
        let shadow_pts = pairs_from_interleaved(&g.shadow_custom_data);
        for &[_, y] in &shadow_pts {
            if y.is_finite() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
        prims.extend(build_dashed_export_primitives(
            &format!("{} (Shadow Baseline)", M::name()),
            &shadow_pts,
            palette.series_color(1),
            1.8,
        ));
        if let Some(meta) = metadata_from_guard::<M>(&g) {
            prims.push(build_fork_vline_primitive(
                &meta,
                y_min,
                y_max,
                palette.accent,
            ));
        }
        prims
    }
}

fn metadata_from_guard<M: UnifiedModel>(
    g: &super::UnifiedBranchSharedState,
) -> Option<BranchMetadata> {
    if !g.fork_active {
        return None;
    }
    Some(BranchMetadata {
        fork_step: g.fork_step.unwrap_or(g.current_step),
        fork_time: g.fork_time.unwrap_or(g.current_time),
        time_unit: M::time_unit(),
        baseline_summary: g.baseline_summary.clone(),
        branch_summary: g.branch_summary.clone(),
        divergence: g.divergence,
    })
}

fn pairs_from_interleaved(custom_data: &[f64]) -> Vec<[f64; 2]> {
    let mut pts = Vec::with_capacity(custom_data.len() / 2);
    for idx in (0..custom_data.len()).step_by(2) {
        if idx + 1 < custom_data.len() {
            pts.push([custom_data[idx], custom_data[idx + 1]]);
        }
    }
    pts
}
