//! Transport, divergence HUD, and comparative dataset/figure export helpers for [`super::SirTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SIR_DT;
use crate::app::SemanticPalette;
use crate::framework::{
    announce_branch_cleared, announce_branch_forked, build_branch_metadata_dataset_2d,
    build_dashed_export_primitives, build_fork_vline_primitive, BranchDivergenceHud,
    BranchMetadata, PlaybackAction, PlaybackTransport, TimelineBranchAction,
    TimelineTransportState,
};
use crate::tabs::epidemiology::sir::SirTool;
use eframe::egui;
use egui_plot::{ExportPrimitive, PlotPoint};

impl SirTool {
    pub(in crate::tabs::epidemiology::sir) fn sir_timeline_state(&self) -> TimelineTransportState {
        let (min_s, max_s) = self
            .branch
            .timeline
            .step_bounds()
            .unwrap_or((0, self.branch.current_step));
        TimelineTransportState {
            current_step: self.branch.current_step,
            min_step: min_s,
            max_step: max_s.max(self.branch.current_step),
            current_time: self.branch.current_step as f64 * SIR_DT,
            time_unit: "days",
            fork_active: self.branch.shadow.is_some(),
            fork_step: self.branch.shadow.as_ref().map(|tw| tw.fork_step),
            fork_time: self.branch.shadow.as_ref().map(|tw| tw.fork_time),
        }
    }

    pub(in crate::tabs::epidemiology::sir) fn apply_sir_timeline_action(
        &mut self,
        action: TimelineBranchAction,
        ctx: Option<&egui::Context>,
        announce: bool,
    ) {
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
                    .branch
                    .timeline
                    .previous_step_before(self.branch.current_step)
                    .unwrap_or_else(|| self.branch.current_step.saturating_sub(1));
                self.scrub_to_step(prev);
            }
            TimelineBranchAction::ScrubTo(step) => self.scrub_to_step(step),
            TimelineBranchAction::ToggleForkBranch => {
                if self.branch.shadow.is_some() {
                    if announce {
                        announce_branch_cleared(actx, "SIR Model");
                    }
                    self.clear_branch();
                } else {
                    let k = self.branch.current_step;
                    if announce {
                        announce_branch_forked(actx, "SIR Model", k, k as f64 * SIR_DT, "days");
                    }
                    self.fork_branch();
                }
            }
            TimelineBranchAction::ClearBranch => {
                if announce {
                    announce_branch_cleared(actx, "SIR Model");
                }
                self.clear_branch();
            }
        }
    }

    fn sir_peak_reduction_summary(&self) -> Option<String> {
        let twin = self.branch.shadow.as_ref()?;
        let base_peak = twin.i_points.iter().map(|p| p.y).fold(0.0_f64, f64::max);
        let branch_peak = self.i_points.iter().map(|p| p.y).fold(0.0_f64, f64::max);
        if base_peak <= 0.0 {
            return None;
        }
        let delta_pct = ((branch_peak - base_peak) / base_peak) * 100.0;
        Some(format!(
            "Peak I: {branch_peak:.1} vs {base_peak:.1} ({delta_pct:+.1}%)"
        ))
    }

    pub(in crate::tabs::epidemiology::sir) fn render_sir_transport_and_hud(
        &mut self,
        ui: &mut egui::Ui,
    ) {
        let ts = self.sir_timeline_state();
        let (play_act, branch_act) =
            PlaybackTransport::in_thread("sir_transport", "SIR Model", self.branch.is_playing)
                .can_reset(true)
                .show_with_timeline(ui, ts);
        if let Some(pa) = play_act {
            match pa {
                PlaybackAction::Play => self.branch.is_playing = true,
                PlaybackAction::Pause => self.branch.is_playing = false,
                PlaybackAction::Step => {
                    self.branch.is_playing = false;
                    self.step_lockstep(1);
                }
                PlaybackAction::StepBack => {
                    self.branch.is_playing = false;
                    self.apply_sir_timeline_action(
                        TimelineBranchAction::StepBack,
                        Some(ui.ctx()),
                        false,
                    );
                }
                PlaybackAction::ScrubTo(step) => {
                    self.branch.is_playing = false;
                    self.scrub_to_step(step);
                }
                PlaybackAction::Reset => {
                    self.branch.is_playing = false;
                    self.recalculate();
                }
            }
        }
        if let Some(ba) = branch_act {
            self.apply_sir_timeline_action(ba, Some(ui.ctx()), false);
        }
        if self.branch.is_playing {
            self.step_lockstep(2);
            ui.ctx().request_repaint();
        }
        let peak_summary = self.sir_peak_reduction_summary();
        if let Some(ref twin) = self.branch.shadow {
            let mut hud = BranchDivergenceHud::new(
                "sir_branch_hud",
                "SIR Model",
                twin.fork_step,
                twin.fork_time,
                "days",
                twin.divergence.metrics(),
            );
            if let Some(ref summary) = peak_summary {
                hud = hud.with_extra_summary(summary);
            }
            if hud.show(ui) {
                announce_branch_cleared(ui.ctx(), "SIR Model");
                self.clear_branch();
            }
        }
    }

    pub(in crate::tabs::epidemiology::sir) fn sir_branch_metadata(&self) -> Option<BranchMetadata> {
        let twin = self.branch.shadow.as_ref()?;
        Some(BranchMetadata {
            fork_step: twin.fork_step,
            fork_time: twin.fork_time,
            time_unit: "days",
            baseline_summary: twin.baseline_summary.clone(),
            branch_summary: self.param_summary(),
            divergence: twin.divergence.metrics(),
        })
    }

    pub(in crate::tabs::epidemiology::sir) fn append_branch_export_datasets(
        &self,
        datasets: &mut Vec<(String, Vec<[f64; 2]>)>,
    ) {
        let Some(ref twin) = self.branch.shadow else {
            return;
        };
        let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
        let t_k = twin.fork_time;
        datasets.push((
            format!("Susceptible (Shadow Baseline @ t_k={t_k:.2})"),
            to_pairs(&twin.s_points),
        ));
        datasets.push((
            format!("Infected (Shadow Baseline @ t_k={t_k:.2})"),
            to_pairs(&twin.i_points),
        ));
        datasets.push((
            format!("Recovered (Shadow Baseline @ t_k={t_k:.2})"),
            to_pairs(&twin.r_points),
        ));
        if let Some(meta) = self.sir_branch_metadata() {
            datasets.push(build_branch_metadata_dataset_2d(&meta));
        }
    }

    pub(in crate::tabs::epidemiology::sir) fn build_sir_export_primitives(
        &self,
    ) -> Vec<ExportPrimitive> {
        let Some(ref twin) = self.branch.shadow else {
            return egui_plot::get_last_plot_primitives();
        };
        let palette = SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
        let mut prims = Vec::new();
        for (name, pts, col) in [
            ("Susceptible", &self.s_points, palette.axis_z),
            ("Infected", &self.i_points, palette.axis_x),
            ("Recovered", &self.r_points, palette.axis_y),
        ] {
            let pairs = to_pairs(pts);
            if pairs.len() >= 2 {
                prims.push(ExportPrimitive::Polyline {
                    name: name.to_string(),
                    points: pairs.into(),
                    color: col,
                    stroke_width: 2.0,
                    closed: false,
                    fill: None,
                });
            }
        }
        for (name, pts, col) in [
            ("Susceptible (Shadow)", &twin.s_points, palette.axis_z),
            ("Infected (Shadow)", &twin.i_points, palette.axis_x),
            ("Recovered (Shadow)", &twin.r_points, palette.axis_y),
        ] {
            prims.extend(build_dashed_export_primitives(
                name,
                &to_pairs(pts),
                col,
                1.7,
            ));
        }
        if let Some(meta) = self.sir_branch_metadata() {
            prims.push(build_fork_vline_primitive(
                &meta,
                0.0,
                self.n.max(1.0),
                palette.accent,
            ));
        }
        prims
    }
}
