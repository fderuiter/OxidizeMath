//! `SimulationFramework` delegation hooks for timeline scrubbing and shadow-twin branch inspection.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BranchDivergenceMetrics, BranchMetadata, SimulationFramework, TimelineBranchAction,
    TimelineTransportState,
};
use eframe::egui;

impl SimulationFramework {
    /// Returns the active tool's timeline scrubber and shadow-branch state, if supported.
    #[must_use]
    pub fn timeline_transport_state(&self) -> Option<TimelineTransportState> {
        self.active_tool
            .as_ref()
            .and_then(|tool| tool.timeline_transport_state())
    }

    /// Dispatches a [`TimelineBranchAction`] (`StepBack`, `ScrubTo`, `ToggleForkBranch`, `ClearBranch`)
    /// to the active tool and invalidates cached export snapshots.
    pub fn apply_timeline_action(
        &mut self,
        action: TimelineBranchAction,
        ctx: Option<&egui::Context>,
    ) -> bool {
        let Some(tool) = self.active_tool.as_mut() else {
            return false;
        };
        if tool.timeline_transport_state().is_none() {
            return false;
        }
        tool.apply_timeline_action(action, ctx);
        if let Some(egui_ctx) = ctx {
            match action {
                TimelineBranchAction::ToggleForkBranch => {
                    if let Some(ts) = tool.timeline_transport_state() {
                        if ts.fork_active {
                            super::announce_branch_forked(
                                egui_ctx,
                                tool.name(),
                                ts.fork_step.unwrap_or(ts.current_step),
                                ts.fork_time.unwrap_or(ts.current_time),
                                ts.time_unit,
                            );
                        } else {
                            super::announce_branch_cleared(egui_ctx, tool.name());
                        }
                    }
                }
                TimelineBranchAction::ClearBranch => {
                    super::announce_branch_cleared(egui_ctx, tool.name());
                }
                TimelineBranchAction::StepBack | TimelineBranchAction::ScrubTo(_) => {}
            }
        }
        true
    }

    /// Returns live [`BranchDivergenceMetrics`] when the active tool has an active shadow twin.
    #[must_use]
    pub fn branch_divergence_metrics(&self) -> Option<BranchDivergenceMetrics> {
        self.active_tool
            .as_ref()
            .and_then(|tool| tool.branch_divergence_metrics())
    }

    /// Returns structured [`BranchMetadata`] from $t_k$ for the active tool when forked.
    #[must_use]
    pub fn branch_export_metadata(&self) -> Option<BranchMetadata> {
        self.active_tool
            .as_ref()
            .and_then(|tool| tool.branch_export_metadata())
    }
}
