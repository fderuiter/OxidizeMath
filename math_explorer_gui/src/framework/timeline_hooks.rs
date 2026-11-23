//! `SimulationFramework` delegation hooks for timeline scrubbing and shadow-twin branch inspection.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BranchDivergenceMetrics, BranchMetadata, InteractiveTool, SimulationFramework,
    TimelineBranchAction, TimelineTransportState,
};
use eframe::egui;

pub(super) fn export_tool_preset_json<T: InteractiveTool + ?Sized>(
    tool: &T,
) -> Result<String, String> {
    if let Some(state_str) = tool.save_state() {
        let config: serde_json::Value =
            serde_json::from_str(&state_str).map_err(|e| format!("Serialization error: {e}"))?;
        crate::presets::serialize_preset_envelope(tool.name(), tool.domain(), config)
    } else {
        Err(format!("Preset export not supported for {}", tool.name()))
    }
}

pub(super) fn import_tool_preset_json<T: InteractiveTool + ?Sized>(
    tool: &mut T,
    json_str: &str,
) -> Result<(), String> {
    let bounds = tool.state_bounds();
    let config: Box<serde_json::value::RawValue> =
        crate::presets::deserialize_preset_envelope_bounded(
            json_str,
            tool.domain(),
            tool.name(),
            bounds,
        )?;
    let config_str = config.get();
    bounds.validate_json(config_str)?;
    tool.try_load_state(config_str)
}

pub(super) fn sample_tool_animation_frames<T: InteractiveTool + ?Sized>(
    tool: &mut T,
    num_frames: usize,
) -> Vec<egui_plot::export::DatasetFrame> {
    let count = num_frames.max(1);
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        tool.step_simulation();
        let ds = tool.export_datasets();
        if !ds.is_empty() {
            frames.push(ds);
        }
    }
    if frames.is_empty() {
        let ds = tool.export_datasets();
        if !ds.is_empty() {
            frames = egui_plot::export::sample_progressive_frames(&ds, count);
        }
    }
    frames
}

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
