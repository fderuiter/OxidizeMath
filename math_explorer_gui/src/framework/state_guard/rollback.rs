#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::StateSnapshot;
use crate::framework::recovery::*;
use crate::framework::SimulationFramework;

impl SimulationFramework {
    fn try_restore_snapshot_or_clamped(&mut self, state_str: &str) -> Option<(&'static str, bool)> {
        let base_ref = self.baseline_tool_state.as_deref();
        let tool = self.active_tool.as_mut()?;
        if tool.try_load_recovery_state(state_str).is_ok() && tool.has_finite_parameters() {
            return Some((
                "Restored exact pre-divergence parameter configuration and plot data",
                false,
            ));
        }
        let constraints = tool.theory().theory_parameters();
        let clamped = crate::framework::clamp::clamp_state_json(state_str, base_ref, &constraints)?;
        if tool.try_load_recovery_state(&clamped).is_ok() && tool.has_finite_parameters() {
            return Some((
                "Snapshot validation failed; restored clamped parameter configuration",
                true,
            ));
        }
        None
    }

    fn restore_verified_recovery_state(
        &mut self,
        snapshot_state: Option<&str>,
    ) -> (&'static str, bool) {
        if self.active_tool.is_none() {
            return (
                "Restored exact pre-divergence parameter configuration and plot data",
                false,
            );
        }
        if let Some(state_str) = snapshot_state {
            if let Some(outcome) = self.try_restore_snapshot_or_clamped(state_str) {
                return outcome;
            }
        } else if self
            .active_tool
            .as_ref()
            .is_some_and(|t| t.has_finite_parameters())
        {
            return (
                "Restored exact pre-divergence parameter configuration and plot data",
                false,
            );
        }

        let base_clone = self.baseline_tool_state.clone();
        if let (Some(tool), Some(ref base)) = (&mut self.active_tool, &base_clone) {
            if tool.try_load_recovery_state(base).is_ok() && tool.has_finite_parameters() {
                return (
                    "Snapshot validation failed; restored baseline parameter configuration",
                    true,
                );
            }
        }
        if let Some(tool) = &mut self.active_tool {
            if tool.clamp_parameters() && tool.has_finite_parameters() {
                return (
                    "Snapshot validation failed; clamped active tool parameters",
                    true,
                );
            }
            if let Some(&meta) = self
                .selected_tool_index
                .and_then(|idx| self.available_tools.get(idx))
            {
                let fresh = (meta.build)();
                self.baseline_tool_state = fresh.save_state();
                *tool = fresh;
            }
        }
        (
            "Snapshot validation failed; rebuilt default baseline tool",
            true,
        )
    }

    /// Executes the Step-Back action: restores pre-divergence parameters without clobbering pinned baselines.
    pub fn rollback_step_back(&mut self) -> bool {
        let Some(snapshot) = self.last_valid_snapshot.clone() else {
            return false;
        };
        let (details, used_fallback) =
            self.restore_verified_recovery_state(snapshot.tool_state.as_deref());
        self.bump_framework_revision();
        let plots = if !used_fallback
            && inspect_shared_datasets_for_divergence(&snapshot.plot_datasets, &[]).is_ok()
        {
            snapshot.plot_datasets
        } else {
            Vec::new()
        };
        self.restore_snapshot_plots(plots);
        if used_fallback {
            self.refresh_saved_and_snapshot_after_reset();
        }

        emit_recovery_telemetry("SimulationFramework", "Step Back", details);
        true
    }

    /// Executes the Auto-Scale Step Size action: halves `dt` on the recovery snapshot while preserving pinned baselines.
    pub fn rollback_auto_scale_step_size(&mut self) -> bool {
        let Some(snapshot) = self.last_valid_snapshot.clone() else {
            return false;
        };
        let scaled = snapshot
            .tool_state
            .as_deref()
            .and_then(auto_scale_state_json);
        let mut applied_scaled = false;
        if let (Some(tool), Some(ref halved_state)) = (&mut self.active_tool, &scaled) {
            if tool.try_load_recovery_state(halved_state).is_ok() && tool.has_finite_parameters() {
                applied_scaled = true;
            }
        }
        let mut used_fallback = false;
        if !applied_scaled {
            let (_, fb) = self.restore_verified_recovery_state(snapshot.tool_state.as_deref());
            used_fallback = fb;
        }
        self.bump_framework_revision();
        let plots = if !used_fallback
            && inspect_shared_datasets_for_divergence(&snapshot.plot_datasets, &[]).is_ok()
        {
            snapshot.plot_datasets
        } else {
            Vec::new()
        };
        self.restore_snapshot_plots(plots);
        if used_fallback {
            self.refresh_saved_and_snapshot_after_reset();
        }

        emit_recovery_telemetry(
            "SimulationFramework",
            "Auto-scale Step Size (dt)",
            "Halved step size and restarted simulation from snapshot",
        );
        true
    }

    pub(super) fn restore_snapshot_plots(
        &mut self,
        plot_datasets: Vec<(String, egui_plot::SharedPoints)>,
    ) {
        egui_plot::clear_last_plot_datasets();
        egui_plot::append_last_plot_shared_datasets(&plot_datasets);
        self.sync_state.datasets_owner_tool = self
            .active_tool
            .as_ref()
            .filter(|_| !plot_datasets.is_empty())
            .map(|t| t.name());
        self.current_valid_plot_datasets = plot_datasets;
        self.rev_cache.last_plot_capture_hash = None;
        self.is_diverged = false;
        self.diagnostic_report = None;
    }

    /// Coordinates parameter clamping on the active tool via domain hooks or JSON bounds fallback.
    pub fn clamp_parameters(&mut self) {
        self.bump_framework_revision();
        let Some(snapshot) = self.last_valid_snapshot.clone() else {
            self.clamp_without_snapshot();
            return;
        };

        let mut telemetry_details = "Fallback to baseline parameters: zero snapshot fields changed";
        if let Some(tool) = &mut self.active_tool {
            let live_diverged = (!tool.has_finite_parameters())
                .then(|| tool.save_recovery_state())
                .flatten();
            if let Some(ref state_str) = snapshot.tool_state {
                tool.load_recovery_state(state_str);
            }
            if tool.clamp_parameters() {
                telemetry_details =
                    "Clamped instability parameters via domain hook and restored plot datasets";
            } else {
                let constraints = tool.theory().theory_parameters();
                let base_ref = self.baseline_tool_state.as_deref();
                let clamped = snapshot
                    .tool_state
                    .as_deref()
                    .and_then(|s| {
                        crate::framework::clamp::clamp_state_json(s, base_ref, &constraints)
                    })
                    .or_else(|| {
                        live_diverged.as_deref().and_then(|s| {
                            crate::framework::clamp::clamp_state_json(s, base_ref, &constraints)
                        })
                    });
                if let Some(clamped_json) = clamped {
                    tool.load_recovery_state(&clamped_json);
                    telemetry_details =
                        "Clamped snapshot JSON parameters to stable bounds and restored plot datasets";
                } else if let Some(base) = base_ref {
                    tool.load_recovery_state(base);
                }
            }
        }

        self.restore_snapshot_plots(snapshot.plot_datasets);
        emit_recovery_telemetry(
            "SimulationFramework",
            "Clamp Instability Parameters",
            telemetry_details,
        );
    }

    fn clamp_without_snapshot(&mut self) {
        if let (Some(tool), Some(ref base)) = (&mut self.active_tool, &self.baseline_tool_state) {
            tool.load_recovery_state(base);
        } else if let Some(tool) = &mut self.active_tool {
            let _ = tool.clamp_parameters();
        }
        let valid_plots = self.current_valid_plot_datasets.clone();
        self.restore_snapshot_plots(valid_plots);
        emit_recovery_telemetry(
            "SimulationFramework",
            "Clamp Instability Parameters",
            "Fallback to baseline parameters: missing last_valid_snapshot",
        );
    }

    /// Coordinates parameter reset to baseline on the active tool.
    pub fn reset_parameters(&mut self) {
        self.bump_framework_revision();
        if let Some(baseline_state) = self.baseline_tool_state.clone() {
            if let Some(tool) = &mut self.active_tool {
                tool.load_state(&baseline_state);
            } else if let Some(&meta) = self
                .selected_tool_index
                .and_then(|idx| self.available_tools.get(idx))
            {
                let mut new_tool = (meta.build)();
                new_tool.load_state(&baseline_state);
                self.active_tool = Some(new_tool);
            }
        } else if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            let new_tool = (meta.build)();
            self.baseline_tool_state = new_tool.save_state();
            self.active_tool = Some(new_tool);
        }
        egui_plot::clear_last_plot_datasets();
        self.current_valid_plot_datasets.clear();
        self.rev_cache.last_plot_capture_hash = None;
        self.sync_state.datasets_owner_tool = None;
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.step_counter = 0;
        self.refresh_saved_and_snapshot_after_reset();
        emit_recovery_telemetry(
            "SimulationFramework",
            "Reset All Parameters",
            "Restored default baseline configuration",
        );
    }

    fn refresh_saved_and_snapshot_after_reset(&mut self) {
        let Some(tool) = &self.active_tool else {
            self.last_valid_snapshot = None;
            return;
        };
        let restored_state = tool.save_state();
        let recovery_state = tool.save_recovery_state();
        let rev = tool.state_revision();
        if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            if meta.id != tool.name() {
                self.saved_tool_states.remove(meta.id);
            }
        }
        if let Some(state) = restored_state {
            self.saved_tool_states
                .insert(tool.name().to_string(), state);
        } else {
            self.saved_tool_states.remove(tool.name());
        }
        self.rev_cache.last_snapshot_rev = (rev > 0).then_some((tool.name(), rev));
        self.rev_cache.last_snapshot_source_datasets.clear();
        let mut snap = StateSnapshot {
            tool_state: recovery_state,
            plot_datasets: Vec::new(),
            step_index: 0,
            timestamp: crate::web_io::PlatformInstant::now(),
        };
        snap.enforce_memory_budget();
        self.last_valid_snapshot = Some(snap);
    }

    pub(crate) fn sync_active_baseline(&mut self) {
        self.bump_framework_revision();
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.step_counter = 0;
        let Some(tool) = &self.active_tool else {
            return;
        };
        let state = tool.save_state();
        let recovery = tool.save_recovery_state();
        let rev = tool.state_revision();
        self.baseline_tool_state = state.clone();
        if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            if meta.id != tool.name() {
                self.saved_tool_states.remove(meta.id);
            }
        }
        if let Some(s) = state {
            self.saved_tool_states.insert(tool.name().to_string(), s);
        } else {
            self.saved_tool_states.remove(tool.name());
        }
        let datasets = crate::framework::persistence::capture_isolated_tool_datasets(tool.as_ref());
        self.sync_state.datasets_owner_tool = Some(tool.name());
        self.current_valid_plot_datasets = datasets.clone();
        self.rev_cache.last_plot_capture_hash = None;
        self.rev_cache.last_snapshot_rev = (rev > 0).then_some((tool.name(), rev));
        self.rev_cache.last_snapshot_source_datasets = datasets.clone();
        let mut snap = StateSnapshot {
            tool_state: recovery,
            plot_datasets: datasets,
            step_index: 0,
            timestamp: crate::web_io::PlatformInstant::now(),
        };
        snap.enforce_memory_budget();
        self.last_valid_snapshot = Some(snap);
    }

    /// Applies a preset JSON envelope to the active tool, refreshing baseline/snapshot on success.
    pub fn apply_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        let res = match &mut self.active_tool {
            Some(tool) => tool.import_preset_json(json_str),
            None => Err("No active tool selected for preset import".to_string()),
        };
        match res {
            Ok(()) => {
                self.preset_dialog.toast_error = None;
                self.sync_active_baseline();
                Ok(())
            }
            Err(err) => {
                self.preset_dialog.record_import_error(err.clone());
                Err(err)
            }
        }
    }

    /// Rebuilds a clean target tool instance for a starter card and synchronizes baseline/reset snapshots.
    pub fn apply_starter_card(
        &mut self,
        card_title: &str,
        tool_name: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        let target_idx = match tool_name {
            Some(target_name) => {
                let Some(idx) = self
                    .available_tools
                    .iter()
                    .position(|t| t.id == target_name || t.display_name == target_name)
                else {
                    let err = format!(
                        "Starter preset '{card_title}' targets unavailable tool '{target_name}'"
                    );
                    self.preset_dialog.record_import_error(err.clone());
                    return Err(err);
                };
                Some(idx)
            }
            None => self.selected_tool_index,
        };

        if let Some(idx) = target_idx {
            self.activate_starter_card_tool(idx, card_title, preset_json)?;
        } else if let Some(json_str) = preset_json {
            let Some(tool) = &mut self.active_tool else {
                let err = format!("No active tool available to launch '{card_title}'");
                self.preset_dialog.record_import_error(err.clone());
                return Err(err);
            };
            if let Err(err) = tool.import_preset_json(json_str) {
                let msg = format!("Failed to launch '{card_title}': {err}");
                self.preset_dialog.record_import_error(msg.clone());
                return Err(msg);
            }
            self.preset_dialog.toast_error = None;
            self.sync_active_baseline();
        }
        self.show_starter_gallery = false;
        Ok(())
    }

    fn activate_starter_card_tool(
        &mut self,
        idx: usize,
        card_title: &str,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        let meta = self.available_tools[idx];
        let mut new_tool = (meta.build)();
        if let Some(json_str) = preset_json {
            if let Err(err) = new_tool.import_preset_json(json_str) {
                let msg = format!("Failed to launch '{card_title}': {err}");
                self.preset_dialog.record_import_error(msg.clone());
                return Err(msg);
            }
        }
        if self.selected_tool_index != Some(idx) {
            if let Some((old_name, old_json)) = self.serialized_active_tool() {
                self.saved_tool_states
                    .insert(old_name.to_string(), old_json.to_string());
            }
        }
        scientific_metadata::citation_registry::CitationRegistry::register(
            new_tool.name().to_string(),
            new_tool.theory().theory_citation(),
        );
        self.selected_tool_index = Some(idx);
        self.active_tool = Some(new_tool);
        self.preset_dialog.toast_error = None;
        self.sync_active_baseline();
        Ok(())
    }

    /// Rebuilds a clean target tool instance for a starter card, applies its preset JSON,
    /// and synchronizes baseline/reset snapshots.
    pub fn launch_starter_card(
        &mut self,
        card: &crate::starter_gallery::StarterCard,
    ) -> Result<(), String> {
        self.apply_starter_card(card.title, card.tool_name, card.preset_json)
    }

    /// Dispatches a StateGuardAction remedy.
    pub fn handle_state_guard_action(&mut self, action: StateGuardAction) {
        match action {
            StateGuardAction::StepBack => {
                self.rollback_step_back();
            }
            StateGuardAction::AutoScaleStepSize => {
                self.rollback_auto_scale_step_size();
            }
            StateGuardAction::ClampParameters => {
                self.clamp_parameters();
            }
            StateGuardAction::ResetParameters => {
                self.reset_parameters();
            }
        }
    }
}
