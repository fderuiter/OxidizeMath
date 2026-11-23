#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::StateSnapshot;
use crate::framework::recovery::emit_recovery_telemetry;
use crate::framework::SimulationFramework;

impl SimulationFramework {
    /// Returns true when the active tool has a loaded preset baseline distinct from or layered atop factory defaults.
    #[must_use]
    pub fn has_preset_baseline(&self) -> bool {
        self.preset_tool_state.is_some()
    }

    fn ensure_factory_baseline(&mut self) {
        if self.factory_tool_state.is_some() {
            return;
        }
        if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            self.factory_tool_state = (meta.build)().save_state();
        }
    }

    fn apply_target_parameter_state(&mut self, target_state: &str) {
        if let Some(tool) = &mut self.active_tool {
            tool.load_state(target_state);
        } else if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            let mut new_tool = (meta.build)();
            new_tool.load_state(target_state);
            self.active_tool = Some(new_tool);
        }
    }

    fn clear_runtime_plots_and_divergence(&mut self) {
        egui_plot::clear_last_plot_datasets();
        self.current_valid_plot_datasets.clear();
        self.current_valid_plot_datasets_3d.clear();
        self.sync_state.datasets_owner_tool = None;
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.step_counter = 0;
    }

    /// Reverts active tool parameters to the active loaded preset baseline while preserving `factory_tool_state`.
    pub fn reset_to_preset(&mut self) -> bool {
        let Some(preset_state) = self.preset_tool_state.clone() else {
            return false;
        };
        self.bump_framework_revision();
        self.ensure_factory_baseline();
        self.baseline_tool_state = Some(preset_state.clone());
        self.apply_target_parameter_state(&preset_state);
        self.clear_runtime_plots_and_divergence();
        self.reset_active_tool_manual_bounds();
        self.on_hud_parameters_reset();
        self.refresh_saved_and_snapshot_after_reset();
        emit_recovery_telemetry(
            "SimulationFramework",
            "Revert to Preset",
            "Reverted parameters to active loaded preset baseline",
        );
        true
    }

    /// Restores active tool parameters to initial factory defaults without erasing any active preset baseline.
    pub fn reset_to_factory_defaults(&mut self) {
        self.bump_framework_revision();
        self.ensure_factory_baseline();
        if let Some(factory_state) = self.factory_tool_state.clone() {
            self.apply_target_parameter_state(&factory_state);
            if self.preset_tool_state.is_none() {
                self.baseline_tool_state = Some(factory_state);
            }
        } else if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            let new_tool = (meta.build)();
            let saved = new_tool.save_state();
            self.factory_tool_state = saved.clone();
            if self.preset_tool_state.is_none() {
                self.baseline_tool_state = saved;
            }
            self.active_tool = Some(new_tool);
        }
        self.clear_runtime_plots_and_divergence();
        self.reset_active_tool_manual_bounds();
        self.on_hud_parameters_reset();
        self.refresh_saved_and_snapshot_after_reset();
        emit_recovery_telemetry(
            "SimulationFramework",
            "Factory Defaults",
            "Restored initial factory default parameters",
        );
    }

    /// Coordinates parameter reset to the active baseline (loaded preset when present, else factory defaults).
    pub fn reset_parameters(&mut self) {
        self.bump_framework_revision();
        let target = self
            .baseline_tool_state
            .clone()
            .or_else(|| self.preset_tool_state.clone())
            .or_else(|| self.factory_tool_state.clone());
        if let Some(baseline_state) = target {
            self.baseline_tool_state = Some(baseline_state.clone());
            self.apply_target_parameter_state(&baseline_state);
        } else if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            let new_tool = (meta.build)();
            let factory = new_tool.save_state();
            self.factory_tool_state = factory.clone();
            self.baseline_tool_state = factory;
            self.active_tool = Some(new_tool);
        }
        self.clear_runtime_plots_and_divergence();
        self.reset_active_tool_manual_bounds();
        self.on_hud_parameters_reset();
        self.refresh_saved_and_snapshot_after_reset();
        emit_recovery_telemetry(
            "SimulationFramework",
            "Reset All Parameters",
            "Restored default baseline configuration",
        );
    }

    pub(in crate::framework) fn refresh_saved_and_snapshot_after_reset(&mut self) {
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
        let mut snap = StateSnapshot {
            tool_state: recovery_state,
            plot_datasets: Vec::new(),
            step_index: 0,
            timestamp: crate::web_io::PlatformInstant::now(),
        };
        snap.enforce_memory_budget();
        self.last_valid_snapshot = Some(snap);
    }

    /// Synchronizes the active loaded preset baseline while preserving `factory_tool_state`.
    pub(crate) fn sync_active_baseline(&mut self) {
        self.bump_framework_revision();
        self.ensure_factory_baseline();
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.step_counter = 0;
        let Some(tool) = &self.active_tool else {
            return;
        };
        let state = tool.save_state();
        let recovery = tool.save_recovery_state();
        let rev = tool.state_revision();
        let tool_name = tool.name().to_string();
        self.preset_tool_state = state.clone();
        self.baseline_tool_state = state.clone().or_else(|| self.factory_tool_state.clone());
        if let Some(&meta) = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx))
        {
            if meta.id != tool.name() {
                self.saved_tool_states.remove(meta.id);
                self.preset_tool_states.remove(meta.id);
            }
        }
        if let Some(s) = state {
            self.preset_tool_states.insert(tool_name.clone(), s.clone());
            self.saved_tool_states.insert(tool_name, s);
        } else {
            self.preset_tool_states.remove(&tool_name);
            self.saved_tool_states.remove(&tool_name);
        }
        let datasets = crate::framework::persistence::capture_isolated_tool_datasets(tool.as_ref());
        self.sync_state.datasets_owner_tool = Some(tool.name());
        self.current_valid_plot_datasets = datasets.clone();
        self.rev_cache.last_snapshot_rev = (rev > 0).then_some((tool.name(), rev));
        let mut snap = StateSnapshot {
            tool_state: recovery,
            plot_datasets: datasets,
            step_index: 0,
            timestamp: crate::web_io::PlatformInstant::now(),
        };
        snap.enforce_memory_budget();
        self.last_valid_snapshot = Some(snap);
    }

    /// Applies a preset JSON envelope to the active tool, refreshing preset baseline/snapshot on success.
    pub fn apply_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        let res = match &mut self.active_tool {
            Some(tool) => tool.import_preset_json(json_str),
            None => Err("No active tool selected for preset import".to_string()),
        };
        match res {
            Ok(()) => {
                self.preset_dialog.toast_error = None;
                self.reset_active_tool_manual_bounds();
                self.sync_active_baseline();
                Ok(())
            }
            Err(err) => {
                self.preset_dialog.record_import_error(err.clone());
                Err(err)
            }
        }
    }

    /// Rebuilds a clean target tool instance for a starter card and synchronizes dual baselines/snapshots.
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
            self.active_challenge = None;
            self.challenge_was_passed = false;
            let _ = tool.apply_challenge(None);
            self.preset_dialog.toast_error = None;
            self.reset_active_tool_manual_bounds();
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
        let factory_state = new_tool.save_state();
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
        self.active_challenge = None;
        self.challenge_was_passed = false;
        self.selected_tool_index = Some(idx);
        self.factory_tool_state = factory_state.clone();
        self.active_tool = Some(new_tool);
        self.promote_legacy_plot_keys_for_active_tool();
        self.preset_dialog.toast_error = None;
        self.reset_active_tool_manual_bounds();
        if preset_json.is_some() {
            self.sync_active_baseline();
        } else {
            self.preset_tool_state = None;
            if let Some(tool) = &self.active_tool {
                self.preset_tool_states.remove(tool.name());
            }
            self.baseline_tool_state = factory_state;
            self.refresh_saved_and_snapshot_after_reset();
        }
        Ok(())
    }

    /// Rebuilds a clean target tool instance for a starter card, applies its preset JSON,
    /// and synchronizes dual baseline/reset snapshots (activating Guided Story or Challenge HUDs when matched).
    pub fn launch_starter_card(
        &mut self,
        card: &crate::starter_gallery::StarterCard,
    ) -> Result<(), String> {
        if let Some(story) =
            crate::starter_gallery::find_story_by_id(card.id, Some(self.domain.as_str()))
        {
            return self.launch_story(story, 0);
        }
        if let Some(challenge) =
            crate::starter_gallery::find_challenge_by_id(card.id, Some(self.domain.as_str()))
        {
            return self.launch_challenge(challenge);
        }
        self.clear_challenge_hud();
        self.apply_starter_card(card.title, card.tool_name, card.preset_json)
    }
}
