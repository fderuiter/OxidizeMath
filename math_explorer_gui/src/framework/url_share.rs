//! URL fragment state export, share-link encoding, and atomic candidate-instance import
//! for [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ActionBarExportStatus, ExportStatusSeverity, SimulationFramework};
use eframe::egui;
use std::hash::{Hash, Hasher};

impl SimulationFramework {
    pub(super) fn poll_action_bar_clipboard_status(&mut self, ctx: &egui::Context) {
        let status_id = egui::Id::new(("framework_action_bar_clipboard_status", self.instance_id));
        if let Some(async_status) =
            ctx.data_mut(|d| d.remove_temp::<ActionBarExportStatus>(status_id))
        {
            let sev = if async_status.is_error {
                ExportStatusSeverity::Error
            } else {
                ExportStatusSeverity::Success
            };
            self.set_export_status(sev, async_status.message);
        }
    }

    pub(super) fn copy_active_share_link(&mut self, ctx: &egui::Context) {
        match self.encode_share_url() {
            Ok(url) => {
                let status_id =
                    egui::Id::new(("framework_action_bar_clipboard_status", self.instance_id));
                let ctx_cb = ctx.clone();
                crate::url_state::copy_share_url_to_clipboard(ctx, url, move |res| {
                    let status = match res {
                        Ok(()) => ActionBarExportStatus {
                            message: "Copied shareable simulation link to clipboard!".to_string(),
                            is_error: false,
                            expires_at_time: None,
                        },
                        Err(err) => ActionBarExportStatus {
                            message: err,
                            is_error: true,
                            expires_at_time: None,
                        },
                    };
                    ctx_cb.data_mut(|d| d.insert_temp(status_id, status));
                });
            }
            Err(err) => self.preset_dialog.record_import_error(err),
        }
    }

    /// Returns the canonical `ToolMetadata::id` of the currently selected tool, if any.
    #[must_use]
    pub fn active_tool_id(&self) -> Option<&'static str> {
        self.selected_tool_index
            .and_then(|idx| self.available_tools.get(idx).map(|meta| meta.id))
    }

    /// Computes an O(1) parameter-only revision token for live URL fragment synchronization.
    #[must_use]
    pub fn url_state_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.selected_tool_index.hash(&mut hasher);
        self.hash_hud_for_url(&mut hasher);
        if let Some(tool) = &self.active_tool {
            tool.name().hash(&mut hasher);
            let tool_rev = tool.state_revision();
            if tool_rev > 0 {
                1_u8.hash(&mut hasher);
                tool_rev.hash(&mut hasher);
            } else {
                0_u8.hash(&mut hasher);
                self.rev_cache.untracked_tool_revision.hash(&mut hasher);
            }
        }
        hasher.finish().max(1)
    }

    /// Exports the active tool's parameters as a compact, deterministic [`crate::presets::PresetWrapper`] JSON string.
    pub fn export_url_preset_json(&self) -> Result<Option<String>, String> {
        let Some(tool) = &self.active_tool else {
            return Ok(None);
        };
        let Some(state_str) = tool.save_state() else {
            return Ok(None);
        };
        let config: serde_json::Value = serde_json::from_str(&state_str)
            .map_err(|e| format!("Failed to serialize state for '{}': {e}", tool.name()))?;
        let envelope_json =
            crate::presets::serialize_compact_preset_envelope(tool.name(), tool.domain(), config)?;
        Ok(Some(envelope_json))
    }

    /// Encodes a `#tab=...` fragment for `tab_slug`, omitting redundant `&state=` when matching an active story/challenge baseline.
    pub fn encode_share_fragment_for_tab(&self, tab_slug: &str) -> Result<String, String> {
        let current_tool_state = self.active_tool.as_ref().and_then(|t| t.save_state());
        match &self.challenge_hud.mode {
            super::ChallengeHudMode::Story {
                story,
                step_index,
                step_state,
            } => {
                let extra_json = if current_tool_state == *step_state {
                    None
                } else {
                    self.export_url_preset_json()?
                };
                crate::url_state::encode_share_fragment_with_hud(
                    tab_slug,
                    self.active_tool_id(),
                    Some(story.id),
                    None,
                    Some(*step_index),
                    extra_json.as_deref(),
                )
            }
            super::ChallengeHudMode::Challenge {
                challenge,
                initial_state,
                ..
            } => {
                let extra_json = if current_tool_state == *initial_state {
                    None
                } else {
                    self.export_url_preset_json()?
                };
                crate::url_state::encode_share_fragment_with_hud(
                    tab_slug,
                    self.active_tool_id(),
                    None,
                    Some(challenge.id),
                    None,
                    extra_json.as_deref(),
                )
            }
            super::ChallengeHudMode::Inactive => {
                let preset_json = self.export_url_preset_json()?;
                crate::url_state::encode_share_fragment(
                    tab_slug,
                    self.active_tool_id(),
                    preset_json.as_deref(),
                )
            }
        }
    }

    /// Encodes a full shareable web URL (`https://...#tab=...&tool=...&state=...`) for the active tool.
    pub fn encode_share_url(&self) -> Result<String, String> {
        let tab_slug = if self.domain == "biology" {
            "morphogenesis"
        } else {
            self.domain.as_str()
        };
        let fragment = self.encode_share_fragment_for_tab(tab_slug)?;
        Ok(crate::url_state::build_share_url_from_fragment(&fragment))
    }

    fn resolve_url_tool_index(&self, tool_id: Option<&str>) -> Result<usize, String> {
        match tool_id {
            Some(requested) => self
                .available_tools
                .iter()
                .position(|meta| {
                    meta.id.eq_ignore_ascii_case(requested)
                        || meta.display_name.eq_ignore_ascii_case(requested)
                        || crate::url_state::matches_tab_slug(requested, meta.id, meta.display_name)
                })
                .ok_or_else(|| {
                    format!(
                        "Tool '{requested}' is not available in domain '{}'",
                        self.domain
                    )
                }),
            None => self
                .selected_tool_index
                .or_else(|| (!self.available_tools.is_empty()).then_some(0))
                .ok_or_else(|| format!("No simulation tools registered for '{}'", self.domain)),
        }
    }

    /// Validates and atomically applies a shared URL tool identifier and `PresetWrapper` JSON payload.
    ///
    /// Builds and validates a candidate tool instance prior to committing so malformed, mismatched,
    /// or non-finite states never mutate the currently active tool.
    pub fn import_url_preset_json(
        &mut self,
        tool_id: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        let target_idx = self.resolve_url_tool_index(tool_id)?;
        let meta = self.available_tools[target_idx];
        let mut candidate = (meta.build)();
        if let Some(json_str) = preset_json {
            crate::url_state::validate_url_preset_json(json_str)?;
            candidate.import_preset_json(json_str)?;
            if !candidate.has_finite_parameters() {
                return Err(format!(
                    "Shared URL parameters for '{}' are non-finite or unstable",
                    candidate.name()
                ));
            }
        }

        if self.selected_tool_index != Some(target_idx) {
            if let Some((old_name, old_json)) = self.serialized_active_tool() {
                self.saved_tool_states
                    .insert(old_name.to_string(), old_json.to_string());
            }
        }
        scientific_metadata::citation_registry::CitationRegistry::register(
            candidate.name().to_string(),
            candidate.theory().theory_citation(),
        );
        self.selected_tool_index = Some(target_idx);
        self.active_tool = Some(candidate);
        self.clear_challenge_hud();
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.show_starter_gallery = false;
        self.preset_dialog.toast_error = None;
        self.sync_active_baseline();
        Ok(())
    }

    fn verify_requested_url_tool(
        &self,
        requested_tool: Option<&str>,
        expected_tool_id: &str,
    ) -> Result<usize, String> {
        let target_idx = self.resolve_url_tool_index(Some(expected_tool_id))?;
        if let Some(req) = requested_tool {
            let req_idx = self.resolve_url_tool_index(Some(req))?;
            if req_idx != target_idx {
                return Err(format!(
                    "Shared URL tool '{req}' does not match target tool '{expected_tool_id}'"
                ));
            }
        }
        Ok(target_idx)
    }

    fn validate_candidate_tool_preset(
        &self,
        target_idx: usize,
        base_preset_json: &str,
        override_state_json: Option<&str>,
    ) -> Result<Box<dyn super::InteractiveTool>, String> {
        let meta = self.available_tools[target_idx];
        let mut candidate = (meta.build)();
        crate::url_state::validate_url_preset_json(base_preset_json)?;
        candidate.import_preset_json(base_preset_json)?;
        if let Some(extra_json) = override_state_json {
            crate::url_state::validate_url_preset_json(extra_json)?;
            candidate.import_preset_json(extra_json)?;
        }
        if !candidate.has_finite_parameters() {
            return Err(format!(
                "Shared URL parameters for '{}' are non-finite or unstable",
                candidate.name()
            ));
        }
        Ok(candidate)
    }

    fn commit_validated_url_candidate(
        &mut self,
        target_idx: usize,
        candidate: Box<dyn super::InteractiveTool>,
    ) {
        if self.selected_tool_index != Some(target_idx) {
            if let Some((old_name, old_json)) = self.serialized_active_tool() {
                self.saved_tool_states
                    .insert(old_name.to_string(), old_json.to_string());
            }
        }
        scientific_metadata::citation_registry::CitationRegistry::register(
            candidate.name().to_string(),
            candidate.theory().theory_citation(),
        );
        self.selected_tool_index = Some(target_idx);
        self.active_tool = Some(candidate);
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.show_starter_gallery = false;
        self.preset_dialog.toast_error = None;
    }

    /// Atomically validates and hydrates `ShareUrlParams` (including `&preset=`, `&challenge=`, and `&step=`).
    pub fn import_url_share_params(
        &mut self,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        if let Some(ref challenge_id) = params.challenge {
            return self.hydrate_url_challenge(challenge_id, params, ctx);
        }
        if let Some(ref preset_id) = params.preset {
            return self.hydrate_url_preset_or_story(preset_id, params, ctx);
        }
        self.import_url_preset_json(params.tool.as_deref(), params.state_json.as_deref())
    }

    fn hydrate_url_challenge(
        &mut self,
        challenge_id: &str,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        let challenge =
            crate::starter_gallery::find_challenge_by_id(challenge_id, Some(self.domain.as_str()))
                .ok_or_else(|| {
                    format!(
                        "Unknown challenge '{challenge_id}' for domain '{}'",
                        self.domain
                    )
                })?;
        let target_idx =
            self.verify_requested_url_tool(params.tool.as_deref(), challenge.tool_name)?;
        let base_json = (challenge.initial_preset_generator)();
        let baseline_candidate =
            self.validate_candidate_tool_preset(target_idx, base_json, None)?;
        let initial_state = baseline_candidate.save_state();
        let candidate = self.validate_candidate_tool_preset(
            target_idx,
            base_json,
            params.state_json.as_deref(),
        )?;
        let compiled = super::CompiledGoalPredicate::compile(challenge.predicate_expr);
        self.commit_validated_url_candidate(target_idx, candidate);
        if let Some(tool) = &mut self.active_tool {
            tool.sync_callouts(challenge.callouts);
        }
        self.challenge_hud
            .sandbox
            .set_formula(challenge.predicate_expr);
        self.challenge_hud.mode = super::ChallengeHudMode::Challenge {
            challenge,
            compiled,
            initial_state,
            currently_satisfied: false,
            solved: false,
            show_hint: false,
        };
        self.sync_active_baseline();
        let _ = self.evaluate_challenge_predicates(ctx);
        Ok(())
    }

    fn hydrate_url_preset_or_story(
        &mut self,
        preset_id: &str,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        if let Some(story) =
            crate::starter_gallery::find_story_by_id(preset_id, Some(self.domain.as_str()))
        {
            return self.hydrate_url_story(story, params, ctx);
        }
        self.hydrate_url_static_preset(preset_id, params)
    }

    fn hydrate_url_story(
        &mut self,
        story: &'static crate::starter_gallery::PresetStoryMetadata,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        let step_idx = params.step.unwrap_or(0);
        let step = story.steps.get(step_idx).ok_or_else(|| {
            format!(
                "Story '{}' step index {step_idx} is out of bounds (max {})",
                story.id,
                story.steps.len().saturating_sub(1)
            )
        })?;
        let target_idx = self.verify_requested_url_tool(params.tool.as_deref(), story.tool_name)?;
        let step_json = (step.preset_generator)();
        let base_candidate = self.validate_candidate_tool_preset(target_idx, step_json, None)?;
        let step_state = base_candidate.save_state();
        let candidate = self.validate_candidate_tool_preset(
            target_idx,
            step_json,
            params.state_json.as_deref(),
        )?;
        self.commit_validated_url_candidate(target_idx, candidate);
        if let Some(tool) = &mut self.active_tool {
            tool.sync_callouts(step.callouts);
        }
        self.challenge_hud.mode = super::ChallengeHudMode::Story {
            story,
            step_index: step_idx,
            step_state,
        };
        self.sync_active_baseline();
        if let Some(egui_ctx) = ctx {
            self.flush_hud_announcements(egui_ctx);
        }
        Ok(())
    }

    fn hydrate_url_static_preset(
        &mut self,
        preset_id: &str,
        params: &crate::url_state::ShareUrlParams,
    ) -> Result<(), String> {
        let starter = crate::starter_gallery::find_starter_preset_by_id(
            preset_id,
            Some(self.domain.as_str()),
        )
        .ok_or_else(|| {
            format!(
                "Unknown preset or story '{preset_id}' for domain '{}'",
                self.domain
            )
        })?;
        if params.step.is_some_and(|s| s > 0) {
            return Err(format!(
                "Static starter preset '{}' does not support multi-step index",
                starter.id
            ));
        }
        let expected_tool = starter
            .tool_name
            .ok_or_else(|| format!("Starter preset '{}' has no target tool", starter.id))?;
        let target_idx = self.verify_requested_url_tool(params.tool.as_deref(), expected_tool)?;
        let candidate = self.validate_candidate_tool_preset(
            target_idx,
            starter.generate_preset(),
            params.state_json.as_deref(),
        )?;
        self.commit_validated_url_candidate(target_idx, candidate);
        self.clear_challenge_hud();
        self.sync_active_baseline();
        Ok(())
    }
}
