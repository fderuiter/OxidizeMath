//! URL parameter hydration for inverse challenges, AST challenges, stories, and presets.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::{ChallengeHudMode, CompiledGoalPredicate, InteractiveTool, SimulationFramework};
use eframe::egui;

impl SimulationFramework {
    /// Atomically validates and hydrates `ShareUrlParams` (supporting inverse challenges, AST challenges, stories, and presets).
    pub fn import_url_share_params(
        &mut self,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        if let Some(ref challenge_id) = params.challenge {
            if crate::starter_gallery::find_challenge_metadata(challenge_id).is_some() {
                return self.import_url_inverse_challenge(
                    params.tool.as_deref(),
                    challenge_id,
                    params.state_json.as_deref(),
                    ctx,
                );
            }
            return self.hydrate_url_goal_challenge(challenge_id, params, ctx);
        }
        if let Some(ref preset_id) = params.preset {
            return self.hydrate_url_preset_or_story(preset_id, params, ctx);
        }
        self.import_url_preset_json(params.tool.as_deref(), params.state_json.as_deref())
    }

    pub(super) fn verify_requested_url_tool(
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

    pub(super) fn validate_candidate_tool_preset(
        &self,
        target_idx: usize,
        base_preset_json: &str,
        override_state_json: Option<&str>,
    ) -> Result<Box<dyn InteractiveTool>, String> {
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

    fn import_url_inverse_challenge(
        &mut self,
        tool_id: Option<&str>,
        challenge_id: &str,
        preset_json: Option<&str>,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        let Some(ch_meta) = crate::starter_gallery::find_challenge_metadata(challenge_id) else {
            return Err(format!("Unknown challenge '{challenge_id}' in shared URL"));
        };
        if !crate::starter_gallery::challenge_matches_domain(ch_meta, &self.domain) {
            return Err(format!(
                "Challenge '{}' belongs to domain '{}', not '{}'",
                ch_meta.id, ch_meta.domain, self.domain
            ));
        }
        let target_idx = self.resolve_challenge_tool_index(ch_meta)?;
        if let Some(explicit_tool) = tool_id {
            let requested_idx = self.resolve_url_tool_index(Some(explicit_tool))?;
            if requested_idx != target_idx {
                return Err(format!(
                    "Challenge '{}' belongs to tool '{}', not '{explicit_tool}'",
                    ch_meta.id, ch_meta.tool_name
                ));
            }
        }
        let clamped_config_json = match preset_json {
            Some(json_str) => extract_and_clamp_challenge_json(
                json_str,
                self.available_tools[target_idx],
                ch_meta,
            )?,
            None => ch_meta.clamp_state_json(&ch_meta.generate_baseline_state())?,
        };
        self.commit_challenge_instance(target_idx, ch_meta, &clamped_config_json)?;
        self.show_starter_gallery = false;
        crate::framework::challenge::dispatch_challenge_announcement(
            ctx,
            &format!("Hydrated challenge from URL: {}", ch_meta.title),
        );
        Ok(())
    }

    fn hydrate_url_goal_challenge(
        &mut self,
        challenge_id: &str,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        let challenge = crate::starter_gallery::find_goal_challenge_by_id(
            challenge_id,
            Some(self.domain.as_str()),
        )
        .ok_or_else(|| {
            format!(
                "Unknown challenge '{challenge_id}' for domain '{}'",
                self.domain
            )
        })?;
        let target_idx =
            self.verify_requested_url_tool(params.tool.as_deref(), challenge.tool_name)?;
        let base_json = (challenge.initial_preset_generator)();
        let initial_state = self
            .validate_candidate_tool_preset(target_idx, base_json, None)?
            .save_url_state();
        let candidate = self.validate_candidate_tool_preset(
            target_idx,
            base_json,
            params.state_json.as_deref(),
        )?;
        let compiled = CompiledGoalPredicate::compile(challenge.predicate_expr);
        self.commit_validated_url_candidate(target_idx, candidate);
        if let Some(tool) = &mut self.active_tool {
            tool.sync_callouts(challenge.callouts);
        }
        self.challenge_hud
            .sandbox
            .set_formula(challenge.predicate_expr);
        self.challenge_hud.mode = ChallengeHudMode::Challenge {
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
        let step_state = self
            .validate_candidate_tool_preset(target_idx, step_json, None)?
            .save_url_state();
        let candidate = self.validate_candidate_tool_preset(
            target_idx,
            step_json,
            params.state_json.as_deref(),
        )?;
        self.commit_validated_url_candidate(target_idx, candidate);
        if let Some(tool) = &mut self.active_tool {
            tool.sync_callouts(step.callouts);
        }
        self.challenge_hud.mode = ChallengeHudMode::Story {
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

fn extract_and_clamp_challenge_json(
    json_str: &str,
    tool_meta: &super::super::ToolMetadata,
    ch_meta: &crate::starter_gallery::ChallengeMetadata,
) -> Result<String, String> {
    crate::url_state::validate_url_preset_json(json_str)?;
    let probe = (tool_meta.build)();
    let bounds = probe.state_bounds();
    let parsed: serde_json::Value =
        serde_json::from_str(egui_plot::export::strip_utf8_bom(json_str))
            .map_err(|e| format!("Invalid URL challenge JSON: {e}"))?;
    let raw_inner = if parsed.get("metadata").is_some() || parsed.get("config").is_some() {
        let raw_cfg: Box<serde_json::value::RawValue> =
            crate::presets::deserialize_preset_envelope_bounded(
                json_str,
                probe.domain(),
                probe.name(),
                bounds,
            )?;
        raw_cfg.get().to_string()
    } else {
        json_str.to_string()
    };
    let clamped = ch_meta.clamp_state_json(&raw_inner)?;
    crate::url_state::validate_url_preset_json(&clamped)?;
    Ok(clamped)
}
