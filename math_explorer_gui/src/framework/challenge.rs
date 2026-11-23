//! Inverse challenge lifecycle, baseline reset, and scoring types for [`super::SimulationFramework`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SimulationFramework;
use crate::starter_gallery::{
    challenge_matches_domain, find_challenge_metadata, ChallengeCard, ChallengeMetadata,
};
use eframe::egui;

/// Live quantitative evaluation of an active inverse challenge on an [`super::InteractiveTool`].
#[derive(Clone, Debug, PartialEq)]
pub struct ChallengeEvaluation {
    /// Normalized challenge score clamped to `0.0..=100.0`.
    pub score: f64,
    /// Whether all quantitative challenge constraints are currently satisfied.
    pub passed: bool,
    /// Short label naming the primary residual or target metric.
    pub residual_label: &'static str,
    /// Numeric value of the primary residual or target metric.
    pub residual_value: f64,
    /// Concise human-readable status detail (e.g., sub-constraints or error breakdown).
    pub detail: String,
}

impl ChallengeEvaluation {
    /// Constructs a [`ChallengeEvaluation`] with `score` clamped to `0.0..=100.0` and finite fallback.
    #[must_use]
    pub fn new(
        score: f64,
        passed: bool,
        residual_label: &'static str,
        residual_value: f64,
        detail: impl Into<String>,
    ) -> Self {
        let clean_score = if score.is_finite() {
            score.clamp(0.0, 100.0)
        } else {
            0.0
        };
        let clean_residual = if residual_value.is_finite() {
            residual_value
        } else {
            f64::INFINITY
        };
        Self {
            score: clean_score,
            passed: passed && score.is_finite() && residual_value.is_finite(),
            residual_label,
            residual_value: clean_residual,
            detail: detail.into(),
        }
    }
}

impl SimulationFramework {
    /// Returns the canonical ID of the active challenge (inverse or AST goal), if any.
    #[must_use]
    pub fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
            .map(|meta| meta.id)
            .or_else(|| self.active_goal_challenge_id())
    }

    /// Resolves a tool index in `available_tools` matching `ChallengeMetadata::tool_name`.
    pub(super) fn resolve_challenge_tool_index(
        &self,
        meta: &ChallengeMetadata,
    ) -> Result<usize, String> {
        self.available_tools
            .iter()
            .position(|t| {
                t.id.eq_ignore_ascii_case(meta.tool_name)
                    || t.display_name.eq_ignore_ascii_case(meta.tool_name)
            })
            .ok_or_else(|| {
                format!(
                    "Challenge '{}' targets unavailable tool '{}' in domain '{}'",
                    meta.id, meta.tool_name, self.domain
                )
            })
    }

    /// Launches a [`ChallengeCard`] selected from `StarterGallery`.
    pub fn launch_challenge_card(&mut self, card: &ChallengeCard) -> Result<(), String> {
        self.launch_challenge_by_id(card.id, None)
    }

    /// Activates a registered inverse [`ChallengeMetadata`] or AST [`crate::starter_gallery::GoalChallengeMetadata`]
    /// by `challenge_id`, loading its canonical baseline.
    pub fn launch_challenge_by_id(
        &mut self,
        challenge_id: &str,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        if let Some(meta) = find_challenge_metadata(challenge_id) {
            if !challenge_matches_domain(meta, &self.domain) {
                let err = format!(
                    "Challenge '{}' belongs to domain '{}', not '{}'",
                    meta.id, meta.domain, self.domain
                );
                self.preset_dialog.record_import_error(err.clone());
                return Err(err);
            }
            let target_idx = self.resolve_challenge_tool_index(meta)?;
            let baseline_raw = meta.generate_baseline_state();
            let clamped_json = meta.clamp_state_json(&baseline_raw)?;
            self.commit_challenge_instance(target_idx, meta, &clamped_json)?;
            self.show_starter_gallery = false;
            let msg = format!(
                "Loaded challenge: {}. Locked: {}.",
                meta.title, meta.locked_inputs
            );
            dispatch_challenge_announcement(ctx, &msg);
            return Ok(());
        }
        if let Some(goal) = crate::starter_gallery::find_goal_challenge_by_id(challenge_id, None) {
            if !crate::starter_gallery::domain_matches_gallery(goal.domain, &self.domain) {
                let err = format!(
                    "Challenge '{}' belongs to domain '{}', not '{}'",
                    goal.id, goal.domain, self.domain
                );
                self.preset_dialog.record_import_error(err.clone());
                return Err(err);
            }
            self.launch_challenge(goal)?;
            self.show_starter_gallery = false;
            if let Some(egui_ctx) = ctx {
                self.flush_hud_announcements(egui_ctx);
            }
            return Ok(());
        }
        let err = format!("Unknown challenge '{challenge_id}'");
        self.preset_dialog.record_import_error(err.clone());
        Err(err)
    }

    pub(super) fn commit_challenge_instance(
        &mut self,
        target_idx: usize,
        meta: &'static ChallengeMetadata,
        clamped_config_json: &str,
    ) -> Result<(), String> {
        let tool_meta = self.available_tools[target_idx];
        let mut candidate = (tool_meta.build)();
        candidate.apply_challenge(Some(meta.id))?;
        candidate
            .state_bounds()
            .validate_json(clamped_config_json)?;
        candidate.try_load_state(clamped_config_json)?;
        candidate.apply_challenge(Some(meta.id))?;
        if !candidate.has_finite_parameters() {
            return Err(format!(
                "Challenge '{}' produced non-finite parameters for '{}'",
                meta.id,
                candidate.name()
            ));
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
        self.clear_challenge_hud();
        self.challenge_was_passed = candidate
            .evaluate_challenge()
            .is_some_and(|eval| eval.passed);
        self.active_challenge = Some(meta);
        self.selected_tool_index = Some(target_idx);
        self.active_tool = Some(candidate);
        self.preset_dialog.toast_error = None;
        self.sync_active_baseline();
        Ok(())
    }

    /// Resets the currently active challenge back to its canonical unsolved baseline state.
    pub fn reset_active_challenge_baseline(
        &mut self,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        let Some(meta) = self.active_challenge else {
            return Err("No active challenge to reset".to_string());
        };
        let target_idx = self.resolve_challenge_tool_index(meta)?;
        let baseline_raw = meta.generate_baseline_state();
        let clamped_json = meta.clamp_state_json(&baseline_raw)?;
        self.commit_challenge_instance(target_idx, meta, &clamped_json)?;
        let msg = format!("Challenge reset to baseline: {}", meta.title);
        dispatch_challenge_announcement(ctx, &msg);
        Ok(())
    }

    /// Exits active challenge mode while keeping the current tool open for free exploration.
    pub fn exit_active_challenge(&mut self, ctx: Option<&egui::Context>) {
        if let Some(meta) = self.active_challenge.take() {
            self.challenge_was_passed = false;
            if let Some(tool) = &mut self.active_tool {
                let _ = tool.apply_challenge(None);
            }
            self.bump_framework_revision();
            let msg = format!("Exited challenge: {}", meta.title);
            dispatch_challenge_announcement(ctx, &msg);
        }
    }
}

pub(super) fn dispatch_challenge_announcement(ctx: Option<&egui::Context>, msg: &str) {
    if let Some(egui_ctx) = ctx {
        crate::accessibility::announce_live_status(egui_ctx, msg);
    } else {
        crate::accessibility::announce_status(msg);
    }
}

// [cite:favorite_child]
