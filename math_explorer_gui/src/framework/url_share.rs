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

    fn cached_untracked_url_digest(&self, tool: &dyn super::InteractiveTool) -> u64 {
        let current_rev = self.rev_cache.untracked_tool_revision;
        let name = tool.name();
        if let Some(cached) = self.rev_cache.untracked_url_digest.get() {
            if cached.tool_name == name && cached.checked_untracked_rev == current_rev {
                return cached.url_digest;
            }
        }
        let digest = compute_untracked_tool_url_digest(tool);
        self.rev_cache
            .untracked_url_digest
            .set(Some(super::UntrackedUrlDigestCache {
                tool_name: name,
                checked_untracked_rev: current_rev,
                url_digest: digest,
            }));
        digest
    }

    /// Computes a parameter-only revision token for live URL fragment synchronization.
    #[must_use]
    pub fn url_state_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.selected_tool_index.hash(&mut hasher);
        self.active_challenge.map(|m| m.id).hash(&mut hasher);
        self.hash_hud_for_url(&mut hasher);
        if let Some(tool) = &self.active_tool {
            tool.name().hash(&mut hasher);
            let tool_rev = tool.url_state_revision();
            if tool_rev > 0 {
                1_u8.hash(&mut hasher);
                tool_rev.hash(&mut hasher);
            } else {
                0_u8.hash(&mut hasher);
                self.cached_untracked_url_digest(tool.as_ref())
                    .hash(&mut hasher);
            }
        }
        hasher.finish().max(1)
    }

    /// Exports the active tool's parameters as a compact, deterministic [`crate::presets::PresetWrapper`] JSON string.
    pub fn export_url_preset_json(&self) -> Result<Option<String>, String> {
        let Some(tool) = &self.active_tool else {
            return Ok(None);
        };
        let Some(state_str) = tool.save_url_state() else {
            return Ok(None);
        };
        let cleaned = egui_plot::export::strip_utf8_bom(&state_str).trim();
        let cap = crate::url_state::MAX_DECODED_URL_STATE_BYTES;
        if cleaned.len() > cap {
            return Err(format!(
                "URL state for '{}' ({} bytes) exceeds maximum allowed limit ({cap} bytes)",
                tool.name(),
                cleaned.len()
            ));
        }
        let raw_config: &serde_json::value::RawValue = serde_json::from_str(cleaned)
            .map_err(|e| format!("Failed to serialize state for '{}': {e}", tool.name()))?;
        let envelope_json = crate::presets::serialize_compact_preset_envelope(
            tool.name(),
            tool.domain(),
            raw_config,
        )?;
        crate::url_state::validate_url_preset_json(&envelope_json)?;
        Ok(Some(envelope_json))
    }

    /// Encodes a `#tab=...` fragment for `tab_slug`, supporting active inverse challenges, guided stories, and AST challenges.
    pub fn encode_share_fragment_for_tab(&self, tab_slug: &str) -> Result<String, String> {
        if let Some(inv_meta) = self.active_challenge {
            let preset_json = self.export_url_preset_json()?;
            return crate::url_state::encode_share_fragment_with_challenge(
                tab_slug,
                self.active_tool_id(),
                Some(inv_meta.id),
                preset_json.as_deref(),
            );
        }
        let current_tool_state = self.active_tool.as_ref().and_then(|t| t.save_url_state());
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

    pub(super) fn resolve_url_tool_index(&self, tool_id: Option<&str>) -> Result<usize, String> {
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
        self.commit_validated_url_candidate(target_idx, candidate);
        self.clear_challenge_hud();
        self.sync_active_baseline();
        Ok(())
    }

    pub(super) fn commit_validated_url_candidate(
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
        self.active_challenge = None;
        self.challenge_was_passed = false;
        self.selected_tool_index = Some(target_idx);
        self.active_tool = Some(candidate);
        self.promote_legacy_plot_keys_for_active_tool();
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.show_starter_gallery = false;
        self.preset_dialog.toast_error = None;
        self.reset_active_tool_manual_bounds();
    }
}

fn compute_untracked_tool_url_digest(tool: &dyn super::InteractiveTool) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match tool.save_url_state() {
        Some(state_str) => {
            let cleaned = egui_plot::export::strip_utf8_bom(&state_str).trim();
            if cleaned.len() <= crate::url_state::MAX_DECODED_URL_STATE_BYTES {
                1_u8.hash(&mut hasher);
                cleaned.hash(&mut hasher);
            } else {
                2_u8.hash(&mut hasher);
            }
        }
        None => {
            0_u8.hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[path = "url_share/hydrate.rs"]
mod hydrate;
