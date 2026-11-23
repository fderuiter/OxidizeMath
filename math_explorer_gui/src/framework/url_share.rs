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

    /// Encodes a full shareable web URL (`https://...#tab=...&tool=...&state=...`) for the active tool.
    pub fn encode_share_url(&self) -> Result<String, String> {
        let tab_slug = if self.domain == "biology" {
            "morphogenesis"
        } else {
            self.domain.as_str()
        };
        let preset_json = self.export_url_preset_json()?;
        let fragment = crate::url_state::encode_share_fragment(
            tab_slug,
            self.active_tool_id(),
            preset_json.as_deref(),
        )?;
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
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.show_starter_gallery = false;
        self.preset_dialog.toast_error = None;
        self.reset_active_tool_manual_bounds();
        self.sync_active_baseline();
        Ok(())
    }
}
