#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::accessibility::AccessibleHoverText;
use crate::presets::{deserialize_preset_envelope_bounded, serialize_compact_preset_envelope};
use serde::{Deserialize, Serialize};

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TraceabilityScanState {
    #[default]
    Idle,
    Scanning,
    Ready,
    Failed(String),
}

#[allow(missing_docs)]
impl TraceabilityScanState {
    pub fn is_scanning(&self) -> bool {
        matches!(self, Self::Scanning)
    }

    #[cfg(test)]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    #[cfg(test)]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    #[cfg(test)]
    pub fn error(&self) -> Option<&str> {
        match self {
            Self::Failed(err) => Some(err.as_str()),
            _ => None,
        }
    }
}

impl std::fmt::Display for TraceabilityScanState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Scanning => write!(f, "Scanning"),
            Self::Ready => write!(f, "Ready"),
            Self::Failed(err) => write!(f, "Failed: {}", err),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct TraceabilityState {
    pub(super) repo_path: Option<String>,
}

/// Validated `PresetWrapper` payload for `TraceabilityTab` URL share state.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TraceabilityPresetConfig {
    /// Optional validated workspace-relative repository directory.
    pub repo_path: Option<String>,
}

impl TraceabilityTab {
    /// Domain identifier used in `PresetWrapper` envelopes for `TraceabilityTab`.
    #[must_use]
    pub fn domain(&self) -> &'static str {
        "traceability"
    }

    /// Exports current `TraceabilityTab` configuration as compact `PresetWrapper` JSON.
    pub fn to_compact_url_json(&self) -> Result<String, String> {
        serialize_compact_preset_envelope(
            "Traceability Portal",
            self.domain(),
            TraceabilityPresetConfig {
                repo_path: self.repo_path.clone(),
            },
        )
    }

    /// Validates and applies a `PresetWrapper` JSON payload to `TraceabilityTab`.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_json_str(&mut self, json_str: &str) -> Result<(), String> {
        let bounds = crate::framework::ToolStateBounds::new(
            crate::url_state::MAX_DECODED_URL_STATE_BYTES,
            crate::framework::MAX_JSON_NESTING_DEPTH,
            crate::framework::MAX_JSON_ARRAY_LEN,
        );
        let config: TraceabilityPresetConfig =
            deserialize_preset_envelope_bounded(json_str, self.domain(), self.name(), bounds)?;
        let validated = match config.repo_path {
            None => None,
            Some(ref raw) => {
                if Self::has_traversal_sequence(raw) || raw.trim().is_empty() {
                    return Err("Invalid repository path in shared Traceability URL".to_string());
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let Some(clean) = Self::validate_repo_path(raw) else {
                        return Err(
                            "Shared Traceability repository path is outside the active workspace"
                                .to_string(),
                        );
                    };
                    Some(clean)
                }
                #[cfg(target_arch = "wasm32")]
                {
                    None
                }
            }
        };
        if self.repo_path != validated {
            self.repo_path = validated;
            if self.has_initial_scan {
                self.refresh();
            }
        }
        Ok(())
    }

    /// Builds a shareable web URL for the `TraceabilityTab` view.
    pub fn encode_share_url(&self) -> Result<String, String> {
        let json = self.to_compact_url_json()?;
        let frag = crate::url_state::encode_share_fragment(self.url_slug(), None, Some(&json))?;
        Ok(crate::url_state::build_share_url_from_fragment(&frag))
    }

    pub(super) fn render_share_link_button(&mut self, ui: &mut egui::Ui) {
        let status_id = egui::Id::new("traceability_share_clipboard_status");
        let toast_id = egui::Id::new("traceability_share_toast");
        if let Some((is_err, msg)) = ui
            .ctx()
            .data_mut(|d| d.remove_temp::<(bool, String)>(status_id))
        {
            ui.ctx()
                .data_mut(|d| d.insert_temp(toast_id, (is_err, msg)));
        }
        if ui
            .button("🔗 Copy Share Link")
            .accessible_hover_text("Copy a shareable link to the Traceability Portal")
            .clicked()
        {
            match self.encode_share_url() {
                Ok(url) => {
                    let ctx_cb = ui.ctx().clone();
                    crate::url_state::copy_share_url_to_clipboard(ui.ctx(), url, move |res| {
                        let entry = match res {
                            Ok(()) => (
                                false,
                                "Copied shareable simulation link to clipboard".to_string(),
                            ),
                            Err(err) => (true, err),
                        };
                        ctx_cb.data_mut(|d| d.insert_temp(status_id, entry));
                    });
                }
                Err(err) => {
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(toast_id, (true, err)));
                }
            }
        }
        if let Some((is_err, msg)) = ui
            .ctx()
            .data_mut(|d| d.get_temp::<(bool, String)>(toast_id))
        {
            if !msg.is_empty() {
                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                let color = if is_err {
                    palette.error
                } else {
                    palette.success
                };
                let prefix = if is_err { "⚠️" } else { "✓" };
                ui.colored_label(color, format!("{prefix} {msg}"));
            }
        }
    }
}

// theory_verification!
