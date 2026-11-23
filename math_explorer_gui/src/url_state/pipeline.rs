//! Bounded multi-pane pipeline URL serialization (`&pipe=<base64url>`) within `URL_JSON_BOUNDS`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::base64url;
use super::codec::{
    MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES, decode_optional_state_json,
    percent_encode_component, validate_url_preset_json,
};
use crate::app::ViewportLayout;
use crate::app::dataflow_bus::{MAX_DATAFLOW_GRAPH_NODES, PaneLink, PanePipelineDag};
use serde::{Deserialize, Serialize};

/// Per-pane serialized entry inside a multi-pane pipeline share URL.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelinePaneUrlEntry {
    /// Canonical tab URL slug.
    pub tab: String,
    /// Optional active `InteractiveTool` ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    /// Optional base64url-encoded `PresetWrapper` JSON state for this pane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_b64: Option<String>,
}

/// Decoded and validated pane descriptor ready for `ExplorerTab::import_url_preset_json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedPipelinePane {
    /// Canonical or normalized tab slug.
    pub tab: String,
    /// Optional active `InteractiveTool` ID.
    pub tool: Option<String>,
    /// Optional validated `PresetWrapper` JSON string.
    pub state_json: Option<String>,
}

/// Wire format and validated in-memory representation of `&pipe=<base64url>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineUrlEnvelope {
    /// Active multi-pane viewport layout (`SplitHorizontal`, `SplitVertical`, or `QuadGrid`).
    pub layout: ViewportLayout,
    /// Active focused pane index (`0..layout.pane_count()`).
    pub focused_pane: usize,
    /// Ordered pane entries (`len == layout.pane_count()`).
    pub panes: Vec<PipelinePaneUrlEntry>,
    /// Active `PanePipelineDag` directed links.
    pub links: Vec<PaneLink>,
}

/// Validated multi-pane pipeline payload attached to [`super::ShareUrlParams`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedPipelineUrl {
    /// Multi-pane layout to restore.
    pub layout: ViewportLayout,
    /// Focused pane index (`0..layout.pane_count()`).
    pub focused_pane: usize,
    /// Validated per-pane tab, tool, and JSON state descriptors.
    pub panes: Vec<DecodedPipelinePane>,
    /// Acyclic validated `PaneLink` edges.
    pub links: Vec<PaneLink>,
}

fn check_graph_nodes_cap_recursive(val: &serde_json::Value) -> Result<(), String> {
    match val {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::Array(nodes)) = map.get("nodes")
                && nodes.len() > MAX_DATAFLOW_GRAPH_NODES
            {
                return Err(format!(
                    "Pipeline URL graph state ({} nodes) exceeds {MAX_DATAFLOW_GRAPH_NODES}-node limit",
                    nodes.len()
                ));
            }
            for child in map.values() {
                check_graph_nodes_cap_recursive(child)?;
            }
            Ok(())
        }
        serde_json::Value::Array(items) => {
            for item in items {
                check_graph_nodes_cap_recursive(item)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Validates that a JSON state string obeys the `200`-node cap for pipeline share links.
pub fn enforce_url_graph_node_cap(json_str: &str) -> Result<(), String> {
    let val: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("Invalid URL state JSON: {e}"))?;
    check_graph_nodes_cap_recursive(&val)
}

/// Builds a `PipelinePaneUrlEntry` from raw `(tab_slug, tool_id, preset_json)`.
pub fn build_pane_url_entry(
    tab_slug: &str,
    tool_id: Option<&str>,
    preset_json: Option<&str>,
) -> Result<PipelinePaneUrlEntry, String> {
    let state_b64 = match preset_json {
        Some(json) => {
            validate_url_preset_json(json)?;
            enforce_url_graph_node_cap(json)?;
            Some(base64url::encode(json.as_bytes()))
        }
        None => None,
    };
    Ok(PipelinePaneUrlEntry {
        tab: tab_slug.to_string(),
        tool: tool_id.map(str::to_string),
        state_b64,
    })
}

/// Encodes a multi-pane pipeline share fragment (`#tab=...&tool=...&pipe=...`) within `4 KiB` / `8 KiB` bounds.
pub fn encode_pipeline_share_fragment(
    focused_tab_slug: &str,
    focused_tool_id: Option<&str>,
    mut envelope: PipelineUrlEnvelope,
) -> Result<String, String> {
    if focused_tab_slug.trim().is_empty() {
        return Err("Cannot encode pipeline URL with an empty tab slug".to_string());
    }
    let pane_count = envelope.layout.pane_count();
    if pane_count <= 1 || envelope.panes.len() != pane_count || envelope.links.is_empty() {
        return Err("Pipeline URL requires a multi-pane layout and at least one link".to_string());
    }
    let mut dag = PanePipelineDag::default();
    dag.restore_links(&envelope.links, pane_count)
        .map_err(|e| e.message().to_string())?;

    let mut pipe_json = serde_json::to_string(&envelope)
        .map_err(|e| format!("Failed to serialize pipeline URL envelope: {e}"))?;
    if pipe_json.len() > MAX_DECODED_URL_STATE_BYTES {
        for (idx, pane) in envelope.panes.iter_mut().enumerate() {
            let in_dag = envelope
                .links
                .iter()
                .any(|l| l.from_pane == idx || l.to_pane == idx);
            if !in_dag {
                pane.state_b64 = None;
            }
        }
        pipe_json = serde_json::to_string(&envelope)
            .map_err(|e| format!("Failed to serialize pipeline URL envelope: {e}"))?;
    }
    validate_url_preset_json(&pipe_json)?;

    let mut fragment = format!("#tab={}", percent_encode_component(focused_tab_slug));
    if let Some(tool) = focused_tool_id.filter(|t| !t.trim().is_empty()) {
        fragment.push_str("&tool=");
        fragment.push_str(&percent_encode_component(tool));
    }
    fragment.push_str("&pipe=");
    fragment.push_str(&base64url::encode(pipe_json.as_bytes()));
    if fragment.len() > MAX_URL_FRAGMENT_BYTES {
        return Err(format!(
            "Encoded pipeline URL fragment ({} bytes) exceeds maximum ({MAX_URL_FRAGMENT_BYTES} bytes)",
            fragment.len()
        ));
    }
    Ok(fragment)
}

/// Decodes and validates an optional `&pipe=<base64url>` parameter.
pub(super) fn decode_optional_pipeline(
    pipe_b64: Option<String>,
) -> Result<Option<ValidatedPipelineUrl>, String> {
    let Some(raw) = pipe_b64 else {
        return Ok(None);
    };
    if raw.trim().is_empty() {
        return Err("Shared URL contains an empty 'pipe' parameter".to_string());
    }
    let decoded_bytes = base64url::decode(raw.trim(), MAX_DECODED_URL_STATE_BYTES)?;
    let json_str = String::from_utf8(decoded_bytes)
        .map_err(|_| "Decoded 'pipe' URL payload is not valid UTF-8".to_string())?;
    validate_url_preset_json(&json_str)?;

    let env: PipelineUrlEnvelope =
        serde_json::from_str(&json_str).map_err(|e| format!("Invalid 'pipe' URL JSON: {e}"))?;
    let pane_count = env.layout.pane_count();
    if pane_count <= 1 {
        return Err("Shared 'pipe' URL specifies single-pane layout".to_string());
    }
    if env.panes.len() != pane_count {
        return Err("Shared 'pipe' URL pane count does not match layout".to_string());
    }
    if env.focused_pane >= pane_count {
        return Err("Shared 'pipe' URL focused_pane index is out of bounds".to_string());
    }
    if env.links.is_empty() || env.links.len() > pane_count {
        return Err("Shared 'pipe' URL must contain 1..=pane_count pipeline links".to_string());
    }
    let mut dag = PanePipelineDag::default();
    dag.restore_links(&env.links, pane_count)
        .map_err(|e| e.message().to_string())?;

    let mut panes = Vec::with_capacity(pane_count);
    for entry in env.panes {
        let tab = entry.tab.trim().to_string();
        if tab.is_empty() {
            return Err("Shared 'pipe' URL contains a pane with an empty tab slug".to_string());
        }
        let tool = match entry.tool {
            Some(t) if t.trim().is_empty() => {
                return Err("Shared 'pipe' URL contains an empty tool identifier".to_string());
            }
            Some(t) => Some(t.trim().to_string()),
            None => None,
        };
        let state_json = decode_optional_state_json(&tab, tool.as_deref(), entry.state_b64)?;
        if let Some(ref s) = state_json {
            enforce_url_graph_node_cap(s)?;
        }
        panes.push(DecodedPipelinePane {
            tab,
            tool,
            state_json,
        });
    }
    Ok(Some(ValidatedPipelineUrl {
        layout: env.layout,
        focused_pane: env.focused_pane,
        panes,
        links: dag.links().to_vec(),
    }))
}
