//! Version 2 DEFLATE-compressed `#ws=<base64url>` workspace share-link codec.
//!
//! Encodes `ViewportLayout`, split ratios, `focused_pane`, `ComparisonSyncConfig`,
//! and up to 4 active simulation panes via raw `flate2` DEFLATE compression while
//! enforcing a strict 32,768-byte decompression ceiling prior to JSON parsing.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::base64url;
use super::codec::{
    clamp_decoded_url_grid_state, encode_share_fragment_with_challenge,
    ensure_all_json_numbers_finite, validate_challenge_id, ShareUrlParams,
    MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES,
};
use crate::app::{ComparisonSyncConfig, ViewportLayout};
use crate::framework::{ToolStateBounds, MAX_JSON_ARRAY_LEN, MAX_JSON_NESTING_DEPTH};
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::io::{Read, Write};

/// Schema version integer embedded in `#ws=<base64url>` compressed workspace envelopes.
pub const WORKSPACE_SHARE_SCHEMA_VERSION: u8 = 2;

/// Maximum UTF-8 byte length permitted for decompressed `#ws=...` JSON payloads (`32 KiB`).
pub const MAX_DECODED_WORKSPACE_URL_BYTES: usize = 32_768;

/// Structural and byte-size limits enforced on decompressed `#ws=...` workspace envelopes.
pub const WORKSPACE_URL_JSON_BOUNDS: ToolStateBounds = ToolStateBounds::new(
    MAX_DECODED_WORKSPACE_URL_BYTES,
    MAX_JSON_NESTING_DEPTH + 3,
    MAX_JSON_ARRAY_LEN,
);

/// Structural and byte-size limits enforced on per-pane `PresetWrapper` JSON inside `#ws=...`.
pub const WORKSPACE_PRESET_JSON_BOUNDS: ToolStateBounds = ToolStateBounds::new(
    MAX_DECODED_WORKSPACE_URL_BYTES,
    MAX_JSON_NESTING_DEPTH + 1,
    MAX_JSON_ARRAY_LEN,
);

const MWS_EXPORT_GUIDANCE: &str =
    "export a .mws workspace session via File -> Save Workspace Session As... instead";

/// Parsed share location supporting both legacy `#tab=...` single-pane links and v2 `#ws=...` workspaces.
#[derive(Clone, Debug, PartialEq)]
pub enum ParsedShareLocation {
    /// Legacy `#tab=<slug>&tool=<id>&state=<base64url>` single-pane share parameters.
    SinglePane(ShareUrlParams),
    /// Compressed `#ws=<base64url>` workspace share parameters (`1..=4` panes).
    Workspace(WorkspaceShareParams),
}

/// Validated workspace share parameters decoded from or ready to encode into `#ws=<base64url>`.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkspaceShareParams {
    /// Active single, split, or quad viewport layout.
    pub layout: ViewportLayout,
    /// Clamped 2-pane split ratio (`0.20..=0.80`).
    pub split_ratio: f32,
    /// Clamped 4-pane `(x, y)` quad split ratios (`0.20..=0.80`).
    pub quad_split_ratios: (f32, f32),
    /// Focused pane index (`0..layout.pane_count()`).
    pub focused_pane: usize,
    /// Multi-pane plot/playback synchronization flags.
    pub compare_sync: ComparisonSyncConfig,
    /// Ordered simulation pane descriptors (`len() == layout.pane_count()`).
    pub panes: Vec<ShareUrlParams>,
}

fn default_split_ratio() -> f32 {
    0.5
}

fn default_quad_ratios() -> (f32, f32) {
    (0.5, 0.5)
}

fn is_default_split_ratio(ratio: &f32) -> bool {
    (*ratio - 0.5).abs() < 1e-6
}

fn is_default_quad_ratios(ratios: &(f32, f32)) -> bool {
    (ratios.0 - 0.5).abs() < 1e-6 && (ratios.1 - 0.5).abs() < 1e-6
}

fn is_zero_usize(val: &usize) -> bool {
    *val == 0
}

fn is_default_compare_sync(cfg: &ComparisonSyncConfig) -> bool {
    *cfg == ComparisonSyncConfig::default()
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePaneShareEntry {
    tab: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    challenge: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    state: Option<Box<RawValue>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireWorkspaceEnvelope {
    v: u8,
    layout: ViewportLayout,
    #[serde(
        default = "default_split_ratio",
        skip_serializing_if = "is_default_split_ratio"
    )]
    split_ratio: f32,
    #[serde(
        default = "default_quad_ratios",
        skip_serializing_if = "is_default_quad_ratios"
    )]
    quad_split_ratios: (f32, f32),
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    focused_pane: usize,
    #[serde(default, skip_serializing_if = "is_default_compare_sync")]
    compare_sync: ComparisonSyncConfig,
    panes: Vec<WirePaneShareEntry>,
}

/// Validates that `json_str` fits within the 32,768-byte `#ws=...` preset cap, obeys structural
/// nesting/array bounds, and contains only finite JSON numbers.
pub fn validate_workspace_url_preset_json(json_str: &str) -> Result<(), String> {
    let cleaned = egui_plot::export::strip_utf8_bom(json_str);
    WORKSPACE_PRESET_JSON_BOUNDS.validate_json(cleaned)?;
    let parsed: serde_json::Value =
        serde_json::from_str(cleaned).map_err(|e| format!("Invalid URL state JSON: {e}"))?;
    ensure_all_json_numbers_finite(&parsed)
}

fn build_wire_panes(panes: &[ShareUrlParams]) -> Result<Vec<WirePaneShareEntry>, String> {
    let mut out = Vec::with_capacity(panes.len());
    for pane in panes {
        let tab = pane.tab.trim();
        if tab.is_empty() {
            return Err("Cannot encode workspace URL with an empty pane tab slug".to_string());
        }
        let tool = pane
            .tool
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string);
        let challenge = match pane.challenge.as_deref().filter(|c| !c.trim().is_empty()) {
            Some(raw_ch) => Some(validate_challenge_id(raw_ch)?),
            None => None,
        };
        let state = match pane.state_json.as_deref() {
            Some(raw_json) => {
                let cleaned = egui_plot::export::strip_utf8_bom(raw_json);
                if cleaned.len() > MAX_DECODED_WORKSPACE_URL_BYTES {
                    return Err(format!(
                        "Workspace URL payload ({} bytes) exceeds maximum limit ({MAX_DECODED_WORKSPACE_URL_BYTES} bytes); {MWS_EXPORT_GUIDANCE}",
                        cleaned.len()
                    ));
                }
                validate_workspace_url_preset_json(cleaned)?;
                let boxed = RawValue::from_string(cleaned.to_string())
                    .map_err(|e| format!("Invalid pane preset JSON: {e}"))?;
                Some(boxed)
            }
            None => None,
        };
        out.push(WirePaneShareEntry {
            tab: tab.to_string(),
            tool,
            challenge,
            state,
        });
    }
    Ok(out)
}

fn validate_workspace_params_invariants(params: &WorkspaceShareParams) -> Result<(), String> {
    let expected_panes = params.layout.pane_count();
    if params.panes.is_empty() || params.panes.len() > 4 || params.panes.len() != expected_panes {
        return Err(format!(
            "Workspace layout '{}' requires {expected_panes} pane(s), found {}",
            params.layout.name(),
            params.panes.len()
        ));
    }
    if params.focused_pane >= expected_panes {
        return Err(format!(
            "Focused pane index {} is out of range for '{}' ({expected_panes} pane(s))",
            params.focused_pane,
            params.layout.name()
        ));
    }
    if !params.split_ratio.is_finite()
        || !params.quad_split_ratios.0.is_finite()
        || !params.quad_split_ratios.1.is_finite()
    {
        return Err("Workspace split ratios must be finite".to_string());
    }
    Ok(())
}

/// Compresses `params` into a `#ws=<base64url>` fragment while enforcing the 32 KiB raw JSON
/// ceiling and 8 KiB URL fragment ceiling.
pub fn encode_workspace_share_fragment(params: &WorkspaceShareParams) -> Result<String, String> {
    validate_workspace_params_invariants(params)?;
    let wire = WireWorkspaceEnvelope {
        v: WORKSPACE_SHARE_SCHEMA_VERSION,
        layout: params.layout,
        split_ratio: params.split_ratio.clamp(0.20, 0.80),
        quad_split_ratios: (
            params.quad_split_ratios.0.clamp(0.20, 0.80),
            params.quad_split_ratios.1.clamp(0.20, 0.80),
        ),
        focused_pane: params.focused_pane,
        compare_sync: if params.layout == ViewportLayout::Single {
            ComparisonSyncConfig::default()
        } else {
            params.compare_sync
        },
        panes: build_wire_panes(&params.panes)?,
    };
    let wire_json = serde_json::to_string(&wire)
        .map_err(|e| format!("Failed to serialize workspace URL envelope: {e}"))?;
    if wire_json.len() > MAX_DECODED_WORKSPACE_URL_BYTES {
        return Err(format!(
            "Workspace URL payload ({} bytes) exceeds maximum limit ({MAX_DECODED_WORKSPACE_URL_BYTES} bytes); {MWS_EXPORT_GUIDANCE}",
            wire_json.len()
        ));
    }
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(wire_json.as_bytes())
        .map_err(|e| format!("Failed to compress workspace URL payload: {e}"))?;
    let compressed = encoder
        .finish()
        .map_err(|e| format!("Failed to finalize DEFLATE stream: {e}"))?;
    let fragment = format!("#ws={}", base64url::encode(&compressed));
    if fragment.len() > MAX_URL_FRAGMENT_BYTES {
        return Err(format!(
            "Compressed workspace URL fragment ({} bytes) exceeds maximum limit ({MAX_URL_FRAGMENT_BYTES} bytes); {MWS_EXPORT_GUIDANCE}",
            fragment.len()
        ));
    }
    Ok(fragment)
}

/// Encodes a single simulation pane as `#tab=...` when raw preset JSON fits in 4,096 bytes,
/// automatically falling back to a compressed 1-pane `#ws=...` envelope up to 32,768 bytes.
pub fn encode_single_pane_smart_fragment(
    tab_slug: &str,
    tool_id: Option<&str>,
    preset_json: Option<&str>,
) -> Result<String, String> {
    encode_single_pane_smart_fragment_with_challenge(tab_slug, tool_id, None, preset_json)
}

/// Encodes a single simulation pane (including optional `challenge_id`) as `#tab=...` when raw
/// preset JSON fits in 4,096 bytes, falling back to a 1-pane `#ws=...` envelope up to 32,768 bytes.
pub fn encode_single_pane_smart_fragment_with_challenge(
    tab_slug: &str,
    tool_id: Option<&str>,
    challenge_id: Option<&str>,
    preset_json: Option<&str>,
) -> Result<String, String> {
    let fits_legacy = preset_json.is_none_or(|j| j.len() <= MAX_DECODED_URL_STATE_BYTES);
    if fits_legacy {
        if let Ok(legacy_frag) =
            encode_share_fragment_with_challenge(tab_slug, tool_id, challenge_id, preset_json)
        {
            return Ok(legacy_frag);
        }
    }
    let single_ws = WorkspaceShareParams {
        layout: ViewportLayout::Single,
        split_ratio: 0.5,
        quad_split_ratios: (0.5, 0.5),
        focused_pane: 0,
        compare_sync: ComparisonSyncConfig::default(),
        panes: vec![ShareUrlParams {
            tab: tab_slug.to_string(),
            tool: tool_id.map(str::to_string),
            challenge: challenge_id.map(str::to_string),
            state_json: preset_json.map(str::to_string),
        }],
    };
    encode_workspace_share_fragment(&single_ws)
}

fn inflate_bounded_deflate(compressed: &[u8]) -> Result<String, String> {
    let decoder = DeflateDecoder::new(compressed);
    let max_plus_one = (MAX_DECODED_WORKSPACE_URL_BYTES + 1) as u64;
    let mut decompressed = Vec::with_capacity(compressed.len().min(4096));
    decoder
        .take(max_plus_one)
        .read_to_end(&mut decompressed)
        .map_err(|e| format!("Corrupted DEFLATE stream in 'ws' URL parameter: {e}"))?;
    if decompressed.len() > MAX_DECODED_WORKSPACE_URL_BYTES {
        return Err(format!(
            "Decompressed workspace URL payload exceeds maximum allowed limit ({MAX_DECODED_WORKSPACE_URL_BYTES} bytes)"
        ));
    }
    String::from_utf8(decompressed)
        .map_err(|_| "Decompressed workspace URL payload is not valid UTF-8".to_string())
}

fn decode_wire_panes(panes: Vec<WirePaneShareEntry>) -> Result<Vec<ShareUrlParams>, String> {
    let mut out = Vec::with_capacity(panes.len());
    for entry in panes {
        let tab = entry.tab.trim().to_string();
        if tab.is_empty() {
            return Err("Workspace URL pane is missing a valid 'tab' slug".to_string());
        }
        let tool = match entry.tool {
            Some(t) if t.trim().is_empty() => {
                return Err("Workspace URL pane contains an empty 'tool' identifier".to_string());
            }
            Some(t) => Some(t.trim().to_string()),
            None => None,
        };
        let challenge = match entry.challenge {
            Some(c) => Some(validate_challenge_id(&c)?),
            None => None,
        };
        let state_json = match entry.state {
            Some(raw_val) => {
                let raw_str = raw_val.get();
                validate_workspace_url_preset_json(raw_str)?;
                Some(clamp_decoded_url_grid_state(
                    &tab,
                    tool.as_deref(),
                    raw_str,
                )?)
            }
            None => None,
        };
        out.push(ShareUrlParams {
            tab,
            tool,
            challenge,
            state_json,
        });
    }
    Ok(out)
}

/// Decodes, inflates (capped at 32,768 bytes), and structurally validates a `#ws=<base64url>` value.
pub fn decode_workspace_share_param(raw_b64: &str) -> Result<WorkspaceShareParams, String> {
    let trimmed = raw_b64.trim();
    if trimmed.is_empty() {
        return Err("Shared URL contains an empty 'ws' parameter".to_string());
    }
    let compressed = base64url::decode(trimmed, MAX_URL_FRAGMENT_BYTES)?;
    let json_str = inflate_bounded_deflate(&compressed)?;
    let cleaned = egui_plot::export::strip_utf8_bom(&json_str);
    WORKSPACE_URL_JSON_BOUNDS.validate_json(cleaned)?;

    let raw_tree: serde_json::Value =
        serde_json::from_str(cleaned).map_err(|e| format!("Invalid workspace URL JSON: {e}"))?;
    ensure_all_json_numbers_finite(&raw_tree)?;

    let wire: WireWorkspaceEnvelope = serde_json::from_str(cleaned)
        .map_err(|e| format!("Invalid workspace URL envelope: {e}"))?;
    if wire.v != WORKSPACE_SHARE_SCHEMA_VERSION {
        return Err(format!(
            "Unsupported workspace URL schema version '{}' (expected {})",
            wire.v, WORKSPACE_SHARE_SCHEMA_VERSION
        ));
    }
    let panes = decode_wire_panes(wire.panes)?;
    let params = WorkspaceShareParams {
        layout: wire.layout,
        split_ratio: wire.split_ratio,
        quad_split_ratios: wire.quad_split_ratios,
        focused_pane: wire.focused_pane,
        compare_sync: if wire.layout == ViewportLayout::Single {
            ComparisonSyncConfig::default()
        } else {
            wire.compare_sync
        },
        panes,
    };
    validate_workspace_params_invariants(&params)?;
    Ok(WorkspaceShareParams {
        split_ratio: params.split_ratio.clamp(0.20, 0.80),
        quad_split_ratios: (
            params.quad_split_ratios.0.clamp(0.20, 0.80),
            params.quad_split_ratios.1.clamp(0.20, 0.80),
        ),
        ..params
    })
}
