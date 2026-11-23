//! URL fragment/query parsing, percent-encoding, slug matching, and JSON state bounds validation.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::base64url;
use crate::framework::{ToolStateBounds, MAX_JSON_ARRAY_LEN, MAX_JSON_NESTING_DEPTH};

/// Maximum UTF-8 byte length permitted for raw URL fragments or query strings (`8 KiB`).
pub const MAX_URL_FRAGMENT_BYTES: usize = 8_192;

/// Maximum UTF-8 byte length permitted for decoded JSON simulation state (`4 KiB`).
pub const MAX_DECODED_URL_STATE_BYTES: usize = 4_096;

/// Canonical public web deployment base URL used when generating share links from desktop builds.
pub const CANONICAL_WEB_BASE_URL: &str = "https://fderuiter.github.io/OxidizeMath/";

/// Structural and byte-size limits enforced on decoded URL JSON payloads prior to deserialization.
pub const URL_JSON_BOUNDS: ToolStateBounds = ToolStateBounds::new(
    MAX_DECODED_URL_STATE_BYTES,
    MAX_JSON_NESTING_DEPTH + 1,
    MAX_JSON_ARRAY_LEN,
);

/// Parsed simulation share-link parameters extracted from `location.hash` or `location.search`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShareUrlParams {
    /// Target domain tab slug or normalized tab identifier.
    pub tab: String,
    /// Optional target `InteractiveTool` identifier or display name.
    pub tool: Option<String>,
    /// Optional inverse `ChallengeMetadata` identifier (`challenge=<id>`).
    pub challenge: Option<String>,
    /// Optional decoded and structurally validated `PresetWrapper` JSON string.
    pub state_json: Option<String>,
}

fn percent_encode_component(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for &b in raw.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            use std::fmt::Write;
            let _ = write!(&mut out, "%{b:02X}");
        }
    }
    out
}

fn percent_decode_component(raw: &str) -> Result<String, String> {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err("Incomplete percent-encoding sequence in URL parameter".to_string());
            }
            let hi = (bytes[i + 1] as char)
                .to_digit(16)
                .ok_or_else(|| "Invalid hex digit in percent-encoded URL parameter".to_string())?;
            let lo = (bytes[i + 2] as char)
                .to_digit(16)
                .ok_or_else(|| "Invalid hex digit in percent-encoded URL parameter".to_string())?;
            out.push(((hi << 4) | lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| "URL parameter is not valid UTF-8".to_string())
}

pub use super::json_validator::validate_url_preset_json;

/// Encodes `tab_slug`, optional `tool_id`, and optional compact `preset_json` into a
/// `#tab=...&tool=...&state=...` URL fragment while enforcing the 4 KiB and 8 KiB caps.
pub fn encode_share_fragment(
    tab_slug: &str,
    tool_id: Option<&str>,
    preset_json: Option<&str>,
) -> Result<String, String> {
    encode_share_fragment_with_challenge(tab_slug, tool_id, None, preset_json)
}

/// Encodes `tab_slug`, optional `tool_id`, optional `challenge_id`, and optional compact
/// `preset_json` into `#tab=<slug>&tool=<id>&challenge=<id>&state=<base64url>`.
pub fn encode_share_fragment_with_challenge(
    tab_slug: &str,
    tool_id: Option<&str>,
    challenge_id: Option<&str>,
    preset_json: Option<&str>,
) -> Result<String, String> {
    if tab_slug.trim().is_empty() {
        return Err("Cannot encode share URL with an empty tab slug".to_string());
    }
    let mut fragment = format!("#tab={}", percent_encode_component(tab_slug));
    if let Some(tool) = tool_id.filter(|t| !t.trim().is_empty()) {
        fragment.push_str("&tool=");
        fragment.push_str(&percent_encode_component(tool));
    }
    if let Some(ch) = challenge_id.filter(|c| !c.trim().is_empty()) {
        validate_challenge_id(ch.trim())?;
        fragment.push_str("&challenge=");
        fragment.push_str(&percent_encode_component(ch.trim()));
    }
    if let Some(json) = preset_json {
        validate_url_preset_json(json)?;
        let encoded_state = base64url::encode(json.as_bytes());
        fragment.push_str("&state=");
        fragment.push_str(&encoded_state);
    }
    if fragment.len() > MAX_URL_FRAGMENT_BYTES {
        return Err(format!(
            "Encoded URL fragment ({} bytes) exceeds maximum allowed limit ({MAX_URL_FRAGMENT_BYTES} bytes)",
            fragment.len()
        ));
    }
    Ok(fragment)
}

fn validate_challenge_id(raw_id: &str) -> Result<String, String> {
    let trimmed = raw_id.trim();
    if trimmed.is_empty() {
        return Err("Shared URL contains an empty 'challenge' parameter".to_string());
    }
    if trimmed.len() > 64
        || !trimmed
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err(format!(
            "Invalid 'challenge' identifier '{trimmed}' in shared URL"
        ));
    }
    Ok(trimmed.to_string())
}

/// Parses `location.hash` first, falling back to `location.search` (ignoring unrelated query
/// parameters like `stall_timeout_ms`) when no hash share parameters are present.
pub fn parse_share_location(hash: &str, search: &str) -> Result<Option<ShareUrlParams>, String> {
    if hash.len() > MAX_URL_FRAGMENT_BYTES {
        return Err(format!(
            "URL fragment length ({} bytes) exceeds maximum allowed limit ({MAX_URL_FRAGMENT_BYTES} bytes)",
            hash.len()
        ));
    }
    let raw_hash = hash.strip_prefix('#').unwrap_or(hash).trim();
    if !raw_hash.is_empty() && raw_hash != "the_canvas_id" {
        return parse_param_pairs(raw_hash, true);
    }

    if search.len() > MAX_URL_FRAGMENT_BYTES {
        return Err(format!(
            "URL query string length ({} bytes) exceeds maximum allowed limit ({MAX_URL_FRAGMENT_BYTES} bytes)",
            search.len()
        ));
    }
    let raw_search = search.strip_prefix('?').unwrap_or(search).trim();
    if raw_search.is_empty() {
        return Ok(None);
    }
    parse_param_pairs(raw_search, false)
}

fn parse_param_pairs(raw: &str, strict_keys: bool) -> Result<Option<ShareUrlParams>, String> {
    let mut tab: Option<String> = None;
    let mut tool: Option<String> = None;
    let mut challenge: Option<String> = None;
    let mut state_b64: Option<String> = None;

    for pair in raw.split('&').filter(|s| !s.is_empty()) {
        let Some((k_raw, v_raw)) = pair.split_once('=') else {
            if strict_keys {
                return Err(format!("Malformed URL fragment parameter '{pair}'"));
            }
            continue;
        };
        match k_raw {
            "tab" => set_unique_param(&mut tab, v_raw, "tab")?,
            "tool" => set_unique_param(&mut tool, v_raw, "tool")?,
            "challenge" => set_unique_param(&mut challenge, v_raw, "challenge")?,
            "state" => set_unique_param(&mut state_b64, v_raw, "state")?,
            _ if strict_keys => {
                return Err(format!("Unknown URL fragment parameter '{k_raw}'"));
            }
            _ => {}
        }
    }

    if tab.is_none() && tool.is_none() && challenge.is_none() && state_b64.is_none() {
        return Ok(None);
    }
    build_validated_share_params(tab, tool, challenge, state_b64)
}

fn set_unique_param(slot: &mut Option<String>, raw_val: &str, name: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("Duplicate '{name}' parameter in shared URL"));
    }
    *slot = Some(percent_decode_component(raw_val)?);
    Ok(())
}

fn clamp_json_u64_in_range(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    cap: u64,
    max_allowed: u64,
) -> bool {
    let Some(slot) = map.get_mut(key) else {
        return false;
    };
    let Some(raw) = slot.as_u64() else {
        return false;
    };
    if raw > cap && raw <= max_allowed {
        *slot = serde_json::Value::Number(serde_json::Number::from(cap));
        true
    } else {
        false
    }
}

fn clamp_decoded_url_grid_state(
    tab: &str,
    tool: Option<&str>,
    json_str: &str,
) -> Result<String, String> {
    let cleaned = egui_plot::export::strip_utf8_bom(json_str);
    let is_fluid = matches_tab_slug(tab, "fluid_dynamics", "Fluid Dynamics")
        || tool.is_some_and(|t| matches_tab_slug(t, "PotentialFlowTool", "Potential Flow"));
    let has_grid_key = cleaned.contains("\"u_resolution\"")
        || cleaned.contains("\"v_resolution\"")
        || cleaned.contains("\"grid_density\"")
        || (is_fluid && cleaned.contains("\"grid_size\""));
    if !has_grid_key {
        return Ok(json_str.to_string());
    }
    let mut root: serde_json::Value =
        serde_json::from_str(cleaned).map_err(|e| format!("Invalid URL state JSON: {e}"))?;
    let max_schema = math_commons::registry::MAX_GRID_SIZE as u64;
    let surf_cap = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION as u64;
    let grid_cap = crate::presets::MAX_INTERACTIVE_GRID_SIZE as u64;

    let target_map = if root.get("config").is_some_and(serde_json::Value::is_object) {
        root.get_mut("config")
            .and_then(serde_json::Value::as_object_mut)
    } else {
        root.as_object_mut()
    };
    let mut clamped = false;
    if let Some(cfg) = target_map {
        clamped |= clamp_json_u64_in_range(cfg, "u_resolution", surf_cap, max_schema);
        clamped |= clamp_json_u64_in_range(cfg, "v_resolution", surf_cap, max_schema);
        clamped |= clamp_json_u64_in_range(cfg, "grid_density", grid_cap, max_schema);
        if is_fluid {
            clamped |= clamp_json_u64_in_range(cfg, "grid_size", grid_cap, max_schema);
        }
    }
    if clamped {
        serde_json::to_string(&root).map_err(|e| format!("Invalid URL state JSON: {e}"))
    } else {
        Ok(json_str.to_string())
    }
}

fn build_validated_share_params(
    tab: Option<String>,
    tool: Option<String>,
    challenge: Option<String>,
    state_b64: Option<String>,
) -> Result<Option<ShareUrlParams>, String> {
    let tab_val = tab
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .ok_or_else(|| "Shared URL is missing a valid 'tab' parameter".to_string())?;
    let tool_val = match tool {
        Some(t) if t.trim().is_empty() => {
            return Err("Shared URL contains an empty 'tool' parameter".to_string());
        }
        Some(t) => Some(t.trim().to_string()),
        None => None,
    };
    let challenge_val = match challenge {
        Some(c) => Some(validate_challenge_id(&c)?),
        None => None,
    };
    let state_json = match state_b64 {
        Some(raw_b64) => {
            if raw_b64.trim().is_empty() {
                return Err("Shared URL contains an empty 'state' parameter".to_string());
            }
            let decoded_bytes = base64url::decode(raw_b64.trim(), MAX_DECODED_URL_STATE_BYTES)?;
            let json_str = String::from_utf8(decoded_bytes)
                .map_err(|_| "Decoded URL state is not valid UTF-8".to_string())?;
            validate_url_preset_json(&json_str)?;
            Some(clamp_decoded_url_grid_state(
                &tab_val,
                tool_val.as_deref(),
                &json_str,
            )?)
        }
        None => None,
    };
    Ok(Some(ShareUrlParams {
        tab: tab_val,
        tool: tool_val,
        challenge: challenge_val,
        state_json,
    }))
}

fn normalize_slug_tokens(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_sep = true;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        } else if !prev_sep {
            out.push('_');
            prev_sep = true;
        }
    }
    if out.ends_with('_') {
        out.pop();
    }
    out
}

/// Returns true if `requested` matches a tab's canonical `url_slug` or display `tab_name`.
#[must_use]
pub fn matches_tab_slug(requested: &str, canonical_slug: &str, tab_name: &str) -> bool {
    let req = normalize_slug_tokens(requested);
    if req.is_empty() {
        return false;
    }
    req == normalize_slug_tokens(canonical_slug)
        || req == normalize_slug_tokens(tab_name)
        || (canonical_slug == "morphogenesis" && req == "biology")
}
