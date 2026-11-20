#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// Maximum UTF-8 byte length (2 MiB) allowed for an individual serialized tool state.
pub const MAX_TOOL_STATE_BYTES: usize = 2 * 1024 * 1024;

/// Additional byte budget (4 KiB) allowed for a `.json` preset metadata envelope around `config`.
pub const MAX_PRESET_ENVELOPE_OVERHEAD_BYTES: usize = 4 * 1024;

/// Maximum UTF-8 byte length (8 MiB) allowed for a serialized `SimulationFramework` session payload.
pub const MAX_SESSION_STATE_BYTES: usize = 8 * 1024 * 1024;

/// Maximum number of saved tool entries permitted inside `SimulationFramework` `tool_states`.
pub const MAX_SESSION_TOOL_ENTRIES: usize = 128;

/// Maximum UTF-8 byte length allowed for a tool identifier key in `tool_states`.
pub const MAX_TOOL_KEY_BYTES: usize = 256;

/// Maximum JSON container (`{}` / `[]`) nesting depth allowed for serialized tool state.
pub const MAX_JSON_NESTING_DEPTH: usize = 32;

/// Maximum number of elements allowed in any single JSON array within a tool state.
pub const MAX_JSON_ARRAY_LEN: usize = 16_384;

/// Structural and byte-size limits enforced on untrusted JSON tool state before `serde_json` allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolStateBounds {
    /// Maximum allowed UTF-8 byte length of the JSON payload (after BOM removal).
    pub max_bytes: usize,
    /// Maximum allowed container (`{}` / `[]`) nesting depth.
    pub max_depth: usize,
    /// Maximum allowed element count within any single JSON array.
    pub max_array_len: usize,
}

impl Default for ToolStateBounds {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy)]
enum JsonScope {
    Object,
    Array { len: usize, empty: bool },
}

impl ToolStateBounds {
    /// Default bounds enforced for individual tool states (`load_state` and `try_load_state`).
    pub const DEFAULT: Self = Self {
        max_bytes: MAX_TOOL_STATE_BYTES,
        max_depth: MAX_JSON_NESTING_DEPTH,
        max_array_len: MAX_JSON_ARRAY_LEN,
    };

    /// Session-level bounds enforced on `SimulationFramework::load_state` payloads.
    pub const SESSION: Self = Self {
        max_bytes: MAX_SESSION_STATE_BYTES,
        max_depth: MAX_JSON_NESTING_DEPTH + 2,
        max_array_len: MAX_JSON_ARRAY_LEN,
    };

    /// Constructs explicit `ToolStateBounds` limits.
    #[must_use]
    pub const fn new(max_bytes: usize, max_depth: usize, max_array_len: usize) -> Self {
        Self {
            max_bytes,
            max_depth,
            max_array_len,
        }
    }

    /// Expands byte and depth headroom by one outer `PresetWrapper` envelope layer.
    #[must_use]
    pub const fn for_preset_envelope(self) -> Self {
        Self {
            max_bytes: self
                .max_bytes
                .saturating_add(MAX_PRESET_ENVELOPE_OVERHEAD_BYTES),
            max_depth: self.max_depth.saturating_add(1),
            max_array_len: self.max_array_len,
        }
    }

    /// Validates that `raw` satisfies byte-length, nesting-depth, and array-length bounds
    /// without allocating a `serde_json` value tree or target struct.
    pub fn validate_json(&self, raw: &str) -> Result<(), String> {
        let cleaned = egui_plot::export::strip_utf8_bom(raw);
        if cleaned.len() > self.max_bytes {
            return Err(format!(
                "JSON state size ({} bytes) exceeds maximum allowed limit ({} bytes)",
                cleaned.len(),
                self.max_bytes
            ));
        }
        scan_json_bounds(cleaned.as_bytes(), self.max_depth, self.max_array_len)
    }
}

fn scan_json_bounds(bytes: &[u8], max_depth: usize, max_array_len: usize) -> Result<(), String> {
    let mut stack: Vec<JsonScope> = Vec::with_capacity(max_depth.min(64).saturating_add(1));
    let mut in_string = false;
    let mut escaped = false;

    for &b in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
            continue;
        }
        step_structural_byte(b, &mut stack, &mut in_string, max_depth, max_array_len)?;
    }
    Ok(())
}

fn step_structural_byte(
    b: u8,
    stack: &mut Vec<JsonScope>,
    in_string: &mut bool,
    max_depth: usize,
    max_array_len: usize,
) -> Result<(), String> {
    if b != b']' {
        if let Some(JsonScope::Array { len, empty }) = stack.last_mut() {
            if *empty {
                *empty = false;
                *len = 1;
                if *len > max_array_len {
                    return Err(format!(
                        "JSON array length ({len}) exceeds maximum allowed limit ({max_array_len})"
                    ));
                }
            }
        }
    }

    match b {
        b'"' => *in_string = true,
        b'[' => push_scope(
            stack,
            JsonScope::Array {
                len: 0,
                empty: true,
            },
            max_depth,
        )?,
        b'{' => push_scope(stack, JsonScope::Object, max_depth)?,
        b']' | b'}' => {
            let _ = stack.pop();
        }
        b',' => {
            if let Some(JsonScope::Array { len, empty: false }) = stack.last_mut() {
                *len = len.saturating_add(1);
                if *len > max_array_len {
                    return Err(format!(
                        "JSON array length ({len}) exceeds maximum allowed limit ({max_array_len})"
                    ));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn push_scope(
    stack: &mut Vec<JsonScope>,
    scope: JsonScope,
    max_depth: usize,
) -> Result<(), String> {
    let next_depth = stack.len().saturating_add(1);
    if next_depth > max_depth {
        return Err(format!(
            "JSON nesting depth ({next_depth}) exceeds maximum allowed limit ({max_depth})"
        ));
    }
    stack.push(scope);
    Ok(())
}

/// Constructs a bounded [`eframe::egui::DragValue`] that clamps finite inputs to `range`
/// and rejects non-finite text literals such as `1e400`, `inf`, or `NaN`.
pub fn bounded_drag_value(
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    speed: f64,
) -> eframe::egui::DragValue<'_> {
    eframe::egui::DragValue::new(value)
        .range(range)
        .speed(speed)
        .custom_parser(|s| s.trim().parse::<f64>().ok().filter(|v| v.is_finite()))
}

/// Enforces strict `min_val < max_val` ordering within `[bound_min, bound_max]` after UI edits.
pub fn enforce_ordered_drag_range(
    min_val: &mut f64,
    max_val: &mut f64,
    bound_min: f64,
    bound_max: f64,
    min_span: f64,
    min_changed: bool,
    max_changed: bool,
) {
    if !min_changed && !max_changed {
        return;
    }
    if !min_val.is_finite() {
        *min_val = bound_min;
    }
    if !max_val.is_finite() {
        *max_val = bound_max;
    }
    *min_val = min_val.clamp(bound_min, bound_max);
    *max_val = max_val.clamp(bound_min, bound_max);
    if *min_val >= *max_val {
        if max_changed && !min_changed {
            *max_val = max_val.clamp(bound_min + min_span, bound_max);
            *min_val = (*max_val - min_span).max(bound_min);
        } else {
            *min_val = min_val.clamp(bound_min, bound_max - min_span);
            *max_val = (*min_val + min_span).min(bound_max);
        }
    }
}
