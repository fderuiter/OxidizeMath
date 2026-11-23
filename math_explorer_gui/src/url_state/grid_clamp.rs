//! Interactive grid resolution clamping for decoded URL simulation presets.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::matches_tab_slug;

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

pub(crate) fn clamp_decoded_url_grid_state(
    tab: &str,
    tool: Option<&str>,
    json_str: &str,
) -> Result<String, String> {
    let cleaned = egui_plot::export::strip_utf8_bom(json_str);
    let mut root: serde_json::Value =
        serde_json::from_str(cleaned).map_err(|e| format!("Invalid URL state JSON: {e}"))?;
    let max_schema = math_commons::registry::MAX_GRID_SIZE as u64;
    let surf_cap = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION as u64;
    let grid_cap = crate::presets::MAX_INTERACTIVE_GRID_SIZE as u64;
    let is_fluid = matches_tab_slug(tab, "fluid_dynamics", "Fluid Dynamics")
        || tool.is_some_and(|t| matches_tab_slug(t, "PotentialFlowTool", "Potential Flow"));

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
