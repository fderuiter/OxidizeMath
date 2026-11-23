//! Structural invariant and HUD field validation helpers for `#ws=<base64url>` workspace URLs.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::codec::{validate_url_identifier, MAX_URL_STORY_STEPS};
use super::workspace_codec::WorkspaceShareParams;
use crate::app::dataflow_bus::PanePipelineDag;

pub(super) struct ValidatedWireHud {
    pub preset: Option<String>,
    pub challenge: Option<String>,
    pub step: Option<usize>,
}

pub(super) fn validate_wire_hud_fields(
    raw_preset: Option<&str>,
    raw_challenge: Option<&str>,
    raw_step: Option<usize>,
) -> Result<ValidatedWireHud, String> {
    let preset = raw_preset
        .filter(|p| !p.trim().is_empty())
        .map(|p| validate_url_identifier(p, "preset"))
        .transpose()?;
    let challenge = raw_challenge
        .filter(|c| !c.trim().is_empty())
        .map(|c| validate_url_identifier(c, "challenge"))
        .transpose()?;
    if preset.is_some() && challenge.is_some() {
        return Err("Shared URL cannot specify both 'preset' and 'challenge'".to_string());
    }
    if let Some(idx) = raw_step {
        if preset.is_none() {
            return Err(
                "Shared URL 'step' parameter requires a 'preset' story identifier".to_string(),
            );
        }
        if idx > MAX_URL_STORY_STEPS {
            return Err(format!(
                "Shared URL 'step' index ({idx}) exceeds maximum ({MAX_URL_STORY_STEPS})"
            ));
        }
    }
    Ok(ValidatedWireHud {
        preset,
        challenge,
        step: raw_step.filter(|&s| s > 0),
    })
}

pub(super) fn validate_workspace_params_invariants(
    params: &WorkspaceShareParams,
) -> Result<(), String> {
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
    if params.bus_links.len() > expected_panes {
        return Err(format!(
            "Workspace URL bus_links length ({}) exceeds pane count ({expected_panes})",
            params.bus_links.len()
        ));
    }
    if expected_panes <= 1 && params.bus_links.iter().any(|&b| b) {
        return Err(
            "Single-pane workspace URL cannot contain active workspace bus links".to_string(),
        );
    }
    if !params.links.is_empty() {
        if expected_panes <= 1 {
            return Err("Single-pane workspace URL cannot contain pipeline links".to_string());
        }
        let mut dag = PanePipelineDag::default();
        dag.restore_links(&params.links, expected_panes)
            .map_err(|e| e.message().to_string())?;
        for pane in &params.panes {
            if let Some(ref s) = pane.state_json {
                super::pipeline::enforce_url_graph_node_cap(s)?;
            }
        }
    }
    Ok(())
}
