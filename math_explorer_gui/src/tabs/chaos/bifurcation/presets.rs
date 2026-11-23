#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BifurcationDiagram, BifurcationState, DEFAULT_CUSTOM_EXPR, DEFAULT_X0};
use crate::framework::MathExpressionInputBar;
use std::collections::BTreeMap;

pub(super) fn default_bifurcation_input_bar() -> MathExpressionInputBar {
    MathExpressionInputBar::new(
        "x_{n+1}",
        DEFAULT_CUSTOM_EXPR,
        vec![
            ("Sine Map", "r * sin(pi * x)"),
            ("Quadratic Map", "x^2 - r"),
            ("Ricker Population", "x * exp(r * (1 - x))"),
            ("Cubic Map", "r * x - x^3"),
            (
                "Perturbed Logistic",
                "r * x * (1 - x) + a * sin(2 * pi * x)",
            ),
        ],
    )
    .with_axes(&["r", "x"])
    .with_param_range(-2.0..=2.0, 0.05)
}

impl BifurcationDiagram {
    pub(super) fn build_state_json(&self, include_hover_r: bool) -> Option<String> {
        let is_custom = self.map_mode.is_custom();
        let midpoint = (self.r_min + self.r_max) * 0.5;
        let has_selected_r = self.r_pinned
            || ((include_hover_r || !self.r_from_hover)
                && (self.selected_r - midpoint).abs() > 1e-12);
        let selected_r = has_selected_r.then_some(self.selected_r);
        let x0 =
            (is_custom || self.r_pinned || (self.x0 - DEFAULT_X0).abs() > 1e-12).then_some(self.x0);
        let trimmed_expr = self.math_input.raw_text.trim();
        let custom_expr = (is_custom || trimmed_expr != DEFAULT_CUSTOM_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        let (xk, yk) = self.atlas.axis_keys();
        serde_json::to_string(&BifurcationState {
            r_min: self.r_min,
            r_max: self.r_max,
            steps: self.steps,
            map_mode: self.map_mode,
            selected_r,
            x0,
            r_pinned: self.r_pinned,
            use_custom_map: false,
            custom_expr,
            custom_params,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }
}

/// Legacy starter preset JSON envelope for the Logistic Map Bifurcation Diagram.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bifurcation Diagram",
    "domain": "chaos",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "r_min": 2.5,
    "r_max": 4.0,
    "steps": 800
  }
}"#;

/// Starter preset JSON envelope pinning the `r = 3.83` period-3 island of stability inside chaos.
pub const STARTER_PERIOD3_WINDOW_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Logistic Period-3 Window (r = 3.83)",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "r_min": 3.7,
    "r_max": 3.9,
    "steps": 800,
    "selected_r": 3.83,
    "x0": 0.1,
    "r_pinned": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bifurcation",
        title: "Bifurcation Diagram",
        description: "x_{n+1} = r·x_n·(1 - x_n): Period-doubling cascade towards deterministic chaos.",
        category: "Non-Linear Dynamics",
        icon: "🌿",
        domain: "chaos",
        tool_name: Some("BifurcationDiagram"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bifurcation_period3_window",
        title: "Logistic Period-3 Window (r = 3.83)",
        description: "Pinned r = 3.83, x_0 = 0.1 period-3 cobweb window inside chaos where λ plunges to -0.345.",
        category: "Non-Linear Dynamics",
        icon: "🎯",
        domain: "chaos",
        tool_name: Some("BifurcationDiagram"),
        preset_generator: || STARTER_PERIOD3_WINDOW_JSON,
    }
}
