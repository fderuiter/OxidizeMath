#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

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
