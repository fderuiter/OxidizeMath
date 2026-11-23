#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BifurcationDiagram, BifurcationMapMode};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

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

impl TheoryDescribable for BifurcationDiagram {
    fn theory_description(&self) -> String {
        match self.map_mode {
            BifurcationMapMode::Logistic => format!(
                "Logistic map bifurcation diagram and maximal Lyapunov exponent λ(r), r range: {:.2} to {:.2}, selected r: {:.3}",
                self.r_min, self.r_max, self.selected_r
            ),
            BifurcationMapMode::Custom => format!(
                "Custom 1D map bifurcation x_{{n+1}} = {} with tangent Lyapunov overlay, r range: {:.2} to {:.2}, selected r: {:.3}",
                self.math_input.raw_text, self.r_min, self.r_max, self.selected_r
            ),
        }
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "May, R. M. (1976). Simple mathematical models with very complicated dynamics. Nature, 261(5560), 459-467; Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear transformations. Journal of Statistical Physics, 19(1), 25-52.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Logistic map bifurcation diagram and maximal Lyapunov exponent λ(r)".to_string(),
        );
        map
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        self.build_theory_parameters()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BifurcationDiagram",
        display_name: "Bifurcation Diagram",
        domain: "chaos",
        tags: &[
            "bifurcation",
            "logistic",
            "logistic map",
            "lyapunov",
            "cobweb",
            "period doubling",
            "feigenbaum",
            "orbit diagram",
            "chaos",
        ],
        build: || Box::new(BifurcationDiagram::default()),
    }
}
