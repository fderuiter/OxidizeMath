//! Starter Template Gallery component providing guided empty-state cards
//! for simulation tabs and instant preset model execution.

use crate::framework::ToolMetadata;
#[cfg(test)]
use eframe::egui;

/// Pre-configured starter simulation template card.
#[derive(Clone, Debug, PartialEq)]
pub struct StarterCard {
    /// Unique identifier for the starter card.
    pub id: &'static str,
    /// User-facing display title.
    pub title: &'static str,
    /// Concise mathematical description or governing equation.
    pub description: &'static str,
    /// Category or domain tag.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain string matching [`crate::framework::InteractiveTool::domain`].
    pub domain: &'static str,
    /// Tool metadata name in the domain to select/activate.
    pub tool_name: Option<&'static str>,
    /// Serialized JSON `PresetWrapper` configuration to load.
    pub preset_json: Option<&'static str>,
}

/// Federated starter preset registration metadata submitted across tools.
#[derive(Clone, Copy, Debug)]
pub struct StarterPresetMetadata {
    /// Unique identifier for the preset template.
    pub id: &'static str,
    /// User-facing display title.
    pub title: &'static str,
    /// Concise mathematical description or governing equation.
    pub description: &'static str,
    /// Category or domain tag.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain string matching the simulation tool domain.
    pub domain: &'static str,
    /// Target tool metadata name (if associated with a registered ToolMetadata).
    pub tool_name: Option<&'static str>,
    /// Function pointer returning the serialized JSON preset configuration envelope.
    pub preset_generator: fn() -> &'static str,
}

impl PartialEq for StarterPresetMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.title == other.title && self.domain == other.domain
    }
}

impl StarterPresetMetadata {
    /// Generates the JSON preset payload.
    #[inline]
    pub fn generate_preset(&self) -> &'static str {
        (self.preset_generator)()
    }
}

inventory::collect!(StarterPresetMetadata);

/// JSON preset envelope for Harmonic Oscillator ODE.
pub const PRESET_HARMONIC_OSCILLATOR: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y'' = -k*y (Harmonic Oscillator)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "HarmonicOscillator",
    "dt": 0.05,
    "total_time": 20.0,
    "param_k": 2.0,
    "param_r": 1.0,
    "param_cap_k": 10.0,
    "ic_y0": 1.5,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Logistic Growth ODE.
pub const PRESET_LOGISTIC_GROWTH: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y' = r*y*(1 - y/K) (Logistic Growth)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "LogisticGrowth",
    "dt": 0.05,
    "total_time": 20.0,
    "param_k": 1.0,
    "param_r": 0.8,
    "param_cap_k": 10.0,
    "ic_y0": 0.5,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Exponential Growth/Decay ODE.
pub const PRESET_EXPONENTIAL_DECAY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y' = k*y (Exponential)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "Exponential",
    "dt": 0.05,
    "total_time": 10.0,
    "param_k": -0.5,
    "param_r": 1.0,
    "param_cap_k": 10.0,
    "ic_y0": 5.0,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Turing Spots reaction-diffusion.
pub const PRESET_TURING_SPOTS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis Simulation - Spots",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 1.0,
    "d_v": 100.0,
    "dt": 0.05,
    "width": 100,
    "height": 100
  }
}"#;

/// JSON preset envelope for Turing Stripes reaction-diffusion.
pub const PRESET_TURING_STRIPES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis Simulation - Stripes",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "a": 0.14,
    "b": 0.86,
    "d_u": 1.0,
    "d_v": 50.0,
    "dt": 0.02,
    "width": 100,
    "height": 100
  }
}"#;

/// JSON preset envelope for Fluid Dynamics Cylinder Vector Field.
pub const PRESET_VECTOR_FIELD: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Cylinder Vector Field",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "elements": [
      {
        "element_type": "Uniform",
        "strength": 10.0,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0
      },
      {
        "element_type": "Doublet",
        "strength": 251.32741228718345,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0
      }
    ],
    "grid_size": 20,
    "view_range": 10.0
  }
}"#;

/// JSON preset envelope for Fluid Dynamics Source-Sink Pair.
pub const PRESET_SOURCE_SINK: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Source-Sink Singularity Pair",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "elements": [
      {
        "element_type": "Source",
        "strength": 10.0,
        "x": -3.0,
        "y": 0.0,
        "angle": 0.0
      },
      {
        "element_type": "Sink",
        "strength": 10.0,
        "x": 3.0,
        "y": 0.0,
        "angle": 0.0
      }
    ],
    "grid_size": 25,
    "view_range": 12.0
  }
}"#;

/// JSON preset envelope for 3D Chaotic Attractor.
pub const PRESET_3D_ATTRACTOR: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lorenz 3D Attractor - Butterfly",
    "domain": "chaos",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665,
    "dt": 0.01,
    "simulation_speed": 5,
    "execution_state": "Running",
    "show_annotations": true,
    "camera": {
      "pitch": 0.35,
      "yaw": -0.45,
      "zoom": 1.0
    }
  }
}"#;

/// JSON preset envelope for Replicator Hawk-Dove Dynamics.
pub const PRESET_HAWK_DOVE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Replicator Dynamics - Hawk-Dove",
    "domain": "game_theory",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "payoff_matrix": [[-1.0, 2.0], [0.0, 1.0]],
    "initial_population": [0.5, 0.5],
    "time_horizon": 20.0,
    "dt": 0.05,
    "strategy_names": ["Hawk", "Dove"]
  }
}"#;

#[path = "starter_gallery/presets_data.rs"]
mod presets_data;
pub use presets_data::*;

/// Returns curated or dynamically generated starter cards for a simulation domain.
pub fn get_starter_cards_for_domain(
    domain: &str,
    available_tools: &[&'static ToolMetadata],
) -> Vec<StarterCard> {
    let mut cards: Vec<StarterCard> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| {
            p.domain == domain
                || (domain == "biology" && p.domain == "morphogenesis")
                || (domain == "morphogenesis" && p.domain == "biology")
        })
        .map(|p| StarterCard {
            id: p.id,
            title: p.title,
            description: p.description,
            category: p.category,
            icon: p.icon,
            domain: p.domain,
            tool_name: p.tool_name,
            preset_json: Some((p.preset_generator)()),
        })
        .collect();

    // Dynamic fallback cards based on available tools in domain if no presets submitted
    if cards.is_empty() {
        for meta in available_tools {
            if meta.domain == domain {
                cards.push(StarterCard {
                    id: meta.id,
                    title: meta.display_name,
                    description:
                        "Interactive simulation tool for mathematical exploration and analysis.",
                    category: meta.domain,
                    icon: "🔬",
                    domain: meta.domain,
                    tool_name: Some(meta.id),
                    preset_json: None,
                });
            }
        }
    }

    cards.sort_by_key(|c| (c.title, c.id));
    cards
}

#[path = "starter_gallery/root_host.rs"]
mod root_host;
pub use root_host::show_starter_gallery_overlay;
pub(crate) use root_host::{
    begin_root_starter_gallery_scope, flush_deferred_starter_galleries, try_defer_starter_gallery,
};

#[cfg(test)]
#[path = "starter_gallery/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "starter_gallery/contract_tests.rs"]
mod contract_tests;
