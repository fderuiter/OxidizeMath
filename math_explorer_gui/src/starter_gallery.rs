//! Starter Template Gallery component providing guided empty-state cards
//! for simulation tabs and instant preset model execution.

use crate::framework::ToolMetadata;
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
    /// Target domain string matching `InteractivePreset::domain`.
    pub domain: &'static str,
    /// Tool metadata name in the domain to select/activate.
    pub tool_name: Option<&'static str>,
    /// Serialized JSON `PresetWrapper` configuration to load.
    pub preset_json: Option<&'static str>,
}

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
    "preset_type": "CylinderFlow",
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
    "preset_type": "SourceSink",
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
    "preset_type": "StandardLorenz",
    "dt": 0.01,
    "simulation_speed": 5
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

/// Returns curated or dynamically generated starter cards for a simulation domain.
#[allow(clippy::too_many_lines)]
pub fn get_starter_cards_for_domain(
    domain: &str,
    available_tools: &[&'static ToolMetadata],
) -> Vec<StarterCard> {
    match domain {
        "analysis" => vec![
            StarterCard {
                id: "harmonic_oscillator",
                title: "Harmonic Oscillator",
                description: "y'' = -k·y: Second-order differential equation modeling undamped periodic spring motion.",
                category: "Differential Equations",
                icon: "〰️",
                domain: "analysis",
                tool_name: Some("OdeSolverTool"),
                preset_json: Some(PRESET_HARMONIC_OSCILLATOR),
            },
            StarterCard {
                id: "logistic_growth",
                title: "Logistic Growth",
                description: "y' = r·y·(1 - y/K): Non-linear growth constrained by carrying capacity K.",
                category: "Population Dynamics",
                icon: "📈",
                domain: "analysis",
                tool_name: Some("OdeSolverTool"),
                preset_json: Some(PRESET_LOGISTIC_GROWTH),
            },
            StarterCard {
                id: "exponential_decay",
                title: "Exponential Decay",
                description: "y' = k·y: First-order differential decay process with constant half-life.",
                category: "Calculus",
                icon: "📉",
                domain: "analysis",
                tool_name: Some("OdeSolverTool"),
                preset_json: Some(PRESET_EXPONENTIAL_DECAY),
            },
        ],
        "fluid_dynamics" => vec![
            StarterCard {
                id: "vector_field",
                title: "Vector Field",
                description: "ψ = U·y·(1 - R²/r²): Potential streamlines and 2D velocity vector field past a cylinder.",
                category: "Fluid Mechanics",
                icon: "🌊",
                domain: "fluid_dynamics",
                tool_name: Some("PotentialFlowTool"),
                preset_json: Some(PRESET_VECTOR_FIELD),
            },
            StarterCard {
                id: "source_sink",
                title: "Source-Sink Pair",
                description: "ψ = (q/2π)·(θ₁ - θ₂): Superposition of equal source and sink singularities.",
                category: "Fluid Mechanics",
                icon: "🔄",
                domain: "fluid_dynamics",
                tool_name: Some("PotentialFlowTool"),
                preset_json: Some(PRESET_SOURCE_SINK),
            },
        ],
        "chaos" => vec![
            StarterCard {
                id: "3d_attractor",
                title: "3D Attractor",
                description: "ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz: Canonical 3D chaotic strange attractor trajectory.",
                category: "Chaos Theory",
                icon: "🦋",
                domain: "chaos",
                tool_name: Some("AttractorPlotter"),
                preset_json: Some(PRESET_3D_ATTRACTOR),
            },
            StarterCard {
                id: "bifurcation",
                title: "Bifurcation Diagram",
                description: "x_{n+1} = r·x_n·(1 - x_n): Period-doubling cascade towards deterministic chaos.",
                category: "Non-Linear Dynamics",
                icon: "🌿",
                domain: "chaos",
                tool_name: Some("BifurcationDiagram"),
                preset_json: None,
            },
        ],
        "biology" | "morphogenesis" => vec![
            StarterCard {
                id: "turing_spots",
                title: "Turing Spots",
                description: "∂u/∂t = D_u∇²u + a - u + u²v: Activator-inhibitor reaction-diffusion generating spot morphology.",
                category: "Reaction-Diffusion",
                icon: "🐆",
                domain: "biology",
                tool_name: None,
                preset_json: Some(PRESET_TURING_SPOTS),
            },
            StarterCard {
                id: "turing_stripes",
                title: "Turing Stripes",
                description: "∂v/∂t = D_v∇²v + b - u²v: Differential diffusion instability yielding labyrinthine stripe patterns.",
                category: "Morphogenesis",
                icon: "🦓",
                domain: "biology",
                tool_name: None,
                preset_json: Some(PRESET_TURING_STRIPES),
            },
        ],
        "game_theory" => vec![
            StarterCard {
                id: "hawk_dove",
                title: "Hawk-Dove Dynamics",
                description: "ẋ_i = x_i((Ax)_i - xᵀAx): Evolutionary game model of conflict and resource competition.",
                category: "Game Theory",
                icon: "🦅",
                domain: "game_theory",
                tool_name: Some("ReplicatorDynamicsTool"),
                preset_json: Some(PRESET_HAWK_DOVE),
            },
        ],
        _ => {
            // Dynamic fallback cards based on available tools in domain
            available_tools
                .iter()
                .map(|meta| StarterCard {
                    id: meta.name,
                    title: meta.name,
                    description: "Interactive simulation tool for mathematical exploration and analysis.",
                    category: meta.domain,
                    icon: "🔬",
                    domain: meta.domain,
                    tool_name: Some(meta.name),
                    preset_json: None,
                })
                .collect()
        }
    }
}

/// Renders the central canvas starter gallery card overlay in an egui Context.
#[allow(clippy::too_many_lines)]
pub fn show_starter_gallery_overlay(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    is_open: &mut bool,
) -> Option<StarterCard> {
    if !*is_open || cards.is_empty() {
        return None;
    }

    let mut selected_card = None;

    egui::Area::new(egui::Id::new(format!("{}_starter_gallery_overlay", id_source)))
        .order(egui::Order::Foreground)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            egui::Frame::window(ui.style())
                .inner_margin(16)
                .corner_radius(12.0)
                .shadow(ui.style().visuals.window_shadow)
                .show(ui, |ui| {
                    ui.set_max_width(740.0);
                    ui.set_max_height(520.0);

                    // Header
                    ui.horizontal(|ui| {
                        ui.heading("🚀 Starter Template Gallery");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("✖ Close Gallery").clicked() {
                                *is_open = false;
                            }
                        });
                    });

                    ui.label(
                        egui::RichText::new("Choose a guided starter model to launch pre-configured simulation parameters:")
                            .color(ui.visuals().weak_text_color()),
                    );
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(10.0);

                    // Cards Grid
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
                            for card in cards {
                                egui::Frame::group(ui.style())
                                    .inner_margin(14.0)
                                    .corner_radius(8.0)
                                    .show(ui, |ui| {
                                        ui.set_width(215.0);
                                        ui.set_min_height(170.0);
                                        ui.vertical(|ui| {
                                            ui.horizontal(|ui| {
                                                ui.label(egui::RichText::new(card.icon).size(26.0));
                                                ui.vertical(|ui| {
                                                    ui.label(egui::RichText::new(card.title).strong().size(15.0));
                                                    ui.label(
                                                        egui::RichText::new(format!("[{}]", card.category))
                                                            .small()
                                                            .color(ui.visuals().weak_text_color()),
                                                    );
                                                });
                                            });
                                            ui.add_space(8.0);
                                            ui.label(
                                                egui::RichText::new(card.description)
                                                    .size(12.0)
                                                    .color(ui.visuals().text_color()),
                                            );
                                            ui.add_space(10.0);
                                            let btn = ui.button(format!("▶ Launch {}", card.title));
                                            if btn.clicked() {
                                                selected_card = Some(card.clone());
                                                *is_open = false;
                                            }
                                        });
                                    });
                            }
                        });
                    });
                });
        });

    selected_card
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starter_cards_presence_and_metadata() {
        let analysis_cards = get_starter_cards_for_domain("analysis", &[]);
        assert!(analysis_cards
            .iter()
            .any(|c| c.title == "Harmonic Oscillator"));
        assert!(analysis_cards.iter().any(|c| c.title == "Logistic Growth"));

        let fluid_cards = get_starter_cards_for_domain("fluid_dynamics", &[]);
        assert!(fluid_cards.iter().any(|c| c.title == "Vector Field"));

        let chaos_cards = get_starter_cards_for_domain("chaos", &[]);
        assert!(chaos_cards.iter().any(|c| c.title == "3D Attractor"));

        let bio_cards = get_starter_cards_for_domain("biology", &[]);
        assert!(bio_cards.iter().any(|c| c.title == "Turing Spots"));
    }

    #[test]
    fn test_preset_json_validity() {
        use crate::presets::{MorphogenesisPresetConfig, OdePresetConfig, PresetWrapper};

        let ode_env: PresetWrapper<OdePresetConfig> =
            serde_json::from_str(PRESET_HARMONIC_OSCILLATOR).unwrap();
        assert_eq!(ode_env.metadata.domain, "analysis");

        let log_env: PresetWrapper<OdePresetConfig> =
            serde_json::from_str(PRESET_LOGISTIC_GROWTH).unwrap();
        assert_eq!(log_env.metadata.domain, "analysis");

        let morph_env: PresetWrapper<MorphogenesisPresetConfig> =
            serde_json::from_str(PRESET_TURING_SPOTS).unwrap();
        assert_eq!(morph_env.metadata.domain, "biology");
    }
}
