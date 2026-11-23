#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{IsingModelTool, DEFAULT_ISING_SIZE};
use math_commons::math_kernel::types::flatten_2d_index;

/// Lattice evolution kernel selection for [`IsingModelTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum IsingInteractionMode {
    /// Classical scalar ferromagnetic/antiferromagnetic Metropolis-Hastings dynamics.
    #[default]
    Ferromagnetic,
    /// Spatial 2x2 evolutionary game with pairwise Fermi imitation.
    FermiGame,
}

/// Canonical 2x2 evolutionary game presets `(R, S, T, P)` for spatial Fermi dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum FermiGamePreset {
    /// Prisoner's Dilemma (`T > R > P > S`): boundary reciprocity sustains cooperator droplets while 0D collapses.
    #[default]
    PrisonersDilemma,
    /// Stag-Hunt (`R > T > P > S`): bistable coordination game with critical nucleation threshold.
    StagHunt,
    /// Hawk-Dove / Snowdrift (`T > R > S > P`): polymorphic coexistence of cooperators and defectors.
    HawkDove,
    /// User-edited 2x2 payoff matrix.
    Custom,
}

impl FermiGamePreset {
    /// Display label for preset buttons and combo selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PrisonersDilemma => "Prisoner's Dilemma",
            Self::StagHunt => "Stag-Hunt",
            Self::HawkDove => "Hawk-Dove",
            Self::Custom => "Custom",
        }
    }

    /// Canonical `[R, S, T, P]` 2x2 payoff matrix entries (`+1` Cooperator = row 0, `-1` Defector = row 1).
    #[must_use]
    pub const fn payoffs(self) -> [f64; 4] {
        match self {
            Self::PrisonersDilemma | Self::Custom => [1.0, -0.1, 1.25, 0.0],
            Self::StagHunt => [2.0, 0.0, 1.2, 1.0],
            Self::HawkDove => [1.0, 0.0, 2.0, -1.0],
        }
    }

    /// Recommended rationality noise temperature $T$ for the preset.
    #[must_use]
    pub const fn default_temperature(self) -> f64 {
        match self {
            Self::PrisonersDilemma => 0.08,
            Self::StagHunt | Self::HawkDove | Self::Custom => 0.15,
        }
    }
}

pub(super) const fn default_game_payoffs() -> [f64; 4] {
    FermiGamePreset::PrisonersDilemma.payoffs()
}

pub(super) const fn default_brush_radius() -> i32 {
    6
}

pub(super) const fn default_brush_cooperator() -> bool {
    true
}

impl scientific_metadata::theory::TheoryDescribable for IsingModelTool {
    fn theory_description(&self) -> String {
        "Two-dimensional square lattice ferromagnetic Ising model governed by Hamiltonian $H = -J \\sum_{\\langle i, j \\rangle} s_i s_j - h \\sum_i s_i$ with Onsager critical temperature $T_c = \\frac{2J}{k_B \\ln(1 + \\sqrt{2})} \\approx 2.269 \\frac{J}{k_B}$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Two-dimensional Ising model ferromagnetic phase transition theory".into()
    }
    fn theory_citation(&self) -> String {
        "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition (Onsager, 1944)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "2D Ising Model Ferromagnetism",
    "domain": "solid_state",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "temperature": 2.269,
    "j_coupling": 1.0,
    "h_field": 0.05,
    "steps_per_frame": 10000
  }
}"#;

fn generate_ising_pinned_domain_wall_preset() -> &'static str {
    static DOMAIN_WALL_PRESET: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    DOMAIN_WALL_PRESET
        .get_or_init(|| {
            let mut pins = vec![0u8; DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE];
            for y in 0..DEFAULT_ISING_SIZE {
                for x in 0..6 {
                    pins[flatten_2d_index(x, y, DEFAULT_ISING_SIZE)] = 1;
                }
                for x in (DEFAULT_ISING_SIZE - 6)..DEFAULT_ISING_SIZE {
                    pins[flatten_2d_index(x, y, DEFAULT_ISING_SIZE)] = 2;
                }
            }
            for x in 35..65 {
                for y in 35..65 {
                    let dx = (x as isize) - 50;
                    let dy = (y as isize) - 50;
                    if dx * dx + dy * dy <= 64 {
                        pins[flatten_2d_index(x, y, DEFAULT_ISING_SIZE)] = 1;
                    }
                }
            }
            let rle = crate::framework::encode_u8_grid(
                DEFAULT_ISING_SIZE,
                DEFAULT_ISING_SIZE,
                &pins,
                2,
            )
            .expect("valid Ising pinned RLE");
            serde_json::json!({
                "metadata": {
                    "schema_version": "1.0",
                    "title": "Ising Pinned Domain Wall",
                    "domain": "solid_state",
                    "timestamp": "2026-11-21T00:00:00Z"
                },
                "config": {
                    "temperature": 1.85,
                    "j_coupling": 1.0,
                    "h_field": 0.0,
                    "steps_per_frame": 12000,
                    "pinned_rle": rle
                }
            })
            .to_string()
        })
        .as_str()
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "ising_model_starter",
        title: "2D Ising Model Ferromagnetism",
        description: "H = -J ∑ s_i·s_j - h ∑ s_i: Monte Carlo Metropolis spin flip dynamics and Curie transition.",
        category: "Statistical Mechanics",
        icon: "🧲",
        domain: "solid_state",
        tool_name: Some("IsingModelTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "ising_pinned_domain_wall",
        title: "Ising Pinned Domain Wall",
        description: "Opposing +1/-1 pinned spin boundaries and central nucleus forcing interfacial domain-wall fluctuations.",
        category: "Statistical Mechanics",
        icon: "🧱",
        domain: "solid_state",
        tool_name: Some("IsingModelTool"),
        preset_generator: generate_ising_pinned_domain_wall_preset,
    }
}
