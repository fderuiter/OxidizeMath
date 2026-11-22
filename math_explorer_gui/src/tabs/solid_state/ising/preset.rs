#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{IsingModelTool, DEFAULT_ISING_SIZE};
use math_commons::math_kernel::types::flatten_2d_index;

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
