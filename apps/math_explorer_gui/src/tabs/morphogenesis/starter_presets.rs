//! Starter gallery preset registrations for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_commons::math_kernel::types::flatten_2d_index;

/// JSON preset envelope for Gray-Scott Mitosis reaction-diffusion.
pub const PRESET_GRAY_SCOTT_MITOSIS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Gray-Scott Mitosis",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 0.16,
    "d_v": 0.08,
    "dt": 1.0,
    "width": 100,
    "height": 100,
    "kinetics_mode": "GrayScott",
    "f": 0.0367,
    "k": 0.0649
  }
}"#;

/// JSON preset envelope for Gray-Scott Coral dendritic branching (`f = 0.0545`, `k = 0.0620`).
pub const PRESET_GRAY_SCOTT_CORAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Gray-Scott Coral",
    "domain": "biology",
    "timestamp": "2026-11-24T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 0.16,
    "d_v": 0.08,
    "dt": 1.0,
    "width": 100,
    "height": 100,
    "kinetics_mode": "GrayScott",
    "f": 0.0545,
    "k": 0.0620
  }
}"#;

/// JSON preset envelope for Gray-Scott Solitons excitable pulses (`f = 0.030`, `k = 0.060`).
pub const PRESET_GRAY_SCOTT_SOLITONS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Gray-Scott Solitons",
    "domain": "biology",
    "timestamp": "2026-11-24T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 0.16,
    "d_v": 0.08,
    "dt": 1.0,
    "width": 100,
    "height": 100,
    "kinetics_mode": "GrayScott",
    "f": 0.030,
    "k": 0.060
  }
}"#;

fn generate_hybrid_turing_mask_preset() -> &'static str {
    static HYBRID_PRESET: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    HYBRID_PRESET
        .get_or_init(|| {
            let mut mask = vec![0u8; 100 * 100];
            for y in 15..85 {
                for x in 50..90 {
                    mask[flatten_2d_index(x, y, 100)] = 1;
                }
            }
            let rle = crate::framework::encode_u8_grid(100, 100, &mask, 2)
                .expect("valid morphogenesis hybrid RLE");
            serde_json::json!({
                "metadata": {
                    "schema_version": "1.0",
                    "title": "Turing Spot-Stripe Mask",
                    "domain": "biology",
                    "timestamp": "2026-11-21T00:00:00Z"
                },
                "config": {
                    "a": 0.08,
                    "b": 1.12,
                    "d_u": 1.0,
                    "d_v": 40.0,
                    "dt": 0.005,
                    "width": 100,
                    "height": 100,
                    "kinetics_mask_rle": rle
                }
            })
            .to_string()
        })
        .as_str()
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turing_spots",
        title: "Turing Spots",
        description: "∂u/∂t = D_u∇²u + a - u + u²v: Activator-inhibitor reaction-diffusion generating spot morphology.",
        category: "Reaction-Diffusion",
        icon: "🐆",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || crate::starter_gallery::PRESET_TURING_SPOTS,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turing_stripes",
        title: "Turing Stripes",
        description: "∂v/∂t = D_v∇²v + b - u²v: Differential diffusion instability yielding labyrinthine stripe patterns.",
        category: "Morphogenesis",
        icon: "🦓",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || crate::starter_gallery::PRESET_TURING_STRIPES,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turing_spot_stripe_mask",
        title: "Turing Spot-Stripe Mask",
        description: "Spatially heterogeneous Schnakenberg (a, b) RLE reaction mask transitioning spots into labyrinthine stripes.",
        category: "Morphogenesis",
        icon: "🎭",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: generate_hybrid_turing_mask_preset,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "morphogenesis_inhibitor_fork",
        title: "Morphogenesis Inhibitor Fork",
        description: "Counterfactual Schnakenberg twin forked at step 60 lowering D_v from 40 to 18 with |u_B - u_A| difference heatmap.",
        category: "Counterfactual Biology",
        icon: "⑂",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || crate::starter_gallery::PRESET_MORPHOGENESIS_INHIBITOR_FORK,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gray_scott_mitosis",
        title: "Gray-Scott Mitosis",
        description: "∂v/∂t = D_v∇²v + uv² - (f+k)v: Autocatalytic spot division and self-replicating wavefronts.",
        category: "Reaction-Diffusion",
        icon: "🦠",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_MITOSIS,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gray_scott_coral",
        title: "Gray-Scott Coral",
        description: "∂v/∂t = D_v∇²v + uv² - (f+k)v: Dendritic coral labyrinth growth at f = 0.0545, k = 0.0620.",
        category: "Morphogenesis",
        icon: "🪸",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_CORAL,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gray_scott_solitons",
        title: "Gray-Scott Solitons",
        description: "∂u/∂t = D_u∇²u - uv² + f(1-u): Localized self-sustaining excitable soliton pulses at f = 0.030, k = 0.060.",
        category: "Reaction-Diffusion",
        icon: "🫧",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_SOLITONS,
    }
}

/// JSON preset envelope coupling Lattice Boltzmann cylinder vortex shedding with Gray-Scott mitosis.
pub const PRESET_VORTEX_WAKE_MORPHOGENESIS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Vortex Wake Morphogenesis",
    "domain": "biology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 0.16,
    "d_v": 0.08,
    "dt": 0.9,
    "width": 100,
    "height": 100,
    "kinetics_mode": "GrayScott",
    "f": 0.0367,
    "k": 0.0649,
    "mask_rle": [[0,4622],[1,1],[0,97],[1,5],[0,94],[1,7],[0,93],[1,7],[0,92],[1,9],[0,92],[1,7],[0,93],[1,7],[0,94],[1,5],[0,97],[1,1],[0,4577]],
    "advection_enabled": true,
    "inlet_u": 0.12,
    "viscosity": 0.015
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vortex_wake_morphogenesis",
        title: "Vortex Wake Morphogenesis",
        description: "∂v/∂t + ∇·(uv) = D_v∇²v + R(u,v): Kármán vortex shedding behind a sculpted cylinder braiding Gray-Scott spots.",
        category: "Multiphysics",
        icon: "🌀",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_VORTEX_WAKE_MORPHOGENESIS,
    }
}
