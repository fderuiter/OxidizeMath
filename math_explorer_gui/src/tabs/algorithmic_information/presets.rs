#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Federated `ToolMetadata` and `StarterPresetMetadata` registrations for [`FractranMachineTool`].

use super::state::FractranMachineTool;

/// Starter preset JSON envelope launching Conway's `PRIMEGAME` past its second prime ($2^3$).
pub const PRESET_FRACTRAN_PRIMEGAME: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "FRACTRAN Virtual Machine - Conway PRIMEGAME",
    "domain": "algorithmic_information",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "playback_rate": 2.0,
    "steps_per_frame": 32,
    "warmup_steps": 75,
    "initial_value": 2,
    "preset": "Primegame",
    "custom_fractions": "17/91, 78/85, 19/51, 23/38, 29/33, 77/29, 95/23, 77/19, 1/17, 11/13, 13/11, 15/2, 1/7, 55/1"
  }
}"#;

/// Starter preset JSON envelope computing $(F_6, F_5) = (8, 5)$ via 9 fractions.
pub const PRESET_FRACTRAN_FIBONACCI: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "FRACTRAN Virtual Machine - Fibonacci Register Machine",
    "domain": "algorithmic_information",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "playback_rate": 1.5,
    "steps_per_frame": 24,
    "warmup_steps": 40,
    "initial_value": 6250,
    "preset": "Fibonacci",
    "custom_fractions": "1547/22, 11/17, 119/33, 19/11, 46/133, 19/23, 69/247, 1/19, 11/5"
  }
}"#;

/// Starter preset JSON envelope multiplying $2^3 \cdot 3^3 \to 5^9$.
pub const PRESET_FRACTRAN_MULTIPLIER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "FRACTRAN Virtual Machine - Minsky Multiplier",
    "domain": "algorithmic_information",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "playback_rate": 1.25,
    "steps_per_frame": 12,
    "warmup_steps": 20,
    "initial_value": 216,
    "preset": "Multiplier",
    "custom_fractions": "455/33, 11/13, 1/11, 3/7, 11/2, 1/3"
  }
}"#;

/// Starter preset JSON envelope adding $2^5 \cdot 3^4 \to 3^9$.
pub const PRESET_FRACTRAN_ADDER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "FRACTRAN Virtual Machine - Minsky Adder",
    "domain": "algorithmic_information",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "playback_rate": 1.0,
    "steps_per_frame": 4,
    "warmup_steps": 5,
    "initial_value": 2592,
    "preset": "Adder",
    "custom_fractions": "3/2"
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "FractranMachineTool",
        display_name: "FRACTRAN Virtual Machine",
        domain: "algorithmic_information",
        tags: &["fractran", "kolmogorov", "minsky", "primes", "bdm", "lz76"],
        build: || Box::new(FractranMachineTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractran_primegame",
        title: "Conway PRIMEGAME",
        description: "N_{t+1} = N_t * f_j: 14 fractions streaming pure powers 2^p for every prime p.",
        category: "Algorithmic Information",
        icon: "🔢",
        domain: "algorithmic_information",
        tool_name: Some("FractranMachineTool"),
        preset_generator: || PRESET_FRACTRAN_PRIMEGAME,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractran_fibonacci",
        title: "Fibonacci Register Machine",
        description: "2^1 * 5^n -> 2^{F_{n+1}} * 3^{F_n}: 9-fraction Minsky machine halting on Fibonacci pairs.",
        category: "Algorithmic Information",
        icon: "🐚",
        domain: "algorithmic_information",
        tool_name: Some("FractranMachineTool"),
        preset_generator: || PRESET_FRACTRAN_FIBONACCI,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractran_multiplier",
        title: "Minsky Multiplier",
        description: "2^a * 3^b -> 5^{a*b}: Nested prime-register loops demonstrating universal arithmetic.",
        category: "Algorithmic Information",
        icon: "✖",
        domain: "algorithmic_information",
        tool_name: Some("FractranMachineTool"),
        preset_generator: || PRESET_FRACTRAN_MULTIPLIER,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractran_adder",
        title: "Minsky Adder",
        description: "2^a * 3^b -> 3^{a+b}: Minimal 1-rule register transfer and Kolmogorov decay.",
        category: "Algorithmic Information",
        icon: "➕",
        domain: "algorithmic_information",
        tool_name: Some("FractranMachineTool"),
        preset_generator: || PRESET_FRACTRAN_ADDER,
    }
}

// [cite:algorithmic_information_rust]
