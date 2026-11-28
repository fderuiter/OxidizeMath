//! `ToolMetadata` and `StarterPresetMetadata` registrations for `QuantumStatsPhononTool`.

use super::QuantumStatsPhononTool;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "QuantumStatsPhononTool",
        display_name: "Quantum Statistics & Phonons",
        domain: "solid_state",
        tags: &["fermi-dirac", "bose-einstein", "phonons", "debye", "screening", "yukawa"],
        build: || Box::new(QuantumStatsPhononTool::default()),
    }
}

const QUANTUM_STATS_STARTER_PRESET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Fermi-Dirac vs Bose-Einstein Quantum Occupancy",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "temperature_k": 300.0,
    "chemical_potential_ev": -0.05,
    "energy_min_ev": 0.02,
    "energy_span_ev": 0.45,
    "sound_velocity_eva": 3.5,
    "optical_omega_mev": 36.0,
    "debye_theta_k": 343.0,
    "k_tf_inv_a": 1.25,
    "exchange_j_mev": 8.0,
    "stats_mode": "BoseEinstein",
    "plot_view": "QuantumOccupancy"
  }
}"#;

const DEBYE_SCREENING_STARTER_PRESET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Debye Phonon Dispersion & Yukawa Screening",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "temperature_k": 45.0,
    "chemical_potential_ev": 0.25,
    "energy_min_ev": 0.0,
    "energy_span_ev": 0.60,
    "sound_velocity_eva": 4.2,
    "optical_omega_mev": 42.0,
    "debye_theta_k": 343.0,
    "k_tf_inv_a": 1.65,
    "exchange_j_mev": 10.0,
    "stats_mode": "FermiDiracCompare",
    "plot_view": "PhononAndDebye"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "quantum_stats_fermi_bose_starter",
        title: "Quantum Statistical Occupancy Lab",
        description: "Compare Fermi-Dirac, Bose-Einstein (μ < E), and Maxwell-Boltzmann distributions in CODATA SI units.",
        category: "Condensed Matter",
        icon: "📊",
        domain: "solid_state",
        tool_name: Some("QuantumStatsPhononTool"),
        preset_generator: || QUANTUM_STATS_STARTER_PRESET,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "debye_phonon_screening_starter",
        title: "Debye Phonons & Yukawa Screening",
        description: "Acoustic/optical phonon branches, Debye T³ heat capacity, and Thomas-Fermi Yukawa screening.",
        category: "Condensed Matter",
        icon: "🔊",
        domain: "solid_state",
        tool_name: Some("QuantumStatsPhononTool"),
        preset_generator: || DEBYE_SCREENING_STARTER_PRESET,
    }
}

// [cite:mmwave_radiotherapy_setup]
