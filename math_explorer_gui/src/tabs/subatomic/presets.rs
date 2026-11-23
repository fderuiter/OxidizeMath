#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Four curated starter presets for the Subatomic Physics (`subatomic`) starter gallery.

use crate::starter_gallery::StarterPresetMetadata;

/// Starter preset 1: Iron-56 Valley of Stability peak & Weizsäcker isobar parabola.
pub const PRESET_VALLEY_OF_STABILITY_FE56: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Nuclear Stability & Decay",
    "domain": "subatomic",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "selected_z": 26,
    "selected_n": 30,
    "orbital_l": 3,
    "spin_aligned": true,
    "alpha_beta": 0.05,
    "resonance_energy_mev": 8.0,
    "resonance_gamma_mev": 1.4,
    "probe_energy_mev": 8.0,
    "half_life_1": 14.0,
    "half_life_2": 7.0,
    "half_life_3": 18.0,
    "max_decay_time": 70.0,
    "plot_mode": "IsobarParabola"
  }
}"#;

/// Starter preset 2: Uranium-238 heavy-nuclide Coulomb competition & 4-step Bateman decay chain.
pub const PRESET_U238_BATEMAN_CHAIN: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Nuclear Stability & Decay",
    "domain": "subatomic",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "selected_z": 92,
    "selected_n": 146,
    "orbital_l": 4,
    "spin_aligned": true,
    "alpha_beta": 0.047,
    "resonance_energy_mev": 6.7,
    "resonance_gamma_mev": 0.8,
    "probe_energy_mev": 6.7,
    "half_life_1": 24.0,
    "half_life_2": 24.0,
    "half_life_3": 10.0,
    "max_decay_time": 110.0,
    "plot_mode": "BatemanDecayChain"
  }
}"#;

/// Starter preset 3: DUNE 1300 km MSW matter neutrino oscillation & electroweak symmetry breaking.
pub const PRESET_DUNE_MSW_NEUTRINOS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Standard Model & Neutrinos",
    "domain": "subatomic",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "g1": 0.357,
    "g2": 0.652,
    "gs": 1.221,
    "higgs_mu2": -7840.0,
    "higgs_lambda": 0.1293,
    "ckm_theta12_deg": 13.04,
    "ckm_theta23_deg": 2.38,
    "ckm_theta13_deg": 0.201,
    "ckm_delta_deg": 68.8,
    "alpha_s_mz": 0.1179,
    "active_flavors": 5.0,
    "probe_scale_gev": 91.1876,
    "initial_flavor": "Muon",
    "pmns_theta12_deg": 33.41,
    "pmns_theta23_deg": 49.10,
    "pmns_theta13_deg": 8.54,
    "pmns_delta_cp_deg": 197.0,
    "inverted_ordering": false,
    "antineutrino": false,
    "neutrino_energy_gev": 2.5,
    "baseline_km": 1300.0,
    "matter_density_g_cm3": 2.85,
    "electron_fraction": 0.5,
    "plot_view": "MswOscillations"
  }
}"#;

/// Starter preset 4: Superluminal blazar jet Doppler beaming, synchrotron power, and >5-sigma Li-Ma flare.
pub const PRESET_BLAZAR_SYNCHROTRON_JET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Relativistic Jet & Observer Lab",
    "domain": "subatomic",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "beta": 0.98,
    "viewing_angle_deg": 8.5,
    "rest_mass_kg": 1.0,
    "comoving_px_factor": 0.6,
    "magnetic_energy_density_ub": 2.5,
    "electron_power_law_p": 2.3,
    "continuous_jet": false,
    "adiabatic_index": 1.3333333333333333,
    "fluid_density": 1.0,
    "pressure_over_rho_c2": 0.33,
    "photon_n_on": 240,
    "photon_n_off": 200,
    "exposure_alpha": 0.2,
    "plot_view": "AngularBeamingAndSuperluminal"
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "subatomic_valley_of_stability_fe56",
        title: "Iron-56 Peak & Isobar Parabola",
        description: "Inspect Weizsacker volume, surface, Coulomb, asymmetry, and pairing terms at Fe-56.",
        category: "Nuclear Structure",
        icon: "⚛️",
        domain: "subatomic",
        tool_name: Some("NuclearStabilityTool"),
        preset_generator: || PRESET_VALLEY_OF_STABILITY_FE56,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "subatomic_u238_bateman_decay_chain",
        title: "Uranium-238 & Degenerate Bateman Chain",
        description: "Explore alpha Q-value, Gamow tunneling, and a 4-step Bateman chain with equal parent/daughter half-lives.",
        category: "Radioactive Decay",
        icon: "☢️",
        domain: "subatomic",
        tool_name: Some("NuclearStabilityTool"),
        preset_generator: || PRESET_U238_BATEMAN_CHAIN,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "subatomic_dune_msw_neutrinos",
        title: "Long-Baseline MSW Matter Neutrinos",
        description: "Compare 3-flavor PMNS crust-matter oscillations at L = 1300 km against vacuum probabilities and QCD running.",
        category: "Standard Model",
        icon: "✨",
        domain: "subatomic",
        tool_name: Some("StandardModelTool"),
        preset_generator: || PRESET_DUNE_MSW_NEUTRINOS,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "subatomic_blazar_synchrotron_jet",
        title: "Superluminal Blazar Synchrotron Jet",
        description: "Boost a 0.98c plasmoid at 8.5 deg viewing angle with beta_app > 4.9c and >10-sigma Li-Ma gamma-ray significance.",
        category: "High-Energy Astrophysics",
        icon: "🚀",
        domain: "subatomic",
        tool_name: Some("RelativisticJetTool"),
        preset_generator: || PRESET_BLAZAR_SYNCHROTRON_JET,
    }
}
