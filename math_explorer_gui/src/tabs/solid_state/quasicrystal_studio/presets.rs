#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Curated starter presets for [`super::QuasicrystalStudio`] in the Solid State gallery.

// [cite:solid_state]

use crate::starter_gallery::StarterPresetMetadata;

/// Starter preset 1: 5D Penrose decagonal quasicrystal with active phason flip comparison.
pub const PRESET_PENROSE_10FOLD: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Quasicrystal & Bragg Studio",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "Quasicrystal",
    "quasicrystal": "Penrose5D",
    "bravais": "Hexagonal",
    "edge_length": 1.0,
    "lattice_b": 1.45,
    "gamma_deg": 72.0,
    "aperture_radius": 28.0,
    "phason_x": 0.03,
    "phason_y": -0.02,
    "reference_phason_x": 0.0,
    "reference_phason_y": 0.0,
    "phason_drift_speed": 0.18,
    "phason_drift_angle_deg": 36.0,
    "highlight_flipped_tiles": true,
    "show_brillouin_zone": true,
    "show_bragg_rings": true,
    "colormap": "Inferno",
    "log_gain": 4.0,
    "is_playing": false
  }
}"#;

/// Starter preset 2: 4D Ammann-Beenker octagonal tiling (silver ratio 2.414214 and C8 Bragg rings).
pub const PRESET_AMMANN_BEENKER_8FOLD: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Quasicrystal & Bragg Studio",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "Quasicrystal",
    "quasicrystal": "AmmannBeenker4D",
    "bravais": "Square",
    "edge_length": 1.0,
    "lattice_b": 1.0,
    "gamma_deg": 90.0,
    "aperture_radius": 28.0,
    "phason_x": 0.04,
    "phason_y": 0.03,
    "reference_phason_x": 0.0,
    "reference_phason_y": 0.0,
    "phason_drift_speed": 0.16,
    "phason_drift_angle_deg": 45.0,
    "highlight_flipped_tiles": true,
    "show_brillouin_zone": true,
    "show_bragg_rings": true,
    "colormap": "Viridis",
    "log_gain": 4.2,
    "is_playing": false
  }
}"#;

/// Starter preset 3: 6D Stampfli-Gähler dodecagonal tiling (C12 Bragg symmetry).
pub const PRESET_STAMPFLI_12FOLD: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Quasicrystal & Bragg Studio",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "Quasicrystal",
    "quasicrystal": "Stampfli6D",
    "bravais": "Hexagonal",
    "edge_length": 1.0,
    "lattice_b": 1.0,
    "gamma_deg": 60.0,
    "aperture_radius": 30.0,
    "phason_x": 0.03,
    "phason_y": -0.02,
    "reference_phason_x": 0.0,
    "reference_phason_y": 0.0,
    "phason_drift_speed": 0.15,
    "phason_drift_angle_deg": 30.0,
    "highlight_flipped_tiles": true,
    "show_brillouin_zone": true,
    "show_bragg_rings": true,
    "colormap": "Inferno",
    "log_gain": 4.0,
    "is_playing": false
  }
}"#;

/// Starter preset 4: Periodic 2D Hexagonal Bravais net with Wigner-Seitz First Brillouin Zone.
pub const PRESET_HEXAGONAL_BRAVAIS_BZ: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Quasicrystal & Bragg Studio",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "PeriodicBravais",
    "quasicrystal": "Penrose5D",
    "bravais": "Hexagonal",
    "edge_length": 1.2,
    "lattice_b": 1.2,
    "gamma_deg": 60.0,
    "aperture_radius": 26.0,
    "phason_x": 0.0,
    "phason_y": 0.0,
    "reference_phason_x": 0.0,
    "reference_phason_y": 0.0,
    "phason_drift_speed": 0.18,
    "phason_drift_angle_deg": 36.0,
    "highlight_flipped_tiles": false,
    "show_brillouin_zone": true,
    "show_bragg_rings": true,
    "colormap": "Monochrome",
    "log_gain": 3.8,
    "is_playing": false
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "solid_state_penrose_10fold",
        title: "Penrose 10-Fold Quasicrystal",
        description: "5D Z^5 projection showing golden-ratio rhombs, live phason flips, and forbidden C10 Bragg rings.",
        category: "Quasicrystals",
        icon: "✴️",
        domain: "solid_state",
        tool_name: Some("QuasicrystalStudio"),
        preset_generator: || PRESET_PENROSE_10FOLD,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "solid_state_ammann_beenker_8fold",
        title: "Ammann-Beenker 8-Fold Tiling",
        description: "4D Z^4 projection with silver-ratio octagonal acceptance window and C8 Bragg diffraction.",
        category: "Quasicrystals",
        icon: "💠",
        domain: "solid_state",
        tool_name: Some("QuasicrystalStudio"),
        preset_generator: || PRESET_AMMANN_BEENKER_8FOLD,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "solid_state_stampfli_12fold",
        title: "Stampfli 12-Fold Dodecagonal",
        description: "6D Z^6 dodecagonal projection tiling triangles, squares, and shields with sharp C12 Bragg rings.",
        category: "Quasicrystals",
        icon: "❄️",
        domain: "solid_state",
        tool_name: Some("QuasicrystalStudio"),
        preset_generator: || PRESET_STAMPFLI_12FOLD,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "solid_state_bravais_hexagonal_bz",
        title: "2D Hexagonal Bravais & Brillouin Zone",
        description: "Periodic p6m lattice contrasting 6-fold reciprocal Bragg spots and Wigner-Seitz First Brillouin Zone.",
        category: "Reciprocal Lattices",
        icon: "⬢",
        domain: "solid_state",
        tool_name: Some("QuasicrystalStudio"),
        preset_generator: || PRESET_HEXAGONAL_BRAVAIS_BZ,
    }
}
