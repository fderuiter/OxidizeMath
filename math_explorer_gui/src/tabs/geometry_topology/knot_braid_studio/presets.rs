#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Starter Gallery presets for the Knot, Braid & Skein Studio.

pub(super) const CHIRAL_TREFOIL_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Chiral Trefoil Braid Closure",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "tube_radius": 0.16,
    "family": "ArtinBraid",
    "braid_strands": 2,
    "braid_generators": [-1, -1, -1],
    "torus_p": 2,
    "torus_q": -3,
    "lissajous_freqs": [3, 4, 7],
    "lissajous_phases": [0.1, 0.7, 0.2],
    "major_radius": 2.0,
    "minor_radius": 0.65,
    "curve_samples": 72,
    "radial_segments": 12,
    "show_bishop_frames": true,
    "show_tube_rings": true,
    "camera": {
      "pitch": 0.55,
      "yaw": 0.45,
      "zoom": 1.0
    }
  }
}"#;

pub(super) const FIGURE_EIGHT_ISOTOPY_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Figure-Eight Reidemeister Isotopy",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "tube_radius": 0.15,
    "family": "FigureEight",
    "braid_strands": 3,
    "braid_generators": [1, -2, 1, -2],
    "torus_p": 2,
    "torus_q": 3,
    "lissajous_freqs": [3, 4, 7],
    "lissajous_phases": [0.1, 0.7, 0.2],
    "major_radius": 2.1,
    "minor_radius": 0.65,
    "curve_samples": 80,
    "radial_segments": 14,
    "show_bishop_frames": false,
    "show_tube_rings": true,
    "camera": {
      "pitch": 0.6,
      "yaw": -0.35,
      "zoom": 1.05
    },
    "reidemeister_ops": [
      {
        "op": "RiAdd",
        "segment": 0,
        "positive": true
      }
    ]
  }
}"#;

pub(super) const BORROMEAN_RINGS_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Borromean Rings Brunnian Link",
    "domain": "geometry_topology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "tube_radius": 0.13,
    "family": "BorromeanRings",
    "braid_strands": 3,
    "braid_generators": [1, -2, 1, -2, 1, -2],
    "torus_p": 2,
    "torus_q": 3,
    "lissajous_freqs": [3, 4, 7],
    "lissajous_phases": [0.1, 0.7, 0.2],
    "major_radius": 2.0,
    "minor_radius": 0.65,
    "curve_samples": 84,
    "radial_segments": 12,
    "show_bishop_frames": false,
    "show_tube_rings": true,
    "camera": {
      "pitch": 0.5,
      "yaw": 0.65,
      "zoom": 0.95
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "knot_studio_chiral_trefoil",
        title: "Chiral Trefoil Braid Closure",
        description: "Left-handed 2-strand Artin braid trefoil closure with Bishop parallel-transport frames and chiral Jones polynomial.",
        category: "Low-Dimensional Topology",
        icon: "🪢",
        domain: "geometry_topology",
        tool_name: Some("KnotBraidStudio"),
        preset_generator: || CHIRAL_TREFOIL_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "knot_studio_figure_eight_isotopy",
        title: "Figure-Eight Reidemeister Isotopy",
        description: "Amphichiral 4_1 figure-eight knot with an active Reidemeister-I curl ready for interactive untwisting.",
        category: "Low-Dimensional Topology",
        icon: "♾️",
        domain: "geometry_topology",
        tool_name: Some("KnotBraidStudio"),
        preset_generator: || FIGURE_EIGHT_ISOTOPY_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "knot_studio_borromean_rings",
        title: "Borromean Rings Brunnian Link",
        description: "3-component Brunnian link with pairwise Gauss linking numbers Lk = 0 yet non-trivial Jones skein invariant.",
        category: "Low-Dimensional Topology",
        icon: "🔗",
        domain: "geometry_topology",
        tool_name: Some("KnotBraidStudio"),
        preset_generator: || BORROMEAN_RINGS_PRESET_JSON,
    }
}
