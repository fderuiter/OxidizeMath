#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Starter presets, state bounds validation, and theoretical metadata for `AbelianSandpileStudio`.

use math_explorer::physics::stat_mech::sandpile::{AvalancheObservable, MAX_SANDPILE_DIM};

use super::{
    AbelianSandpileStudio, SandpileBrush, SandpileColormap, SandpilePreset, SandpileStudioState,
    SandpileViewMode, MIN_SANDPILE_DIM,
};

pub(super) fn is_valid_state(s: &SandpileStudioState) -> bool {
    (MIN_SANDPILE_DIM..=MAX_SANDPILE_DIM).contains(&s.grid_size)
        && (1..=64).contains(&s.drops_per_frame)
        && (1..=16).contains(&s.brush_grains)
        && s.cursor_x < s.grid_size
        && s.cursor_y < s.grid_size
}

pub(super) fn clamp_state_fields(s: &mut SandpileStudioState) {
    s.grid_size = s.grid_size.clamp(MIN_SANDPILE_DIM, MAX_SANDPILE_DIM);
    s.drops_per_frame = s.drops_per_frame.clamp(1, 64);
    s.brush_grains = s.brush_grains.clamp(1, 16);
    s.cursor_x = s.cursor_x.min(s.grid_size - 1);
    s.cursor_y = s.cursor_y.min(s.grid_size - 1);
}

pub(super) fn default_state_for_preset(preset: SandpilePreset) -> SandpileStudioState {
    match preset {
        SandpilePreset::CriticalAvalanches1F => SandpileStudioState {
            preset,
            grid_size: 31,
            drops_per_frame: 4,
            animate_wavefronts: false,
            brush: SandpileBrush::SingleGrain,
            brush_grains: 4,
            view_mode: SandpileViewMode::GrainHeights,
            colormap: SandpileColormap::CriticalThermal,
            observable: AvalancheObservable::Size,
            show_burning_overlay: false,
            cursor_x: 15,
            cursor_y: 15,
            rng_seed: 2026,
        },
        SandpilePreset::CreutzIdentityCrystal63 => SandpileStudioState {
            preset,
            grid_size: 63,
            drops_per_frame: 1,
            animate_wavefronts: false,
            brush: SandpileBrush::SymmetricCross,
            brush_grains: 4,
            view_mode: SandpileViewMode::GrainHeights,
            colormap: SandpileColormap::ApollonianHighContrast,
            observable: AvalancheObservable::Size,
            show_burning_overlay: false,
            cursor_x: 31,
            cursor_y: 31,
            rng_seed: 1991,
        },
        SandpilePreset::AbelianGroupIdentityProof => SandpileStudioState {
            preset,
            grid_size: 25,
            drops_per_frame: 2,
            animate_wavefronts: false,
            brush: SandpileBrush::CriticalTower,
            brush_grains: 8,
            view_mode: SandpileViewMode::DharBurningWaves,
            colormap: SandpileColormap::BurningWavePhase,
            observable: AvalancheObservable::Area,
            show_burning_overlay: true,
            cursor_x: 12,
            cursor_y: 12,
            rng_seed: 1990,
        },
    }
}

impl scientific_metadata::theory::TheoryDescribable for AbelianSandpileStudio {
    fn theory_description(&self) -> String {
        "2D Bak-Tang-Wiesenfeld Abelian sandpile cellular automaton exhibiting parameter-free self-organized criticality (1/f power-law avalanche distributions) and finite Abelian group structure G ~= Z^(W*H) / Delta Z^(W*H) with Creutz recurrent identity e = (2*z_max - (2*z_max)^o)^o, Dhar's burning recurrence waves, and exact Bareiss Kirchhoff determinants.".into()
    }

    fn phonetic_description(&self) -> String {
        "Two dimensional Bak Tang Wiesenfeld Abelian sandpile self organized criticality and recurrent group studio"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Bak, Tang & Wiesenfeld, Phys. Rev. Lett. 59, 381 (1987); Dhar, Phys. Rev. Lett. 64, 1613 (1990); Creutz, Comput. Phys. 5, 198 (1991)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "CriticalAvalanches1F".into(),
            "Parameter-free slow grain driving onto the self-organized critical attractor with log-log power-law fits and seismic sonification".into(),
        );
        map.insert(
            "CreutzIdentityCrystal63".into(),
            "Exact 63x63 Apollonian-like recurrent identity fractal e = (2*z_max - (2*z_max)^o)^o satisfying (e + e)^o = e".into(),
        );
        map.insert(
            "AbelianGroupIdentityProof".into(),
            "Constructive verification of (r + e)^o = r, recurrent inverse (r + r^-1)^o = e, Zn cyclic cosets, and Dhar burning wave depths".into(),
        );
        map
    }
}

const PRESET_SOC_AVALANCHES_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "1/f Critical Avalanche Cascades",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "CriticalAvalanches1F",
    "grid_size": 31,
    "drops_per_frame": 4,
    "animate_wavefronts": false,
    "brush": "SingleGrain",
    "brush_grains": 4,
    "view_mode": "GrainHeights",
    "colormap": "CriticalThermal",
    "observable": "Size",
    "show_burning_overlay": false,
    "cursor_x": 15,
    "cursor_y": 15,
    "rng_seed": 2026
  }
}"#;

const PRESET_IDENTITY_CRYSTAL_63_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "63x63 Recurrent Identity Crystal",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "CreutzIdentityCrystal63",
    "grid_size": 63,
    "drops_per_frame": 1,
    "animate_wavefronts": false,
    "brush": "SymmetricCross",
    "brush_grains": 4,
    "view_mode": "GrainHeights",
    "colormap": "ApollonianHighContrast",
    "observable": "Size",
    "show_burning_overlay": false,
    "cursor_x": 31,
    "cursor_y": 31,
    "rng_seed": 1991
  }
}"#;

const PRESET_ABELIAN_GROUP_PROOF_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "(r + e)° = r Abelian Group Proof",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "AbelianGroupIdentityProof",
    "grid_size": 25,
    "drops_per_frame": 2,
    "animate_wavefronts": false,
    "brush": "CriticalTower",
    "brush_grains": 8,
    "view_mode": "DharBurningWaves",
    "colormap": "BurningWavePhase",
    "observable": "Area",
    "show_burning_overlay": true,
    "cursor_x": 12,
    "cursor_y": 12,
    "rng_seed": 1990
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "AbelianSandpileStudio",
        display_name: "Abelian Sandpile Studio",
        domain: "solid_state",
        tags: &["sandpile", "self-organized-criticality", "abelian-group", "fractals", "avalanches"],
        build: || Box::new(AbelianSandpileStudio::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sandpile_soc_avalanches_starter",
        title: "1/f Critical Avalanche Cascades",
        description: "Parameter-free Bak-Tang-Wiesenfeld grain cascades with live log-log power-law fitting and seismic crackle audio.",
        category: "Self-Organized Criticality",
        icon: "⏳",
        domain: "solid_state",
        tool_name: Some("AbelianSandpileStudio"),
        preset_generator: || PRESET_SOC_AVALANCHES_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sandpile_identity_crystal_63_starter",
        title: "63x63 Recurrent Identity Crystal",
        description: "Exact Creutz iterative fractal identity e = (2*z_max - (2*z_max)^o)^o of the finite Abelian sandpile group.",
        category: "Algebraic Lattice Crystals",
        icon: "🔷",
        domain: "solid_state",
        tool_name: Some("AbelianSandpileStudio"),
        preset_generator: || PRESET_IDENTITY_CRYSTAL_63_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sandpile_abelian_group_proof_starter",
        title: "(r + e)° = r Abelian Group Proof",
        description: "Verify recurrent identity (r + e)^o = r, group inverse r^-1, Zn cyclic cosets, and Dhar burning wave depths.",
        category: "Algebraic Lattice Crystals",
        icon: "🧮",
        domain: "solid_state",
        tool_name: Some("AbelianSandpileStudio"),
        preset_generator: || PRESET_ABELIAN_GROUP_PROOF_JSON,
    }
}
