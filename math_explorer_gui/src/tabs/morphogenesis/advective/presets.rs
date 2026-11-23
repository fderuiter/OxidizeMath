//! Reaction-advection preset schemas, [`TheoryDescribable`] metadata, and Starter Gallery cards
//! for [`super::AdvectiveMorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{decode_mask_hex, BrushMode, FairingStamp, LBM_CELL_COUNT};
use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(crate) const fn default_brush_radius() -> f64 {
    2.0
}

pub(crate) const fn default_brush_mode() -> BrushMode {
    BrushMode::Solid
}

pub(crate) const fn default_rng_seed() -> u64 {
    42
}

/// Reaction kinetics preset selectable in [`super::AdvectiveMorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdvectiveKineticsPreset {
    /// Schnakenberg activator-substrate spots (`a = 0.10, b = 0.90`).
    #[default]
    SchnakenbergSpots,
    /// Schnakenberg labyrinthine stripes (`a = 0.08, b = 1.15`).
    SchnakenbergStripes,
    /// Gray-Scott self-dividing mitosis spots (`F = 0.0367, k = 0.0649`).
    GrayScottMitosis,
    /// Gray-Scott dendritic coral fronds (`F = 0.0545, k = 0.0620`).
    GrayScottCoral,
}

impl AdvectiveKineticsPreset {
    /// Ordered list of all studio kinetics presets.
    pub const ALL: [AdvectiveKineticsPreset; 4] = [
        Self::SchnakenbergSpots,
        Self::SchnakenbergStripes,
        Self::GrayScottMitosis,
        Self::GrayScottCoral,
    ];

    /// Display label for UI combo boxes and accessibility status strings.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SchnakenbergSpots => "Schnakenberg Spots",
            Self::SchnakenbergStripes => "Schnakenberg Stripes",
            Self::GrayScottMitosis => "Gray-Scott Mitosis",
            Self::GrayScottCoral => "Gray-Scott Coral",
        }
    }

    /// Returns `true` when this preset uses [`GrayScottKinetics`].
    #[must_use]
    pub const fn is_gray_scott(self) -> bool {
        matches!(self, Self::GrayScottMitosis | Self::GrayScottCoral)
    }

    /// Default `(a, b, f, k, d_u, d_v, dt, peclet)` tuple for `self`.
    #[must_use]
    pub const fn default_parameters(self) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        match self {
            Self::SchnakenbergSpots => (0.10, 0.90, 0.0367, 0.0649, 0.15, 3.50, 0.035, 12.0),
            Self::SchnakenbergStripes => (0.08, 1.15, 0.0367, 0.0649, 0.15, 3.00, 0.035, 12.0),
            Self::GrayScottMitosis => (0.10, 0.90, 0.0367, 0.0649, 0.16, 0.08, 0.50, 4.0),
            Self::GrayScottCoral => (0.10, 0.90, 0.0545, 0.0620, 0.16, 0.08, 0.50, 4.5),
        }
    }
}

/// Scalar field visualization mode for the `100 x 50` coupled wind-tunnel canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdvectiveViewMode {
    /// Activator / pattern concentration overlaid with wake vorticity contours.
    #[default]
    MorphogenWithVorticity,
    /// Pure morphogen concentration field (`u` for Schnakenberg, `v` for Gray-Scott).
    MorphogenOnly,
    /// Signed fluid curl $\omega_z = \partial_x u_y - \partial_y u_x$.
    VorticityOnly,
}

impl AdvectiveViewMode {
    /// Ordered list of view modes.
    pub const ALL: [AdvectiveViewMode; 3] = [
        Self::MorphogenWithVorticity,
        Self::MorphogenOnly,
        Self::VorticityOnly,
    ];

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::MorphogenWithVorticity => "Morphogen + Vorticity",
            Self::MorphogenOnly => "Morphogen Field",
            Self::VorticityOnly => "Wake Vorticity (ω_z)",
        }
    }
}

/// Compact `< 4 KiB` URL-safe and preset-safe state for [`super::AdvectiveMorphogenesisTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvectiveToolState {
    /// Active reaction kinetics family and preset.
    pub preset: AdvectiveKineticsPreset,
    /// Kinematic fluid viscosity $\nu$ (`0.005..=0.25`).
    pub viscosity: f64,
    /// Dimensionless Péclet advection gain (`0.0..=30.0`).
    pub peclet: f64,
    /// Requested reaction-diffusion step size `dt` (`0.001..=1.0`), auto-clamped by CFL.
    pub dt: f64,
    /// Schnakenberg feed rate `a` (`0.01..=0.60`).
    pub a: f64,
    /// Schnakenberg feed rate `b` (`0.10..=2.00`).
    pub b: f64,
    /// Gray-Scott feed rate `F` (`0.01..=0.10`).
    pub f: f64,
    /// Gray-Scott kill rate `k` (`0.03..=0.08`).
    pub k: f64,
    /// Species-0 diffusivity `D_u` (`0.02..=2.00`).
    pub d_u: f64,
    /// Species-1 diffusivity `D_v` (`0.02..=12.00`).
    pub d_v: f64,
    /// Selected built-in fairing stamp.
    pub stamp: FairingStamp,
    /// Optional `1,250`-char hex-encoded `100 x 50` obstacle mask.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_hex: Option<String>,
    /// Active scalar view mode.
    #[serde(default)]
    pub view_mode: AdvectiveViewMode,
    /// Brush disk radius in lattice cells (`0.0..=8.0`).
    #[serde(default = "default_brush_radius")]
    pub brush_radius: f64,
    /// Active sculpting brush mode (`Solid` or `Erase`).
    #[serde(default = "default_brush_mode")]
    pub brush_mode: BrushMode,
    /// Deterministic PRNG seed for chemical nucleation.
    #[serde(default = "default_rng_seed")]
    pub rng_seed: u64,
    /// Whether continuous co-stepping is running.
    #[serde(default)]
    pub playing: bool,
}

impl Default for AdvectiveToolState {
    fn default() -> Self {
        let (a, b, f, k, d_u, d_v, dt, peclet) =
            AdvectiveKineticsPreset::SchnakenbergSpots.default_parameters();
        Self {
            preset: AdvectiveKineticsPreset::SchnakenbergSpots,
            viscosity: 0.020,
            peclet,
            dt,
            a,
            b,
            f,
            k,
            d_u,
            d_v,
            stamp: FairingStamp::BluffCylinder,
            mask_hex: None,
            view_mode: AdvectiveViewMode::MorphogenWithVorticity,
            brush_radius: default_brush_radius(),
            brush_mode: default_brush_mode(),
            rng_seed: default_rng_seed(),
            playing: false,
        }
    }
}

impl AdvectiveToolState {
    /// Validates all scalar bounds and optional `1,250`-char `mask_hex` under `URL_JSON_BOUNDS`.
    pub fn validate(&self) -> Result<(), String> {
        let bounds = [
            ("viscosity", self.viscosity, 0.018, 0.25),
            ("peclet", self.peclet, 0.0, 30.0),
            ("dt", self.dt, 0.001, 1.0),
            ("a", self.a, 0.01, 0.60),
            ("b", self.b, 0.10, 2.00),
            ("f", self.f, 0.01, 0.10),
            ("k", self.k, 0.03, 0.08),
            ("d_u", self.d_u, 0.02, 2.00),
            ("d_v", self.d_v, 0.02, 12.00),
            ("brush_radius", self.brush_radius, 0.0, 8.0),
        ];
        for (name, val, lo, hi) in bounds {
            if !val.is_finite() || val < lo || val > hi {
                return Err(format!(
                    "Advective morphogenesis '{name}' = {val} must be finite within [{lo}, {hi}]"
                ));
            }
        }
        if !matches!(self.brush_mode, BrushMode::Solid | BrushMode::Erase) {
            return Err("Advective morphogenesis brush_mode must be Solid or Erase".to_string());
        }
        if let Some(hex) = &self.mask_hex {
            let mut tmp = [false; LBM_CELL_COUNT];
            decode_mask_hex(hex, &mut tmp)?;
        }
        Ok(())
    }
}

/// [`TheoryDescribable`] metadata for coupled D2Q9 Lattice Boltzmann + reaction-diffusion transport.
#[derive(Clone, Debug, Default)]
pub struct AdvectiveMorphogenesisTheory {
    schnak: SchnakenbergKinetics,
    gs: GrayScottKinetics,
}

impl TheoryDescribable for AdvectiveMorphogenesisTheory {
    fn theory_description(&self) -> String {
        "Co-steps a 2D D2Q9 BGK Lattice Boltzmann wind tunnel with Schnakenberg and Gray-Scott reaction-diffusion PDEs under conservative first-order upwind advection and zero-flux bounce-back obstacles."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        format!(
            "{} Coupled with D2Q9 fluid wake advection scaled by Peclet gain Pe.",
            self.schnak.phonetic_description()
        )
    }

    fn theory_citation(&self) -> String {
        format!(
            "{}; {}",
            self.schnak.theory_citation(),
            self.gs.theory_citation()
        )
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = self.schnak.available_descriptions();
        map.extend(self.gs.available_descriptions());
        map.insert(
            "peclet_advection".to_string(),
            "Conservative 2D upwind flux divergence -Pe * div(u c) with exact zero normal flux across solid obstacles."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\partial_t \mathbf{c} + \operatorname{Pe}\,\nabla\cdot(\mathbf{u}\,\mathbf{c}) = \mathbf{D}\nabla^2\mathbf{c} + \mathbf{R}(\mathbf{c}), \quad \mathbf{n}\cdot\nabla\mathbf{c}\big|_{\partial\Omega_s}=0"
                .to_string(),
        )
    }
}

/// Starter preset JSON for vortex-sheared Schnakenberg spots behind a bluff cylinder.
pub const PRESET_ADVECTIVE_VORTEX_SPOTS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Advective Vortex-Sheared Turing Spots",
    "domain": "biology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "SchnakenbergSpots",
    "viscosity": 0.022,
    "peclet": 14.0,
    "dt": 0.035,
    "a": 0.1,
    "b": 0.9,
    "f": 0.0367,
    "k": 0.0649,
    "d_u": 0.15,
    "d_v": 3.5,
    "stamp": "BluffCylinder",
    "view_mode": "MorphogenWithVorticity",
    "brush_radius": 2.0,
    "brush_mode": "Solid",
    "rng_seed": 42,
    "playing": true
  }
}"#;

/// Starter preset JSON for Gray-Scott coral plumes steered by a teardrop fairing wake.
pub const PRESET_ADVECTIVE_CORAL_PLUME: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gray-Scott Coral Wake Plume",
    "domain": "biology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "preset": "GrayScottCoral",
    "viscosity": 0.02,
    "peclet": 5.0,
    "dt": 0.5,
    "a": 0.1,
    "b": 0.9,
    "f": 0.0545,
    "k": 0.062,
    "d_u": 0.16,
    "d_v": 0.08,
    "stamp": "TeardropFairing",
    "view_mode": "MorphogenWithVorticity",
    "brush_radius": 2.0,
    "brush_mode": "Solid",
    "rng_seed": 42,
    "playing": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "advective_vortex_spots",
        title: "Advective Vortex-Sheared Spots",
        description: "∂_t c + Pe∇·(uc) = D∇²c + R(c): Unsteady D2Q9 vortex street stretching Turing spots into shear-aligned wake plumes.",
        category: "Reaction-Advection",
        icon: "🌪",
        domain: "biology",
        tool_name: Some("AdvectiveMorphogenesisTool"),
        preset_generator: || PRESET_ADVECTIVE_VORTEX_SPOTS,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "advective_coral_plume",
        title: "Gray-Scott Coral Wake Plume",
        description: "Streamlined teardrop fairing steering dendritic Gray-Scott coral plumes across zero-flux bounce-back boundaries.",
        category: "Reaction-Advection",
        icon: "🪸",
        domain: "biology",
        tool_name: Some("AdvectiveMorphogenesisTool"),
        preset_generator: || PRESET_ADVECTIVE_CORAL_PLUME,
    }
}
