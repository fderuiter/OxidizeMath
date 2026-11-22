#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Validated serialization schema and discrete UI selectors for [`super::VortexMorphogenesisTool`].

use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use math_explorer::vortex_morphogenesis::{
    MorphogenKinetics, VELOCITY_SAFETY_CAP, VortexMorphogenesisConfig, VortexRegimePreset,
};
use serde::{Deserialize, Serialize};

fn default_brush_radius() -> usize {
    3
}

/// Interactive canvas painting brush mode for [`super::VortexMorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VortexBrushMode {
    /// Paint solid bounce-back / zero-flux obstacles.
    #[default]
    Obstacle,
    /// Paint continuous autocatalytic morphogen emitters.
    Emitter,
    /// Erase obstacles and chemical emitters back to open fluid.
    Erase,
}

impl VortexBrushMode {
    /// Human-readable label for brush buttons and screen-reader announcements.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Obstacle => "Obstacle",
            Self::Emitter => "Emitter",
            Self::Erase => "Erase",
        }
    }
}

/// Field visualization mode for [`super::VortexMorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VortexDisplayMode {
    /// Composite overlay of fluid vorticity wake, morphogen catalyst `v`, obstacles, and emitters.
    #[default]
    CoupledWake,
    /// Morphogen species `v` (Gray-Scott autocatalyst or Schnakenberg inhibitor) heatmap.
    MorphogenV,
    /// Morphogen species `u` (Gray-Scott substrate or Schnakenberg activator) heatmap.
    MorphogenU,
    /// Fluid scalar curl $\omega_z = \partial_x u_y - \partial_y u_x$ wake visualization.
    VorticityCurl,
}

impl VortexDisplayMode {
    /// Ordered list of all display layer modes.
    pub const ALL: [Self; 4] = [
        Self::CoupledWake,
        Self::MorphogenV,
        Self::MorphogenU,
        Self::VorticityCurl,
    ];

    /// Display label for UI controls.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CoupledWake => "Coupled Wake + Morphogen",
            Self::MorphogenV => "Catalyst / Inhibitor (v)",
            Self::MorphogenU => "Substrate / Activator (u)",
            Self::VorticityCurl => "Fluid Vorticity (ω_z)",
        }
    }
}

/// Active reaction kinetics selector for [`super::VortexMorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VortexKineticsKind {
    /// Gray-Scott cubic autocatalysis ($U + 2V \to 3V$).
    #[default]
    GrayScott,
    /// Schnakenberg activator-inhibitor Turing kinetics.
    Schnakenberg,
}

/// Canonical preset tag persisted inside [`VortexToolState`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VortexPresetTag {
    /// Unsteady Karman vortex street trapping Gray-Scott spots.
    #[default]
    KarmanVortexSpots,
    /// Streamwise Couette shear stretching Schnakenberg stripes.
    ShearAlignedStripes,
    /// Bluff-body stagnation zone pinning autocatalytic plumes.
    StagnationPlumes,
    /// High-velocity low-Damkohler washout bifurcation.
    DamkohlerWashout,
    /// User-customized configuration.
    Custom,
}

impl VortexPresetTag {
    /// Maps a [`VortexRegimePreset`] to its serializable tag.
    pub const fn from_regime(preset: VortexRegimePreset) -> Self {
        match preset {
            VortexRegimePreset::KarmanVortexSpots => Self::KarmanVortexSpots,
            VortexRegimePreset::ShearAlignedStripes => Self::ShearAlignedStripes,
            VortexRegimePreset::StagnationPlumes => Self::StagnationPlumes,
            VortexRegimePreset::DamkohlerWashout => Self::DamkohlerWashout,
        }
    }

    /// Converts back to [`VortexRegimePreset`] when not [`Self::Custom`].
    pub const fn as_regime(self) -> Option<VortexRegimePreset> {
        match self {
            Self::KarmanVortexSpots => Some(VortexRegimePreset::KarmanVortexSpots),
            Self::ShearAlignedStripes => Some(VortexRegimePreset::ShearAlignedStripes),
            Self::StagnationPlumes => Some(VortexRegimePreset::StagnationPlumes),
            Self::DamkohlerWashout => Some(VortexRegimePreset::DamkohlerWashout),
            Self::Custom => None,
        }
    }
}

/// Validated JSON state schema for [`super::VortexMorphogenesisTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VortexToolState {
    /// Dimensionless velocity coupling multiplier (`0.0..=2.0`).
    pub advection_coupling: f64,
    /// Species `u` diffusion coefficient (`0.001..=5.0`).
    pub d_u: f64,
    /// Species `v` diffusion coefficient (`0.001..=200.0`).
    pub d_v: f64,
    /// Macroscopic time step `dt` (`0.01..=10.0`).
    pub dt: f64,
    /// Spatial step `dx` (`0.05..=5.0`).
    pub dx: f64,
    /// Chemical emitter source injection rate (`0.0..=1.0`).
    pub emitter_strength: f64,
    /// Whether inlet column supplies fresh homogeneous feed.
    pub fresh_inlet_feed: bool,
    /// Gray-Scott feed rate `f` (`0.0..=0.2`).
    pub gs_f: f64,
    /// Gray-Scott kill rate `k` (`0.0..=0.2`).
    pub gs_k: f64,
    /// Grid height in lattice cells (`8..=256`).
    pub height: usize,
    /// Horizontal inlet speed (`-0.15..=0.15`).
    pub inlet_ux: f64,
    /// Selected reaction kinetics family.
    pub kinetics: VortexKineticsKind,
    /// Whether live frame stepping is paused.
    #[serde(default)]
    pub paused: bool,
    /// Active regime preset tag.
    #[serde(default)]
    pub preset: VortexPresetTag,
    /// Whether streamwise Couette shear profile is active.
    pub shear_profile: bool,
    /// Schnakenberg activator source `a` (`0.0..=1.0`).
    pub sn_a: f64,
    /// Schnakenberg inhibitor source `b` (`0.0..=2.0`).
    pub sn_b: f64,
    /// LBM BGK relaxation time `tau` (`0.51..=2.0`).
    pub tau: f64,
    /// Grid width in lattice cells (`8..=256`).
    pub width: usize,
    /// Active interactive brush mode.
    #[serde(default)]
    pub brush_mode: VortexBrushMode,
    /// Brush disc radius in lattice cells (`1..=12`).
    #[serde(default = "default_brush_radius")]
    pub brush_radius: usize,
    /// Active field visualization layer.
    #[serde(default)]
    pub display_mode: VortexDisplayMode,
}

impl VortexToolState {
    /// Builds a [`VortexToolState`] corresponding to one of the four canonical presets.
    pub fn for_preset(preset: VortexRegimePreset) -> Self {
        let cfg = VortexMorphogenesisConfig::for_preset(preset);
        let (kinetics, gs_f, gs_k, sn_a, sn_b) = match cfg.kinetics {
            MorphogenKinetics::GrayScott(gs) => (VortexKineticsKind::GrayScott, gs.f, gs.k, 0.10, 0.90),
            MorphogenKinetics::Schnakenberg(sn) => {
                (VortexKineticsKind::Schnakenberg, 0.038, 0.060, sn.a, sn.b)
            }
        };
        Self {
            advection_coupling: cfg.advection_coupling,
            d_u: cfg.d_u,
            d_v: cfg.d_v,
            dt: cfg.dt,
            dx: cfg.dx,
            emitter_strength: cfg.emitter_strength,
            fresh_inlet_feed: cfg.fresh_inlet_feed,
            gs_f,
            gs_k,
            height: cfg.height,
            inlet_ux: cfg.inlet_ux,
            kinetics,
            paused: false,
            preset: VortexPresetTag::from_regime(preset),
            shear_profile: cfg.shear_profile,
            sn_a,
            sn_b,
            tau: cfg.tau,
            width: cfg.width,
            brush_mode: VortexBrushMode::Obstacle,
            brush_radius: 3,
            display_mode: VortexDisplayMode::CoupledWake,
        }
    }

    /// Validates finiteness and physical stability bounds for every serialized parameter.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("advection_coupling", self.advection_coupling, 0.0, 2.0),
            ("d_u", self.d_u, 0.001, 5.0),
            ("d_v", self.d_v, 0.001, 200.0),
            ("dt", self.dt, 0.01, 10.0),
            ("dx", self.dx, 0.05, 5.0),
            ("emitter_strength", self.emitter_strength, 0.0, 1.0),
            ("gs_f", self.gs_f, 0.0, 0.2),
            ("gs_k", self.gs_k, 0.0, 0.2),
            ("inlet_ux", self.inlet_ux, -VELOCITY_SAFETY_CAP, VELOCITY_SAFETY_CAP),
            ("sn_a", self.sn_a, 0.0, 1.0),
            ("sn_b", self.sn_b, 0.0, 2.0),
            ("tau", self.tau, 0.51, 2.0),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Vortex Morphogenesis parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(8..=256).contains(&self.width) || !(8..=256).contains(&self.height) {
            return Err(format!(
                "Vortex Morphogenesis grid {}x{} must be within 8..=256",
                self.width, self.height
            ));
        }
        if !(1..=12).contains(&self.brush_radius) {
            return Err(format!(
                "Brush radius {} must be within 1..=12",
                self.brush_radius
            ));
        }
        Ok(())
    }

    /// Converts this state into a [`VortexMorphogenesisConfig`].
    pub fn to_engine_config(&self) -> VortexMorphogenesisConfig {
        let kinetics = match self.kinetics {
            VortexKineticsKind::GrayScott => {
                MorphogenKinetics::GrayScott(GrayScottKinetics::new(self.gs_f, self.gs_k))
            }
            VortexKineticsKind::Schnakenberg => {
                MorphogenKinetics::Schnakenberg(SchnakenbergKinetics::new(self.sn_a, self.sn_b))
            }
        };
        VortexMorphogenesisConfig {
            width: self.width,
            height: self.height,
            dx: self.dx,
            tau: self.tau,
            inlet_ux: self.inlet_ux,
            advection_coupling: self.advection_coupling,
            d_u: self.d_u,
            d_v: self.d_v,
            dt: self.dt,
            lbm_steps_per_tick: 1,
            kinetics,
            fresh_inlet_feed: self.fresh_inlet_feed,
            shear_profile: self.shear_profile,
            emitter_strength: self.emitter_strength,
        }
    }
}
