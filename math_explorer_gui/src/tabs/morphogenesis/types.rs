#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use crate::presets::MorphogenesisKineticsMode;
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics,
};
use serde::{Deserialize, Serialize};

/// Selectable hydrodynamic velocity coupling mode for [`super::MorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphogenesisFlowMode {
    /// Pure reaction-diffusion with zero advective drift (`u_x = u_y = 0`).
    #[default]
    None,
    /// Mesoscopic D2Q9 Lattice Boltzmann BGK solver shedding vortex wakes past obstacles.
    LbmVortexWake,
    /// Superposed inviscid analytical `PotentialFlowField` (uniform stream + cylinder doublet + circulation vortex).
    PotentialFlow,
}

impl MorphogenesisFlowMode {
    /// Display label shown in UI selectors.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None (Zero Drift)",
            Self::LbmVortexWake => "LBM Vortex Wake",
            Self::PotentialFlow => "Potential Flow",
        }
    }
}

/// Interactive pointer brush behavior on the morphogenesis canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphogenesisBrushMode {
    /// Primary click/drag paints solid no-flux obstacles; secondary click/drag erases obstacles.
    #[default]
    Obstacle,
    /// Primary click/drag erases solid obstacles.
    EraseObstacle,
    /// Primary click/drag injects autocatalytic reagent perturbations; secondary resets to baseline.
    Reagent,
    /// Disables brush sculpting so pointer drag pans and zooms the plot viewport.
    PanZoom,
}

impl MorphogenesisBrushMode {
    /// Display label shown in UI selectors.
    pub fn label(self) -> &'static str {
        match self {
            Self::Obstacle => "Paint Obstacle",
            Self::EraseObstacle => "Erase Obstacle",
            Self::Reagent => "Seed Reagent",
            Self::PanZoom => "Pan / Zoom",
        }
    }
}

/// Scalar/vector field visualization mode for the 2D morphogenesis canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphogenesisOverlayMode {
    /// Normalized activator/autocatalyst concentration heatmap.
    #[default]
    Concentration,
    /// Signed hydrodynamic vorticity curl field `\omega_z = \partial_x u_y - \partial_y u_x`.
    Vorticity,
    /// Dual composite blending reagent concentration patterns with signed shear vorticity contours.
    DualOverlay,
}

impl MorphogenesisOverlayMode {
    /// Display label shown in UI selectors.
    pub fn label(self) -> &'static str {
        match self {
            Self::Concentration => "Concentration",
            Self::Vorticity => "Vorticity (ω_z)",
            Self::DualOverlay => "Dual (Conc + ω_z)",
        }
    }
}

/// Unified zero-allocation kinetics adapter switching cleanly between Schnakenberg `(a, b)`
/// and Gray-Scott `(f, k)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphogenesisKinetics {
    /// Active reaction model variant.
    pub mode: MorphogenesisKineticsMode,
    /// Schnakenberg activator production rate `a`.
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b`.
    pub b: f64,
    /// Gray-Scott substrate feed rate `f`.
    pub f: f64,
    /// Gray-Scott autocatalyst kill rate `k`.
    pub k: f64,
}

impl Default for MorphogenesisKinetics {
    fn default() -> Self {
        let gs = GrayScottKinetics::mitosis();
        Self {
            mode: MorphogenesisKineticsMode::Schnakenberg,
            a: 0.1,
            b: 0.9,
            f: gs.f,
            k: gs.k,
        }
    }
}

impl ReactionKinetics<2> for MorphogenesisKinetics {
    fn reaction(&self, concentrations: [f64; 2]) -> [f64; 2] {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                SchnakenbergKinetics::new(self.a, self.b).reaction(concentrations)
            }
            MorphogenesisKineticsMode::GrayScott => {
                GrayScottKinetics::new(self.f, self.k).reaction(concentrations)
            }
        }
    }
}

/// Structured worker snapshot payload enabling theme-reactive recoloring and dimensionless readouts.
#[derive(Clone, Debug)]
pub struct MorphogenesisSnapshotData {
    /// Visualized species concentration slice (`u` for Schnakenberg, `v` for Gray-Scott).
    pub concentration: Vec<f64>,
    /// Row-major discrete vorticity field `\omega_z`.
    pub vorticity: Vec<f64>,
    /// Row-major solid obstacle mask.
    pub obstacles: Vec<bool>,
    /// Dimensionless Péclet number `\mathrm{Pe} = U L / D_u`.
    pub peclet: f64,
    /// Dimensionless Damköhler number `\mathrm{Da}`.
    pub damkohler: f64,
}

/// Computes live dimensionless Péclet (`Pe = U L / D_u`) and Damköhler (`Da`) numbers.
#[must_use]
pub fn compute_peclet_damkohler(
    u_char: f64,
    length: f64,
    d_u: f64,
    rate_primary: f64,
    rate_secondary: f64,
    _mode: MorphogenesisKineticsMode,
) -> (f64, f64) {
    let safe_u = if u_char.is_finite() && u_char > 0.0 {
        u_char
    } else {
        0.0
    };
    let safe_l = if length.is_finite() && length > 0.0 {
        length
    } else {
        1.0
    };
    let safe_du = if d_u.is_finite() && d_u > 1e-9 {
        d_u
    } else {
        1e-9
    };
    let sum_rates = rate_primary + rate_secondary;
    let k_rxn = if sum_rates.is_finite() && sum_rates > 1e-6 {
        sum_rates
    } else {
        1e-6
    };
    let peclet = (safe_u * safe_l) / safe_du;
    let damkohler = if safe_u > 1e-6 {
        (k_rxn * safe_l) / safe_u
    } else {
        (k_rxn * safe_l * safe_l) / safe_du
    };
    (peclet, damkohler)
}

pub(crate) fn default_viscosity() -> f64 {
    0.02
}

pub(crate) fn default_brush_radius() -> i32 {
    3
}

/// Thread-safe shared configuration synced between UI controls and [`super::runner::MorphogenesisRunner`].
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub struct MorphogenesisSharedParams {
    pub a: f64,
    pub b: f64,
    pub f: f64,
    pub k: f64,
    pub d_u: f64,
    pub d_v: f64,
    pub dt: f64,
    pub high_contrast: bool,
    pub theme_mode: crate::app::ThemeMode,
    pub kinetics_mode: MorphogenesisKineticsMode,
    pub flow_mode: MorphogenesisFlowMode,
    pub inlet_velocity: f64,
    pub viscosity: f64,
    pub circulation: f64,
    pub overlay_mode: MorphogenesisOverlayMode,
    pub brush_mode: MorphogenesisBrushMode,
    pub brush_radius: i32,
}

impl Default for MorphogenesisSharedParams {
    fn default() -> Self {
        let gs = GrayScottKinetics::mitosis();
        Self {
            a: 0.1,
            b: 0.9,
            f: gs.f,
            k: gs.k,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            high_contrast: false,
            theme_mode: crate::app::ThemeMode::Dark,
            kinetics_mode: MorphogenesisKineticsMode::Schnakenberg,
            flow_mode: MorphogenesisFlowMode::None,
            inlet_velocity: 0.0,
            viscosity: 0.02,
            circulation: 0.0,
            overlay_mode: MorphogenesisOverlayMode::Concentration,
            brush_mode: MorphogenesisBrushMode::Obstacle,
            brush_radius: 3,
        }
    }
}
