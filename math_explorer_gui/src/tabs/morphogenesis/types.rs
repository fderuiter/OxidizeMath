//! Coupled reaction, hydrodynamic flow, brush, and overlay mode enums and dimensionless metrics.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use crate::presets::MorphogenesisKineticsMode;
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
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "None (Zero Drift)",
            Self::LbmVortexWake => "LBM Vortex Wake",
            Self::PotentialFlow => "Potential Flow",
        }
    }

    /// Returns `true` when this flow mode equals [`Self::None`].
    #[must_use]
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// Interactive pointer brush behavior for coupled obstacle and reagent sculpting.
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
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Obstacle => "Paint Obstacle",
            Self::EraseObstacle => "Erase Obstacle",
            Self::Reagent => "Seed Reagent",
            Self::PanZoom => "Pan / Zoom",
        }
    }

    /// Returns `true` when this brush mode equals [`Self::Obstacle`].
    #[must_use]
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Obstacle)
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
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Concentration => "Concentration",
            Self::Vorticity => "Vorticity (ω_z)",
            Self::DualOverlay => "Dual (Conc + ω_z)",
        }
    }

    /// Returns `true` when this overlay mode equals [`Self::Concentration`].
    #[must_use]
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Concentration)
    }
}

/// Shared reaction-diffusion parameters synchronized between the UI thread and worker runner.
#[derive(Clone, Copy, Debug)]
pub struct MorphogenesisSharedParams {
    /// Schnakenberg activator feed rate `a`.
    pub a: f64,
    /// Schnakenberg inhibitor feed rate `b`.
    pub b: f64,
    /// Primary species diffusion coefficient `D_u`.
    pub d_u: f64,
    /// Secondary species diffusion coefficient `D_v`.
    pub d_v: f64,
    /// Integration step size `dt`.
    pub dt: f64,
    /// Whether high-contrast palette rendering is active.
    pub high_contrast: bool,
}

impl Default for MorphogenesisSharedParams {
    fn default() -> Self {
        Self {
            a: 0.1,
            b: 0.9,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            high_contrast: false,
        }
    }
}

/// Shared hydrodynamic coupling, overlay, and reagent brush configuration synchronized with the worker.
#[derive(Clone, Copy, Debug)]
pub struct MorphogenesisFlowState {
    /// Active hydrodynamic velocity coupling mode.
    pub flow_mode: MorphogenesisFlowMode,
    /// Horizontal freestream / inlet velocity magnitude `U`.
    pub inlet_velocity: f64,
    /// Kinematic fluid viscosity `\nu` for D2Q9 BGK relaxation.
    pub viscosity: f64,
    /// Analytical vortex circulation `\Gamma` for potential flow.
    pub circulation: f64,
    /// Active visualization overlay mode.
    pub overlay_mode: MorphogenesisOverlayMode,
    /// Current theme mode for theme-reactive snapshot rendering.
    pub theme_mode: crate::app::ThemeMode,
    /// Coupled obstacle/reagent brush mode used by `SimCommand::ApplyBrush`.
    pub brush_mode: MorphogenesisBrushMode,
    /// Coupled brush radius in grid cells.
    pub brush_radius: i32,
}

impl Default for MorphogenesisFlowState {
    fn default() -> Self {
        Self {
            flow_mode: MorphogenesisFlowMode::None,
            inlet_velocity: 0.0,
            viscosity: default_viscosity(),
            circulation: 0.0,
            overlay_mode: MorphogenesisOverlayMode::Concentration,
            theme_mode: crate::app::ThemeMode::Dark,
            brush_mode: MorphogenesisBrushMode::Obstacle,
            brush_radius: default_brush_radius(),
        }
    }
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

pub(crate) fn is_zero_f64(val: &f64) -> bool {
    *val == 0.0
}

pub(crate) fn default_viscosity() -> f64 {
    0.02
}

pub(crate) fn is_default_viscosity(val: &f64) -> bool {
    (*val - default_viscosity()).abs() < 1e-12
}

pub(crate) fn default_brush_radius() -> i32 {
    3
}

pub(crate) fn is_default_brush_radius(val: &i32) -> bool {
    *val == default_brush_radius()
}
