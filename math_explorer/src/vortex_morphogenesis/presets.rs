//! Kinetics variants, regime presets, and dimensionless numbers for coupled vortex morphogenesis.

use domain_biology::biology::morphogenesis::{
    GrayScottKinetics, ReactionKinetics, SchnakenbergKinetics,
};

/// Maximum permitted lattice velocity magnitude ($|\mathbf{u}| \le 0.15$) to keep low-Mach D2Q9 BGK stable.
pub const VELOCITY_SAFETY_CAP: f64 = 0.15;

/// Maximum explicit advective Courant number per morphogen sub-step ($|u_x|\Delta t/\Delta x + |u_y|\Delta t/\Delta y \le 0.5$).
pub const MAX_ADVECTIVE_CFL: f64 = 0.5;

/// Maximum explicit 2D diffusive Fourier number per morphogen sub-step ($2 D_{\max} \Delta t (\Delta x^{-2} + \Delta y^{-2}) \le 0.45$).
pub const MAX_DIFFUSIVE_CFL: f64 = 0.45;

/// Fixed upper bound on explicit transport sub-steps per coupled frame (NASA Power of 10 Rule 2).
pub const MAX_TRANSPORT_SUBSTEPS: usize = 64;

/// Selectable two-species reaction kinetics for coupled vortex morphogenesis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MorphogenKinetics {
    /// Gray-Scott cubic autocatalysis ($U + 2V \to 3V$).
    GrayScott(GrayScottKinetics),
    /// Schnakenberg activator-inhibitor Turing kinetics.
    Schnakenberg(SchnakenbergKinetics),
}

impl Default for MorphogenKinetics {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::GrayScott(GrayScottKinetics::new(0.038, 0.061))
    }
}

impl ReactionKinetics<2> for MorphogenKinetics {
    #[inline(always)]
    #[verified_engine::verified]
    fn reaction(&self, concentrations: [f64; 2]) -> [f64; 2] {
        match self {
            Self::GrayScott(gs) => gs.reaction(concentrations),
            Self::Schnakenberg(sn) => sn.reaction(concentrations),
        }
    }
}

impl MorphogenKinetics {
    /// Homogeneous feed/equilibrium concentrations `(u_0, v_0)` used at fresh channel inlets.
    #[verified_engine::verified]
    pub fn homogeneous_feed(&self) -> (f64, f64) {
        match self {
            Self::GrayScott(_) => (1.0, 0.0),
            Self::Schnakenberg(sn) => {
                let u0 = (sn.a + sn.b).max(1e-3);
                let v0 = sn.b / (u0 * u0);
                (u0, v0)
            }
        }
    }

    /// Characteristic feed-limited autocatalytic reaction rate $k_{\text{react}}$ for Damkohler scaling.
    #[verified_engine::verified]
    pub fn characteristic_rate(&self) -> f64 {
        match self {
            Self::GrayScott(gs) => gs.f.max(1e-6),
            Self::Schnakenberg(sn) => (sn.a + sn.b).max(1e-6),
        }
    }
}

/// Canonical physical-biological regime presets for fluid-coupled morphogenesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexRegimePreset {
    /// Unsteady cylinder wake shedding Karman eddies that trap autocatalytic spots.
    KarmanVortexSpots,
    /// Channel velocity shear stretching Turing instabilities into longitudinal stripes.
    ShearAlignedStripes,
    /// Bluff-body stagnation zone pinning a stationary autocatalytic chemical plume.
    StagnationPlumes,
    /// High-velocity low-Damkohler flushing sweeping autocatalysts past washout bifurcation.
    DamkohlerWashout,
}

impl VortexRegimePreset {
    /// Ordered list of all four coupled flow-morphogenesis presets.
    pub const ALL: [Self; 4] = [
        Self::KarmanVortexSpots,
        Self::ShearAlignedStripes,
        Self::StagnationPlumes,
        Self::DamkohlerWashout,
    ];

    /// Human-readable display title for UI buttons and telemetry.
    #[verified_engine::verified]
    pub const fn title(self) -> &'static str {
        match self {
            Self::KarmanVortexSpots => "Karman Vortex Spots",
            Self::ShearAlignedStripes => "Shear-Aligned Stripes",
            Self::StagnationPlumes => "Stagnation Plumes",
            Self::DamkohlerWashout => "Damkohler Washout",
        }
    }

    /// Concise physical description of the emergent coupled regime.
    #[verified_engine::verified]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::KarmanVortexSpots => {
                "Alternating Karman eddies behind a cylinder trap slow-diffusing Gray-Scott spots."
            }
            Self::ShearAlignedStripes => {
                "Streamwise Couette shear damps x-gradients into flow-parallel Turing stripes."
            }
            Self::StagnationPlumes => {
                "Low-velocity bluff-body stagnation zones pin stationary autocatalytic plumes."
            }
            Self::DamkohlerWashout => {
                "High-velocity flushing (Da < 1) sweeps autocatalyst out of the channel into washout."
            }
        }
    }
}

/// Parameters governing coupled D2Q9 LBM fluid flow and 2D reaction-advection-diffusion transport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VortexMorphogenesisConfig {
    /// Grid width in lattice cells (`100` by default).
    pub width: usize,
    /// Grid height in lattice cells (`50` by default).
    pub height: usize,
    /// Spatial step size $\Delta x = \Delta y > 0$ for morphogen reaction-diffusion-advection.
    pub dx: f64,
    /// LBM BGK relaxation time $\tau > 0.5$ ($\nu = (\tau - 0.5)/3$).
    pub tau: f64,
    /// Horizontal inlet driving speed ($|u_{\text{in}}| \le 0.15$).
    pub inlet_ux: f64,
    /// Dimensionless velocity coupling multiplier mapping lattice speed to morphogen advection ($\le 1.0$ typical).
    pub advection_coupling: f64,
    /// Diffusion coefficient $D_u$ of species $u$.
    pub d_u: f64,
    /// Diffusion coefficient $D_v$ of species $v$.
    pub d_v: f64,
    /// Macroscopic reaction-diffusion step size $\Delta t$ per coupled tick (auto-substepped for CFL).
    pub dt: f64,
    /// Number of LBM fluid steps executed per coupled tick (`>= 1`).
    pub lbm_steps_per_tick: usize,
    /// Active two-species reaction kinetics.
    pub kinetics: MorphogenKinetics,
    /// Whether column `x = 0` supplies fresh homogeneous feed where inflow enters (`ux > 0`).
    pub fresh_inlet_feed: bool,
    /// Whether the channel uses a streamwise-periodic Couette shear profile.
    pub shear_profile: bool,
    /// Continuous source injection rate on painted chemical emitter cells (`[0.0, 1.0]`).
    pub emitter_strength: f64,
}

impl Default for VortexMorphogenesisConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::for_preset(VortexRegimePreset::KarmanVortexSpots)
    }
}

impl VortexMorphogenesisConfig {
    /// Returns the calibrated configuration for one of the four canonical coupled regimes.
    #[verified_engine::verified]
    pub fn for_preset(preset: VortexRegimePreset) -> Self {
        match preset {
            VortexRegimePreset::KarmanVortexSpots => Self {
                width: 100,
                height: 50,
                dx: 1.0,
                tau: 0.56,
                inlet_ux: 0.10,
                advection_coupling: 0.45,
                d_u: 0.16,
                d_v: 0.06,
                dt: 1.2,
                lbm_steps_per_tick: 1,
                kinetics: MorphogenKinetics::GrayScott(GrayScottKinetics::new(0.038, 0.060)),
                fresh_inlet_feed: true,
                shear_profile: false,
                emitter_strength: 0.35,
            },
            VortexRegimePreset::ShearAlignedStripes => Self {
                width: 100,
                height: 50,
                dx: 1.0,
                tau: 0.62,
                inlet_ux: 0.15,
                advection_coupling: 1.0,
                d_u: 0.05,
                d_v: 1.10,
                dt: 3.0,
                lbm_steps_per_tick: 1,
                kinetics: MorphogenKinetics::Schnakenberg(SchnakenbergKinetics::new(0.10, 0.90)),
                fresh_inlet_feed: false,
                shear_profile: true,
                emitter_strength: 0.25,
            },
            VortexRegimePreset::StagnationPlumes => Self {
                width: 100,
                height: 50,
                dx: 1.0,
                tau: 0.70,
                inlet_ux: 0.09,
                advection_coupling: 0.75,
                d_u: 0.16,
                d_v: 0.06,
                dt: 1.1,
                lbm_steps_per_tick: 1,
                kinetics: MorphogenKinetics::GrayScott(GrayScottKinetics::new(0.036, 0.059)),
                fresh_inlet_feed: true,
                shear_profile: false,
                emitter_strength: 0.45,
            },
            VortexRegimePreset::DamkohlerWashout => Self {
                width: 100,
                height: 50,
                dx: 0.35,
                tau: 0.60,
                inlet_ux: 0.145,
                advection_coupling: 1.0,
                d_u: 0.04,
                d_v: 0.02,
                dt: 2.6,
                lbm_steps_per_tick: 1,
                kinetics: MorphogenKinetics::GrayScott(GrayScottKinetics::new(0.026, 0.060)),
                fresh_inlet_feed: true,
                shear_profile: false,
                emitter_strength: 0.0,
            },
        }
    }
}

/// Live dimensionless transport metrics computed from the coupled fluid and morphogen fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimensionlessRegimeMetrics {
    /// Peclet number $\mathrm{Pe} = U_{\text{eff}} L_{\text{char}} / D_{\text{slow}}$.
    pub peclet: f64,
    /// Damkohler number $\mathrm{Da} = k_{\text{react}} L_{\text{char}} / U_{\text{eff}}$.
    pub damkohler: f64,
    /// Peak clamped advective speed $|\mathbf{u}|_{\max}$ on the grid ($\le 0.15$).
    pub max_speed: f64,
    /// Number of CFL-safe transport sub-steps used on the most recent tick.
    pub substeps: usize,
}

impl DimensionlessRegimeMetrics {
    /// Human-readable transport regime label for Peclet badges.
    #[verified_engine::verified]
    pub const fn peclet_label(&self) -> &'static str {
        if self.peclet < 1.0 {
            "Diffusion-Dominated"
        } else if self.peclet <= 18.0 {
            "Vortex-Trapping"
        } else {
            "Advection-Dominated"
        }
    }

    /// Human-readable reaction-vs-flow label for Damkohler badges.
    #[verified_engine::verified]
    pub const fn damkohler_label(&self) -> &'static str {
        if self.damkohler < 1.0 {
            "Washout"
        } else if self.damkohler <= 2.5 {
            "Critical Balance"
        } else {
            "Reaction-Sustained"
        }
    }
}
