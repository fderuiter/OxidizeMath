//! Spacetime geometry, worldline causality, gesture, viewport, and lensing source enums
//! for [`super::GeodesicPlaygroundTool`].

use math_explorer::physics::astrophysics::{
    DwarfMorphology, GalacticDiskSystem, HaloProfile, Mpc, OrbitPopulationConfig,
};
use serde::{Deserialize, Serialize};

pub(super) const fn default_fan_count() -> usize {
    9
}

pub(super) const fn default_fan_spread() -> f64 {
    2.4
}

pub(super) fn is_zero_f64(v: &f64) -> bool {
    v.abs() <= 1e-12
}

pub(super) fn build_default_lensing_galaxy() -> GalacticDiskSystem {
    GalacticDiskSystem::from_morphology(
        DwarfMorphology::TypeCode10,
        Mpc(12.0),
        0.60,
        0.85,
        HaloProfile::BurkertCore {
            virial_mass_solar: 1.2e10,
            core_radius_kpc: 2.2,
            concentration: 12.0,
        },
        OrbitPopulationConfig {
            disk_tracers: 14,
            tidal_tracers: 4,
            trail_capacity: 16,
            tidal_tangential_ratio: 0.65,
        },
    )
}

#[cfg(feature = "biology")]
pub(super) fn build_default_lenia_engine() -> math_explorer::biology::lenia::LeniaEngine {
    let mut rng = oxidize_core::rng::OxidizeRng::from_seed(0x51A7_E5E1);
    math_explorer::biology::lenia::LeniaEngine::from_species(
        64,
        64,
        math_explorer::biology::lenia::LeniaSpecies::Orbium,
        0.0,
        &mut rng,
    )
}

/// Curved spacetime background selectable in [`super::GeodesicPlaygroundTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpacetimePreset {
    /// Static Schwarzschild black hole ($R_s = 2 M_{\text{geom}}$) with Flamm paraboloid embedding.
    #[default]
    Schwarzschild,
    /// Rotating equatorial Kerr black hole with spin parameter $a$ and frame dragging.
    KerrEquatorial,
    /// Horizon-free traversable Morris-Thorne wormhole with throat radius $b_0$.
    MorrisThorneWormhole,
    /// User-authored static spherically symmetric lapse $f(r)$ via `StaticRiemannianMetric`.
    CustomRadialLapse,
    /// User-authored traversable wormhole throat profile $b(l)$ via `StaticRiemannianMetric`.
    CustomWormholeProfile,
}

impl SpacetimePreset {
    /// Ordered list of all selectable spacetime geometries.
    pub const ALL: [Self; 5] = [
        Self::Schwarzschild,
        Self::KerrEquatorial,
        Self::MorrisThorneWormhole,
        Self::CustomRadialLapse,
        Self::CustomWormholeProfile,
    ];

    /// Display label for combo boxes and telemetry readouts.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Schwarzschild => "Schwarzschild Black Hole",
            Self::KerrEquatorial => "Kerr Equatorial (Rotating)",
            Self::MorrisThorneWormhole => "Morris-Thorne Wormhole",
            Self::CustomRadialLapse => "Custom Radial Lapse f(r)",
            Self::CustomWormholeProfile => "Custom Wormhole Profile b(l)",
        }
    }

    /// Returns `true` when the spacetime uses proper wormhole coordinate $l \in (-\infty, \infty)$.
    pub const fn is_wormhole(self) -> bool {
        matches!(
            self,
            Self::MorrisThorneWormhole | Self::CustomWormholeProfile
        )
    }
}

/// Causal worldline character for geodesic integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GeodesicMode {
    /// Massless photon ray ($g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = 0$, normalized $e = 1$).
    #[default]
    NullPhoton,
    /// Massive test particle ($g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = -1$) with proper clock $\tau$.
    TimelikeParticle,
}

impl GeodesicMode {
    /// Human-readable worldline label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::NullPhoton => "Null Photon (Light Ray)",
            Self::TimelikeParticle => "Timelike Particle (Massive)",
        }
    }
}

/// Pointer and keyboard gesture mode on the geodesic canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GeodesicInteractionMode {
    /// Rotate and zoom `Camera3D` freely.
    #[default]
    Orbit,
    /// Aim and sling a single timelike or null probe with `Camera3D` rotation locked.
    Slingshot,
    /// Aim and fan up to 16 simultaneous null/timelike geodesics across critical impact parameters.
    PhotonFan,
}

impl GeodesicInteractionMode {
    /// Short toolbar label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Orbit => "Orbit",
            Self::Slingshot => "Slingshot",
            Self::PhotonFan => "Photon Fan",
        }
    }
}

/// Central viewport visualization mode for [`super::GeodesicPlaygroundTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ViewportViewMode {
    /// Isometric 3D Flamm paraboloid or two-sheet Morris-Thorne embedding.
    #[default]
    Embedding3D,
    /// 2D deflection-table gravitational lensing camera.
    LensingCamera2D,
}

impl ViewportViewMode {
    /// Short selector label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Embedding3D => "3D Embedding",
            Self::LensingCamera2D => "2D Lensing Camera",
        }
    }
}

/// Cross-domain background warped by the 2D gravitational lensing camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LensingBackgroundKind {
    /// Symplectic spiral dwarf galaxy (`GalacticDiskSystem`).
    #[default]
    GalacticDisk,
    /// Continuous Lenia soliton portal (`LeniaEngine` across $l < 0$ when `biology` is enabled).
    LeniaPortal,
    /// High-contrast procedural deep-field stargrid.
    Starfield,
}

impl LensingBackgroundKind {
    /// Human-readable background label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::GalacticDisk => "Spiral Dwarf Galaxy",
            Self::LeniaPortal => "Lenia Soliton Portal",
            Self::Starfield => "Deep-Field Stargrid",
        }
    }
}

// [cite:general_relativity_geodesics]
