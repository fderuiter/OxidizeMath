#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Spacetime geometry, worldline causality, gesture, viewport, and lensing source enums
//! for [`super::GeodesicPlaygroundTool`].

use serde::{Deserialize, Serialize};

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
