#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Configuration, manifold topology, PDE regime, and strict serde schema for [`super::RiemannianPdeStudio`].

use super::super::custom_surface::SurfaceType;
use super::super::surface_brush::{BrushPolarity, ViewportBrushMode};
use crate::camera::Camera3D;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Maximum vertex count allowed for interactive 3D manifold PDEs (`64 x 64` equivalent).
pub const MAX_MANIFOLD_VERTICES: usize = 4096;
/// Minimum interactive grid resolution per parameter axis.
pub const MIN_STUDIO_RES: usize = 8;
/// Maximum interactive grid resolution per parameter axis.
pub const MAX_STUDIO_RES: usize = 64;

/// 3D 2-manifold family discretized into a CSR cotangent mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ManifoldKind {
    /// Genus-0 sphere with welded north and south poles.
    #[default]
    Sphere,
    /// Genus-1 doubly periodic torus.
    Torus,
    /// Non-orientable figure-8 / classic Klein bottle immersion.
    KleinBottle,
    /// Genus-0 gyral-sulcal folded cortical brain shell with paired white and pial surfaces.
    FoldedCorticalShell,
    /// User-authored `CompiledExpr` heightfield `z = f(x, y)`.
    CustomHeightfield,
    /// User-authored `CompiledExpr` parametric surface `(x(u,v), y(u,v), z(u,v))`.
    CustomParametric,
}

impl ManifoldKind {
    /// All selectable manifold geometries.
    pub const ALL: [Self; 6] = [
        Self::Sphere,
        Self::FoldedCorticalShell,
        Self::Torus,
        Self::KleinBottle,
        Self::CustomHeightfield,
        Self::CustomParametric,
    ];

    /// Human-readable UI label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sphere => "Welded Sphere",
            Self::Torus => "Periodic Torus",
            Self::KleinBottle => "Klein Bottle",
            Self::FoldedCorticalShell => "Folded Cortical Shell",
            Self::CustomHeightfield => "Custom Heightfield z=f(x,y)",
            Self::CustomParametric => "Custom Parametric r(u,v)",
        }
    }

    /// Maps to [`SurfaceType`] when backed by [`super::super::custom_surface::CustomSurfaceSandbox`].
    #[must_use]
    pub const fn as_custom_surface_type(self) -> Option<SurfaceType> {
        match self {
            Self::CustomHeightfield => Some(SurfaceType::CustomHeightfield),
            Self::CustomParametric => Some(SurfaceType::CustomParametric),
            _ => None,
        }
    }
}

/// Mathematical PDE or morphometry regime evaluated over the cotangent manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PdeRegime {
    /// Intrinsic heat diffusion coupled with `HeatEquationSolver` and `laplace_beltrami` chart verification.
    #[default]
    HeatBeltrami,
    /// Geometric vertex smoothing via `MeanCurvatureFlow` (`dr/dt = H n`).
    MeanCurvatureSmoothing,
    /// Two-species `SchnakenbergKinetics` reaction-diffusion over `CotangentMeshDiffusion`.
    SchnakenbergTuring,
    /// Two-species `GrayScottKinetics` mitosis/coral reaction-diffusion over `CotangentMeshDiffusion`.
    GrayScottTuring,
    /// Excitable `HodgkinHuxleyNeuron` cortical waves scaled by FreeSurfer `cortical_thickness`.
    CorticalHodgkinHuxley,
    /// FreeSurfer cortical morphometry (`cortical_thickness`, `evolve_surface`, `bayesian_classification`) and group GLM (`t_statistic`).
    FreeSurferMorphometryGlm,
}

impl PdeRegime {
    /// Regimes available under the active Cargo feature set.
    #[must_use]
    pub fn available_regimes() -> &'static [Self] {
        #[cfg(all(feature = "biology", feature = "applied"))]
        {
            &[
                Self::HeatBeltrami,
                Self::MeanCurvatureSmoothing,
                Self::SchnakenbergTuring,
                Self::GrayScottTuring,
                Self::CorticalHodgkinHuxley,
                Self::FreeSurferMorphometryGlm,
            ]
        }
        #[cfg(all(not(feature = "biology"), feature = "applied"))]
        {
            &[
                Self::HeatBeltrami,
                Self::MeanCurvatureSmoothing,
                Self::FreeSurferMorphometryGlm,
            ]
        }
        #[cfg(all(feature = "biology", not(feature = "applied")))]
        {
            &[
                Self::HeatBeltrami,
                Self::MeanCurvatureSmoothing,
                Self::SchnakenbergTuring,
                Self::GrayScottTuring,
                Self::CorticalHodgkinHuxley,
            ]
        }
        #[cfg(all(not(feature = "biology"), not(feature = "applied")))]
        {
            &[Self::HeatBeltrami, Self::MeanCurvatureSmoothing]
        }
    }

    /// Coerces regimes whose domain crate feature is disabled back to `HeatBeltrami`.
    #[must_use]
    pub const fn coerce_to_enabled_features(self) -> Self {
        match self {
            Self::HeatBeltrami | Self::MeanCurvatureSmoothing => self,
            Self::SchnakenbergTuring | Self::GrayScottTuring | Self::CorticalHodgkinHuxley => {
                #[cfg(feature = "biology")]
                {
                    self
                }
                #[cfg(not(feature = "biology"))]
                {
                    Self::HeatBeltrami
                }
            }
            Self::FreeSurferMorphometryGlm => {
                #[cfg(feature = "applied")]
                {
                    self
                }
                #[cfg(not(feature = "applied"))]
                {
                    Self::HeatBeltrami
                }
            }
        }
    }

    /// Human-readable UI label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HeatBeltrami => "Heat & Laplace-Beltrami",
            Self::MeanCurvatureSmoothing => "Mean Curvature Flow",
            Self::SchnakenbergTuring => "Schnakenberg Turing",
            Self::GrayScottTuring => "Gray-Scott Mitosis",
            Self::CorticalHodgkinHuxley => "Cortical Hodgkin-Huxley",
            Self::FreeSurferMorphometryGlm => "FreeSurfer GLM & Thickness",
        }
    }
}

/// Scalar map displayed when `PdeRegime::FreeSurferMorphometryGlm` is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum FreeSurferOverlayKind {
    /// Vertex-wise symmetric white-to-pial `cortical_thickness` (mm).
    #[default]
    CorticalThickness,
    /// Mass-univariate group contrast `t_statistic` map.
    GlmTStatistic,
    /// Group contrast effect size `estimate_beta` contrast component.
    GlmBetaContrast,
    /// Synthetic T1 tissue class from `bayesian_classification`.
    BayesianTissueClass,
}

#[cfg(feature = "applied")]
impl FreeSurferOverlayKind {
    /// All selectable FreeSurfer overlays.
    pub const ALL: [Self; 4] = [
        Self::CorticalThickness,
        Self::GlmTStatistic,
        Self::GlmBetaContrast,
        Self::BayesianTissueClass,
    ];

    /// Human-readable UI label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::CorticalThickness => "Cortical Thickness (mm)",
            Self::GlmTStatistic => "Group GLM t-Statistic",
            Self::GlmBetaContrast => "Group GLM Beta Contrast",
            Self::BayesianTissueClass => "Bayesian Tissue Prior",
        }
    }
}

fn default_u_res() -> usize {
    24
}
fn default_v_res() -> usize {
    18
}
fn default_dt() -> f64 {
    0.005
}
fn default_heat_alpha() -> f64 {
    0.08
}
fn default_schnak_a() -> f64 {
    0.10
}
fn default_schnak_b() -> f64 {
    0.90
}
fn default_gs_f() -> f64 {
    0.0367
}
fn default_gs_k() -> f64 {
    0.0649
}
fn default_d_u() -> f64 {
    0.004
}
fn default_d_v() -> f64 {
    0.08
}
fn default_hh_cond() -> f64 {
    0.035
}
fn default_hh_stim() -> f64 {
    35.0
}
fn default_sulcal_depth() -> f64 {
    0.22
}
fn default_fs_grad() -> f64 {
    0.35
}
fn default_brush_radius() -> f64 {
    0.28
}
fn default_brush_strength() -> f64 {
    0.65
}

/// Persistable parameter state for `RiemannianPdeStudio` (`< 4 KiB` URL-safe JSON).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiemannianStudioConfig {
    /// Selected 3D manifold topology.
    #[serde(default)]
    pub manifold: ManifoldKind,
    /// Active PDE or morphometry regime.
    #[serde(default)]
    pub regime: PdeRegime,
    /// Active FreeSurfer morphometry / GLM scalar overlay.
    #[serde(default)]
    pub overlay: FreeSurferOverlayKind,
    /// Longitudinal / primary parameter resolution (`8..=64`).
    #[serde(default = "default_u_res")]
    pub u_res: usize,
    /// Latitudinal / secondary parameter resolution (`8..=64`).
    #[serde(default = "default_v_res")]
    pub v_res: usize,
    /// Explicit integration step size `dt`.
    #[serde(default = "default_dt")]
    pub dt: f64,
    /// Thermal diffusivity `alpha` for `HeatBeltrami`.
    #[serde(default = "default_heat_alpha")]
    pub heat_alpha: f64,
    /// Schnakenberg source rate `a`.
    #[serde(default = "default_schnak_a")]
    pub schnak_a: f64,
    /// Schnakenberg source rate `b`.
    #[serde(default = "default_schnak_b")]
    pub schnak_b: f64,
    /// Gray-Scott feed rate `f`.
    #[serde(default = "default_gs_f")]
    pub gs_f: f64,
    /// Gray-Scott kill rate `k`.
    #[serde(default = "default_gs_k")]
    pub gs_k: f64,
    /// Activator surface diffusivity `D_u`.
    #[serde(default = "default_d_u")]
    pub d_u: f64,
    /// Inhibitor surface diffusivity `D_v`.
    #[serde(default = "default_d_v")]
    pub d_v: f64,
    /// Cortical lateral sheet coupling conductance.
    #[serde(default = "default_hh_cond")]
    pub hh_conductance: f64,
    /// Depolarizing brush current amplitude ($\mu\text{A}/\text{cm}^2$).
    #[serde(default = "default_hh_stim")]
    pub hh_stimulus: f64,
    /// Gyral-sulcal folding depth for `FoldedCorticalShell` (`0.0..=0.45`).
    #[serde(default = "default_sulcal_depth")]
    pub sulcal_depth: f64,
    /// FreeSurfer deformable surface image-gradient strength.
    #[serde(default = "default_fs_grad")]
    pub fs_grad_strength: f64,
    /// Viewport gesture mode (`Orbit` vs `Sculpt`).
    #[serde(default)]
    pub brush_mode: ViewportBrushMode,
    /// Brush polarity (`Excite` vs `Inhibit`).
    #[serde(default)]
    pub brush_polarity: BrushPolarity,
    /// Geodesic brush kernel sigma (`0.05..=1.20`).
    #[serde(default = "default_brush_radius")]
    pub brush_radius: f64,
    /// Geodesic brush amplitude (`0.05..=2.0`).
    #[serde(default = "default_brush_strength")]
    pub brush_strength: f64,
    /// Whether continuous playback is active.
    #[serde(default)]
    pub running: bool,
    /// 3D viewport camera pose.
    #[serde(default)]
    pub camera: Camera3D,
    /// Non-mute acoustic gain and pitch scale settings persisted across presets and share URLs.
    #[serde(
        default,
        skip_serializing_if = "crate::audio::AcousticSettings::is_default"
    )]
    pub audio: crate::audio::AcousticSettings,
    /// Optional custom heightfield formula `z(x, y)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_expr: Option<String>,
    /// Optional custom parametric triple `[x(u,v), y(u,v), z(u,v)]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parametric_exprs: Option<[String; 3]>,
    /// Synthesized expression parameter slider values.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom_params: BTreeMap<String, f64>,
    /// Optional custom `u` domain bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u_bounds: Option<[f64; 2]>,
    /// Optional custom `v` domain bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v_bounds: Option<[f64; 2]>,
}

impl Default for RiemannianStudioConfig {
    fn default() -> Self {
        Self {
            manifold: ManifoldKind::Sphere,
            regime: PdeRegime::HeatBeltrami,
            overlay: FreeSurferOverlayKind::CorticalThickness,
            u_res: default_u_res(),
            v_res: default_v_res(),
            dt: default_dt(),
            heat_alpha: default_heat_alpha(),
            schnak_a: default_schnak_a(),
            schnak_b: default_schnak_b(),
            gs_f: default_gs_f(),
            gs_k: default_gs_k(),
            d_u: default_d_u(),
            d_v: default_d_v(),
            hh_conductance: default_hh_cond(),
            hh_stimulus: default_hh_stim(),
            sulcal_depth: default_sulcal_depth(),
            fs_grad_strength: default_fs_grad(),
            brush_mode: ViewportBrushMode::Orbit,
            brush_polarity: BrushPolarity::Excite,
            brush_radius: default_brush_radius(),
            brush_strength: default_brush_strength(),
            running: false,
            camera: Camera3D::default(),
            audio: crate::audio::AcousticSettings::default(),
            custom_expr: None,
            parametric_exprs: None,
            custom_params: BTreeMap::new(),
            u_bounds: None,
            v_bounds: None,
        }
    }
}

impl RiemannianStudioConfig {
    /// Validates numeric finiteness, parameter bounds, and the `<= 4,096` vertex cap.
    pub fn validate(&self) -> Result<(), String> {
        if !(MIN_STUDIO_RES..=MAX_STUDIO_RES).contains(&self.u_res)
            || !(MIN_STUDIO_RES..=MAX_STUDIO_RES).contains(&self.v_res)
        {
            return Err("Mesh resolution must stay within [8, 64]".to_string());
        }
        if self.estimated_vertices() > MAX_MANIFOLD_VERTICES {
            return Err("Mesh exceeds 4,096 vertex budget".to_string());
        }
        let scalars = [
            self.dt,
            self.heat_alpha,
            self.schnak_a,
            self.schnak_b,
            self.gs_f,
            self.gs_k,
            self.d_u,
            self.d_v,
            self.hh_conductance,
            self.hh_stimulus,
            self.sulcal_depth,
            self.fs_grad_strength,
            self.brush_radius,
            self.brush_strength,
        ];
        if !scalars.iter().all(|v| v.is_finite()) {
            return Err("All PDE and morphometry parameters must be finite".to_string());
        }
        if !(1e-5..=0.1).contains(&self.dt) {
            return Err("Integration step dt must stay within [1e-5, 0.1]".to_string());
        }
        if !self.camera.is_valid() {
            return Err("Camera3D parameters are out of valid bounds".to_string());
        }
        self.audio.validate()?;
        Ok(())
    }

    /// Upper bound on generated vertex count for `(u_res, v_res)`.
    #[must_use]
    pub const fn estimated_vertices(&self) -> usize {
        match self.manifold {
            ManifoldKind::Sphere | ManifoldKind::FoldedCorticalShell => {
                2 + self.v_res.saturating_sub(1) * self.u_res
            }
            ManifoldKind::Torus | ManifoldKind::KleinBottle => self.u_res * self.v_res,
            ManifoldKind::CustomHeightfield | ManifoldKind::CustomParametric => {
                (self.u_res + 1) * (self.v_res + 1)
            }
        }
    }

    /// Clamps unstable or out-of-range parameters back into safe finite intervals.
    pub fn clamp_in_place(&mut self) {
        self.u_res = self.u_res.clamp(MIN_STUDIO_RES, 62);
        self.v_res = self.v_res.clamp(MIN_STUDIO_RES, 62);
        while self.estimated_vertices() > MAX_MANIFOLD_VERTICES && self.v_res > MIN_STUDIO_RES {
            self.v_res -= 1;
        }
        self.dt = if self.dt.is_finite() {
            self.dt.clamp(1e-5, 0.05)
        } else {
            default_dt()
        };
        self.heat_alpha = self.heat_alpha.clamp(0.001, 1.0);
        self.schnak_a = self.schnak_a.clamp(0.01, 1.0);
        self.schnak_b = self.schnak_b.clamp(0.1, 2.5);
        self.gs_f = self.gs_f.clamp(0.005, 0.10);
        self.gs_k = self.gs_k.clamp(0.02, 0.10);
        self.d_u = self.d_u.clamp(1e-4, 0.5);
        self.d_v = self.d_v.clamp(1e-4, 0.5);
        self.hh_conductance = self.hh_conductance.clamp(0.0, 2.0);
        self.hh_stimulus = self.hh_stimulus.clamp(-50.0, 50.0);
        self.sulcal_depth = self.sulcal_depth.clamp(0.0, 0.45);
        self.fs_grad_strength = self.fs_grad_strength.clamp(0.01, 2.0);
        self.brush_radius = self.brush_radius.clamp(0.05, 1.20);
        self.brush_strength = self.brush_strength.clamp(0.05, 2.0);
        self.regime = self.regime.coerce_to_enabled_features();
        if !self.camera.is_valid() {
            self.camera = Camera3D::default();
        }
        self.audio = self.audio.sanitized();
    }
}
