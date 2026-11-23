#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Geodesic Playground tool: 3D Flamm and wormhole embeddings, 2D gravitational lensing,
//! `Orbit` / `Slingshot` / `PhotonFan` modes, $V_{\text{eff}}(r)$, twin clocks, and sonification.

use crate::camera::Camera3D;
use crate::framework::{
    ExportCapabilities, InteractiveTool, ShaderDialect, ToolMetadata, ToolStateBounds,
};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::physics::astrophysics::{
    DwarfMorphology, GalacticDiskSystem, HaloProfile, Mpc, OrbitPopulationConfig,
};
use math_explorer::physics::high_energy::{
    DeflectionTable, LensingMetricSpec, DEFAULT_DEFLECTION_BINS,
};
use serde::{Deserialize, Serialize};

#[path = "playground/lensing_shader.rs"]
mod lensing_shader;
#[path = "playground/lensing_view.rs"]
mod lensing_view;
#[path = "playground/sim.rs"]
mod sim;
#[path = "playground/sim_integrators.rs"]
mod sim_integrators;
#[path = "playground/slingshot.rs"]
mod slingshot;
#[path = "playground/ui.rs"]
mod ui;
#[path = "playground/viewport_3d.rs"]
mod viewport_3d;

pub use sim::{BundleRay, TrajectorySample, MAX_BUNDLE_RAYS, MAX_TRAJECTORY_SAMPLES};
pub use slingshot::SlingshotDragState;

/// Default horizontal pixel width of the 2D gravitational lensing camera.
pub const LENSING_WIDTH: usize = 160;
/// Default vertical pixel height of the 2D gravitational lensing camera.
pub const LENSING_HEIGHT: usize = 120;

const fn default_fan_count() -> usize {
    9
}

const fn default_fan_spread() -> f64 {
    2.4
}

const fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

/// Curved spacetime background selectable in [`GeodesicPlaygroundTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpacetimePreset {
    /// Static Schwarzschild black hole ($R_s = 2 M_{\text{geom}}$) with Flamm paraboloid embedding.
    #[default]
    Schwarzschild,
    /// Rotating equatorial Kerr black hole with spin parameter $a$ and frame dragging.
    KerrEquatorial,
    /// Horizon-free traversable Morris-Thorne wormhole with throat radius $b_0$.
    MorrisThorneWormhole,
}

impl SpacetimePreset {
    /// Display label for combo boxes and telemetry readouts.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Schwarzschild => "Schwarzschild Black Hole",
            Self::KerrEquatorial => "Kerr Equatorial (Rotating)",
            Self::MorrisThorneWormhole => "Morris-Thorne Wormhole",
        }
    }
}

/// Causal worldline character for geodesic integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GeodesicMode {
    /// Massless photon ray ($g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = 0$).
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

/// Central viewport visualization mode for [`GeodesicPlaygroundTool`].
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

/// Bounded JSON state persisted for [`GeodesicPlaygroundTool`] URL sharing and starter presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeodesicPlaygroundState {
    /// Selected spacetime metric.
    pub spacetime: SpacetimePreset,
    /// Null photon vs massive timelike particle.
    pub mode: GeodesicMode,
    /// Schwarzschild radius $R_s$ (or wormhole throat scale $b_0$).
    pub scale_radius: f64,
    /// Dimensionless Kerr spin $\chi = 2a / R_s \in (-0.95, 0.95)$.
    pub spin_chi: f64,
    /// Initial radial coordinate $r_0$ (or proper distance $l_0$ for wormhole).
    pub initial_radius: f64,
    /// Impact parameter $b$ (null) or specific angular momentum $\ell$ (timelike).
    pub impact_or_momentum: f64,
    /// Inward radial launch angle bias in $[-1.0, 1.0]$.
    pub radial_bias: f64,
    /// Affine step size $\Delta\lambda$.
    pub step_size: f64,
    /// Steps advanced per playback tick.
    pub steps_per_frame: usize,
    /// Whether continuous trajectory playback is active.
    #[serde(default)]
    pub is_playing: bool,
    /// Scrubbed sample index along the precomputed/integrated trajectory.
    #[serde(default)]
    pub scrub_index: usize,
    /// 3D viewport camera orientation and zoom.
    pub camera: Camera3D,
    /// Active canvas gesture mode (`Orbit`, `Slingshot`, `PhotonFan`).
    #[serde(default)]
    pub interaction_mode: GeodesicInteractionMode,
    /// Active central viewport (`Embedding3D` vs `LensingCamera2D`).
    #[serde(default)]
    pub view_mode: ViewportViewMode,
    /// Source plane warped by the 2D gravitational lensing camera.
    #[serde(default)]
    pub lensing_background: LensingBackgroundKind,
    /// Number of rays in `PhotonFan` mode (`2..=16`).
    #[serde(default = "default_fan_count")]
    pub fan_count: usize,
    /// Total impact-parameter spread $\Delta b$ across a `PhotonFan`.
    #[serde(default = "default_fan_spread")]
    pub fan_spread: f64,
    /// Initial azimuthal angle $\varphi_0$ in radians.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub launch_azimuth: f64,
}

impl Default for GeodesicPlaygroundState {
    fn default() -> Self {
        Self {
            spacetime: SpacetimePreset::Schwarzschild,
            mode: GeodesicMode::NullPhoton,
            scale_radius: 2.0,
            spin_chi: 0.65,
            initial_radius: 10.0,
            impact_or_momentum: 5.35,
            radial_bias: -0.82,
            step_size: 0.04,
            steps_per_frame: 6,
            is_playing: false,
            scrub_index: 120,
            camera: Camera3D::new(0.48, -0.62, 1.0),
            interaction_mode: GeodesicInteractionMode::Orbit,
            view_mode: ViewportViewMode::Embedding3D,
            lensing_background: LensingBackgroundKind::GalacticDisk,
            fan_count: default_fan_count(),
            fan_spread: default_fan_spread(),
            launch_azimuth: 0.0,
        }
    }
}

/// Interactive curved-spacetime geodesic, slingshot, gravitational lensing, and sonification studio.
pub struct GeodesicPlaygroundTool {
    pub(crate) state: GeodesicPlaygroundState,
    pub(crate) samples: Vec<TrajectorySample>,
    pub(crate) bundle_rays: Vec<BundleRay>,
    pub(crate) active_ray_count: usize,
    pub(crate) halted_near_singularity: bool,
    pub(crate) deflection_table: DeflectionTable,
    pub(crate) lensing_dirty: bool,
    pub(crate) lensing_pixels: Vec<egui::Color32>,
    pub(crate) lensing_texture: Option<egui::TextureHandle>,
    pub(crate) lensing_galaxy: GalacticDiskSystem,
    #[cfg(feature = "biology")]
    pub(crate) lenia_engine: math_explorer::biology::lenia::LeniaEngine,
    pub(crate) slingshot: SlingshotDragState,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) high_contrast: bool,
    pub(crate) last_tick: PlatformInstant,
    pub(crate) revision: u64,
}

impl Default for GeodesicPlaygroundTool {
    fn default() -> Self {
        let state = GeodesicPlaygroundState::default();
        let deflection_table = DeflectionTable::with_capacity(
            LensingMetricSpec::Schwarzschild {
                schwarzschild_radius: state.scale_radius,
            },
            DEFAULT_DEFLECTION_BINS,
        )
        .expect("default Schwarzschild deflection table");
        #[cfg(feature = "biology")]
        let lenia_engine = build_default_lenia_engine();
        let mut bundle_rays = Vec::with_capacity(MAX_BUNDLE_RAYS);
        for _ in 0..MAX_BUNDLE_RAYS {
            bundle_rays.push(BundleRay::with_capacity(MAX_TRAJECTORY_SAMPLES));
        }
        let mut tool = Self {
            state,
            samples: Vec::with_capacity(MAX_TRAJECTORY_SAMPLES),
            bundle_rays,
            active_ray_count: 1,
            halted_near_singularity: false,
            deflection_table,
            lensing_dirty: true,
            lensing_pixels: vec![egui::Color32::BLACK; LENSING_WIDTH * LENSING_HEIGHT],
            lensing_texture: None,
            lensing_galaxy: build_default_lensing_galaxy(),
            #[cfg(feature = "biology")]
            lenia_engine,
            slingshot: SlingshotDragState::default(),
            focused_canvas: None,
            high_contrast: false,
            last_tick: PlatformInstant::now(),
            revision: 1,
        };
        tool.rebuild_trajectory(260);
        tool
    }
}

fn build_default_lensing_galaxy() -> GalacticDiskSystem {
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
fn build_default_lenia_engine() -> math_explorer::biology::lenia::LeniaEngine {
    let mut rng = oxidize_core::rng::OxidizeRng::from_seed(0x51A7_E5E1);
    math_explorer::biology::lenia::LeniaEngine::from_species(
        64,
        64,
        math_explorer::biology::lenia::LeniaSpecies::Orbium,
        0.0,
        &mut rng,
    )
}

impl InteractiveTool for GeodesicPlaygroundTool {
    fn name(&self) -> &'static str {
        "Geodesic Playground"
    }

    fn domain(&self) -> &'static str {
        "relativity"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn has_finite_parameters(&self) -> bool {
        let s = &self.state;
        s.scale_radius.is_finite()
            && s.spin_chi.is_finite()
            && s.initial_radius.is_finite()
            && s.impact_or_momentum.is_finite()
            && s.radial_bias.is_finite()
            && s.step_size.is_finite()
            && s.fan_spread.is_finite()
            && s.launch_azimuth.is_finite()
            && s.camera.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_state_in_place();
        self.rebuild_trajectory(260);
        self.lensing_dirty = true;
        if self.state.view_mode == ViewportViewMode::LensingCamera2D {
            self.refresh_lensing_pixels_in_place();
        }
        self.revision = self.revision.wrapping_add(1);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, raw: &str) -> Result<(), String> {
        self.state_bounds().validate_json(raw)?;
        let loaded: GeodesicPlaygroundState =
            serde_json::from_str(raw).map_err(|e| format!("Invalid geodesic state JSON: {e}"))?;
        if !loaded.camera.is_valid() {
            return Err("Camera parameters out of valid bounds".to_string());
        }
        self.state = loaded;
        self.clamp_state_in_place();
        let target_steps = self.state.scrub_index.clamp(80, MAX_TRAJECTORY_SAMPLES - 1);
        self.rebuild_trajectory(target_steps);
        self.lensing_dirty = true;
        if self.state.view_mode == ViewportViewMode::LensingCamera2D {
            self.refresh_lensing_pixels_in_place();
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_shader()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        Some(self.build_lensing_color_image())
    }

    fn export_glsl_shader(&self, dialect: ShaderDialect) -> Result<String, String> {
        self.build_lensing_glsl_shader(dialect)
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn step_simulation(&mut self) {
        self.advance_steps(self.state.steps_per_frame.clamp(1, 32));
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_tool(ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for GeodesicPlaygroundTool {
    fn theory_description(&self) -> String {
        format!(
            "Integrates stack-allocated RK4 geodesic bundles and deflection-table lensing on {} ({}, {}) with scale R = {:.2}, evaluating exact equatorial Kerr clocks, potentials, Doppler shifts, and synchrotron beaming.",
            self.state.spacetime.label(),
            self.state.mode.label(),
            self.state.interaction_mode.label(),
            self.state.scale_radius
        )
    }

    fn phonetic_description(&self) -> String {
        "Curved spacetime geodesic slingshot, gravitational lensing, and redshift sonification studio"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Schwarzschild (1916); Kerr (1963); Bardeen (1973); Morris & Thorne (1988)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "GeodesicPlaygroundTool",
        display_name: "Geodesic Playground",
        domain: "relativity",
        tags: &["general-relativity", "black-hole", "wormhole", "lensing", "geodesic"],
        build: || Box::new(GeodesicPlaygroundTool::default()),
    }
}
