#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Geodesic Playground tool: 3D Flamm and wormhole embeddings, 2D gravitational lensing,
//! `Orbit` / `Slingshot` / `PhotonFan` modes, $V_{\text{eff}}(r)$ with $\frac{1}{2}e^2$, twin worldlines,
//! custom AST metrics, 2D lensing regime cartography, and sonification.

use std::collections::{BTreeMap, HashMap};

use crate::async_sim::atlas::ParameterAtlasController;
use crate::camera::Camera3D;
use crate::framework::{
    CalloutManager, CalloutSpec, ExportCapabilities, InteractiveTool, ShaderDialect,
    ShaderExportFormat, ToolMetadata, ToolStateBounds,
};
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::physics::astrophysics::GalacticDiskSystem;
use math_explorer::physics::high_energy::{
    DeflectionTable, LensingMetricSpec, DEFAULT_DEFLECTION_BINS,
};
use scientific_metadata::theory::ParameterConstraint;
use serde::{Deserialize, Serialize};

#[path = "playground/atlas.rs"]
pub mod atlas;
#[path = "playground/embedding.rs"]
mod embedding;
#[path = "playground/lensing_shader.rs"]
mod lensing_shader;
#[path = "playground/lensing_view.rs"]
mod lensing_view;
#[path = "playground/sandbox.rs"]
pub mod sandbox;
#[path = "playground/sim.rs"]
mod sim;
#[path = "playground/sim_integrators.rs"]
mod sim_integrators;
#[path = "playground/slingshot.rs"]
mod slingshot;
#[path = "playground/telemetry.rs"]
mod telemetry;
#[path = "playground/types.rs"]
mod types;
#[path = "playground/ui.rs"]
mod ui;
#[path = "playground/viewport_3d.rs"]
mod viewport_3d;

pub use atlas::{kerr_prograde_critical_b, RELATIVITY_ATLAS_AXIS_PAIRS};
pub use sandbox::MetricSandbox;
pub use sim::{BundleRay, TrajectorySample, MAX_BUNDLE_RAYS, MAX_TRAJECTORY_SAMPLES};
pub use slingshot::SlingshotDragState;
pub use types::{
    GeodesicInteractionMode, GeodesicMode, LensingBackgroundKind, SpacetimePreset, ViewportViewMode,
};

/// Default horizontal pixel width of the 2D gravitational lensing camera.
pub const LENSING_WIDTH: usize = 160;
/// Default vertical pixel height of the 2D gravitational lensing camera.
pub const LENSING_HEIGHT: usize = 120;

use atlas::{default_relativity_atlas_axes, is_default_relativity_atlas_axes};
use sandbox::{
    default_custom_lapse_expr, default_custom_wormhole_expr, is_default_custom_lapse_expr,
    is_default_custom_wormhole_expr,
};
#[cfg(feature = "biology")]
use types::build_default_lenia_engine;
use types::{build_default_lensing_galaxy, default_fan_count, default_fan_spread, is_zero_f64};

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
    /// Impact parameter $b = \ell / e$ (null) or specific angular momentum $\ell$ (timelike).
    pub impact_or_momentum: f64,
    /// Offset $\Delta b$ applied to the diverging twin worldline (`0.0` disables twin ray).
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub twin_delta_b: f64,
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
    /// Custom 4D radial lapse formula $f(r)$.
    #[serde(
        default = "default_custom_lapse_expr",
        skip_serializing_if = "is_default_custom_lapse_expr"
    )]
    pub custom_lapse_expr: String,
    /// Custom 3D wormhole profile formula $b(l)$.
    #[serde(
        default = "default_custom_wormhole_expr",
        skip_serializing_if = "is_default_custom_wormhole_expr"
    )]
    pub custom_wormhole_expr: String,
    /// Synthesized free-parameter values for custom AST metric formulas.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom_params: BTreeMap<String, f64>,
    /// Whether the progressive 2D parameter regime atlas is open.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub atlas_visible: bool,
    /// Selected `(x, y)` axis keys for the 2D parameter regime atlas.
    #[serde(
        default = "default_relativity_atlas_axes",
        skip_serializing_if = "is_default_relativity_atlas_axes"
    )]
    pub atlas_axes: [String; 2],
    /// Persisted non-mute acoustic synthesis settings.
    #[serde(
        default,
        skip_serializing_if = "crate::audio::AcousticSettings::is_default"
    )]
    pub audio: crate::audio::AcousticSettings,
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
            twin_delta_b: 0.0,
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
            custom_lapse_expr: default_custom_lapse_expr(),
            custom_wormhole_expr: default_custom_wormhole_expr(),
            custom_params: BTreeMap::new(),
            atlas_visible: false,
            atlas_axes: default_relativity_atlas_axes(),
            audio: crate::audio::AcousticSettings::default(),
        }
    }
}

/// Interactive curved-spacetime geodesic, slingshot, gravitational lensing, and sonification studio.
pub struct GeodesicPlaygroundTool {
    pub(crate) state: GeodesicPlaygroundState,
    pub(crate) samples: Vec<TrajectorySample>,
    pub(crate) twin_samples: Vec<TrajectorySample>,
    pub(crate) bundle_rays: Vec<BundleRay>,
    pub(crate) active_ray_count: usize,
    pub(crate) halted_near_singularity: bool,
    pub(crate) twin_halted_near_singularity: bool,
    pub(crate) launch_energy: f64,
    pub(crate) twin_launch_energy: f64,
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
    pub(crate) sandbox: MetricSandbox,
    pub(crate) atlas: ParameterAtlasController,
    pub(crate) callouts: CalloutManager,
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
            bundle_rays.push(BundleRay::with_capacity(MAX_TRAJECTORY_SAMPLES + 8));
        }
        let mut tool = Self {
            state,
            samples: Vec::with_capacity(MAX_TRAJECTORY_SAMPLES + 8),
            twin_samples: Vec::with_capacity(MAX_TRAJECTORY_SAMPLES + 8),
            bundle_rays,
            active_ray_count: 1,
            halted_near_singularity: false,
            twin_halted_near_singularity: false,
            launch_energy: 1.0,
            twin_launch_energy: 1.0,
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
            sandbox: MetricSandbox::default(),
            atlas: ParameterAtlasController::new(RELATIVITY_ATLAS_AXIS_PAIRS.to_vec()),
            callouts: CalloutManager::new(),
            last_tick: PlatformInstant::now(),
            revision: 1,
        };
        tool.rebuild_trajectory(560);
        tool
    }
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
            && s.twin_delta_b.is_finite()
            && s.radial_bias.is_finite()
            && s.step_size.is_finite()
            && s.fan_spread.is_finite()
            && s.launch_azimuth.is_finite()
            && s.custom_params.values().all(|v| v.is_finite())
            && self.sandbox.has_finite_params()
            && s.audio.validate().is_ok()
            && s.camera.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_state_in_place();
        self.state.audio = self.state.audio.sanitized();
        self.sandbox.lapse_bar.clamp_active_params(-5.0..=5.0);
        self.sandbox.wormhole_bar.clamp_active_params(-5.0..=5.0);
        self.sync_sandbox_to_state();
        self.rebuild_trajectory(560);
        self.lensing_dirty = true;
        if self.state.view_mode == ViewportViewMode::LensingCamera2D {
            self.refresh_lensing_pixels_in_place();
        }
        self.revision = self.revision.wrapping_add(1);
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        self.collect_observables()
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callouts.sync_specs(callouts);
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
        loaded.audio.validate()?;
        let mut next_sandbox = MetricSandbox::default();
        next_sandbox.apply_persisted(
            &loaded.custom_lapse_expr,
            &loaded.custom_wormhole_expr,
            &loaded.custom_params,
            loaded.spacetime == SpacetimePreset::CustomRadialLapse,
            loaded.spacetime == SpacetimePreset::CustomWormholeProfile,
        )?;
        self.atlas
            .set_axis_keys(&loaded.atlas_axes[0], &loaded.atlas_axes[1])?;
        self.atlas.set_visible(loaded.atlas_visible);
        self.sandbox = next_sandbox;
        self.state = loaded;
        self.clamp_state_in_place();
        self.sync_sandbox_to_state();
        let target_steps = self.state.scrub_index.clamp(480, MAX_TRAJECTORY_SAMPLES);
        self.rebuild_trajectory(target_steps);
        self.lensing_dirty = true;
        if self.state.view_mode == ViewportViewMode::LensingCamera2D {
            self.refresh_lensing_pixels_in_place();
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_audio().with_shader()
    }
    fn sonification_frame(&self) -> Option<crate::audio::SonificationFrame> {
        Some(if self.has_finite_parameters() {
            self.build_audio_frame()
        } else {
            crate::audio::SonificationFrame::silent()
        })
    }
    fn acoustic_settings(&self) -> crate::audio::AcousticSettings {
        self.state.audio
    }
    fn set_acoustic_settings(&mut self, settings: crate::audio::AcousticSettings) {
        self.state.audio = settings.sanitized();
        self.revision = self.revision.wrapping_add(1);
    }
    fn sample_sonification_frames(
        &mut self,
        num_frames: usize,
    ) -> Vec<crate::audio::SonificationFrame> {
        self.sample_geodesic_wav_frames(num_frames)
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        Some(self.build_lensing_color_image())
    }

    fn export_shader_source(&self, format: ShaderExportFormat) -> Result<String, String> {
        match format {
            ShaderExportFormat::Wgsl => self.build_lensing_wgsl_shader(),
            ShaderExportFormat::Glsl => self.build_lensing_glsl_shader(ShaderDialect::WebGl2Es300),
        }
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
            "Integrates stack-allocated RK4 geodesic bundles d^2x^mu/dlambda^2 + Gamma^mu_(alpha beta) dx^alpha/dlambda dx^beta/dlambda = 0 and deflection-table lensing on {} ({}, {}) with scale R = {:.2}, visualizing 3D embeddings, V_eff(r) vs 1/2 e^2, twin worldline bifurcation, 2D lensing cartography, and Doppler sonification.",
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

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        self.build_theory_parameters()
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
