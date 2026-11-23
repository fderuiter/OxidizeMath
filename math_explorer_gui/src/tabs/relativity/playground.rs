#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Geodesic Playground tool: 3D Flamm and wormhole embeddings, $V_{\text{eff}}(r)$, twin worldlines,
//! custom AST metrics, and 2D lensing regime cartography.

use std::collections::{BTreeMap, HashMap};

use crate::async_sim::atlas::ParameterAtlasController;
use crate::camera::Camera3D;
use crate::framework::{
    CalloutManager, CalloutSpec, ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds,
};
use crate::web_io::PlatformInstant;
use eframe::egui;
use scientific_metadata::theory::ParameterConstraint;
use serde::{Deserialize, Serialize};

#[path = "playground/atlas.rs"]
pub mod atlas;
#[path = "playground/embedding.rs"]
mod embedding;
#[path = "playground/sandbox.rs"]
pub mod sandbox;
#[path = "playground/sim.rs"]
mod sim;
#[path = "playground/telemetry.rs"]
mod telemetry;
#[path = "playground/ui.rs"]
mod ui;

pub use atlas::{kerr_prograde_critical_b, RELATIVITY_ATLAS_AXIS_PAIRS};
pub use sandbox::MetricSandbox;
pub use sim::{TrajectorySample, MAX_TRAJECTORY_SAMPLES};

use atlas::{default_relativity_atlas_axes, is_default_relativity_atlas_axes};
use sandbox::{
    default_custom_lapse_expr, default_custom_wormhole_expr, is_default_custom_lapse_expr,
    is_default_custom_wormhole_expr,
};

fn is_zero_f64(v: &f64) -> bool {
    v.abs() <= 1e-12
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
            custom_lapse_expr: default_custom_lapse_expr(),
            custom_wormhole_expr: default_custom_wormhole_expr(),
            custom_params: BTreeMap::new(),
            atlas_visible: false,
            atlas_axes: default_relativity_atlas_axes(),
        }
    }
}

/// Interactive 3D curved-spacetime geodesic laboratory.
pub struct GeodesicPlaygroundTool {
    pub(crate) state: GeodesicPlaygroundState,
    pub(crate) samples: Vec<TrajectorySample>,
    pub(crate) twin_samples: Vec<TrajectorySample>,
    pub(crate) halted_near_singularity: bool,
    pub(crate) twin_halted_near_singularity: bool,
    pub(crate) launch_energy: f64,
    pub(crate) twin_launch_energy: f64,
    pub(crate) sandbox: MetricSandbox,
    pub(crate) atlas: ParameterAtlasController,
    pub(crate) callouts: CalloutManager,
    pub(crate) last_tick: PlatformInstant,
    pub(crate) revision: u64,
}

impl Default for GeodesicPlaygroundTool {
    fn default() -> Self {
        let mut tool = Self {
            state: GeodesicPlaygroundState::default(),
            samples: Vec::with_capacity(MAX_TRAJECTORY_SAMPLES + 8),
            twin_samples: Vec::with_capacity(MAX_TRAJECTORY_SAMPLES + 8),
            halted_near_singularity: false,
            twin_halted_near_singularity: false,
            launch_energy: 1.0,
            twin_launch_energy: 1.0,
            sandbox: MetricSandbox::default(),
            atlas: ParameterAtlasController::new(RELATIVITY_ATLAS_AXIS_PAIRS.to_vec()),
            callouts: CalloutManager::new(),
            last_tick: PlatformInstant::now(),
            revision: 1,
        };
        tool.rebuild_trajectory(260);
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
            && s.custom_params.values().all(|v| v.is_finite())
            && self.sandbox.has_finite_params()
            && s.camera.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_state_in_place();
        self.sandbox.lapse_bar.clamp_active_params(-5.0..=5.0);
        self.sandbox.wormhole_bar.clamp_active_params(-5.0..=5.0);
        self.sync_sandbox_to_state();
        self.rebuild_trajectory(560);
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
        ToolStateBounds::DEFAULT
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
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn step_simulation(&mut self) {
        self.advance_steps(self.state.steps_per_frame.clamp(1, 32));
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
            "Integrates stack-allocated RK4 geodesics d^2x^mu/dlambda^2 + Gamma^mu_(alpha beta) dx^alpha/dlambda dx^beta/dlambda = 0 on {} ({}) with scale R = {:.2}, visualizing 3D embeddings, radial effective potential V_eff(r) vs 1/2 e^2, twin worldline bifurcation, and 2D lensing cartography.",
            self.state.spacetime.label(),
            self.state.mode.label(),
            self.state.scale_radius
        )
    }

    fn phonetic_description(&self) -> String {
        "Curved spacetime geodesic integration, Flamm paraboloid, and twin clock dilation".into()
    }

    fn theory_citation(&self) -> String {
        "Schwarzschild (1916); Kerr (1963); Misner, Thorne, & Wheeler, Gravitation (1973); Morris & Thorne (1988)".into()
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
        tags: &["general-relativity", "black-hole", "wormhole", "geodesic"],
        build: || Box::new(GeodesicPlaygroundTool::default()),
    }
}
