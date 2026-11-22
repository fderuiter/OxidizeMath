#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Geodesic Playground tool: 3D Flamm and wormhole embeddings, $V_{\text{eff}}(r)$, and twin clocks.

use crate::camera::Camera3D;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use crate::web_io::PlatformInstant;
use eframe::egui;
use serde::{Deserialize, Serialize};

#[path = "playground/sim.rs"]
mod sim;
#[path = "playground/ui.rs"]
mod ui;

pub use sim::{TrajectorySample, MAX_TRAJECTORY_SAMPLES};

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
        }
    }
}

/// Interactive 3D curved-spacetime geodesic laboratory.
pub struct GeodesicPlaygroundTool {
    pub(crate) state: GeodesicPlaygroundState,
    pub(crate) samples: Vec<TrajectorySample>,
    pub(crate) halted_near_singularity: bool,
    pub(crate) last_tick: PlatformInstant,
    pub(crate) revision: u64,
}

impl Default for GeodesicPlaygroundTool {
    fn default() -> Self {
        let mut tool = Self {
            state: GeodesicPlaygroundState::default(),
            samples: Vec::with_capacity(MAX_TRAJECTORY_SAMPLES),
            halted_near_singularity: false,
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
            && s.radial_bias.is_finite()
            && s.step_size.is_finite()
            && s.camera.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_state_in_place();
        self.rebuild_trajectory(260);
        self.revision = self.revision.wrapping_add(1);
        true
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
        self.state = loaded;
        self.clamp_state_in_place();
        let target_steps = self.state.scrub_index.clamp(80, MAX_TRAJECTORY_SAMPLES - 1);
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
            "Integrates stack-allocated RK4 geodesics d^2x^mu/dlambda^2 + Gamma^mu_(alpha beta) dx^alpha/dlambda dx^beta/dlambda = 0 on {} ({}) with scale R = {:.2}, visualizing isometric 3D Flamm or wormhole embeddings, radial effective potential V_eff(r), and combined gravitational-kinematic clock dilation.",
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

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
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
