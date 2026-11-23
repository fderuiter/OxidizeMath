// @explorer_feature = "biology"
use crate::async_sim::SimWorkerState;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::presets::MorphogenesisPresetConfig;
#[cfg(test)]
use crate::presets::{deserialize_preset_envelope, serialize_preset_envelope};
use crate::tabs::ExplorerTab;
use eframe::egui;
#[cfg(test)]
use eframe::egui::ColorImage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Progressive 2D dispersion & CFL parameter regime cartography.
pub mod atlas;
/// Snapshot texture sync, sculpted canvas plot, and side-by-side parameter atlas view.
pub mod canvas;
/// 2D FFT Lenia continuous cellular automaton studio.
pub mod lenia;
/// Theme-aware concentration, vorticity, and dual composite pixel overlays.
pub mod overlay;
/// Starter gallery registrations, PRNG helper, and dynamic theory adapter.
pub mod presets;
/// Background simulation runner coupling LBM / PotentialFlow to AdvectionDiffusion2D.
pub mod runner;
/// Obstacle/reagent brush sculpting and fluid velocity field coupling helpers.
pub mod sculpt;
/// Validated serializable state schema for MorphogenesisTool.
pub mod state;
/// Interactive tool lifecycle and framework integration.
pub mod tool;
/// Coupled reaction, flow, brush, and overlay mode enums and dimensionless metrics.
pub mod types;
/// Control panel widgets, dimensionless badges, and CFL stabilization actions.
pub mod ui;

#[cfg(test)]
pub use atlas::{
    classify_schnakenberg_regime, grid_u_variance, MorphogenesisCartographer, MorphogenesisRegime,
};
#[cfg(test)]
pub use lenia::LeniaTool;
#[cfg(test)]
pub use overlay::render_morphogenesis_overlay;
#[cfg(test)]
pub use presets::MorphogenesisTheory;
pub use presets::SimpleRng;
#[cfg(test)]
pub use runner::MorphogenesisRunner;
pub use state::MorphogenesisToolState;
pub use tool::MorphogenesisTool;
pub use types::{
    MorphogenesisBrushMode, MorphogenesisFlowMode, MorphogenesisKineticsMode,
    MorphogenesisOverlayMode, MorphogenesisSharedParams,
};

/// Legacy flat `MorphoState` schema saved by pre-framework `.mws` workspaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyMorphoState {
    /// Time step size `dt`.
    pub dt: f64,
    /// Optional worker execution state.
    #[serde(default)]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Legacy starter gallery visibility flag (ignored on load; managed by framework).
    #[serde(default)]
    pub show_starter_gallery: bool,
    /// Deferred start flag keeping hidden tabs idle until first rendered frame.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending_initial_start: bool,
}

/// Biology tab hosting [`MorphogenesisTool`] and [`lenia::LeniaTool`] inside [`crate::framework::SimulationFramework`].
pub struct MorphogenesisTab {
    /// Underlying simulation framework managing exports, presets, theory portal, and state guard.
    pub framework: crate::framework::SimulationFramework,
    pub(crate) params: Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
}

impl Default for MorphogenesisTab {
    fn default() -> Self {
        let mut framework = crate::framework::SimulationFramework::new("biology");
        framework.selected_tool_index = framework
            .available_tools
            .iter()
            .position(|m| m.id == "MorphogenesisTool");

        let tool = MorphogenesisTool::default();
        let params = Arc::clone(&tool.params);
        let name = crate::framework::InteractiveTool::name(&tool).to_string();
        let baseline = crate::framework::InteractiveTool::save_state(&tool);
        framework.active_tool = Some(Box::new(tool));
        if let Some(b) = baseline {
            framework.saved_tool_states.insert(name, b.clone());
            framework.baseline_tool_state = Some(b);
        }
        framework.cache_valid_snapshot();
        Self { framework, params }
    }
}

impl MorphogenesisTab {
    /// Domain category string used for preset envelope validation and filenames.
    #[cfg(test)]
    pub fn domain(&self) -> &'static str {
        "biology"
    }

    /// Maps a 2D concentration slice onto a normalized heatmap [`ColorImage`].
    #[cfg(test)]
    pub fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        MorphogenesisTool::plot_concentration(data, width, height)
    }

    #[cfg(test)]
    pub(crate) fn create_runner(
        params: &Arc<std::sync::RwLock<MorphogenesisSharedParams>>,
        width: usize,
        height: usize,
    ) -> MorphogenesisRunner {
        MorphogenesisTool::create_runner(params, width, height)
    }

    /// Exports active [`MorphogenesisTool`] simulation parameters into [`MorphogenesisPresetConfig`].
    pub fn export_config(&self) -> MorphogenesisPresetConfig {
        if let Some(json) = self
            .framework
            .active_tool
            .as_ref()
            .filter(|t| t.name() == "Turing Morphogenesis")
            .and_then(|t| t.save_state())
        {
            if let Ok(s) = serde_json::from_str::<MorphogenesisToolState>(&json) {
                return MorphogenesisPresetConfig {
                    a: s.a,
                    b: s.b,
                    d_u: s.d_u,
                    d_v: s.d_v,
                    dt: s.dt,
                    width: s.width,
                    height: s.height,
                    kinetics_mode: s.kinetics_mode,
                    f: s.f,
                    k: s.k,
                    flow_mode: s.flow_mode,
                    inlet_velocity: s.inlet_velocity,
                    viscosity: s.viscosity,
                    circulation: s.circulation,
                    overlay_mode: s.overlay_mode,
                    brush_mode: s.brush_mode,
                    brush_radius: s.brush_radius,
                };
            }
        }
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        MorphogenesisPresetConfig {
            a: p.a,
            b: p.b,
            d_u: p.d_u,
            d_v: p.d_v,
            dt: p.dt,
            width: 100,
            height: 100,
            kinetics_mode: p.kinetics_mode,
            f: p.f,
            k: p.k,
            flow_mode: p.flow_mode,
            inlet_velocity: p.inlet_velocity,
            viscosity: p.viscosity,
            circulation: p.circulation,
            overlay_mode: p.overlay_mode,
            brush_mode: p.brush_mode,
            brush_radius: p.brush_radius,
        }
    }

    /// Applies [`MorphogenesisPresetConfig`] to [`MorphogenesisTool`].
    pub fn apply_config(&mut self, config: MorphogenesisPresetConfig) -> Result<(), String> {
        if self.framework.active_tool_id() != Some("MorphogenesisTool") {
            let _ = self.framework.select_tool_by_name("MorphogenesisTool");
        }
        let Some(tool) = self.framework.active_tool.as_mut() else {
            return Err("No active MorphogenesisTool".to_string());
        };
        let existing = tool
            .save_state()
            .and_then(|s| serde_json::from_str::<MorphogenesisToolState>(&s).ok());
        let current_state = existing
            .as_ref()
            .and_then(|s| s.state)
            .unwrap_or(SimWorkerState::Ready);
        let atlas_visible = existing.as_ref().is_some_and(|s| s.atlas_visible);
        let atlas_axes = existing
            .map(|s| s.atlas_axes)
            .unwrap_or_else(atlas::default_morpho_atlas_axes);
        let target_state = if current_state == SimWorkerState::Running {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        let state = MorphogenesisToolState {
            a: config.a,
            b: config.b,
            d_u: config.d_u,
            d_v: config.d_v,
            dt: config.dt,
            width: config.width,
            height: config.height,
            kinetics_mode: config.kinetics_mode,
            f: config.f,
            k: config.k,
            flow_mode: config.flow_mode,
            inlet_velocity: config.inlet_velocity,
            viscosity: config.viscosity,
            circulation: config.circulation,
            overlay_mode: config.overlay_mode,
            brush_mode: config.brush_mode,
            brush_radius: config.brush_radius,
            state: Some(target_state),
            paused: None,
            pending_initial_start: false,
            atlas_visible,
            atlas_axes,
        };
        let payload = serde_json::to_string(&state).map_err(|e| e.to_string())?;
        tool.try_load_state(&payload)?;
        self.sync_shared_params_from_config(&config);
        self.sync_framework_after_direct_tool_mutation();
        Ok(())
    }

    fn sync_shared_params_from_config(&mut self, cfg: &MorphogenesisPresetConfig) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = cfg.a;
        p.b = cfg.b;
        p.f = cfg.f;
        p.k = cfg.k;
        p.d_u = cfg.d_u;
        p.d_v = cfg.d_v;
        p.dt = cfg.dt;
        p.kinetics_mode = cfg.kinetics_mode;
        p.flow_mode = cfg.flow_mode;
        p.inlet_velocity = cfg.inlet_velocity;
        p.viscosity = cfg.viscosity;
        p.circulation = cfg.circulation;
        p.overlay_mode = cfg.overlay_mode;
        p.brush_mode = cfg.brush_mode;
        p.brush_radius = cfg.brush_radius;
    }

    /// Serializes active simulation parameters to a formatted preset envelope JSON string.
    #[cfg(test)]
    pub fn to_json_string(&self) -> Result<String, String> {
        serialize_preset_envelope(
            "Turing Morphogenesis Simulation",
            self.domain(),
            self.export_config(),
        )
    }

    /// Validates a preset envelope JSON string and applies its configuration.
    #[cfg(test)]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_json_str(&mut self, json_str: &str) -> Result<(), String> {
        let config: MorphogenesisPresetConfig =
            deserialize_preset_envelope(json_str, self.domain(), self.name())?;
        self.apply_config(config)
    }

    fn sync_framework_after_direct_tool_mutation(&mut self) {
        if let Some(tool) = self.framework.active_tool.as_ref() {
            if let Some(saved) = tool.save_state() {
                self.framework
                    .saved_tool_states
                    .insert(tool.name().to_string(), saved);
            }
        }
        self.framework.cache_valid_snapshot();
    }

    fn try_load_legacy_state(&mut self, state: &str) -> bool {
        let Ok(legacy) = serde_json::from_str::<LegacyMorphoState>(state) else {
            return false;
        };
        if self.framework.active_tool_id() != Some("MorphogenesisTool") {
            let _ = self.framework.select_tool_by_name("MorphogenesisTool");
        }
        let Some(tool) = self.framework.active_tool.as_mut() else {
            return true;
        };
        let Some(current_json) = tool.save_state() else {
            return true;
        };
        let Ok(mut current_state) = serde_json::from_str::<MorphogenesisToolState>(&current_json)
        else {
            return true;
        };
        current_state.dt = legacy.dt.clamp(1e-4, 2.0);
        current_state.state = legacy.state;
        current_state.paused = legacy.paused;
        current_state.pending_initial_start = legacy.pending_initial_start;
        if let Ok(updated_json) = serde_json::to_string(&current_state) {
            if tool.try_load_state(&updated_json).is_ok() {
                let cfg = self.export_config();
                self.sync_shared_params_from_config(&cfg);
                self.framework.show_starter_gallery = false;
                self.sync_framework_after_direct_tool_mutation();
            }
        }
        true
    }
}

impl FiniteStateGuard for MorphogenesisTab {
    fn is_finite_state(&self) -> bool {
        let p = self.params.read().unwrap_or_else(|e| e.into_inner());
        p.a.is_finite()
            && p.b.is_finite()
            && p.f.is_finite()
            && p.k.is_finite()
            && p.d_u.is_finite()
            && p.d_v.is_finite()
            && p.dt.is_finite()
            && p.inlet_velocity.is_finite()
            && p.viscosity.is_finite()
            && p.circulation.is_finite()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let mut cfg = self.export_config();
        let current_dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
        cfg.dt = (current_dt * 0.5).clamp(1e-4, 2.0);
        let _ = self.apply_config(cfg);
    }

    fn clamp_parameters(&mut self) {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut cfg = self.export_config();
        cfg.a = p.a.clamp(0.0, 1.0);
        cfg.b = p.b.clamp(0.0, 2.0);
        cfg.f = p.f.clamp(0.0, 0.12);
        cfg.k = p.k.clamp(0.0, 0.12);
        cfg.d_u = p.d_u.clamp(0.005, 5.0);
        cfg.d_v = p.d_v.clamp(0.005, 200.0);
        cfg.dt = p.dt.clamp(0.001, 2.0);
        cfg.inlet_velocity = p.inlet_velocity.clamp(0.0, 2.0);
        cfg.viscosity = p.viscosity.clamp(0.003, 0.5);
        cfg.circulation = p.circulation.clamp(-25.0, 25.0);
        cfg.brush_radius = p.brush_radius.clamp(1, 25);
        let _ = self.apply_config(cfg);
    }

    fn reset_parameters(&mut self) {
        let mut cfg = self.export_config();
        cfg.a = 0.1;
        cfg.b = 0.9;
        cfg.f = 0.0367;
        cfg.k = 0.0649;
        cfg.d_u = 1.0;
        cfg.d_v = 100.0;
        cfg.dt = 0.05;
        cfg.kinetics_mode = MorphogenesisKineticsMode::Schnakenberg;
        cfg.flow_mode = MorphogenesisFlowMode::None;
        cfg.inlet_velocity = 0.0;
        cfg.viscosity = 0.02;
        cfg.circulation = 0.0;
        let _ = self.apply_config(cfg);
    }
}

impl ExplorerTab for MorphogenesisTab {
    fn name(&self) -> &'static str {
        "Morphogenesis"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "biology");
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        if !self.try_load_legacy_state(state) {
            self.framework.load_state(state);
        }
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }

    fn import_url_preset_json(
        &mut self,
        tool_id: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        self.framework
            .import_url_preset_json(tool_id, preset_json)?;
        if self.framework.active_tool_id() == Some("MorphogenesisTool") {
            let cfg = self.export_config();
            self.sync_shared_params_from_config(&cfg);
        }
        Ok(())
    }
}

#[cfg(test)]
mod coupled_tests;
#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

// [cite:favorite_child]
