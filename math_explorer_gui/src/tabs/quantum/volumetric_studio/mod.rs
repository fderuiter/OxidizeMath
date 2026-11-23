#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! `VolumetricFieldStudioTool`: unified 3D hydrogenic quantum & 3D Turing morphogenesis studio
//! rendering `MarchingCubes` nodal meshes and `Gaussian3D` EWA radiance clouds on `Plot3DViewport`.
// [cite:quantum_mechanics]
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]
// [cite:favorite_child]

pub mod engine;
pub mod presets;
mod render;
pub mod state;
mod ui;

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use crate::web_io::PlatformInstant;
use eframe::egui;
use engine::VolumetricEngine;
use scientific_metadata::theory::TheoryDescribable;
use state::VolumetricStudioConfig;
use std::hash::{Hash, Hasher};
use std::sync::mpsc::Receiver;

/// Interactive 3D volumetric field studio uniting quantum eigenstates, 3D Turing PDEs,
/// `MarchingCubes` nodal surfaces, `.obj` Wavefront export, and `Gaussian3D` radiance splatting.
pub struct VolumetricFieldStudioTool {
    /// Validated serializable studio configuration.
    pub config: VolumetricStudioConfig,
    /// Pre-allocated zero-allocation 3D PDE, isosurface, and splatting workspace.
    pub engine: VolumetricEngine,
    pub(super) splat_texture: Option<egui::TextureHandle>,
    pub(super) splat_dirty: bool,
    pub(super) last_step_tick: PlatformInstant,
    pub(super) obj_save_rx: Option<Receiver<Result<String, String>>>,
    pub(super) export_status: Option<(String, bool)>,
}

impl Default for VolumetricFieldStudioTool {
    fn default() -> Self {
        let config = VolumetricStudioConfig::default();
        let engine = VolumetricEngine::new(&config);
        Self {
            config,
            engine,
            splat_texture: None,
            splat_dirty: true,
            last_step_tick: PlatformInstant::now(),
            obj_save_rx: None,
            export_status: None,
        }
    }
}

impl VolumetricFieldStudioTool {
    /// Advances the active 3D simulation and re-extracts surfaces/splats with zero heap allocations.
    pub fn step_zero_alloc(&mut self) {
        self.engine.step_and_extract_zero_alloc(&self.config, true);
        self.splat_dirty = true;
    }

    /// Re-extracts `MarchingCubes` and `Gaussian3D` outputs at the current `config` without advancing time.
    pub fn refresh_zero_alloc(&mut self) {
        self.engine.sync_structural_config(&self.config);
        self.engine.step_and_extract_zero_alloc(&self.config, false);
        self.splat_dirty = true;
    }

    /// Exports the active `MarchingCubes` nodal surface to a Wavefront `.obj` string.
    pub fn export_obj_string(&self) -> Result<String, String> {
        self.engine.export_obj_string(&self.config)
    }
}

impl FiniteStateGuard for VolumetricFieldStudioTool {
    fn is_finite_state(&self) -> bool {
        !self.engine.diverged
            && self.config.validate().is_ok()
            && self.engine.voxel_grid.data().iter().all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.engine.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.engine.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    fn reset_parameters(&mut self) {
        self.config = VolumetricStudioConfig::default();
        self.engine.reset_simulation(&self.config);
        self.splat_dirty = true;
    }
}

impl InteractiveTool for VolumetricFieldStudioTool {
    fn name(&self) -> &'static str {
        "Volumetric Field Studio"
    }

    fn domain(&self) -> &'static str {
        "quantum"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.config.source.hash(&mut h);
        self.config.render_mode.hash(&mut h);
        self.config.grid_resolution.hash(&mut h);
        (self.config.n1, self.config.l1, self.config.m1).hash(&mut h);
        (self.config.n2, self.config.l2, self.config.m2).hash(&mut h);
        self.config.superposition_mix.to_bits().hash(&mut h);
        self.config.relative_phase.to_bits().hash(&mut h);
        self.config.quantum_dt.to_bits().hash(&mut h);
        self.config.observable.hash(&mut h);
        self.config.iso_threshold.to_bits().hash(&mut h);
        self.config.splat_scale.to_bits().hash(&mut h);
        self.config.turing_a.to_bits().hash(&mut h);
        self.config.turing_b.to_bits().hash(&mut h);
        self.config.turing_du.to_bits().hash(&mut h);
        self.config.turing_dv.to_bits().hash(&mut h);
        self.config.turing_dt.to_bits().hash(&mut h);
        self.config.cut_enabled.hash(&mut h);
        self.config.cut_axis.hash(&mut h);
        self.config.cut_offset.to_bits().hash(&mut h);
        self.config.probe_brush_enabled.hash(&mut h);
        self.config.running.hash(&mut h);
        self.config.camera.hash_state(&mut h);
        h.finish()
    }

    fn has_finite_parameters(&self) -> bool {
        self.config.validate().is_ok() && !self.engine.diverged
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: VolumetricStudioConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid VolumetricFieldStudioTool state: {e}"))?;
        parsed.validate()?;
        self.config = parsed;
        self.engine.diverged = false;
        self.engine.diagnostic_report = None;
        self.refresh_zero_alloc();
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.config.clamp_in_place();
        self.engine.diverged = false;
        self.engine.diagnostic_report = None;
        self.refresh_zero_alloc();
        true
    }

    fn step_simulation(&mut self) {
        self.step_zero_alloc();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.poll_async_and_playback(ui.ctx());
        self.draw_side_panel(ui);
        self.draw_viewport(ui);
    }
}

impl TheoryDescribable for VolumetricFieldStudioTool {
    fn theory_description(&self) -> String {
        format!(
            "3D Volumetric Field Studio ({}) evolves hydrogenic superpositions $\\Psi(\\mathbf{{r}}, t) = c_1(t) R_{{n_1 \\ell_1}}(r) Y_{{\\ell_1 m_1}}(\\theta,\\phi) + c_2(t) R_{{n_2 \\ell_2}}(r) Y_{{\\ell_2 m_2}}(\\theta,\\phi)$ and 3D Schnakenberg reaction-diffusion $\\partial_t u = D_u \\nabla^2_{{3D}} u + a - u + u^2 v$ across a {}^3 VoxelGrid using MarchingCubes and EWA Gaussian3D splatting.",
            self.config.source.label(),
            self.config.grid_resolution
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Three-dimensional Volumetric Field Studio running {} in {} mode at {} cubed voxels.",
            self.config.source.label(),
            self.config.render_mode.label(),
            self.config.grid_resolution
        )
    }

    fn theory_citation(&self) -> String {
        "Lorensen & Cline (1987) Marching Cubes (SIGGRAPH); Kerbl et al. (2023) 3D Gaussian Splatting (ACM TOG); Turing (1952) Chemical Basis of Morphogenesis.".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "iso_threshold".into(),
            "Scalar level-set threshold tau extracted into dual positive and negative MarchingCubes nodal shells.".into(),
        );
        map.insert(
            "cut_offset".into(),
            "Normalized 3D cutting-plane position slicing open interior radial orbital shells and Turing gyroid channels.".into(),
        );
        map.insert(
            "splat_scale".into(),
            "Multiplier applied to gradient-aligned 3D Gaussian covariance axes prior to EWA projection.".into(),
        );
        map
    }
}

inventory::submit! {
    ToolMetadata {
        id: "VolumetricFieldStudioTool",
        display_name: "Volumetric Field Studio",
        domain: "quantum",
        tags: &["volumetric", "3d", "orbital", "marching-cubes", "gaussian-splatting", "turing", "isosurface"],
        build: || Box::new(VolumetricFieldStudioTool::default()),
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
