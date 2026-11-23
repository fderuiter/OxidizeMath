#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! `VolumetricFieldStudioTool`: unified 3D hydrogenic quantum & 3D Turing morphogenesis studio
//! rendering `MarchingCubes` nodal meshes and `Gaussian3D` EWA radiance clouds on `Plot3DViewport`.
// [cite:quantum_mechanics]
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]
// [cite:favorite_child]

pub mod ast_field;
pub mod engine;
pub mod presets;
mod render;
pub mod state;
mod ui;

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use crate::web_io::PlatformInstant;
use ast_field::CustomAstFieldController;
use eframe::egui;
use engine::VolumetricEngine;
use scientific_metadata::theory::TheoryDescribable;
use state::VolumetricStudioConfig;
use std::hash::{Hash, Hasher};
use std::sync::mpsc::Receiver;

/// Interactive 3D volumetric field studio uniting quantum eigenstates, 3D Turing PDEs,
/// custom 3D implicit AST surfaces, `MarchingCubes` nodal meshes, and `Gaussian3D` splatting.
pub struct VolumetricFieldStudioTool {
    /// Validated serializable studio configuration.
    pub config: VolumetricStudioConfig,
    /// Pre-allocated zero-allocation 3D PDE, AST, isosurface, and splatting workspace.
    pub engine: VolumetricEngine,
    /// Formula bar & synthesized `ParameterSliderBank` for `FieldSourceMode::CustomImplicitAst`.
    pub ast_controller: CustomAstFieldController,
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
        let mut ast_controller = CustomAstFieldController::default();
        ast_controller.load_from_config(&config.custom_expr, &config.custom_params);
        Self {
            config,
            engine,
            ast_controller,
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
        self.refresh_with_drag_preview(false);
    }

    /// Re-extracts `MarchingCubes`, `Gaussian3D`, and slice contours, using a `20^3` preview grid
    /// when `drag_preview` is active in `CustomImplicitAst` mode and full `grid_resolution^3` on release.
    pub fn refresh_with_drag_preview(&mut self, drag_preview: bool) {
        if let Some(compiled) = self.ast_controller.bar.active_compiled() {
            self.engine
                .ast_kernel
                .sync_from_bar(compiled, self.ast_controller.bar.param_slice());
        }
        self.engine
            .sync_structural_config_with_preview(&self.config, drag_preview);
        self.engine.step_and_extract_zero_alloc(&self.config, false);
        self.splat_dirty = true;
    }

    /// Compiles and applies a new 3D custom implicit AST expression.
    #[cfg(test)]
    pub fn set_custom_expression(&mut self, expr: &str) -> Result<(), String> {
        ast_field::validate_custom_3d_formula(expr)?;
        self.ast_controller.bar.set_expression(expr);
        self.config.custom_expr = expr.to_string();
        self.config.custom_params = self.ast_controller.bar.params_map();
        if let Some(compiled) = self.ast_controller.bar.active_compiled() {
            self.engine
                .ast_kernel
                .sync_from_bar(compiled, self.ast_controller.bar.param_slice());
        }
        self.refresh_with_drag_preview(false);
        Ok(())
    }

    /// Updates a synthesized 3D AST parameter in place without heap allocations during drag previews.
    #[cfg(test)]
    pub fn set_custom_param_zero_alloc(&mut self, slot: usize, value: f64, drag_preview: bool) {
        let mut buf = [0.0; crate::framework::math_input::MAX_SYNTHESIZED_PARAMS];
        let src = self.ast_controller.bar.param_slice();
        let len = src.len().min(buf.len());
        buf[..len].copy_from_slice(&src[..len]);
        if slot < len {
            buf[slot] = value;
        }
        self.engine.ast_kernel.set_params_slice(&buf[..len]);
        self.engine
            .sync_structural_config_with_preview(&self.config, drag_preview);
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
        self.ast_controller
            .load_from_config(&self.config.custom_expr, &self.config.custom_params);
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
        self.config.custom_expr.hash(&mut h);
        for (k, v) in &self.config.custom_params {
            k.hash(&mut h);
            v.to_bits().hash(&mut h);
        }
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
        self.ast_controller
            .load_from_config(&self.config.custom_expr, &self.config.custom_params);
        self.engine.diverged = false;
        self.engine.diagnostic_report = None;
        self.engine
            .ast_kernel
            .sync_from_config_expr(&self.config.custom_expr, &self.config.custom_params);
        self.refresh_zero_alloc();
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.config.clamp_in_place();
        self.ast_controller
            .load_from_config(&self.config.custom_expr, &self.config.custom_params);
        self.engine.diverged = false;
        self.engine.diagnostic_report = None;
        self.engine
            .ast_kernel
            .sync_from_config_expr(&self.config.custom_expr, &self.config.custom_params);
        self.refresh_zero_alloc();
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let tris = (self.engine.pos_mesh.indices.len() + self.engine.neg_mesh.indices.len()) / 3;
        let splats = self.engine.splat_ws.scene.gaussians.len();
        let radial_nodes = self.engine.count_cutting_plane_radial_nodes(&self.config);
        vec![
            ("grid_resolution", self.engine.voxel_grid.width() as f64),
            ("iso_threshold", self.config.iso_threshold),
            ("mesh_triangles", tris as f64),
            ("active_splats", splats as f64),
            ("slice_contours", self.engine.slice_contours.len() as f64),
            ("radial_nodes", radial_nodes as f64),
            ("superposition_mix", self.config.superposition_mix),
        ]
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
            "3D Volumetric Field Studio ({}) evolves hydrogenic superpositions $\\Psi(\\mathbf{{r}}, t) = c_1(t) R_{{n_1 \\ell_1}}(r) Y_{{\\ell_1 m_1}}(\\theta,\\phi) + c_2(t) R_{{n_2 \\ell_2}}(r) Y_{{\\ell_2 m_2}}(\\theta,\\phi)$ and 3D Schnakenberg reaction-diffusion $\\partial_t u = D_u \\nabla^2_{{3D}} u + a - u + u^2 v$ across a ${}^3$ VoxelGrid using MarchingCubes and EWA Gaussian3D splatting.",
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
