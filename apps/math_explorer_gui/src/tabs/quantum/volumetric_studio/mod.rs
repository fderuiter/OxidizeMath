#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! `VolumetricFieldStudioTool`: unified 3D implicit CAD, signed AST fields, hydrogenic quantum,
//! and 3D Turing studio rendering `MarchingCubes` watertight solids and `Gaussian3D` radiance clouds.
// [cite:quantum_mechanics]
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]
// [cite:favorite_child]

pub mod ast_field;
pub mod compute_wgsl;
pub mod engine;
mod export_3d;
mod hud;
pub mod implicit;
pub mod presets;
mod render;
pub mod state;
mod ui;
pub mod webgpu_runner;

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use crate::web_io::PlatformInstant;
use ast_field::CustomAstFieldController;
use eframe::egui;
use engine::VolumetricEngine;
use implicit::ImplicitFormulaUi;
use oxidize_core::mesh::{ColoredMesh, Mesh};
use scientific_metadata::theory::TheoryDescribable;
use state::{FieldSourceMode, VolumetricRenderMode, VolumetricStudioConfig};
use std::hash::{Hash, Hasher};
use webgpu_runner::{ComputeBackend, WebGpuVolumetricRunner};

/// Interactive 3D volumetric field and implicit CAD studio uniting $f(x, y, z) = c$ solids,
/// signed 3D AST formulas, quantum eigenstates, 3D Turing PDEs, watertight `.stl`/`.ply`/`.obj`
/// export, and `Gaussian3D` splats.
pub struct VolumetricFieldStudioTool {
    /// Validated serializable studio configuration.
    pub config: VolumetricStudioConfig,
    /// Pre-allocated zero-allocation 3D PDE, implicit, AST, isosurface, and splatting workspace.
    pub engine: VolumetricEngine,
    /// `128^3` WebGPU `@compute` / off-main-thread Web Worker volumetric pipeline.
    pub webgpu_runner: WebGpuVolumetricRunner,
    /// Formula bar & synthesized `ParameterSliderBank` for `FieldSourceMode::CustomImplicitAst`.
    pub ast_controller: CustomAstFieldController,
    pub(super) implicit_ui: ImplicitFormulaUi,
    pub(super) splat_texture: Option<egui::TextureHandle>,
    pub(super) splat_dirty: bool,
    pub(super) last_step_tick: PlatformInstant,
}

impl Default for VolumetricFieldStudioTool {
    fn default() -> Self {
        let config = VolumetricStudioConfig::default();
        let engine = VolumetricEngine::new(&config);
        let mut ast_controller = CustomAstFieldController::default();
        ast_controller.load_from_config(&config.custom_expr, &config.custom_params);
        let implicit_ui = ImplicitFormulaUi::new(&config);
        Self {
            config,
            engine,
            webgpu_runner: WebGpuVolumetricRunner::default(),
            ast_controller,
            implicit_ui,
            splat_texture: None,
            splat_dirty: true,
            last_step_tick: PlatformInstant::now(),
        }
    }
}

impl VolumetricFieldStudioTool {
    /// Active cubic voxel grid side length (`16..=48` on CPU, `16..=128` on WebGPU Compute).
    #[must_use]
    pub fn active_voxel_dimension(&self) -> usize {
        if self.config.compute_backend == ComputeBackend::WebGpuCompute {
            self.webgpu_runner
                .high_res_grid
                .as_ref()
                .map_or(self.config.grid_resolution, |g| g.width())
        } else {
            self.engine.voxel_grid.width()
        }
    }

    /// Advances the active 3D simulation synchronously and re-extracts surfaces/splats.
    pub fn step_zero_alloc(&mut self) {
        self.run_step_internal(true, false, false);
    }

    pub(super) fn step_playback_frame(&mut self) {
        self.run_step_internal(true, false, true);
    }

    /// Re-extracts `MarchingCubes` and `Gaussian3D` outputs at the current `config` without advancing time.
    pub fn refresh_zero_alloc(&mut self) {
        self.refresh_with_drag_preview(false);
    }

    /// Re-extracts `MarchingCubes`, `Gaussian3D`, and slice contours, using a `20^3` preview grid
    /// when `drag_preview` is active in `CustomImplicitAst` mode and full `grid_resolution^3` on release.
    pub fn refresh_with_drag_preview(&mut self, drag_preview: bool) {
        if self.config.source == FieldSourceMode::CustomImplicitAst {
            if let Some(compiled) = self.ast_controller.bar.active_compiled() {
                self.engine
                    .ast_kernel
                    .sync_from_bar(compiled, self.ast_controller.bar.param_slice());
            }
        }
        self.run_step_internal(false, drag_preview, false);
    }

    fn run_step_internal(&mut self, advance_time: bool, drag_preview: bool, async_playback: bool) {
        if self.config.compute_backend == ComputeBackend::WebGpuCompute {
            let preview = drag_preview && self.config.source == FieldSourceMode::CustomImplicitAst;
            self.engine.preview_drag_active = preview;
            self.webgpu_runner.evaluate_and_sync(
                &self.config,
                advance_time,
                preview,
                async_playback,
                &mut self.engine,
            );
            self.engine.extract_current_grid_zero_alloc(&self.config);
        } else {
            let t0 = PlatformInstant::now();
            if !advance_time {
                self.engine
                    .sync_structural_config_with_preview(&self.config, drag_preview);
            }
            self.engine
                .step_and_extract_zero_alloc(&self.config, advance_time);
            self.webgpu_runner.last_step_ms = (t0.elapsed().as_secs_f64() * 1_000.0).max(0.05);
        }
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
    #[cfg(test)]
    pub fn export_obj_string(&self) -> Result<String, String> {
        self.engine.export_obj_string(&self.config)
    }

    fn sync_ui_controllers_from_config(&mut self) {
        self.ast_controller
            .load_from_config(&self.config.custom_expr, &self.config.custom_params);
        self.implicit_ui.sync_from_config(&self.config);
        self.engine
            .ast_kernel
            .sync_from_config_expr(&self.config.custom_expr, &self.config.custom_params);
    }
}

impl FiniteStateGuard for VolumetricFieldStudioTool {
    fn is_finite_state(&self) -> bool {
        !self.engine.diverged
            && self.config.validate().is_ok()
            && self.engine.voxel_grid.data().iter().all(|v| v.is_finite())
            && self.webgpu_runner.is_finite()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.engine.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.engine.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    fn reset_parameters(&mut self) {
        self.config = VolumetricStudioConfig::default();
        self.sync_ui_controllers_from_config();
        self.webgpu_runner.reset();
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
        let base = if self.config.render_mode == VolumetricRenderMode::GaussianRadianceCloud {
            ExportCapabilities::plot().with_3d_splat_ply()
        } else {
            ExportCapabilities::plot().with_3d_solid()
        };
        base.with_compute_shader()
    }

    fn export_compute_shader_wgsl(&self) -> Result<String, String> {
        compute_wgsl::build_volumetric_compute_wgsl(&self.config)
    }

    fn export_mesh_3d(&self) -> Option<Result<Mesh, String>> {
        self.config
            .render_mode
            .uses_mesh()
            .then(|| self.engine.export_solid_mesh_3d(&self.config))
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<ColoredMesh, String>> {
        self.config
            .render_mode
            .uses_mesh()
            .then(|| self.build_colored_solid_mesh_3d())
    }

    fn export_splat_ply_3d(&self) -> Option<Result<Vec<u8>, String>> {
        (self.config.render_mode == VolumetricRenderMode::GaussianRadianceCloud)
            .then(|| self.export_splat_ply_bytes())
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.config.source.hash(&mut h);
        self.config.render_mode.hash(&mut h);
        self.config.compute_backend.hash(&mut h);
        self.config.grid_resolution.hash(&mut h);
        (self.config.n1, self.config.l1, self.config.m1).hash(&mut h);
        (self.config.n2, self.config.l2, self.config.m2).hash(&mut h);
        hash_f64_slice(
            &mut h,
            &[
                self.config.superposition_mix,
                self.config.relative_phase,
                self.config.quantum_dt,
                self.config.iso_threshold,
                self.config.splat_scale,
                self.config.turing_a,
                self.config.turing_b,
                self.config.turing_du,
                self.config.turing_dv,
                self.config.turing_dt,
                self.config.implicit_level,
                self.config.wall_thickness,
                self.config.export_size_mm,
                self.config.cut_offset,
            ],
        );
        self.config.observable.hash(&mut h);
        hash_formula_maps_and_bounds(&mut h, &self.config);
        self.config.cut_enabled.hash(&mut h);
        self.config.cut_axis.hash(&mut h);
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
        self.sync_ui_controllers_from_config();
        self.engine.diverged = false;
        self.engine.diagnostic_report = None;
        self.refresh_zero_alloc();
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        self.config.clamp_in_place();
        self.sync_ui_controllers_from_config();
        self.engine.diverged = false;
        self.engine.diagnostic_report = None;
        self.refresh_zero_alloc();
        true
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let tris = (self.engine.pos_mesh.indices.len() + self.engine.neg_mesh.indices.len()) / 3;
        let splats = self.engine.splat_ws.scene.gaussians.len();
        let radial_nodes = self.engine.count_cutting_plane_radial_nodes(&self.config);
        vec![
            ("grid_resolution", self.active_voxel_dimension() as f64),
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

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
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
        map.insert(
            "implicit_expr".into(),
            "Implicit 3D equation f(x, y, z) = c closed via half-space (f <= c) or thickened wall (|f - c| <= w/2) into watertight STL/PLY/OBJ solids.".into(),
        );
        map
    }
}

fn hash_f64_slice<H: Hasher>(h: &mut H, vals: &[f64]) {
    for &v in vals {
        v.to_bits().hash(h);
    }
}

fn hash_formula_maps_and_bounds<H: Hasher>(h: &mut H, cfg: &VolumetricStudioConfig) {
    cfg.custom_expr.hash(h);
    for (k, v) in &cfg.custom_params {
        k.hash(h);
        v.to_bits().hash(h);
    }
    cfg.implicit_expr.hash(h);
    cfg.implicit_closure.hash(h);
    for (k, v) in &cfg.implicit_params {
        k.hash(h);
        v.to_bits().hash(h);
    }
    for bounds in [cfg.x_bounds, cfg.y_bounds, cfg.z_bounds] {
        bounds[0].to_bits().hash(h);
        bounds[1].to_bits().hash(h);
    }
}

inventory::submit! {
    ToolMetadata {
        id: "VolumetricFieldStudioTool",
        display_name: "Volumetric Field Studio",
        domain: "quantum",
        tags: &[
            "volumetric",
            "3d",
            "orbital",
            "marching-cubes",
            "gaussian-splatting",
            "turing",
            "isosurface",
            "implicit",
            "cad",
            "stl",
            "ply",
            "gyroid",
            "tpms",
            "schwarz-p",
            "csg",
        ],
        build: || Box::new(VolumetricFieldStudioTool::default()),
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
