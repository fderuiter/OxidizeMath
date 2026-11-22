//! Interactive Implicit Isosurface Workbench (`IsosurfaceTool`) with live 3D AST
//! equation evaluation, watertight slab/solid Marching Cubes extraction, and
//! takeaway binary `.stl` + `.obj` mesh export.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    expression_bar_presets, find_preset_by_expression, parse_validated_state, IsosurfacePresetSpec,
    IsosurfaceState, BUILTIN_ISOSURFACE_PRESETS,
};
use crate::accessibility::announce_live_status;
use crate::camera::Camera3D;
use crate::framework::{InteractiveTool, MathExpressionInputBar, ToolMetadata};
use crate::web_io::{spawn_save_as_file_dialog, PlatformInstant, SaveAsDialogConfig};
use crate::web_io_bytes::{spawn_save_as_bytes_dialog, SaveAsBytesDialogConfig};
use eframe::egui;
use math_commons::ast::CompiledExpr;
use math_explorer::applied::isosurface::{
    analyze_mesh_manifold, extract_isosurface, sample_watertight_grid, IsosurfaceMode,
    ManifoldStats, Mesh, WatertightGridConfig,
};
use oxidize_core::mesh::{export_mesh_to_binary_stl, export_mesh_to_obj_string};
use std::hash::{Hash, Hasher};
use std::sync::mpsc::{channel, Receiver};
use std::sync::OnceLock;

static DEFAULT_GYROID_MESH: OnceLock<(Mesh, ManifoldStats)> = OnceLock::new();

fn grid_config(
    half_extent: f64,
    resolution: usize,
    iso_level: f64,
    sheet_mode: bool,
    thickness: f64,
) -> WatertightGridConfig {
    let mode = if sheet_mode {
        IsosurfaceMode::Sheet { thickness }
    } else {
        IsosurfaceMode::Solid
    };
    WatertightGridConfig::cubic(half_extent, resolution, iso_level, mode)
}

fn build_preset_mesh(spec: &IsosurfacePresetSpec) -> (Mesh, ManifoldStats) {
    let expr = math_commons::ast::parse(spec.expression).expect("valid builtin preset expression");
    let compiled = CompiledExpr::compile_with_vars(&expr, &["x", "y", "z"]).expect("compile 3d");
    let cfg = grid_config(
        spec.half_extent,
        spec.resolution,
        spec.iso_level,
        spec.sheet_mode,
        spec.thickness,
    );
    let grid = sample_watertight_grid(&cfg, |x, y, z| compiled.eval_3d(x, y, z, &[])).unwrap();
    let mesh = extract_isosurface(&grid, 0.0).expect("extract builtin preset mesh");
    let stats = analyze_mesh_manifold(&mesh);
    (mesh, stats)
}

/// Interactive implicit surface and TPMS lattice generator with watertight `.stl` / `.obj` export.
pub struct IsosurfaceTool {
    pub(crate) input_bar: MathExpressionInputBar,
    pub(crate) half_extent: f64,
    pub(crate) iso_level: f64,
    pub(crate) thickness: f64,
    pub(crate) resolution: usize,
    pub(crate) sheet_mode: bool,
    pub(crate) show_wireframe: bool,
    pub(crate) camera: Camera3D,
    pub(crate) mesh: Mesh,
    pub(crate) stats: ManifoldStats,
    pub(crate) extraction_ms: f64,
    pub(crate) field_error: Option<String>,
    pub(crate) export_status: Option<(bool, String)>,
    pub(super) pending_announce: Option<String>,
    save_rx: Option<Receiver<Result<String, String>>>,
}

impl Default for IsosurfaceTool {
    fn default() -> Self {
        let spec = &BUILTIN_ISOSURFACE_PRESETS[0];
        let (mesh, stats) = DEFAULT_GYROID_MESH
            .get_or_init(|| build_preset_mesh(spec))
            .clone();
        Self {
            input_bar: MathExpressionInputBar::new(
                "f(x, y, z)",
                spec.expression,
                expression_bar_presets(),
            ),
            half_extent: spec.half_extent,
            iso_level: spec.iso_level,
            thickness: spec.thickness,
            resolution: spec.resolution,
            sheet_mode: spec.sheet_mode,
            show_wireframe: false,
            camera: Camera3D::new(0.55, 0.65, 1.0),
            mesh,
            stats,
            extraction_ms: 0.0,
            field_error: None,
            export_status: None,
            pending_announce: None,
            save_rx: None,
        }
    }
}

impl IsosurfaceTool {
    /// Applies one of the curated [`IsosurfacePresetSpec`] configurations and rebuilds the mesh.
    pub fn apply_preset_spec(&mut self, spec: &IsosurfacePresetSpec, announce: bool) {
        self.input_bar.set_expression(spec.expression);
        self.half_extent = spec.half_extent;
        self.iso_level = spec.iso_level;
        self.sheet_mode = spec.sheet_mode;
        self.thickness = spec.thickness;
        self.resolution = spec.resolution;
        self.export_status = None;
        self.rebuild_mesh(announce);
    }

    fn current_filename_stem(&self) -> &'static str {
        find_preset_by_expression(&self.input_bar.raw_text)
            .map_or("implicit_isosurface", |spec| spec.filename_stem)
    }

    /// Re-evaluates the 3D AST across the `(N + 2)^3` padded grid and extracts the watertight mesh.
    pub fn rebuild_mesh(&mut self, announce: bool) {
        let start = PlatformInstant::now();
        let Some(expr) = self.input_bar.expr().cloned() else {
            let msg = self
                .input_bar
                .error_message()
                .unwrap_or_else(|| "Invalid expression".to_string());
            self.record_extraction_failure(start, msg);
            return;
        };

        let compiled = match CompiledExpr::compile_with_vars(&expr, &["x", "y", "z"]) {
            Ok(c) => c,
            Err(err) => {
                self.record_extraction_failure(start, err.to_string());
                return;
            }
        };

        let cfg = grid_config(
            self.half_extent,
            self.resolution,
            self.iso_level,
            self.sheet_mode,
            self.thickness,
        );
        match sample_watertight_grid(&cfg, |x, y, z| compiled.eval_3d(x, y, z, &[])) {
            Ok(grid) => match extract_isosurface(&grid, 0.0) {
                Ok(mesh) => self.finish_mesh_update(start, mesh, announce),
                Err(err) => self.record_extraction_failure(start, err.to_string()),
            },
            Err(err) => self.record_extraction_failure(
                start,
                format!(
                    "{err}. Check for division by zero, domain singularities, or undefined variables (use x, y, z)."
                ),
            ),
        }
    }

    fn finish_mesh_update(&mut self, start: PlatformInstant, mesh: Mesh, announce: bool) {
        let stats = analyze_mesh_manifold(&mesh);
        self.extraction_ms = start.elapsed().as_secs_f64() * 1_000.0;
        self.field_error = (stats.triangle_count == 0).then(|| {
            "No surface intersects the current domain bounds and iso-level. Adjust Iso-Level (c) or Domain Extent (±L)."
                .to_string()
        });
        self.mesh = mesh;
        self.stats = stats;
        if announce {
            self.pending_announce = Some(if self.stats.triangle_count == 0 {
                "Isosurface empty within current bounds; 0 triangles generated.".to_string()
            } else {
                format!(
                    "Rebuilt watertight isosurface: {} triangles, {} open boundary edges in {:.1} milliseconds.",
                    self.stats.triangle_count, self.stats.open_boundary_edges, self.extraction_ms
                )
            });
        }
    }

    fn record_extraction_failure(&mut self, start: PlatformInstant, message: String) {
        self.mesh = Mesh::default();
        self.stats = ManifoldStats::default();
        self.extraction_ms = start.elapsed().as_secs_f64() * 1_000.0;
        self.field_error = Some(message);
    }

    /// Triggers a binary `.stl` takeaway save dialog or browser `Blob` download.
    pub fn trigger_stl_export(&mut self, ctx: &egui::Context) {
        if self.mesh.indices.is_empty() {
            self.report_export_error(ctx, "Cannot export STL: current mesh contains 0 triangles.");
            return;
        }
        match export_mesh_to_binary_stl(&self.mesh) {
            Ok(bytes) => {
                let default_name = format!("{}.stl", self.current_filename_stem());
                let byte_len = bytes.len();
                let (tx, rx) = channel();
                let cfg = SaveAsBytesDialogConfig {
                    filter_name: "Binary STL Solid (*.stl)",
                    extensions: &["stl"],
                    fallback_filename: "isosurface.stl",
                    write_error_prefix: "Failed to export binary STL",
                };
                spawn_save_as_bytes_dialog(tx, cfg, &default_name, bytes, move |_path, leaf| {
                    format!("Exported watertight binary STL {leaf} ({byte_len} bytes).")
                });
                self.save_rx = Some(rx);
                self.poll_export_channel(ctx);
            }
            Err(err) => self.report_export_error(ctx, &format!("STL serialization error: {err}")),
        }
    }

    /// Triggers an ASCII `.obj` takeaway save dialog or browser download.
    pub fn trigger_obj_export(&mut self, ctx: &egui::Context) {
        if self.mesh.indices.is_empty() {
            self.report_export_error(ctx, "Cannot export OBJ: current mesh contains 0 triangles.");
            return;
        }
        match export_mesh_to_obj_string(&self.mesh) {
            Ok(obj_text) => {
                let default_name = format!("{}.obj", self.current_filename_stem());
                let tri_count = self.stats.triangle_count;
                let (tx, rx) = channel();
                let cfg = SaveAsDialogConfig {
                    filter_name: "Wavefront OBJ Mesh (*.obj)",
                    extensions: &["obj"],
                    fallback_filename: "isosurface.obj",
                    write_error_prefix: "Failed to export OBJ mesh",
                };
                spawn_save_as_file_dialog(tx, cfg, &default_name, obj_text, move |_path, leaf| {
                    format!("Exported Wavefront OBJ {leaf} ({tri_count} triangles).")
                });
                self.save_rx = Some(rx);
                self.poll_export_channel(ctx);
            }
            Err(err) => self.report_export_error(ctx, &format!("OBJ serialization error: {err}")),
        }
    }

    fn report_export_error(&mut self, ctx: &egui::Context, msg: &str) {
        self.export_status = Some((false, msg.to_string()));
        announce_live_status(ctx, msg);
    }

    pub(super) fn poll_export_channel(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.save_rx.as_ref() else {
            return;
        };
        if let Ok(outcome) = rx.try_recv() {
            let (ok, msg) = match outcome {
                Ok(success_msg) => (true, success_msg),
                Err(err_msg) => (false, err_msg),
            };
            self.export_status = Some((ok, msg.clone()));
            announce_live_status(ctx, msg);
            self.save_rx = None;
        }
    }
}

impl InteractiveTool for IsosurfaceTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Implicit Isosurface Workbench"
    }

    fn domain(&self) -> &'static str {
        "isosurface"
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.half_extent.to_bits().hash(&mut h);
        self.iso_level.to_bits().hash(&mut h);
        self.thickness.to_bits().hash(&mut h);
        self.resolution.hash(&mut h);
        self.sheet_mode.hash(&mut h);
        self.show_wireframe.hash(&mut h);
        self.input_bar.raw_text.hash(&mut h);
        self.camera.pitch.to_bits().hash(&mut h);
        self.camera.yaw.to_bits().hash(&mut h);
        self.camera.zoom.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.half_extent.is_finite() && self.iso_level.is_finite() && self.thickness.is_finite()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&IsosurfaceState {
            half_extent: self.half_extent,
            iso_level: self.iso_level,
            thickness: self.thickness,
            resolution: self.resolution,
            sheet_mode: self.sheet_mode,
            show_wireframe: self.show_wireframe,
            expression: self.input_bar.raw_text.clone(),
            camera: Some(self.camera),
            pitch: None,
            yaw: None,
            zoom: None,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let (loaded, camera) = parse_validated_state(state)?;
        self.half_extent = loaded.half_extent;
        self.iso_level = loaded.iso_level;
        self.thickness = loaded.thickness;
        self.resolution = loaded.resolution;
        self.sheet_mode = loaded.sheet_mode;
        self.show_wireframe = loaded.show_wireframe;
        self.camera.apply_pose(camera);
        self.input_bar.set_expression(&loaded.expression);
        self.rebuild_mesh(false);
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.show_tool_ui(ui);
    }
}

inventory::submit! {
    ToolMetadata {
        id: "IsosurfaceTool",
        display_name: "Implicit Isosurface Workbench",
        domain: "isosurface",
        tags: &["isosurface", "marching cubes", "stl", "gyroid", "tpms", "sdf"],
        build: || Box::new(IsosurfaceTool::default()),
    }
}
