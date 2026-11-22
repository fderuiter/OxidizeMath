//! Interactive Implicit Isosurface Workbench (`IsosurfaceTool`) with live 3D AST
//! equation evaluation, watertight slab/solid Marching Cubes extraction, and
//! takeaway binary `.stl` + `.obj` mesh export.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    expression_bar_presets, find_preset_by_expression, parse_validated_state, IsosurfacePresetSpec,
    IsosurfaceState, BUILTIN_ISOSURFACE_PRESETS, MAX_GUI_RESOLUTION, MIN_GUI_RESOLUTION,
};
use super::render::{draw_isosurface_scene, project_mesh_facets};
use crate::accessibility::announce_live_status;
use crate::app::SemanticPalette;
use crate::camera::{Camera3D, Plot3DViewport};
use crate::framework::{
    scoped_panel_id, InteractiveTool, MathExpressionInputBar, PaneSidePanel, ToolMetadata,
};
use crate::reflective_ui::render_copyable_metric;
use crate::web_io::{spawn_save_as_file_dialog, PlatformInstant, SaveAsDialogConfig};
use crate::web_io_bytes::{spawn_save_as_bytes_dialog, SaveAsBytesDialogConfig};
use eframe::egui;
use math_explorer::applied::isosurface::{
    analyze_mesh_manifold, extract_isosurface, sample_watertight_grid, IsosurfaceMode,
    ManifoldStats, Mesh, WatertightGridConfig,
};
use oxidize_core::mesh::{export_mesh_to_binary_stl, export_mesh_to_obj_string};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver};

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
    pending_announce: Option<String>,
    save_rx: Option<Receiver<Result<String, String>>>,
}

impl Default for IsosurfaceTool {
    fn default() -> Self {
        let spec = &BUILTIN_ISOSURFACE_PRESETS[0];
        let mut tool = Self {
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
            mesh: Mesh::default(),
            stats: ManifoldStats::default(),
            extraction_ms: 0.0,
            field_error: None,
            export_status: None,
            pending_announce: None,
            save_rx: None,
        };
        tool.rebuild_mesh(false);
        tool
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

        let mode = if self.sheet_mode {
            IsosurfaceMode::Sheet {
                thickness: self.thickness,
            }
        } else {
            IsosurfaceMode::Solid
        };
        let cfg =
            WatertightGridConfig::cubic(self.half_extent, self.resolution, self.iso_level, mode);
        let params = HashMap::new();

        match sample_watertight_grid(&cfg, |x, y, z| expr.eval_3d(x, y, z, &params)) {
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

    fn poll_export_channel(&mut self, ctx: &egui::Context) {
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

    fn render_preset_and_equation_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let current_title = find_preset_by_expression(&self.input_bar.raw_text)
            .map_or("Custom Equation", |s| s.title);
        ui.label("Surface Preset:");
        egui::ComboBox::from_id_salt(ui.id().with("isosurface_preset_combo"))
            .selected_text(current_title)
            .width(ui.available_width() - 8.0)
            .show_ui(ui, |ui| {
                for spec in BUILTIN_ISOSURFACE_PRESETS {
                    if ui
                        .selectable_label(current_title == spec.title, spec.title)
                        .clicked()
                    {
                        self.apply_preset_spec(spec, true);
                    }
                }
            });
        let hint = find_preset_by_expression(&self.input_bar.raw_text).map_or(
            "Variables: x, y, z, pi, e, tau | SDF functions: min(a,b), max(a,b), atan2(y,x)",
            |spec| spec.summary,
        );
        ui.small(hint);
        ui.add_space(4.0);
        self.input_bar.ui(ui)
    }

    fn render_sampling_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.label("Offsetting Mode:");
        ui.horizontal(|ui| {
            dirty |= ui
                .radio_value(&mut self.sheet_mode, true, "Thickened Sheet (TPMS)")
                .changed();
            dirty |= ui
                .radio_value(&mut self.sheet_mode, false, "Solid Sublevel (f <= c)")
                .changed();
        });
        dirty |= ui
            .add(egui::Slider::new(&mut self.iso_level, -4.0..=4.0).text("Iso-Level (c)"))
            .changed();
        if self.sheet_mode {
            dirty |= ui
                .add(egui::Slider::new(&mut self.thickness, 0.08..=2.0).text("Slab Thickness (t)"))
                .changed();
        }
        dirty |= ui
            .add(egui::Slider::new(&mut self.half_extent, 0.5..=8.0).text("Domain Extent (±L)"))
            .changed();
        dirty |= ui
            .add(
                egui::Slider::new(
                    &mut self.resolution,
                    MIN_GUI_RESOLUTION..=MAX_GUI_RESOLUTION,
                )
                .text("Interior Grid (N³)"),
            )
            .changed();
        ui.checkbox(&mut self.show_wireframe, "Show Triangle Wireframe");
        dirty
    }

    fn render_telemetry_and_export_controls(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Manifold & Takeaway Export");
        render_copyable_metric(ui, "Triangles", &self.stats.triangle_count.to_string());
        render_copyable_metric(ui, "Vertices", &self.stats.vertex_count.to_string());
        render_copyable_metric(
            ui,
            "Open Half-Edges",
            &self.stats.open_boundary_edges.to_string(),
        );
        render_copyable_metric(
            ui,
            "Extraction Latency",
            &format!("{:.2} ms", self.extraction_ms),
        );

        let (badge_color, badge_text) = if self.stats.is_watertight_two_manifold() {
            (palette.success, "✓ Watertight 2-Manifold (0 open edges)")
        } else if self.stats.triangle_count == 0 {
            (palette.warning, "⚠ Empty volume (0 triangles)")
        } else {
            (palette.error, "⚠ Non-manifold mesh detected")
        };
        ui.colored_label(badge_color, badge_text);

        ui.add_space(6.0);
        ui.add_enabled_ui(self.stats.triangle_count > 0, |ui| {
            ui.horizontal(|ui| {
                if ui.button("⬇ Download .STL").clicked() {
                    self.trigger_stl_export(ui.ctx());
                }
                if ui.button("⬇ Download .OBJ").clicked() {
                    self.trigger_obj_export(ui.ctx());
                }
            });
        });

        if let Some((ok, ref msg)) = self.export_status {
            ui.colored_label(if ok { palette.success } else { palette.error }, msg);
        }
    }

    fn render_viewport(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let dark_mode = ui.visuals().dark_mode;
        if let Some(ref err) = self.field_error {
            egui::Frame::group(ui.style())
                .fill(palette.error_banner_bg)
                .stroke(egui::Stroke::new(1.0_f32, palette.error_banner_stroke))
                .show(ui, |ui| {
                    ui.colored_label(palette.warning, format!("ℹ {err}"));
                });
            ui.add_space(4.0);
        }

        let facets = project_mesh_facets(
            &self.mesh,
            &self.camera,
            &palette,
            dark_mode,
            self.show_wireframe,
        );
        let viewport_extent = (self.half_extent * 1.45).max(1.0);
        let half_extent = self.half_extent;
        let cam_snapshot = self.camera;

        Plot3DViewport::new("isosurface_3d_viewport", viewport_extent).show(
            ui,
            &mut self.camera,
            |plot_ui| {
                draw_isosurface_scene(plot_ui, &facets, half_extent, &cam_snapshot, &palette);
            },
        );
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
        self.poll_export_channel(ui.ctx());
        if let Some(msg) = self.pending_announce.take() {
            announce_live_status(ui.ctx(), msg);
        }

        let panel_id = scoped_panel_id(ui, "isosurface_controls");
        let mut needs_rebuild = false;
        PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Implicit Isosurface Workbench");
                ui.separator();
                needs_rebuild |= self.render_preset_and_equation_controls(ui);
                ui.separator();
                needs_rebuild |= self.render_sampling_controls(ui);
                ui.separator();
                egui::CollapsingHeader::new("Camera 3D")
                    .default_open(false)
                    .show(ui, |ui| self.camera.ui(ui));
                ui.separator();
                self.render_telemetry_and_export_controls(ui);
            });
        });

        if needs_rebuild {
            self.rebuild_mesh(true);
            if let Some(msg) = self.pending_announce.take() {
                announce_live_status(ui.ctx(), msg);
            }
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_viewport(ui);
        });
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
