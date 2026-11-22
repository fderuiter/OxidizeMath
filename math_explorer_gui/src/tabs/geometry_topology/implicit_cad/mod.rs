//! Browser and desktop studio for implicit 3D equations, TPMS lattices, CSG solids,
//! manifold verification, and millimeter `.stl` / `.obj` downloads.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:isosurface_extraction]

/// Pre-allocated Marching Cubes sampling and shaded viewport projection.
pub mod mesher;
/// Starter gallery presets and scientific theory metadata.
pub mod presets;
#[cfg(test)]
mod tests;

use crate::framework::{Camera3D, InteractiveTool, Plot3DViewport};
use crate::presets::serialize_compact_preset_envelope;
use crate::url_state::{
    build_share_url_from_fragment, copy_share_url_to_clipboard, encode_share_fragment,
};
use crate::web_io::{SaveAsDialogConfig, download_mesh_bytes};
use eframe::egui::{self, Color32, RichText, Stroke};
use egui_plot::{PlotPoints, Polygon};
use math_commons::ast::parse_equation;
use mesher::{ImplicitCadParams, ImplicitCadWorkspace, MAX_GRID_RES, MIN_GRID_RES};
use oxidize_core::mesh::{export_mesh_to_obj_string, export_mesh_to_stl_bytes};
use presets::{GYROID_EQUATION, SCHWARZ_P_EQUATION, TORUS_BOX_CSG_EQUATION};

const fn default_true() -> bool {
    true
}

#[derive(serde::Serialize)]
struct ImplicitCadSaveState<'a> {
    equation: &'a str,
    box_size_mm: f64,
    shell_thickness_mm: f64,
    resolution: usize,
    seal_boundaries: bool,
    show_wireframe: bool,
    camera: Camera3D,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ImplicitCadLoadState {
    equation: String,
    box_size_mm: f64,
    shell_thickness_mm: f64,
    resolution: usize,
    #[serde(default = "default_true")]
    seal_boundaries: bool,
    #[serde(default)]
    show_wireframe: bool,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
}

/// Interactive Implicit CAD and TPMS Studio for printable millimeter solids.
pub struct ImplicitCadStudio {
    equation: String,
    box_size_mm: f64,
    shell_thickness_mm: f64,
    resolution: usize,
    seal_boundaries: bool,
    show_wireframe: bool,
    camera: Camera3D,
    share_feedback: Option<String>,
    workspace: ImplicitCadWorkspace,
}

impl Default for ImplicitCadStudio {
    fn default() -> Self {
        Self {
            equation: GYROID_EQUATION.to_string(),
            box_size_mm: 40.0,
            shell_thickness_mm: 1.6,
            resolution: 32,
            seal_boundaries: true,
            show_wireframe: false,
            camera: Camera3D::new(0.52, 0.55, 1.0),
            share_feedback: None,
            workspace: ImplicitCadWorkspace::default(),
        }
    }
}

impl ImplicitCadStudio {
    fn current_params(&self) -> ImplicitCadParams {
        ImplicitCadParams {
            equation: self.equation.clone(),
            box_size_mm: self.box_size_mm,
            shell_thickness_mm: self.shell_thickness_mm,
            resolution: self.resolution.clamp(MIN_GRID_RES, MAX_GRID_RES),
            seal_boundaries: self.seal_boundaries,
        }
    }

    /// Synchronizes the pre-allocated mesher workspace with current parameters.
    pub fn refresh_mesh(&mut self) {
        self.resolution = self.resolution.clamp(MIN_GRID_RES, MAX_GRID_RES);
        let params = self.current_params();
        self.workspace.ensure_up_to_date(&params);
    }

    /// Encodes a bookmarkable `#tab=geometry_topology&tool=ImplicitCadStudio&state=...` URL.
    pub fn encode_share_url(&self) -> Result<String, String> {
        let raw_state = self
            .save_state()
            .ok_or_else(|| "Failed to serialize studio state".to_string())?;
        let state_val: serde_json::Value =
            serde_json::from_str(&raw_state).map_err(|e| e.to_string())?;
        let envelope_json =
            serialize_compact_preset_envelope(self.name(), self.domain(), state_val)?;
        let fragment = encode_share_fragment(
            "geometry_topology",
            Some("ImplicitCadStudio"),
            Some(&envelope_json),
        )?;
        Ok(build_share_url_from_fragment(&fragment))
    }

    fn apply_quick_template(&mut self, eq: &str, box_mm: f64, shell_mm: f64, res: usize) {
        self.equation = eq.to_string();
        self.box_size_mm = box_mm;
        self.shell_thickness_mm = shell_mm;
        self.resolution = res.clamp(MIN_GRID_RES, MAX_GRID_RES);
        self.seal_boundaries = true;
        self.share_feedback = None;
    }

    fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Implicit CAD & TPMS Studio");
        ui.small("Zero-install millimeter lattice & boolean CSG generator");
        ui.separator();

        ui.label(RichText::new("Starter Templates").strong());
        ui.horizontal_wrapped(|ui| {
            if ui.button("🧬 Gyroid").clicked() {
                self.apply_quick_template(GYROID_EQUATION, 40.0, 1.6, 36);
            }
            if ui.button("🧊 Schwarz P").clicked() {
                self.apply_quick_template(SCHWARZ_P_EQUATION, 40.0, 2.0, 36);
            }
            if ui.button("⚙️ Torus-Box CSG").clicked() {
                self.apply_quick_template(TORUS_BOX_CSG_EQUATION, 40.0, 0.0, 40);
            }
        });

        ui.separator();
        ui.label(RichText::new("Implicit Field f(x, y, z)").strong());
        ui.small("Supports x, y, z (mm), pi, sin, cos, sqrt, abs, min(a, b), max(a, b):");
        ui.add(
            egui::TextEdit::multiline(&mut self.equation)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .code_editor(),
        );
        if let Some(err) = &self.workspace.last_error {
            ui.colored_label(Color32::from_rgb(235, 95, 95), err);
        }

        ui.separator();
        ui.label(RichText::new("Millimeter Domain & Shell").strong());
        ui.add(
            egui::Slider::new(&mut self.box_size_mm, 4.0..=200.0)
                .suffix(" mm")
                .text("Box Size"),
        );
        ui.add(
            egui::Slider::new(&mut self.shell_thickness_mm, 0.0..=20.0)
                .suffix(" mm")
                .text("Wall w (|f|-w/2)"),
        );
        ui.add(
            egui::Slider::new(&mut self.resolution, MIN_GRID_RES..=MAX_GRID_RES)
                .text("Grid Resolution"),
        );
        ui.checkbox(
            &mut self.seal_boundaries,
            "Seal clipped box faces (watertight solid)",
        );
        ui.checkbox(&mut self.show_wireframe, "Show triangle edges");

        ui.separator();
        self.camera.ui(ui);

        ui.separator();
        self.render_diagnostics_summary(ui);
        ui.add_space(4.0);
        self.render_export_buttons(ui);
    }

    fn render_diagnostics_summary(&self, ui: &mut egui::Ui) {
        let diag = self.workspace.diagnostics;
        let (badge_text, badge_color) = if diag.is_watertight() {
            ("● Watertight Manifold Solid", Color32::from_rgb(56, 196, 118))
        } else if diag.triangle_count == 0 {
            ("○ Empty Isosurface (0 facets)", Color32::from_rgb(230, 175, 65))
        } else {
            ("▲ Open Surface Shell", Color32::from_rgb(230, 175, 65))
        };
        ui.colored_label(badge_color, RichText::new(badge_text).strong());
        ui.small(format!(
            "Facets: {} | Vertices: {} | Open edges: {} | NaN normals: {}",
            diag.triangle_count, diag.vertex_count, diag.open_edges, diag.nan_normals
        ));
        ui.small(format!(
            "Bounds: {:.2} × {:.2} × {:.2} mm",
            diag.dimensions_mm.x, diag.dimensions_mm.y, diag.dimensions_mm.z
        ));
        ui.small(format!(
            "Area: {:.1} mm² | Volume: {:.1} mm³",
            diag.area_mm2, diag.signed_volume_mm3
        ));
    }

    fn render_export_buttons(&mut self, ui: &mut egui::Ui) {
        let has_mesh = self.workspace.diagnostics.triangle_count > 0;
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(has_mesh, egui::Button::new("⬇ Binary .STL (mm)"))
                .clicked()
            {
                if let Ok(bytes) = export_mesh_to_stl_bytes(&self.workspace.mesh) {
                    download_mesh_bytes(
                        SaveAsDialogConfig {
                            filter_name: "Binary STL Mesh (*.stl)",
                            extensions: &["stl"],
                            fallback_filename: "implicit_solid.stl",
                            write_error_prefix: "Failed to write STL file",
                        },
                        "implicit_solid_mm.stl",
                        bytes,
                    );
                }
            }
            if ui
                .add_enabled(has_mesh, egui::Button::new("⬇ Wavefront .OBJ (mm)"))
                .clicked()
            {
                if let Ok(obj) = export_mesh_to_obj_string(&self.workspace.mesh) {
                    download_mesh_bytes(
                        SaveAsDialogConfig {
                            filter_name: "Wavefront OBJ Mesh (*.obj)",
                            extensions: &["obj"],
                            fallback_filename: "implicit_solid.obj",
                            write_error_prefix: "Failed to write OBJ file",
                        },
                        "implicit_solid_mm.obj",
                        obj.into_bytes(),
                    );
                }
            }
            if ui.button("🔗 Copy Share Link").clicked() {
                match self.encode_share_url() {
                    Ok(url) => {
                        copy_share_url_to_clipboard(ui.ctx(), url, |_| {});
                        self.share_feedback = Some("Copied bookmarkable URL!".to_string());
                    }
                    Err(e) => self.share_feedback = Some(e),
                }
            }
        });
        if let Some(msg) = &self.share_feedback {
            ui.small(msg);
        }
    }
}

impl InteractiveTool for ImplicitCadStudio {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Implicit CAD & TPMS Studio"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&ImplicitCadSaveState {
            equation: &self.equation,
            box_size_mm: self.box_size_mm,
            shell_thickness_mm: self.shell_thickness_mm,
            resolution: self.resolution.clamp(MIN_GRID_RES, MAX_GRID_RES),
            seal_boundaries: self.seal_boundaries,
            show_wireframe: self.show_wireframe,
            camera: self.camera,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: ImplicitCadLoadState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid ImplicitCadStudio state: {e}"))?;
        if loaded.equation.trim().is_empty() || loaded.equation.len() > 1024 {
            return Err("Equation must be non-empty and at most 1024 bytes".to_string());
        }
        parse_equation(&loaded.equation)
            .map_err(|e| format!("Invalid implicit equation: {}", e.user_message()))?;
        if !loaded.box_size_mm.is_finite() || !(4.0..=200.0).contains(&loaded.box_size_mm) {
            return Err("box_size_mm must be between 4.0 and 200.0 mm".to_string());
        }
        if !loaded.shell_thickness_mm.is_finite()
            || !(0.0..=50.0).contains(&loaded.shell_thickness_mm)
        {
            return Err("shell_thickness_mm must be between 0.0 and 50.0 mm".to_string());
        }
        if !(MIN_GRID_RES..=MAX_GRID_RES).contains(&loaded.resolution) {
            return Err(format!(
                "resolution must be within {MIN_GRID_RES}..={MAX_GRID_RES}"
            ));
        }
        let camera = Camera3D::from_preset_fields(
            loaded.camera,
            loaded.pitch,
            loaded.yaw,
            loaded.zoom,
            Some(Camera3D::new(0.52, 0.55, 1.0)),
        )?;
        self.equation = loaded.equation;
        self.box_size_mm = loaded.box_size_mm;
        self.shell_thickness_mm = loaded.shell_thickness_mm;
        self.resolution = loaded.resolution;
        self.seal_boundaries = loaded.seal_boundaries;
        self.show_wireframe = loaded.show_wireframe;
        self.camera.apply_pose(camera);
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.refresh_mesh();
        let panel_id = crate::framework::scoped_panel_id(ui, "implicit_cad_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .default_width(310.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.render_controls_panel(ui));
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let cull = self.seal_boundaries && self.workspace.diagnostics.is_watertight();
            let box_mm = self.box_size_mm;
            let wireframe = self.show_wireframe;
            let cam_snapshot = self.camera;
            let (facets, _front_total) =
                self.workspace
                    .project_facets(&cam_snapshot, box_mm, cull);

            Plot3DViewport::new("implicit_cad_viewport", 1.85).show(
                ui,
                &mut self.camera,
                |plot_ui| {
                    for facet in facets {
                        let edge_stroke = if wireframe {
                            Stroke::new(0.65_f32, Color32::from_rgba_unmultiplied(15, 25, 38, 190))
                        } else {
                            Stroke::new(0.35_f32, facet.fill)
                        };
                        plot_ui.polygon(
                            Polygon::new("", PlotPoints::new(facet.pts.to_vec()))
                                .fill_color(facet.fill)
                                .stroke(edge_stroke),
                        );
                    }
                },
            );
        });
    }
}
