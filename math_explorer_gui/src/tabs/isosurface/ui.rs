//! Sidebar controls, manifold telemetry panel, and 3D viewport rendering for [`super::tool::IsosurfaceTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    find_preset_by_expression, BUILTIN_ISOSURFACE_PRESETS, MAX_GUI_RESOLUTION, MIN_GUI_RESOLUTION,
};
use super::render::{draw_isosurface_scene, project_mesh_facets};
use super::tool::IsosurfaceTool;
use crate::accessibility::announce_live_status;
use crate::app::SemanticPalette;
use crate::camera::{Plot3DViewport, CAMERA_3D_PERSPECTIVE_DISTANCE};
use crate::framework::{scoped_panel_id, PaneSidePanel};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;

impl IsosurfaceTool {
    pub(super) fn show_tool_ui(&mut self, ui: &mut egui::Ui) {
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

        self.camera = self.camera.with_perspective_distance(
            (self.half_extent * 3.2).max(CAMERA_3D_PERSPECTIVE_DISTANCE),
        );
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
        let mesh_ref = &self.mesh;

        Plot3DViewport::new("isosurface_3d_viewport", viewport_extent).show(
            ui,
            &mut self.camera,
            |plot_ui| {
                draw_isosurface_scene(
                    plot_ui,
                    mesh_ref,
                    &facets,
                    half_extent,
                    &cam_snapshot,
                    &palette,
                );
            },
        );
    }
}
