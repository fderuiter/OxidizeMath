#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ParametricSurfaceKind, SurfaceViewer};
use crate::framework::Plot3DViewport;
use eframe::egui;

impl SurfaceViewer {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "surface_viewer_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Surface Viewer");
            ui.separator();

            ui.label("Select Surface:");
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::Sphere,
                    "Sphere",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::Torus,
                    "Torus",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::KleinBottle,
                    "Klein Bottle",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::CustomHeightfield,
                    "Custom z=f(x,y)",
                );
                ui.radio_value(
                    &mut self.surface_type,
                    ParametricSurfaceKind::CustomParametric,
                    "Custom r(u,v)",
                );
            });

            ui.separator();
            self.draw_parameter_controls(ui);

            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));

            ui.collapsing("Resolution", |ui| {
                let max_res = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
                self.u_resolution = self.u_resolution.clamp(10, max_res);
                self.v_resolution = self.v_resolution.clamp(10, max_res);
                ui.add(egui::Slider::new(&mut self.u_resolution, 10..=max_res).text("Grid Size U"));
                ui.add(egui::Slider::new(&mut self.v_resolution, 10..=max_res).text("Grid Size V"));
            });
        });
    }

    fn draw_parameter_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Parameters")
            .default_open(matches!(
                self.surface_type,
                ParametricSurfaceKind::CustomHeightfield | ParametricSurfaceKind::CustomParametric
            ))
            .show(ui, |ui| match self.surface_type {
                ParametricSurfaceKind::Sphere => {
                    ui.add(egui::Slider::new(&mut self.sphere_radius, 0.1..=5.0).text("Radius"));
                }
                ParametricSurfaceKind::Torus => {
                    ui.add(
                        egui::Slider::new(&mut self.torus_major_radius, 1.0..=5.0)
                            .text("Major Radius (R)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.torus_minor_radius, 0.1..=2.0)
                            .text("Minor Radius (r)"),
                    );
                }
                ParametricSurfaceKind::KleinBottle => {
                    ui.add(egui::Slider::new(&mut self.klein_radius, 0.5..=5.0).text("Radius"));
                    self.draw_wall_thickness_slider(ui);
                }
                ParametricSurfaceKind::CustomHeightfield
                | ParametricSurfaceKind::CustomParametric => {
                    self.custom_sandbox.ui_for_type(ui, self.surface_type);
                    self.draw_wall_thickness_slider(ui);
                }
            });
    }

    fn draw_wall_thickness_slider(&mut self, ui: &mut egui::Ui) {
        use crate::tabs::geometry_topology::export_utils::{
            MAX_WALL_THICKNESS, MIN_WALL_THICKNESS,
        };
        ui.add(
            egui::Slider::new(
                &mut self.wall_thickness,
                MIN_WALL_THICKNESS..=MAX_WALL_THICKNESS,
            )
            .text("Shell Wall Thickness"),
        );
    }

    pub(super) fn draw_viewport(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            self.populate_scene(palette.series_color(0));
            let mut camera = self.camera;
            Plot3DViewport::new("surface_plot", 4.0).show(ui, &mut camera, |plot_ui| {
                self.scene_builder.finish(plot_ui);
            });
            self.camera = camera;
        });
    }
}
