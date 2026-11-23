//! `Plot3DViewport` multi-sheet Riemann surface renderer, 2D Schwarz-Christoffel / Cauchy
//! companion plot, and export primitives for [`super::RiemannSurfaceStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CompanionInspectorMode, RiemannSurfaceStudio};
use crate::app::SemanticPalette;
use crate::camera::{Plot3DViewport, Scene3DBuilder, CAMERA_3D_HELPER_TEXT};
use eframe::egui;
use egui_plot::{ExportPrimitive, Line, LineStyle, MarkerShape, PlotPoints, Points, Polygon};
use nalgebra::Point3;

type ProjectedSheetQuad = (f64, [[f64; 2]; 4], egui::Color32);

fn sheet_scalar_color(val: f64, range: (f64, f64), alpha: u8) -> egui::Color32 {
    let span = (range.1 - range.0).abs().max(1e-6);
    let t = ((val - range.0) / span).clamp(0.0, 1.0) as f32;
    let r = (36.0 + 205.0 * t).round() as u8;
    let g = (110.0 + 85.0 * (1.0 - (2.0 * t - 1.0).abs())).round() as u8;
    let b = (220.0 - 175.0 * t).round() as u8;
    egui::Color32::from_rgba_unmultiplied(r, g, b, alpha)
}

impl RiemannSurfaceStudio {
    pub(super) fn render_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let avail = ui.available_size();
            if avail.x >= 680.0 {
                ui.columns(2, |cols| {
                    self.render_3d_riemann_viewport(&mut cols[0]);
                    self.render_2d_companion_plot(&mut cols[1]);
                });
            } else {
                let half_h = (avail.y * 0.52).max(180.0);
                ui.allocate_ui(egui::vec2(avail.x, half_h), |ui| {
                    self.render_3d_riemann_viewport(ui);
                });
                self.render_2d_companion_plot(ui);
            }
        });
    }

    fn build_riemann_3d_scene(
        &self,
        camera: crate::camera::Camera3D,
        dark: bool,
        palette: &SemanticPalette,
    ) -> (Scene3DBuilder, Vec<ProjectedSheetQuad>) {
        let mut scene = Scene3DBuilder::new(camera);
        scene.add_axes(2.2, dark);
        let mut sorted_quads = Vec::with_capacity(self.sheet_quads.len());
        let wire_stroke = egui::Stroke::new(
            0.75_f32,
            egui::Color32::from_gray(if dark { 85 } else { 150 }),
        );

        for q in &self.sheet_quads {
            let mut proj = [[0.0_f64; 2]; 4];
            let mut vz_sum = 0.0_f64;
            for (k, &w) in q.corners.iter().enumerate() {
                let (p2, vz) = camera.project_with_depth(&w);
                proj[k] = p2;
                vz_sum += vz;
            }
            sorted_quads.push((
                0.25 * vz_sum,
                proj,
                sheet_scalar_color(q.color_scalar, self.color_range, 165),
            ));
            if self.show_wireframe {
                let [c0, c1, c2, _] = q.corners.map(Point3::from);
                scene.add_segment("", c0, c1, wire_stroke, false);
                scene.add_segment("", c1, c2, wire_stroke, false);
            }
        }
        sorted_quads.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.append_monodromy_overlays(&mut scene, palette);
        (scene, sorted_quads)
    }

    fn append_monodromy_overlays(&self, scene: &mut Scene3DBuilder, palette: &SemanticPalette) {
        if let Some(first_quad) = self.sheet_quads.first() {
            scene.record_series(
                "Riemann Sheet Surface",
                &[Point3::from(first_quad.corners[0])],
                palette.series_color(0),
            );
        }
        if !self.show_monodromy_path || self.lifted_path_3d.len() < 2 {
            return;
        }
        let pts: Vec<Point3<f64>> = self
            .lifted_path_3d
            .iter()
            .copied()
            .map(Point3::from)
            .collect();
        scene.add_polyline(
            "Monodromy Lifted Path",
            &pts,
            egui::Stroke::new(3.0_f32, palette.series_color(3)),
            LineStyle::Solid,
            true,
        );
        if let (Some(&p_start), Some(&p_end)) = (pts.first(), pts.last()) {
            scene.add_points(
                "Loop Start (t=0)",
                &[p_start],
                5.5_f32,
                palette.series_color(1),
                MarkerShape::Circle,
                true,
            );
            scene.add_points(
                "Loop End (t=1)",
                &[p_end],
                6.0_f32,
                palette.series_color(2),
                MarkerShape::Diamond,
                true,
            );
        }
    }

    fn render_3d_riemann_viewport(&mut self, ui: &mut egui::Ui) {
        let dark = ui.visuals().dark_mode;
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let mut camera = self.camera;
        let (mut scene, sorted_quads) = self.build_riemann_3d_scene(camera, dark, &palette);
        Plot3DViewport::new("riemann_surface_3d_viewport", 2.65)
            .helper_text(CAMERA_3D_HELPER_TEXT)
            .show(ui, &mut camera, |plot_ui| {
                for (_, corners, fill) in &sorted_quads {
                    plot_ui.polygon(
                        Polygon::new("", PlotPoints::new(corners.to_vec()))
                            .fill_color(*fill)
                            .stroke(egui::Stroke::NONE)
                            .allow_hover(false),
                    );
                }
                scene.finish(plot_ui);
            });
        self.camera = camera;
    }

    fn render_2d_companion_plot(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let plot = egui_plot::Plot::new("riemann_companion_2d")
            .data_aspect(1.0)
            .allow_2d_analysis(false)
            .allow_csv_import(false)
            .show_navigation_controls(false)
            .show_toolbar(false)
            .show_settings_button(false);

        plot.show(ui, |plot_ui| match self.companion_mode {
            CompanionInspectorMode::SchwarzChristoffelPolygon => {
                self.draw_sc_companion(plot_ui, &palette);
            }
            CompanionInspectorMode::CauchyResidueContour => {
                self.draw_cauchy_companion(plot_ui, &palette);
            }
        });
    }

    fn draw_sc_companion(&self, plot_ui: &mut egui_plot::PlotUi, palette: &SemanticPalette) {
        for (i, curve) in self.sc_upper_curves.iter().enumerate() {
            let lbl = if i == 0 { "SC Upper Half-Plane" } else { "" };
            plot_ui.line(
                Line::new(lbl, PlotPoints::new(curve.clone()))
                    .color(palette.series_color(0))
                    .width(2.0_f32),
            );
        }
        for (i, curve) in self.sc_reflected_curves.iter().enumerate() {
            let lbl = if i == 0 { "Schwarz Reflected" } else { "" };
            plot_ui.line(
                Line::new(lbl, PlotPoints::new(curve.clone()))
                    .color(palette.series_color(2))
                    .style(LineStyle::Dashed { length: 6.0 })
                    .width(1.8_f32),
            );
        }
        plot_ui.points(
            Points::new("f(z_0)", vec![self.sc_probe_image])
                .radius(5.5_f32)
                .color(palette.series_color(1)),
        );
        if self.enable_schwarz_reflection {
            plot_ui.points(
                Points::new("f(conj(z_0))", vec![self.sc_reflected_probe_image])
                    .radius(5.5_f32)
                    .shape(MarkerShape::Diamond)
                    .color(palette.series_color(3)),
            );
        }
    }

    fn draw_cauchy_companion(&self, plot_ui: &mut egui_plot::PlotUi, palette: &SemanticPalette) {
        plot_ui.line(
            Line::new(
                "Branch Cut (-inf, 0]",
                PlotPoints::new(vec![[-3.0, 0.0], [0.0, 0.0]]),
            )
            .color(palette.series_color(4))
            .style(LineStyle::Dashed { length: 5.0 })
            .width(2.0_f32),
        );
        plot_ui.line(
            Line::new(
                "Contour gamma(t)",
                PlotPoints::new(self.base_loop_2d.clone()),
            )
            .color(palette.series_color(0))
            .width(2.4_f32),
        );
        plot_ui.points(
            Points::new("Probe z_0", vec![self.probe_z])
                .radius(5.5_f32)
                .color(palette.series_color(1)),
        );
        plot_ui.points(
            Points::new("Pole z_p", vec![self.pole_z])
                .radius(6.0_f32)
                .shape(MarkerShape::Cross)
                .color(palette.series_color(2)),
        );
    }

    pub(super) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = vec![("Contour_Gamma_Re_Im".to_string(), self.base_loop_2d.clone())];
        let lifted_proj: Vec<[f64; 2]> = self
            .lifted_path_3d
            .iter()
            .enumerate()
            .map(|(i, p)| [i as f64 / self.lifted_path_3d.len().max(1) as f64, p[2]])
            .collect();
        out.push(("Monodromy_Height_Profile".to_string(), lifted_proj));
        if let Some(first_sc) = self.sc_upper_curves.first() {
            out.push((
                "SchwarzChristoffel_Boundary_Ray".to_string(),
                first_sc.clone(),
            ));
        }
        out
    }

    pub(super) fn collect_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut prims = Vec::new();
        let lifted_2d: Vec<[f64; 2]> = self
            .lifted_path_3d
            .iter()
            .map(|w| self.camera.project_with_depth(w).0)
            .collect();
        if lifted_2d.len() >= 2 {
            prims.push(ExportPrimitive::Polyline {
                name: "Monodromy Lifted Path".to_string(),
                points: lifted_2d.into(),
                color: crate::app::palette::DARK_PALETTE.series[1],
                stroke_width: 2.5_f32,
                closed: false,
                fill: None,
            });
        }
        for q in self.sheet_quads.iter().step_by(2) {
            let ring: Vec<[f64; 2]> = q
                .corners
                .iter()
                .map(|w| self.camera.project_with_depth(w).0)
                .collect();
            prims.push(ExportPrimitive::Polyline {
                name: String::new(),
                points: ring.into(),
                color: sheet_scalar_color(q.color_scalar, self.color_range, 220),
                stroke_width: 1.0_f32,
                closed: true,
                fill: None,
            });
        }
        prims
    }
}
