#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ComplexMappingTool;
use eframe::egui;
use egui_plot::export_primitives::ExportPrimitive;
use egui_plot::ExportSubplotPanel;
use num_complex::Complex64;

#[derive(Default)]
struct PlanePanelAccum {
    datasets: Vec<(String, Vec<[f64; 2]>)>,
    primitives: Vec<ExportPrimitive>,
}

impl PlanePanelAccum {
    fn push_marker(&mut self, name: &str, point: [f64; 2], color: egui::Color32) {
        self.datasets.push((name.to_string(), vec![point]));
        self.primitives.push(ExportPrimitive::Points {
            name: name.to_string(),
            points: vec![point],
            color,
            radius: 5.0,
        });
    }

    fn into_panel(
        self,
        id: &str,
        captured: &[ExportSubplotPanel],
        fallback: egui::Rect,
    ) -> ExportSubplotPanel {
        let meta = captured.iter().find(|p| p.id == id);
        ExportSubplotPanel {
            id: id.to_string(),
            screen_rect: meta
                .map(|p| p.screen_rect)
                .filter(|r| r.width() > 1.0 && r.height() > 1.0)
                .unwrap_or(fallback),
            log_x: meta.is_some_and(|p| p.log_x),
            log_y: meta.is_some_and(|p| p.log_y),
            datasets: self.datasets,
            primitives: self.primitives,
        }
    }
}

impl ComplexMappingTool {
    pub(super) fn cold_export_subplot_panels(&self) -> Vec<ExportSubplotPanel> {
        let captured = egui_plot::get_last_plot_subplots();
        let palette = crate::app::palette::DARK_PALETTE;
        let mut z_panel = PlanePanelAccum::default();
        let mut w_panel = PlanePanelAccum::default();

        if self.grid_max > self.grid_min && self.grid_density > 1 {
            self.append_vertical_contours(&mut z_panel, &mut w_panel, palette.series_color(0));
            self.append_horizontal_contours(&mut z_panel, &mut w_panel, palette.series_color(1));
        }

        z_panel.push_marker(
            "z_plane: inspection_point",
            self.inspection_point,
            palette.error,
        );
        let w0 = self.eval_z(Complex64::new(
            self.inspection_point[0],
            self.inspection_point[1],
        ));
        w_panel.push_marker("w_plane: mapped_point", [w0.re, w0.im], palette.error);

        vec![
            z_panel.into_panel(
                "z_plane",
                &captured,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(396.0, 600.0)),
            ),
            w_panel.into_panel(
                "w_plane",
                &captured,
                egui::Rect::from_min_max(egui::pos2(404.0, 0.0), egui::pos2(800.0, 600.0)),
            ),
        ]
    }

    fn append_vertical_contours(
        &self,
        z_panel: &mut PlanePanelAccum,
        w_panel: &mut PlanePanelAccum,
        color: egui::Color32,
    ) {
        let step = (self.grid_max - self.grid_min) / (self.grid_density as f64 - 1.0);
        let resolution = 50usize;
        for i in 0..self.grid_density {
            let x = self.grid_min + (i as f64) * step;
            let z_name = format!("z_plane: vertical_grid_{i}");
            let z_pts = vec![[x, self.grid_min], [x, self.grid_max]];
            z_panel.datasets.push((z_name.clone(), z_pts.clone()));
            z_panel.primitives.push(ExportPrimitive::Polyline {
                name: z_name,
                points: z_pts,
                color,
                stroke_width: 1.5,
                closed: false,
                fill: None,
            });

            let w_name = format!("w_plane: vertical_contour_{i}");
            let w_pts: Vec<[f64; 2]> = (0..resolution)
                .map(|j| {
                    let y = self.grid_min
                        + (self.grid_max - self.grid_min) * (j as f64)
                            / (resolution as f64 - 1.0);
                    let w = self.eval_z(Complex64::new(x, y));
                    [w.re, w.im]
                })
                .collect();
            w_panel.datasets.push((w_name.clone(), w_pts.clone()));
            w_panel.primitives.push(ExportPrimitive::Polyline {
                name: w_name,
                points: w_pts,
                color,
                stroke_width: 1.5,
                closed: false,
                fill: None,
            });
        }
    }

    fn append_horizontal_contours(
        &self,
        z_panel: &mut PlanePanelAccum,
        w_panel: &mut PlanePanelAccum,
        color: egui::Color32,
    ) {
        let step = (self.grid_max - self.grid_min) / (self.grid_density as f64 - 1.0);
        let resolution = 50usize;
        for i in 0..self.grid_density {
            let y = self.grid_min + (i as f64) * step;
            let z_name = format!("z_plane: horizontal_grid_{i}");
            let z_pts = vec![[self.grid_min, y], [self.grid_max, y]];
            z_panel.datasets.push((z_name.clone(), z_pts.clone()));
            z_panel.primitives.push(ExportPrimitive::Polyline {
                name: z_name,
                points: z_pts,
                color,
                stroke_width: 1.5,
                closed: false,
                fill: None,
            });

            let w_name = format!("w_plane: horizontal_contour_{i}");
            let w_pts: Vec<[f64; 2]> = (0..resolution)
                .map(|j| {
                    let x = self.grid_min
                        + (self.grid_max - self.grid_min) * (j as f64)
                            / (resolution as f64 - 1.0);
                    let w = self.eval_z(Complex64::new(x, y));
                    [w.re, w.im]
                })
                .collect();
            w_panel.datasets.push((w_name.clone(), w_pts.clone()));
            w_panel.primitives.push(ExportPrimitive::Polyline {
                name: w_name,
                points: w_pts,
                color,
                stroke_width: 1.5,
                closed: false,
                fill: None,
            });
        }
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut all = Vec::new();
        for panel in self.cold_export_subplot_panels() {
            all.extend(panel.datasets);
        }
        all
    }
}
