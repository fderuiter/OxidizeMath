#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::TrainingMonitorTool;
use eframe::egui;
use egui_plot::{ExportPrimitive, ExportSubplotPanel};

fn subplot_rect_or(captured: &[ExportSubplotPanel], idx: usize, fallback: egui::Rect) -> egui::Rect {
    captured
        .get(idx)
        .map(|p| p.screen_rect)
        .filter(|r| r.width() > 1.0 && r.height() > 1.0)
        .unwrap_or(fallback)
}

fn subplot_log_flags(captured: &[ExportSubplotPanel], idx: usize) -> (bool, bool) {
    captured
        .get(idx)
        .map_or((false, false), |p| (p.log_x, p.log_y))
}

fn build_history_panel(
    id: &str,
    label: &str,
    points: &[[f64; 2]],
    color: egui::Color32,
    screen_rect: egui::Rect,
    log_flags: (bool, bool),
) -> ExportSubplotPanel {
    let series_name = format!("{id}: {label}");
    let datasets = if points.is_empty() {
        Vec::new()
    } else {
        vec![(series_name.clone(), points.to_vec())]
    };
    let primitives = if points.is_empty() {
        Vec::new()
    } else {
        vec![ExportPrimitive::Polyline {
            name: series_name,
            points: points.to_vec(),
            color,
            stroke_width: 2.0,
            closed: false,
            fill: None,
        }]
    };
    ExportSubplotPanel {
        id: id.to_string(),
        screen_rect,
        log_x: log_flags.0,
        log_y: log_flags.1,
        datasets,
        primitives,
    }
}

impl TrainingMonitorTool {
    fn build_decision_boundary_panel(
        &self,
        screen_rect: egui::Rect,
        log_flags: (bool, bool),
    ) -> ExportSubplotPanel {
        let mut class_0 = Vec::new();
        let mut class_1 = Vec::new();
        for (x, y) in &self.data {
            let pt = [x[0], x[1]];
            if y[0] > 0.5 {
                class_0.push(pt);
            } else {
                class_1.push(pt);
            }
        }
        let datasets = vec![
            ("decision_boundary: Class 0".to_string(), class_0.clone()),
            ("decision_boundary: Class 1".to_string(), class_1.clone()),
        ];
        let primitives = vec![
            ExportPrimitive::Points {
                name: "decision_boundary: Class 0".to_string(),
                points: class_0,
                color: egui::Color32::from_rgb(240, 95, 95),
                radius: 4.0,
            },
            ExportPrimitive::Points {
                name: "decision_boundary: Class 1".to_string(),
                points: class_1,
                color: egui::Color32::from_rgb(85, 165, 250),
                radius: 4.0,
            },
        ];
        ExportSubplotPanel {
            id: "decision_boundary".to_string(),
            screen_rect,
            log_x: log_flags.0,
            log_y: log_flags.1,
            datasets,
            primitives,
        }
    }

    pub(super) fn cold_export_subplot_panels(&self) -> Vec<ExportSubplotPanel> {
        let captured = egui_plot::get_last_plot_subplots();
        vec![
            build_history_panel(
                "loss_history",
                "Loss",
                &self.loss_history,
                egui::Color32::from_rgb(80, 175, 250),
                subplot_rect_or(
                    &captured,
                    0,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(390.0, 220.0)),
                ),
                subplot_log_flags(&captured, 0),
            ),
            build_history_panel(
                "accuracy_history",
                "Accuracy",
                &self.accuracy_history,
                egui::Color32::from_rgb(74, 214, 122),
                subplot_rect_or(
                    &captured,
                    1,
                    egui::Rect::from_min_max(egui::pos2(410.0, 0.0), egui::pos2(800.0, 220.0)),
                ),
                subplot_log_flags(&captured, 1),
            ),
            self.build_decision_boundary_panel(
                subplot_rect_or(
                    &captured,
                    2,
                    egui::Rect::from_min_max(egui::pos2(0.0, 240.0), egui::pos2(800.0, 600.0)),
                ),
                subplot_log_flags(&captured, 2),
            ),
        ]
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        for panel in self.cold_export_subplot_panels() {
            out.extend(panel.datasets);
        }
        out
    }
}
