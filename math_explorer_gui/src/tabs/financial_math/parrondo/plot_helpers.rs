#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ParrondoParadoxTool;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{ExportPrimitive, PlotPoint};

impl ParrondoParadoxTool {
    pub(super) fn export_occupancy_primitives(
        &self,
        pal: &SemanticPalette,
    ) -> Vec<ExportPrimitive> {
        let Some(ref r) = self.report else {
            return Vec::new();
        };
        let m = self.modulus.min(r.pi_b.len());
        let mut prims = Vec::with_capacity(m * 3);
        for s in 0..m {
            let x = s as f64;
            for (name, dx, val, col) in [
                ("pi_B", -0.22, r.pi_b[s], pal.series[1]),
                ("pi_mix", 0.0, r.pi_mix[s], pal.series[2]),
                ("pi_schedule", 0.22, r.pi_schedule_mean[s], pal.series[3]),
            ] {
                prims.push(ExportPrimitive::Bar {
                    name: format!("{name}[{s}]"),
                    min: [x + dx - 0.09, 0.0],
                    max: [x + dx + 0.09, val],
                    fill: col,
                    stroke_color: pal.accent,
                    stroke_width: 1.0,
                });
            }
        }
        prims
    }
}

pub(super) fn to_pairs(pts: &[PlotPoint]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| [p.x, p.y]).collect()
}

pub(super) fn scatter_prim(
    name: &str,
    points: Vec<[f64; 2]>,
    color: egui::Color32,
    radius: f32,
) -> ExportPrimitive {
    ExportPrimitive::Points {
        name: name.into(),
        points,
        color,
        radius,
    }
}

pub(super) fn polyline(
    name: &str,
    pts: &[PlotPoint],
    color: egui::Color32,
    width: f32,
) -> ExportPrimitive {
    ExportPrimitive::Polyline {
        name: name.to_string(),
        points: to_pairs(pts),
        color,
        stroke_width: width,
        closed: false,
        fill: None,
    }
}
