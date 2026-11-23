//! Tabular CSV/JSON dataset and vector SVG/PNG figure primitive assembly for [`DynamicalBilliardsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BilliardGeometryKind, DynamicalBilliardsTool};
use egui_plot::ExportPrimitive;
use std::f64::consts::TAU;

impl DynamicalBilliardsTool {
    pub(crate) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        for (idx, slot) in self.rays.iter().enumerate() {
            if !slot.birkhoff_points.is_empty() {
                datasets.push((
                    format!("Birkhoff Ray {} (s/L, sin phi)", idx + 1),
                    slot.birkhoff_points.clone(),
                ));
            }
            if slot.vertices.len() >= 2 {
                datasets.push((
                    format!("Spatial Chord Ray {} (x, y)", idx + 1),
                    slot.vertices.clone(),
                ));
            }
        }
        datasets
    }

    pub(crate) fn collect_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut prims = Vec::new();
        let boundary = self.outer_boundary_polyline();
        if boundary.len() >= 3 {
            prims.push(ExportPrimitive::Polyline {
                name: "Billiard Outer Boundary".to_string(),
                points: boundary,
                color: palette.info,
                stroke_width: 2.0_f32,
                closed: true,
                fill: None,
            });
        }
        if self.geometry == BilliardGeometryKind::Sinai {
            let r = self.sinai.scatterer_radius;
            let disk: Vec<[f64; 2]> = (0..64)
                .map(|i| {
                    let th = (i as f64 / 64.0) * TAU;
                    [r * th.cos(), r * th.sin()]
                })
                .collect();
            prims.push(ExportPrimitive::Polyline {
                name: "Sinai Central Scatterer".to_string(),
                points: disk,
                color: palette.warning,
                stroke_width: 1.8_f32,
                closed: true,
                fill: None,
            });
        }
        for (idx, slot) in self.rays.iter().enumerate() {
            if slot.vertices.len() >= 2 {
                prims.push(ExportPrimitive::Polyline {
                    name: format!("Ray {} Chords", idx + 1),
                    points: slot.vertices.clone(),
                    color: palette.series_color(slot.color_index),
                    stroke_width: 1.3_f32,
                    closed: false,
                    fill: None,
                });
            }
        }
        prims
    }
}
