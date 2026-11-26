//! Offscreen SVG/PNG primitive and dataset builders for the Vietoris-Rips TDA workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::plots::{visible_intervals, PersistencePlotMode};
use super::VietorisRipsTool;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{ExportPrimitive, PlotPoint};
use math_explorer::pure_math::statistics::tda::RipsInterval;

/// Builds offscreen `ExportPrimitive` shapes combining the 2D complex and persistence diagram/barcode.
#[must_use]
pub fn build_export_primitives(tool: &VietorisRipsTool) -> Vec<ExportPrimitive> {
    let palette = SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
    let mut prims = Vec::new();
    append_complex_primitives(tool, &palette, &mut prims);
    append_persistence_primitives(tool, &palette, &mut prims);
    prims
}

/// Builds tabular `(name, points)` series for dataset export and animation fallbacks.
#[must_use]
pub fn build_export_datasets(tool: &VietorisRipsTool) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut datasets = Vec::new();
    let verts: Vec<[f64; 2]> = tool.cached_vertices.iter().map(|p| [p.x, p.y]).collect();
    if !verts.is_empty() {
        datasets.push(("Vertices (2D)".to_string(), verts));
    }
    for dim in [0_usize, 1_usize] {
        let pts: Vec<[f64; 2]> = visible_intervals(tool)
            .into_iter()
            .filter(|iv| iv.dimension == dim)
            .map(|iv| [iv.birth, iv.death.max(iv.birth)])
            .collect();
        if !pts.is_empty() {
            datasets.push((format!("H{dim} Persistence (birth, death)"), pts));
        }
    }
    datasets
}

fn append_complex_primitives(
    tool: &VietorisRipsTool,
    palette: &SemanticPalette,
    prims: &mut Vec<ExportPrimitive>,
) {
    if tool.cached_vertices.is_empty() {
        return;
    }
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for p in &tool.cached_vertices {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let span = (max_x - min_x).max(max_y - min_y).max(1e-6);
    let map_pt = |p: PlotPoint| [(p.x - min_x) / span, (p.y - min_y) / span];
    let s0 = palette.series_color(0);
    let tri_fill = egui::Color32::from_rgba_unmultiplied(s0.r(), s0.g(), s0.b(), 65);

    for tri in &tool.cached_triangles {
        prims.push(ExportPrimitive::Polyline {
            name: "2-Simplex".to_string(),
            points: vec![map_pt(tri[0]), map_pt(tri[1]), map_pt(tri[2])].into(),
            color: egui::Color32::TRANSPARENT,
            stroke_width: 0.5,
            closed: true,
            fill: Some(tri_fill),
        });
    }
    for edge in &tool.cached_edges {
        prims.push(ExportPrimitive::Polyline {
            name: "1-Simplex".to_string(),
            points: vec![map_pt(edge[0]), map_pt(edge[1])].into(),
            color: palette.graph_edge,
            stroke_width: 1.5,
            closed: false,
            fill: None,
        });
    }
    prims.push(ExportPrimitive::Points {
        name: "Vertices".to_string(),
        points: tool
            .cached_vertices
            .iter()
            .copied()
            .map(map_pt)
            .collect::<Vec<_>>()
            .into(),
        color: s0,
        radius: 4.0,
    });
}

fn append_persistence_primitives(
    tool: &VietorisRipsTool,
    palette: &SemanticPalette,
    prims: &mut Vec<ExportPrimitive>,
) {
    let intervals = visible_intervals(tool);
    if tool.plot_mode == PersistencePlotMode::Diagram {
        append_diagram_primitives(tool, palette, &intervals, prims);
    } else {
        append_barcode_primitives(tool, palette, &intervals, prims);
    }
}

fn append_diagram_primitives(
    tool: &VietorisRipsTool,
    palette: &SemanticPalette,
    intervals: &[&RipsInterval],
    prims: &mut Vec<ExportPrimitive>,
) {
    let x0 = 1.25;
    let max_s = tool.max_slider_radius.max(tool.radius).max(1e-6);
    let eps_n = (tool.radius / max_s).clamp(0.0, 1.0);
    prims.push(ExportPrimitive::Polyline {
        name: "Diagonal y = x".to_string(),
        points: vec![[x0, 0.0], [x0 + 1.0, 1.0]].into(),
        color: palette.muted_text,
        stroke_width: 1.2,
        closed: false,
        fill: None,
    });
    prims.push(ExportPrimitive::Polyline {
        name: "Active ε Quadrant".to_string(),
        points: vec![[x0, eps_n], [x0 + eps_n, eps_n], [x0 + eps_n, 1.0]].into(),
        color: palette.accent,
        stroke_width: 1.6,
        closed: false,
        fill: None,
    });
    for dim in [0_usize, 1_usize] {
        let pts: Vec<[f64; 2]> = intervals
            .iter()
            .filter(|iv| iv.dimension == dim)
            .map(|iv| {
                [
                    x0 + (iv.birth / max_s).clamp(0.0, 1.0),
                    (iv.death.max(iv.birth) / max_s).clamp(0.0, 1.0),
                ]
            })
            .collect();
        if !pts.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: format!("H{dim} Diagram"),
                points: pts.into(),
                color: palette.series_color(dim),
                radius: if dim == 0 { 4.5 } else { 5.2 },
            });
        }
    }
}

fn append_barcode_primitives(
    tool: &VietorisRipsTool,
    palette: &SemanticPalette,
    intervals: &[&RipsInterval],
    prims: &mut Vec<ExportPrimitive>,
) {
    let x0 = 1.25;
    let max_s = tool.max_slider_radius.max(tool.radius).max(1e-6);
    let eps_n = (tool.radius / max_s).clamp(0.0, 1.0);
    let denom = (intervals.len().max(1) + 1) as f64;
    for (idx, iv) in intervals.iter().enumerate() {
        let y = ((idx + 1) as f64) / denom;
        let b = x0 + (iv.birth / max_s).clamp(0.0, 1.0);
        let d = x0 + (iv.death.max(iv.birth) / max_s).clamp(0.0, 1.0);
        prims.push(ExportPrimitive::Polyline {
            name: format!("H{} Interval", iv.dimension),
            points: vec![[b, y], [d.max(b + 0.005), y]].into(),
            color: palette.series_color(iv.dimension.min(1)),
            stroke_width: if iv.is_alive_at(tool.radius) {
                3.0
            } else {
                1.6
            },
            closed: false,
            fill: None,
        });
    }
    prims.push(ExportPrimitive::Polyline {
        name: "ε Threshold".to_string(),
        points: vec![[x0 + eps_n, 0.0], [x0 + eps_n, 1.0]].into(),
        color: palette.accent,
        stroke_width: 2.0,
        closed: false,
        fill: None,
    });
}

// [cite:graph_parameters_rust]
