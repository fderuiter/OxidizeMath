//! Offscreen `egui_plot::ExportPrimitive` and tabular dataset builders for `GraphTheoryTab` tools.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui::{Color32, Pos2};
use egui_plot::{ExportPrimitive, SharedPoints};
use std::collections::BTreeMap;

/// Visual styling descriptor for an exported network edge primitive.
#[derive(Clone, Debug)]
pub struct EdgePlotSpec {
    /// Edge series name.
    pub name: String,
    /// Tail node ID.
    pub u: usize,
    /// Head node ID.
    pub v: usize,
    /// Stroke color.
    pub color: Color32,
    /// Stroke width in UI points.
    pub width: f32,
    /// Whether to emit an arrowhead from `u` to `v`.
    pub directed: bool,
}

/// Visual styling descriptor for a group or individual exported vertex marker.
#[derive(Clone, Debug)]
pub struct NodePlotGroup {
    /// Series name.
    pub name: String,
    /// Screen-space node positions (`y` is negated automatically into plot space).
    pub positions: Vec<Pos2>,
    /// Marker color.
    pub color: Color32,
    /// Marker radius in UI points.
    pub radius: f32,
}

/// Converts a screen-space `Pos2` (`+y` down) into plot-space `[x, -y]` (`+y` up).
#[must_use]
pub fn screen_to_plot(pos: Pos2) -> [f64; 2] {
    [f64::from(pos.x), -f64::from(pos.y)]
}

/// Builds offscreen `ExportPrimitive` items for a styled network topology (or a reference frame when empty).
#[must_use]
pub fn build_network_plot_primitives(
    node_positions: &BTreeMap<usize, Pos2>,
    edges: &[EdgePlotSpec],
    node_groups: &[NodePlotGroup],
) -> Vec<ExportPrimitive> {
    if node_positions.is_empty() {
        return empty_canvas_reference_primitives();
    }
    let mut prims = Vec::with_capacity(edges.len() * 2 + node_groups.len());
    for spec in edges {
        let (Some(&pu), Some(&pv)) = (node_positions.get(&spec.u), node_positions.get(&spec.v))
        else {
            continue;
        };
        let p0 = screen_to_plot(pu);
        let p1 = screen_to_plot(pv);
        prims.push(ExportPrimitive::Polyline {
            name: spec.name.clone(),
            points: SharedPoints::from(vec![p0, p1]),
            color: spec.color,
            stroke_width: spec.width,
            closed: false,
            fill: None,
        });
        if spec.directed && (pu - pv).length() > 1e-3 {
            prims.push(ExportPrimitive::Arrow {
                name: spec.name.clone(),
                origin: p0,
                tip: p1,
                tip_length: Some(9.0),
                color: spec.color,
            });
        }
    }
    for group in node_groups {
        if group.positions.is_empty() {
            continue;
        }
        let pts: Vec<[f64; 2]> = group
            .positions
            .iter()
            .copied()
            .map(screen_to_plot)
            .collect();
        prims.push(ExportPrimitive::Points {
            name: group.name.clone(),
            points: SharedPoints::from(pts),
            color: group.color,
            radius: group.radius,
        });
    }
    if prims.is_empty() {
        return empty_canvas_reference_primitives();
    }
    prims
}

/// Builds deterministic `[x, -y]` datasets for CSV and JSON exports via `SimulationFramework`.
#[must_use]
pub fn build_network_datasets(
    node_positions: &BTreeMap<usize, Pos2>,
    edges: &[(usize, usize, f64)],
) -> Vec<(String, Vec<[f64; 2]>)> {
    if node_positions.is_empty() {
        return vec![(
            "Empty Graph Canvas".to_string(),
            vec![[0.0, 0.0], [100.0, 0.0], [100.0, -100.0], [0.0, -100.0]],
        )];
    }
    let mut datasets = Vec::new();
    let node_pts: Vec<[f64; 2]> = node_positions
        .values()
        .copied()
        .map(screen_to_plot)
        .collect();
    datasets.push(("Vertices".to_string(), node_pts));

    for (idx, &(u, v, w)) in edges.iter().enumerate() {
        if let (Some(&pu), Some(&pv)) = (node_positions.get(&u), node_positions.get(&v)) {
            datasets.push((
                format!("Edge {idx} ({u}-{v}, w={w:.2})"),
                vec![screen_to_plot(pu), screen_to_plot(pv)],
            ));
        }
    }
    datasets
}

fn empty_canvas_reference_primitives() -> Vec<ExportPrimitive> {
    let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
    vec![
        ExportPrimitive::Polyline {
            name: "Canvas Frame".to_string(),
            points: SharedPoints::from(vec![
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, -100.0],
                [0.0, -100.0],
            ]),
            color: palette.graph_edge,
            stroke_width: 1.5,
            closed: true,
            fill: None,
        },
        ExportPrimitive::Points {
            name: "Origin Reference".to_string(),
            points: SharedPoints::from(vec![[50.0, -50.0]]),
            color: palette.accent,
            radius: 5.0,
        },
    ]
}

// [cite:graph_parameters_rust]
