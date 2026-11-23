//! Reusable grid and node-link graph [`super::ExportPrimitive`] builders for custom-painter tools.

use std::f64::consts::TAU;

use super::ExportPrimitive;

/// Specification for a single 2D grid cell rendered offscreen as an [`ExportPrimitive::Bar`].
#[derive(Clone, Debug)]
pub struct GridCellSpec {
    /// Zero-based row index from top (`0`) to bottom (`rows - 1`).
    pub row: usize,
    /// Zero-based column index from left (`0`) to right.
    pub col: usize,
    /// Total number of rows in the grid (used to invert Y so row `0` sits at the top).
    pub total_rows: usize,
    /// Interior cell fill color.
    pub fill: egui::Color32,
    /// Cell border stroke color.
    pub stroke_color: egui::Color32,
    /// Cell border stroke width in UI points.
    pub stroke_width: f32,
    /// Optional series label for debugging and SVG metadata.
    pub label: String,
}

/// Converts a slice of [`GridCellSpec`] items into axis-aligned [`ExportPrimitive::Bar`] rectangles.
#[must_use]
pub fn build_grid_primitives(cells: &[GridCellSpec]) -> Vec<ExportPrimitive> {
    let mut out = Vec::with_capacity(cells.len());
    for cell in cells {
        let rows = cell.total_rows.max(1);
        let x0 = cell.col as f64;
        let x1 = x0 + 0.94;
        let inv_row = rows
            .saturating_sub(1)
            .saturating_sub(cell.row.min(rows - 1));
        let y0 = inv_row as f64;
        let y1 = y0 + 0.94;
        out.push(ExportPrimitive::Bar {
            name: cell.label.clone(),
            min: [x0, y0],
            max: [x1, y1],
            fill: cell.fill,
            stroke_color: cell.stroke_color,
            stroke_width: cell.stroke_width,
        });
    }
    out
}

/// Specification for a node in a 2D node-link graph scene.
#[derive(Clone, Debug)]
pub struct GraphNodeSpec {
    /// Unique node identifier referenced by [`GraphEdgeSpec`].
    pub id: usize,
    /// 2D node center `[x, y]` in layout or screen coordinates.
    pub pos: [f64; 2],
    /// Interior fill color of the node marker.
    pub fill: egui::Color32,
    /// Outline stroke color of the node marker.
    pub stroke_color: egui::Color32,
    /// Visual marker radius in UI points (and relative plot-space disc scaling).
    pub radius: f32,
    /// Series/node label.
    pub label: String,
}

/// Specification for an undirected or directed edge in a 2D node-link graph scene.
#[derive(Clone, Debug)]
pub struct GraphEdgeSpec {
    /// Source node identifier matching [`GraphNodeSpec::id`].
    pub from_id: usize,
    /// Target node identifier matching [`GraphNodeSpec::id`].
    pub to_id: usize,
    /// Edge stroke color.
    pub color: egui::Color32,
    /// Edge stroke width in UI points.
    pub width: f32,
    /// Whether to render a directional arrowhead pointing at `to_id`.
    pub directed: bool,
    /// Edge label.
    pub label: String,
}

fn find_node_pos(nodes: &[GraphNodeSpec], id: usize, flip_y: bool) -> Option<[f64; 2]> {
    nodes.iter().find(|n| n.id == id).map(|n| {
        let y = if flip_y { -n.pos[1] } else { n.pos[1] };
        [n.pos[0], y]
    })
}

fn node_disc_radius_world(nodes: &[GraphNodeSpec], flip_y: bool) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for node in nodes {
        let y = if flip_y { -node.pos[1] } else { node.pos[1] };
        if node.pos[0].is_finite() && y.is_finite() {
            min_x = min_x.min(node.pos[0]);
            max_x = max_x.max(node.pos[0]);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    let span = (max_x - min_x).max(max_y - min_y);
    if span.is_finite() && span > 1e-6 {
        span * 0.035
    } else {
        0.08
    }
}

fn build_circle_polygon(center: [f64; 2], radius: f64, segments: usize) -> Vec<[f64; 2]> {
    let count = segments.max(8);
    (0..count)
        .map(|i| {
            let theta = TAU * (i as f64) / (count as f64);
            [
                center[0] + radius * theta.cos(),
                center[1] + radius * theta.sin(),
            ]
        })
        .collect()
}

/// Converts [`GraphNodeSpec`] and [`GraphEdgeSpec`] slices into offscreen [`ExportPrimitive`] shapes.
///
/// Set `flip_y` to `true` when node coordinates originate from top-down UI screen coordinates.
#[must_use]
pub fn build_graph_primitives(
    nodes: &[GraphNodeSpec],
    edges: &[GraphEdgeSpec],
    flip_y: bool,
) -> Vec<ExportPrimitive> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let world_r = node_disc_radius_world(nodes, flip_y);
    let mut out = Vec::with_capacity(edges.len() + nodes.len() * 2);

    for edge in edges {
        let (Some(p0), Some(p1)) = (
            find_node_pos(nodes, edge.from_id, flip_y),
            find_node_pos(nodes, edge.to_id, flip_y),
        ) else {
            continue;
        };
        if edge.directed {
            let dx = p1[0] - p0[0];
            let dy = p1[1] - p0[1];
            let dist = dx.hypot(dy);
            let trim = if dist > world_r * 2.4 {
                world_r * 1.05 / dist
            } else {
                0.0
            };
            let origin = [p0[0] + dx * trim, p0[1] + dy * trim];
            let tip = [p1[0] - dx * trim, p1[1] - dy * trim];
            out.push(ExportPrimitive::Arrow {
                name: edge.label.clone(),
                origin,
                tip,
                tip_length: Some((edge.width * 4.5).clamp(7.0, 14.0)),
                color: edge.color,
            });
        } else {
            out.push(ExportPrimitive::Polyline {
                name: edge.label.clone(),
                points: vec![p0, p1],
                color: edge.color,
                stroke_width: edge.width.max(1.2),
                closed: false,
                fill: None,
            });
        }
    }

    for node in nodes {
        let y = if flip_y { -node.pos[1] } else { node.pos[1] };
        if !node.pos[0].is_finite() || !y.is_finite() {
            continue;
        }
        let scale = (node.radius / 14.0).clamp(0.65, 1.6) as f64;
        let poly = build_circle_polygon([node.pos[0], y], world_r * scale, 18);
        out.push(ExportPrimitive::Polyline {
            name: node.label.clone(),
            points: poly,
            color: node.stroke_color,
            stroke_width: 1.8,
            closed: true,
            fill: Some(node.fill),
        });
        out.push(ExportPrimitive::Points {
            name: node.label.clone(),
            points: vec![[node.pos[0], y]],
            color: node.fill,
            radius: node.radius.clamp(3.0, 12.0),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_grid_primitives_inverts_rows_into_bars() {
        let cells = vec![
            GridCellSpec {
                row: 0,
                col: 0,
                total_rows: 3,
                fill: egui::Color32::RED,
                stroke_color: egui::Color32::WHITE,
                stroke_width: 1.0,
                label: "top_left".into(),
            },
            GridCellSpec {
                row: 2,
                col: 1,
                total_rows: 3,
                fill: egui::Color32::BLUE,
                stroke_color: egui::Color32::WHITE,
                stroke_width: 1.0,
                label: "bottom_mid".into(),
            },
        ];
        let prims = build_grid_primitives(&cells);
        assert_eq!(prims.len(), 2);
        match &prims[0] {
            ExportPrimitive::Bar { min, max, .. } => {
                assert!((min[0] - 0.0).abs() < 1e-9);
                assert!((min[1] - 2.0).abs() < 1e-9);
                assert!(max[0] > min[0] && max[1] > min[1]);
            }
            other => panic!("Expected Bar, got {other:?}"),
        }
    }

    #[test]
    fn test_build_graph_primitives_handles_directed_and_undirected_edges() {
        let nodes = vec![
            GraphNodeSpec {
                id: 10,
                pos: [0.0, 10.0],
                fill: egui::Color32::GREEN,
                stroke_color: egui::Color32::BLACK,
                radius: 10.0,
                label: "N10".into(),
            },
            GraphNodeSpec {
                id: 20,
                pos: [100.0, 10.0],
                fill: egui::Color32::YELLOW,
                stroke_color: egui::Color32::BLACK,
                radius: 10.0,
                label: "N20".into(),
            },
        ];
        let edges = vec![
            GraphEdgeSpec {
                from_id: 10,
                to_id: 20,
                color: egui::Color32::GRAY,
                width: 2.0,
                directed: true,
                label: "dir".into(),
            },
            GraphEdgeSpec {
                from_id: 20,
                to_id: 10,
                color: egui::Color32::WHITE,
                width: 1.5,
                directed: false,
                label: "undir".into(),
            },
        ];
        let prims = build_graph_primitives(&nodes, &edges, true);
        assert_eq!(prims.len(), 6);
        assert!(matches!(prims[0], ExportPrimitive::Arrow { .. }));
        assert!(matches!(prims[1], ExportPrimitive::Polyline { .. }));
    }
}
