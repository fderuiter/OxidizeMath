use crate::pure_math::graph_theory::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Bounding box representation: (min_x, min_y, max_x, max_y)
pub type LayoutBounds = (f32, f32, f32, f32);

/// Computes a circular layout for the nodes in the graph inside the given bounds.
///
/// Nodes are placed evenly along the perimeter of a circle centered within the bounds.
#[verified_engine::verified]
pub fn circular_layout<N, E>(
    graph: &Graph<N, E>,
    bounds: LayoutBounds,
) -> HashMap<NodeIndex, (f32, f32)> {
    let mut positions = HashMap::new();
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return positions;
    }

    let (min_x, min_y, max_x, max_y) = bounds;
    let width = (max_x - min_x).abs();
    let height = (max_y - min_y).abs();
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;

    if n == 1 {
        positions.insert(nodes[0], (center_x, center_y));
        return positions;
    }

    let radius = (width.min(height) * 0.35).max(1.0);
    let step = 2.0 * std::f32::consts::PI / (n as f32);

    for (i, &node) in nodes.iter().enumerate() {
        let angle = (i as f32) * step;
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        positions.insert(node, (x, y));
    }

    positions
}

/// Computes a grid layout for the nodes in the graph inside the given bounds.
///
/// Nodes are arranged in a 2D rectangular grid.
#[verified_engine::verified]
pub fn grid_layout<N, E>(
    graph: &Graph<N, E>,
    bounds: LayoutBounds,
) -> HashMap<NodeIndex, (f32, f32)> {
    let mut positions = HashMap::new();
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n == 0 {
        return positions;
    }

    let (min_x, min_y, max_x, max_y) = bounds;
    let width = (max_x - min_x).abs();
    let height = (max_y - min_y).abs();
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;

    if n == 1 {
        positions.insert(nodes[0], (center_x, center_y));
        return positions;
    }

    let cols = (n as f32).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);

    let margin_x = width * 0.1;
    let margin_y = height * 0.1;
    let usable_w = width - 2.0 * margin_x;
    let usable_h = height - 2.0 * margin_y;

    let col_step = if cols > 1 {
        usable_w / ((cols - 1) as f32)
    } else {
        0.0
    };
    let row_step = if rows > 1 {
        usable_h / ((rows - 1) as f32)
    } else {
        0.0
    };

    let start_x = if cols > 1 { min_x + margin_x } else { center_x };
    let start_y = if rows > 1 { min_y + margin_y } else { center_y };

    for (i, &node) in nodes.iter().enumerate() {
        let r = i / cols;
        let c = i % cols;
        let x = start_x + (c as f32) * col_step;
        let y = start_y + (r as f32) * row_step;
        positions.insert(node, (x, y));
    }

    positions
}

#[verified_engine::verified]
fn compute_repulsive_forces(
    nodes: &[NodeIndex],
    positions: &HashMap<NodeIndex, (f32, f32)>,
    k2: f32,
    disp: &mut HashMap<NodeIndex, (f32, f32)>,
) {
    let n = nodes.len();
    for i in 0..n {
        let u = nodes[i];
        let pos_u = *positions.get(&u).unwrap();

        for &v in &nodes[(i + 1)..] {
            let pos_v = *positions.get(&v).unwrap();

            let mut dx = pos_u.0 - pos_v.0;
            let mut dy = pos_u.1 - pos_v.1;
            let mut dist = (dx * dx + dy * dy).sqrt();

            if dist < 0.001 {
                dx = 0.01;
                dy = 0.01;
                dist = 0.01414;
            }

            let fr = k2 / dist;
            let fx = (dx / dist) * fr;
            let fy = (dy / dist) * fr;

            if let Some(d_u) = disp.get_mut(&u) {
                d_u.0 += fx;
                d_u.1 += fy;
            }
            if let Some(d_v) = disp.get_mut(&v) {
                d_v.0 -= fx;
                d_v.1 -= fy;
            }
        }
    }
}

#[verified_engine::verified]
fn compute_attractive_forces(
    edges: &[(NodeIndex, NodeIndex)],
    positions: &HashMap<NodeIndex, (f32, f32)>,
    k: f32,
    disp: &mut HashMap<NodeIndex, (f32, f32)>,
) {
    for &(u, v) in edges {
        if u == v {
            continue;
        }
        if let (Some(&pos_u), Some(&pos_v)) = (positions.get(&u), positions.get(&v)) {
            let mut dx = pos_u.0 - pos_v.0;
            let mut dy = pos_u.1 - pos_v.1;
            let mut dist = (dx * dx + dy * dy).sqrt();

            if dist < 0.001 {
                dx = 0.01;
                dy = 0.01;
                dist = 0.01414;
            }

            let fa = (dist * dist) / k;
            let fx = (dx / dist) * fa;
            let fy = (dy / dist) * fa;

            if let Some(d_u) = disp.get_mut(&u) {
                d_u.0 -= fx;
                d_u.1 -= fy;
            }
            if let Some(d_v) = disp.get_mut(&v) {
                d_v.0 += fx;
                d_v.1 += fy;
            }
        }
    }
}

#[verified_engine::verified]
fn apply_displacements(
    nodes: &[NodeIndex],
    positions: &mut HashMap<NodeIndex, (f32, f32)>,
    disp: &HashMap<NodeIndex, (f32, f32)>,
    temp: f32,
    bounds: (f32, f32, f32, f32),
) {
    let (min_bound_x, min_bound_y, max_bound_x, max_bound_y) = bounds;
    for &u in nodes {
        if let (Some(pos), Some(d)) = (positions.get_mut(&u), disp.get(&u)) {
            let d_len = (d.0 * d.0 + d.1 * d.1).sqrt();
            if d_len > 0.0001 {
                let cap = d_len.min(temp);
                pos.0 += (d.0 / d_len) * cap;
                pos.1 += (d.1 / d_len) * cap;
            }
            pos.0 = pos.0.clamp(min_bound_x, max_bound_x);
            pos.1 = pos.1.clamp(min_bound_y, max_bound_y);
        }
    }
}

/// Computes a force-directed layout (Fruchterman-Reingold) for the nodes in the graph inside the given bounds.
///
/// Simulates repulsive forces between nodes and attractive spring forces along edges over `iterations` steps.
#[verified_engine::verified]
pub fn force_directed_layout<N, E>(
    graph: &Graph<N, E>,
    bounds: LayoutBounds,
    iterations: usize,
) -> HashMap<NodeIndex, (f32, f32)> {
    let mut positions = circular_layout(graph, bounds);
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let n = nodes.len();

    if n <= 1 || iterations == 0 {
        return positions;
    }

    let (min_x, min_y, max_x, max_y) = bounds;
    let width = (max_x - min_x).abs();
    let height = (max_y - min_y).abs();
    let k = ((width * height) / (n as f32)).sqrt();
    let k2 = k * k;

    let edges: Vec<(NodeIndex, NodeIndex)> = graph
        .graph
        .edge_references()
        .map(|e| (e.source(), e.target()))
        .collect();

    let margin = 20.0;
    let clamp_bounds = (
        min_x + margin,
        min_y + margin,
        max_x - margin,
        max_y - margin,
    );

    let mut temp = (width.min(height) * 0.1).max(1.0);
    let dt = temp / (iterations as f32);

    for _step in 0..iterations {
        let mut disp: HashMap<NodeIndex, (f32, f32)> =
            nodes.iter().map(|&n| (n, (0.0, 0.0))).collect();
        compute_repulsive_forces(&nodes, &positions, k2, &mut disp);
        compute_attractive_forces(&edges, &positions, k, &mut disp);
        apply_displacements(&nodes, &mut positions, &disp, temp, clamp_bounds);
        temp = (temp - dt).max(0.001);
    }

    positions
}
