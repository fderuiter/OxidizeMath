//! Standalone LaTeX `TikZ` (`.tex`) topology diagram serializer.

use super::escape::{escape_latex_text, format_compact_f64, parse_hex_rgb};
use super::{
    DEFAULT_TOPOLOGY_LAYOUT_BOUNDS, GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle,
    TopologyNode, TopologyNodeShape,
};
use std::collections::HashMap;
use std::fmt::Write as _;

/// Serializes a [`GraphTopologyDocument`] into a standalone compilable LaTeX `TikZ` document.
#[must_use]
#[verified_engine::verified]
pub fn to_tikz_topology(doc: &GraphTopologyDocument) -> String {
    let mut positioned = doc.clone();
    positioned.ensure_positions(DEFAULT_TOPOLOGY_LAYOUT_BOUNDS);
    let coords = normalize_tikz_coordinates(&positioned.nodes);
    let id_to_idx: HashMap<&str, usize> = positioned
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();

    let mut out = String::from(
        "\\documentclass[tikz,border=10pt]{standalone}\n\
         \\usepackage[utf8]{inputenc}\n\
         \\usepackage{tikz}\n\
         \\usetikzlibrary{arrows.meta,positioning,calc}\n\
         \\begin{document}\n\
         \\begin{tikzpicture}[>=Stealth, every node/.style={font=\\sffamily\\scriptsize}]\n",
    );
    let _ = writeln!(
        out,
        "  % Graph: {} (nodes={}, edges={})",
        escape_latex_text(&positioned.name),
        positioned.nodes.len(),
        positioned.edges.len()
    );

    for (idx, node) in positioned.nodes.iter().enumerate() {
        write_tikz_node(&mut out, idx, node, coords[idx]);
    }

    let pair_counts = count_undirected_multiplicities(&positioned.edges);
    let mut pair_seen: HashMap<(String, String), usize> = HashMap::new();
    for (eidx, edge) in positioned.edges.iter().enumerate() {
        let Some(&u) = id_to_idx.get(edge.source.as_str()) else {
            continue;
        };
        let Some(&v) = id_to_idx.get(edge.target.as_str()) else {
            continue;
        };
        let key = canonical_pair(&edge.source, &edge.target);
        let mult = pair_counts.get(&key).copied().unwrap_or(1);
        let ord = *pair_seen
            .entry(key)
            .and_modify(|c| *c += 1)
            .or_insert(0usize);
        write_tikz_edge(
            &mut out,
            eidx,
            edge,
            (u, v),
            (positioned.directed, mult, ord),
        );
    }

    out.push_str("\\end{tikzpicture}\n\\end{document}\n");
    out
}

#[verified_engine::verified]
fn normalize_tikz_coordinates(nodes: &[TopologyNode]) -> Vec<[f64; 2]> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for node in nodes {
        let [x, y] = node.position.unwrap_or([0.0, 0.0]);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let span_x = (max_x - min_x).max(1.0);
    let span_y = (max_y - min_y).max(1.0);
    nodes
        .iter()
        .map(|n| {
            let [x, y] = n.position.unwrap_or([min_x, min_y]);
            let tx = ((x - min_x) / span_x) * 10.0;
            // Invert screen Y so top-of-canvas maps to top of TikZ figure.
            let ty = ((max_y - y) / span_y) * 7.5;
            [tx, ty]
        })
        .collect()
}

#[verified_engine::verified]
fn write_tikz_node(out: &mut String, idx: usize, node: &TopologyNode, xy: [f64; 2]) {
    let fill_rgb = node
        .fill_hex
        .as_deref()
        .and_then(parse_hex_rgb)
        .unwrap_or([232, 240, 254]);
    let stroke_rgb = node
        .stroke_hex
        .as_deref()
        .and_then(parse_hex_rgb)
        .unwrap_or([35, 68, 122]);
    let _ = writeln!(
        out,
        "  \\definecolor{{nf{idx}}}{{RGB}}{{{},{},{}}}",
        fill_rgb[0], fill_rgb[1], fill_rgb[2]
    );
    let _ = writeln!(
        out,
        "  \\definecolor{{ns{idx}}}{{RGB}}{{{},{},{}}}",
        stroke_rgb[0], stroke_rgb[1], stroke_rgb[2]
    );
    let shape_opts = match node.shape {
        TopologyNodeShape::Circle => "circle, minimum size=16pt",
        TopologyNodeShape::DoubleCircle => {
            "circle, double, double distance=1.5pt, minimum size=17pt"
        }
        TopologyNodeShape::Box => "rectangle, rounded corners=2pt, minimum size=16pt",
    };
    let _ = writeln!(
        out,
        "  \\node[draw=ns{idx}, fill=nf{idx}, line width=0.8pt, {shape_opts}, inner sep=2pt] (v{idx}) at ({},{}) {{{}}};",
        format_compact_f64(xy[0]),
        format_compact_f64(xy[1]),
        escape_latex_text(&node.label)
    );
}

#[verified_engine::verified]
fn write_tikz_edge(
    out: &mut String,
    eidx: usize,
    edge: &TopologyEdge,
    (u, v): (usize, usize),
    (directed, mult, ord): (bool, usize, usize),
) {
    let rgb = edge
        .color_hex
        .as_deref()
        .and_then(parse_hex_rgb)
        .unwrap_or([90, 104, 124]);
    let _ = writeln!(
        out,
        "  \\definecolor{{ec{eidx}}}{{RGB}}{{{},{},{}}}",
        rgb[0], rgb[1], rgb[2]
    );
    let arrow = if directed { "->" } else { "-" };
    let stroke = match edge.style {
        TopologyEdgeStyle::Solid => "line width=0.75pt",
        TopologyEdgeStyle::Dashed => "dashed, line width=0.75pt",
        TopologyEdgeStyle::Thick => "line width=1.4pt",
    };
    let bend = if u == v {
        format!("loop above, looseness={}", 6 + ord * 2)
    } else if mult > 1 {
        let angle = bend_angle(mult, ord, u < v);
        format!("bend left={angle}")
    } else {
        String::new()
    };
    let style_sep = if bend.is_empty() { "" } else { ", " };
    let label_str = edge_caption(edge);
    if label_str.is_empty() {
        let _ = writeln!(
            out,
            "  \\draw[{arrow}, draw=ec{eidx}, {stroke}{style_sep}{bend}] (v{u}) to (v{v});"
        );
    } else {
        let _ = writeln!(
            out,
            "  \\draw[{arrow}, draw=ec{eidx}, {stroke}{style_sep}{bend}] (v{u}) to node[midway, fill=white, fill opacity=0.85, text opacity=1, inner sep=1pt] {{{}}} (v{v});",
            escape_latex_text(&label_str)
        );
    }
}

#[verified_engine::verified]
fn edge_caption(edge: &TopologyEdge) -> String {
    if let Some(lbl) = &edge.label
        && !lbl.trim().is_empty()
    {
        return lbl.clone();
    }
    if (edge.weight - 1.0).abs() > 1e-9 {
        format_compact_f64(edge.weight)
    } else {
        String::new()
    }
}

#[verified_engine::verified]
pub(super) fn canonical_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

#[verified_engine::verified]
pub(super) fn count_undirected_multiplicities(
    edges: &[TopologyEdge],
) -> HashMap<(String, String), usize> {
    let mut counts = HashMap::with_capacity(edges.len());
    for edge in edges {
        let key = canonical_pair(&edge.source, &edge.target);
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

#[verified_engine::verified]
fn bend_angle(mult: usize, ord: usize, forward: bool) -> i32 {
    let step = 18_i32;
    let centered = (ord as i32) * 2 - (mult as i32 - 1);
    let raw = centered * (step / 2).max(8);
    if forward { raw } else { -raw }
}

// [cite:graph_parameters_rust]
