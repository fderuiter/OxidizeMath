//! Standalone vector `.svg` diagram serializer for [`super::GraphTopologyDocument`].

use super::escape::{escape_xml, format_compact_f64};
use super::tikz::{canonical_pair, count_undirected_multiplicities};
use super::{
    DEFAULT_TOPOLOGY_LAYOUT_BOUNDS, GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle,
    TopologyNode, TopologyNodeShape,
};
use std::collections::HashMap;
use std::fmt::Write as _;

/// Serializes a [`GraphTopologyDocument`] into a standalone Scalable Vector Graphics (`.svg`) figure.
#[must_use]
#[verified_engine::verified]
pub fn to_svg_topology(doc: &GraphTopologyDocument) -> String {
    let mut positioned = doc.clone();
    positioned.ensure_positions(DEFAULT_TOPOLOGY_LAYOUT_BOUNDS);
    let coords = normalize_svg_coordinates(&positioned.nodes);
    let id_to_idx: HashMap<&str, usize> = positioned
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();

    let mut out = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"520\" viewBox=\"0 0 720 520\">\n\
         <defs>\n\
           <marker id=\"topo-arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\">\n\
             <path d=\"M 0 1.5 L 10 5 L 0 8.5 z\" fill=\"#64748B\"/>\n\
           </marker>\n\
         </defs>\n\
         <rect width=\"720\" height=\"520\" fill=\"#0F172A\" rx=\"8\"/>\n",
    );
    let _ = writeln!(out, "  <title>{}</title>", escape_xml(&positioned.name));

    let pair_counts = count_undirected_multiplicities(&positioned.edges);
    let mut pair_seen: HashMap<(String, String), usize> = HashMap::new();
    for edge in &positioned.edges {
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
        write_svg_edge(
            &mut out,
            edge,
            (coords[u], coords[v]),
            (u == v, positioned.directed, mult, ord),
        );
    }

    for (idx, node) in positioned.nodes.iter().enumerate() {
        write_svg_node(&mut out, node, coords[idx]);
    }

    out.push_str("</svg>\n");
    out
}

#[verified_engine::verified]
fn normalize_svg_coordinates(nodes: &[TopologyNode]) -> Vec<[f64; 2]> {
    if nodes.is_empty() {
        return Vec::new();
    }
    if nodes.len() == 1 {
        return vec![[360.0, 260.0]];
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for node in nodes {
        let [x, y] = node.position.unwrap_or([360.0, 260.0]);
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
            [
                70.0 + ((x - min_x) / span_x) * 580.0,
                70.0 + ((y - min_y) / span_y) * 380.0,
            ]
        })
        .collect()
}

#[verified_engine::verified]
fn write_svg_edge(
    out: &mut String,
    edge: &TopologyEdge,
    (p1, p2): ([f64; 2], [f64; 2]),
    (self_loop, directed, mult, ord): (bool, bool, usize, usize),
) {
    let color = edge.color_hex.as_deref().unwrap_or("#94A3B8");
    let width = match edge.style {
        TopologyEdgeStyle::Thick => "2.8",
        TopologyEdgeStyle::Solid | TopologyEdgeStyle::Dashed => "1.8",
    };
    let dash = if edge.style == TopologyEdgeStyle::Dashed {
        " stroke-dasharray=\"6,4\""
    } else {
        ""
    };
    let marker = if directed {
        " marker-end=\"url(#topo-arrow)\""
    } else {
        ""
    };

    let label_pos = if self_loop {
        let r = 22.0 + (ord as f64) * 8.0;
        let _ = writeln!(
            out,
            "  <path d=\"M {},{} C {},{} {},{} {},{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{width}\"{dash}{marker}/>",
            format_compact_f64(p1[0] - 8.0),
            format_compact_f64(p1[1] - 12.0),
            format_compact_f64(p1[0] - r),
            format_compact_f64(p1[1] - 2.0 * r),
            format_compact_f64(p1[0] + r),
            format_compact_f64(p1[1] - 2.0 * r),
            format_compact_f64(p1[0] + 8.0),
            format_compact_f64(p1[1] - 12.0),
            escape_xml(color)
        );
        [p1[0], p1[1] - 1.7 * r]
    } else {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let len = (dx * dx + dy * dy).sqrt().max(1.0);
        let ux = dx / len;
        let uy = dy / len;
        let sx = p1[0] + ux * 18.0;
        let sy = p1[1] + uy * 18.0;
        let ex = p2[0] - ux * (if directed { 21.0 } else { 18.0 });
        let ey = p2[1] - uy * (if directed { 21.0 } else { 18.0 });
        if mult > 1 {
            let offset = ((ord as f64) - (mult as f64 - 1.0) * 0.5) * 26.0;
            let cx = (sx + ex) * 0.5 - uy * offset;
            let cy = (sy + ey) * 0.5 + ux * offset;
            let _ = writeln!(
                out,
                "  <path d=\"M {},{} Q {},{} {},{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{width}\"{dash}{marker}/>",
                format_compact_f64(sx),
                format_compact_f64(sy),
                format_compact_f64(cx),
                format_compact_f64(cy),
                format_compact_f64(ex),
                format_compact_f64(ey),
                escape_xml(color)
            );
            [(sx + 2.0 * cx + ex) * 0.25, (sy + 2.0 * cy + ey) * 0.25]
        } else {
            let _ = writeln!(
                out,
                "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{width}\"{dash}{marker}/>",
                format_compact_f64(sx),
                format_compact_f64(sy),
                format_compact_f64(ex),
                format_compact_f64(ey),
                escape_xml(color)
            );
            [(sx + ex) * 0.5, (sy + ey) * 0.5 - 6.0]
        }
    };

    let caption = svg_edge_caption(edge);
    if !caption.is_empty() {
        let _ = writeln!(
            out,
            "  <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#E2E8F0\" font-family=\"sans-serif\" font-size=\"11\">{}</text>",
            format_compact_f64(label_pos[0]),
            format_compact_f64(label_pos[1]),
            escape_xml(&caption)
        );
    }
}

#[verified_engine::verified]
fn svg_edge_caption(edge: &TopologyEdge) -> String {
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
fn write_svg_node(out: &mut String, node: &TopologyNode, xy: [f64; 2]) {
    let fill = node.fill_hex.as_deref().unwrap_or("#1E3A8A");
    let stroke = node.stroke_hex.as_deref().unwrap_or("#60A5FA");
    let x = format_compact_f64(xy[0]);
    let y = format_compact_f64(xy[1]);
    match node.shape {
        TopologyNodeShape::Box => {
            let _ = writeln!(
                out,
                "  <rect x=\"{}\" y=\"{}\" width=\"36\" height=\"28\" rx=\"5\" fill=\"{}\" stroke=\"{}\" stroke-width=\"2\"/>",
                format_compact_f64(xy[0] - 18.0),
                format_compact_f64(xy[1] - 14.0),
                escape_xml(fill),
                escape_xml(stroke)
            );
        }
        TopologyNodeShape::DoubleCircle => {
            let _ = writeln!(
                out,
                "  <circle cx=\"{x}\" cy=\"{y}\" r=\"20\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.6\"/>",
                escape_xml(stroke)
            );
            let _ = writeln!(
                out,
                "  <circle cx=\"{x}\" cy=\"{y}\" r=\"16\" fill=\"{}\" stroke=\"{}\" stroke-width=\"2\"/>",
                escape_xml(fill),
                escape_xml(stroke)
            );
        }
        TopologyNodeShape::Circle => {
            let _ = writeln!(
                out,
                "  <circle cx=\"{x}\" cy=\"{y}\" r=\"17\" fill=\"{}\" stroke=\"{}\" stroke-width=\"2\"/>",
                escape_xml(fill),
                escape_xml(stroke)
            );
        }
    }
    let _ = writeln!(
        out,
        "  <text x=\"{x}\" y=\"{}\" text-anchor=\"middle\" fill=\"#F8FAFC\" font-family=\"sans-serif\" font-size=\"11\" font-weight=\"600\">{}</text>",
        format_compact_f64(xy[1] + 4.0),
        escape_xml(&node.label)
    );
}

// [cite:graph_parameters_rust]
