//! Escaping, quoting, XML entity decoding, color parsing, and style mapping helpers for `interchange` codecs.

use super::{
    GraphTheoryError, MAX_TOPOLOGY_NODES, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};
use std::collections::HashMap;
use std::fmt::Write as _;

#[verified_engine::verified]
pub(super) fn format_compact_f64(value: f64) -> String {
    if !value.is_finite() {
        return "0.0".to_string();
    }
    let rounded = (value * 1e6).round() / 1e6;
    if (rounded - rounded.round()).abs() < 1e-9 {
        format!("{:.1}", rounded.round())
    } else {
        let s = format!("{rounded:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[verified_engine::verified]
pub(super) fn escape_dot_quoted(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 4);
    out.push('"');
    for ch in raw.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => {}
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[verified_engine::verified]
pub(super) fn escape_xml(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if c.is_control() && c != '\n' && c != '\t' => {}
            c => out.push(c),
        }
    }
    out
}

#[verified_engine::verified]
pub(super) fn unescape_xml(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(amp_idx) = rest.find('&') {
        out.push_str(&rest[..amp_idx]);
        let after = &rest[amp_idx + 1..];
        if let Some(semi_idx) = after.find(';') {
            decode_xml_entity(&mut out, &after[..semi_idx]);
            rest = &after[semi_idx + 1..];
        } else {
            out.push('&');
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

#[verified_engine::verified]
fn decode_xml_entity(out: &mut String, entity: &str) {
    match entity {
        "amp" => out.push('&'),
        "lt" => out.push('<'),
        "gt" => out.push('>'),
        "quot" => out.push('"'),
        "apos" => out.push('\''),
        num if num.starts_with("#x") || num.starts_with("#X") => {
            if let Ok(code) = u32::from_str_radix(&num[2..], 16)
                && let Some(c) = char::from_u32(code)
            {
                out.push(c);
            }
        }
        num if num.starts_with('#') => {
            if let Ok(code) = num[1..].parse::<u32>()
                && let Some(c) = char::from_u32(code)
            {
                out.push(c);
            }
        }
        other => {
            let _ = write!(out, "&{other};");
        }
    }
}

#[verified_engine::verified]
pub(super) fn escape_csv_field(raw: &str) -> String {
    if raw.contains([',', '"', '\n', '\r', '\t']) {
        let mut out = String::with_capacity(raw.len() + 2);
        out.push('"');
        for ch in raw.chars() {
            if ch == '"' {
                out.push_str("\"\"");
            } else {
                out.push(ch);
            }
        }
        out.push('"');
        out
    } else {
        raw.to_string()
    }
}

#[verified_engine::verified]
pub(super) fn escape_latex_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 4);
    for ch in raw.chars() {
        match ch {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '$' => out.push_str("\\$"),
            '&' => out.push_str("\\&"),
            '#' => out.push_str("\\#"),
            '_' => out.push_str("\\_"),
            '%' => out.push_str("\\%"),
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            '\n' | '\r' => out.push(' '),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

#[verified_engine::verified]
pub(super) fn parse_hex_rgb(hex: &str) -> Option<[u8; 3]> {
    let trimmed = hex.trim().strip_prefix('#').unwrap_or(hex.trim());
    if trimmed.len() == 6 {
        let r = u8::from_str_radix(&trimmed[0..2], 16).ok()?;
        let g = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
        let b = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
        Some([r, g, b])
    } else {
        None
    }
}

#[verified_engine::verified]
pub(super) fn shape_to_str(shape: TopologyNodeShape) -> &'static str {
    match shape {
        TopologyNodeShape::Circle => "circle",
        TopologyNodeShape::DoubleCircle => "doublecircle",
        TopologyNodeShape::Box => "box",
    }
}

#[verified_engine::verified]
pub(super) fn str_to_shape(raw: &str) -> TopologyNodeShape {
    match raw.trim().to_ascii_lowercase().as_str() {
        "doublecircle" => TopologyNodeShape::DoubleCircle,
        "box" | "rect" | "rectangle" | "square" => TopologyNodeShape::Box,
        _ => TopologyNodeShape::Circle,
    }
}

#[verified_engine::verified]
pub(super) fn edge_style_to_str(style: TopologyEdgeStyle) -> &'static str {
    match style {
        TopologyEdgeStyle::Solid => "solid",
        TopologyEdgeStyle::Dashed => "dashed",
        TopologyEdgeStyle::Thick => "bold",
    }
}

#[verified_engine::verified]
pub(super) fn str_to_edge_style(raw: &str) -> TopologyEdgeStyle {
    let lower = raw.trim().to_ascii_lowercase();
    if lower.contains("dashed") || lower.contains("dotted") {
        TopologyEdgeStyle::Dashed
    } else if lower.contains("bold") || lower.contains("thick") {
        TopologyEdgeStyle::Thick
    } else {
        TopologyEdgeStyle::Solid
    }
}

#[verified_engine::verified]
pub(super) fn ensure_node_index(
    nodes: &mut Vec<TopologyNode>,
    index_map: &mut HashMap<String, usize>,
    id: &str,
) -> Result<usize, GraphTheoryError> {
    if let Some(&idx) = index_map.get(id) {
        return Ok(idx);
    }
    if nodes.len() >= MAX_TOPOLOGY_NODES {
        return Err(GraphTheoryError::VertexLimitExceeded {
            count: nodes.len() + 1,
            max: MAX_TOPOLOGY_NODES,
        });
    }
    let idx = nodes.len();
    nodes.push(TopologyNode::new(id, id));
    index_map.insert(id.to_string(), idx);
    Ok(idx)
}

#[verified_engine::verified]
pub(super) fn looks_like_dot(text: &str) -> bool {
    let trimmed = text.trim_start();
    let head = trimmed
        .lines()
        .find(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with("//") && !t.starts_with('#')
        })
        .unwrap_or("")
        .trim_start()
        .to_ascii_lowercase();
    (head.starts_with("graph ")
        || head.starts_with("graph{")
        || head.starts_with("digraph ")
        || head.starts_with("digraph{")
        || head.starts_with("strict graph")
        || head.starts_with("strict digraph"))
        && trimmed.contains('{')
}

// [cite:graph_parameters_rust]
