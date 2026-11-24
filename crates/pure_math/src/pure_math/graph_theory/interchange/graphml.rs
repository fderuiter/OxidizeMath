//! Deterministic `.graphml` XML parser and serializer for weighted graphs.

use super::escape::{escape_xml, unescape_xml};
use super::{
    GraphTheoryError, InterchangeGraph, InterchangeNode, clamp_interchange_limit, format_f64,
    parse_graphml_topology,
};
use std::collections::BTreeMap;

/// Serializes `graph` into deterministic GraphML XML text with node coordinates and edge weights.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::interchange::{
///     InterchangeGraph, parse_graphml, to_graphml,
/// };
///
/// let mut g = InterchangeGraph::new(true);
/// g.push_edge(0, 1, 4.25)?;
/// let xml = to_graphml(&g)?;
/// let round_trip = parse_graphml(&xml)?;
/// assert!(round_trip.directed);
/// assert!((round_trip.edges[0].weight - 4.25).abs() < 1e-9);
/// # Ok::<(), math_explorer::error::Error>(())
/// ```
#[verified_engine::verified]
pub fn to_graphml(graph: &InterchangeGraph) -> Result<String, GraphTheoryError> {
    graph.validate()?;
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n");
    out.push_str("  <key id=\"d_label\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>\n");
    out.push_str("  <key id=\"d_x\" for=\"node\" attr.name=\"x\" attr.type=\"double\"/>\n");
    out.push_str("  <key id=\"d_y\" for=\"node\" attr.name=\"y\" attr.type=\"double\"/>\n");
    out.push_str(
        "  <key id=\"d_weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n",
    );
    let def = if graph.directed {
        "directed"
    } else {
        "undirected"
    };
    out.push_str(&format!("  <graph id=\"G\" edgedefault=\"{def}\">\n"));
    for (&id, node) in &graph.nodes {
        append_graphml_node(&mut out, id, node);
    }
    for (idx, edge) in graph.edges.iter().enumerate() {
        out.push_str(&format!(
            "    <edge id=\"e{idx}\" source=\"n{}\" target=\"n{}\">\n      <data key=\"d_weight\">{}</data>\n    </edge>\n",
            edge.source,
            edge.target,
            format_f64(edge.weight)
        ));
    }
    out.push_str("  </graph>\n</graphml>\n");
    Ok(out)
}

#[verified_engine::verified]
fn append_graphml_node(out: &mut String, id: usize, node: &InterchangeNode) {
    if node.label.is_none() && node.pos.is_none() {
        out.push_str(&format!("    <node id=\"n{id}\"/>\n"));
        return;
    }
    out.push_str(&format!("    <node id=\"n{id}\">\n"));
    if let Some(lbl) = &node.label {
        out.push_str(&format!(
            "      <data key=\"d_label\">{}</data>\n",
            escape_xml(lbl)
        ));
    }
    if let Some((x, y)) = node.pos {
        out.push_str(&format!(
            "      <data key=\"d_x\">{}</data>\n",
            format_f64(x)
        ));
        out.push_str(&format!(
            "      <data key=\"d_y\">{}</data>\n",
            format_f64(y)
        ));
    }
    out.push_str("    </node>\n");
}

/// Parses a `.graphml` XML document into a validated [`InterchangeGraph`].
#[verified_engine::verified]
pub fn parse_graphml(raw: &str) -> Result<InterchangeGraph, GraphTheoryError> {
    parse_graphml_topology(raw)
        .and_then(|doc| doc.to_interchange())
        .map_err(clamp_interchange_limit)
}

#[verified_engine::verified]
pub(super) fn reject_xml_dtd(input: &str) -> Result<(), GraphTheoryError> {
    let lower = input.to_ascii_lowercase();
    if lower.contains("<!doctype") || lower.contains("<!entity") {
        return Err(GraphTheoryError::ForbiddenXmlDeclaration);
    }
    Ok(())
}

#[verified_engine::verified]
pub(super) fn detect_graphml_header(input: &str) -> (bool, Option<String>) {
    let mut search = input;
    while let Some(pos) = search.find("<graph") {
        let after = &search[(pos + 6)..];
        let next_ch = after.chars().next();
        if matches!(next_ch, Some(ch) if ch.is_whitespace() || ch == '>') {
            if let Some(end) = after.find('>') {
                let attrs = parse_xml_attrs(&after[..end]);
                let dir = attrs
                    .get("edgedefault")
                    .is_some_and(|v| v.eq_ignore_ascii_case("directed"));
                return (dir, attrs.get("id").cloned());
            }
            break;
        }
        search = after;
    }
    (false, None)
}

#[verified_engine::verified]
pub(super) fn extract_element_body<'a>(
    after_open: &'a str,
    close_tag: &str,
    self_closing: bool,
) -> Result<(&'a str, &'a str), GraphTheoryError> {
    if self_closing {
        return Ok(("", after_open));
    }
    let Some(close_pos) = after_open.find(close_tag) else {
        return Err(GraphTheoryError::ParseError {
            format: "GraphML",
            message: format!("missing closing tag {close_tag}"),
        });
    };
    Ok((
        &after_open[..close_pos],
        &after_open[(close_pos + close_tag.len())..],
    ))
}

#[verified_engine::verified]
pub(super) fn parse_data_entries(inner: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut rest = inner;
    while let Some(pos) = rest.find("<data") {
        let after = &rest[(pos + 5)..];
        let Some(tag_end) = after.find('>') else {
            break;
        };
        let attrs = parse_xml_attrs(after[..tag_end].trim_end_matches('/'));
        let after_open = &after[(tag_end + 1)..];
        let Some(close_pos) = after_open.find("</data>") else {
            break;
        };
        if let Some(k) = attrs.get("key") {
            map.insert(k.clone(), after_open[..close_pos].trim().to_string());
        }
        rest = &after_open[(close_pos + 7)..];
    }
    map
}

#[verified_engine::verified]
pub(super) fn parse_xml_attrs(raw: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key = raw[key_start..i].trim();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            i += 1;
            continue;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
            let quote = bytes[i];
            i += 1;
            let val_start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            if !key.is_empty() {
                map.insert(key.to_string(), unescape_xml(&raw[val_start..i]));
            }
            i = (i + 1).min(bytes.len());
        }
    }
    map
}

// [cite:graph_parameters_rust]
