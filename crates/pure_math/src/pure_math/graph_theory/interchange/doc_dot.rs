//! Rich `.dot` serializer and parser for [`super::GraphTopologyDocument`],
//! reusing the shared statement and escaping primitives in `dot` and `escape`.

use super::document::{check_import_byte_limit, strip_utf8_bom_str};
use super::dot::{
    extract_dot_edge_weight, is_global_dot_attr, parse_dot_attrs, parse_dot_header_and_name,
    parse_dot_pos, split_dot_statements, split_head_and_attrs, strip_dot_comments, unquote_dot,
};
use super::escape::{
    edge_style_to_str, ensure_node_index, escape_dot_quoted, format_compact_f64, shape_to_str,
    str_to_edge_style, str_to_shape,
};
use super::{
    GraphTheoryError, GraphTopologyDocument, MAX_TOPOLOGY_EDGES, TopologyEdge, TopologyEdgeStyle,
    TopologyNode, TopologyNodeShape,
};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

/// Serializes a [`GraphTopologyDocument`] deterministically into Graphviz `.dot` format.
#[must_use]
#[verified_engine::verified]
pub fn to_dot_topology(doc: &GraphTopologyDocument) -> String {
    let mut out = String::new();
    let keyword = if doc.directed { "digraph" } else { "graph" };
    let connector = if doc.directed { "->" } else { "--" };
    let graph_name = if doc.name.trim().is_empty() {
        "G"
    } else {
        doc.name.trim()
    };
    let _ = writeln!(out, "{keyword} {} {{", escape_dot_quoted(graph_name));
    for (k, v) in &doc.attributes {
        let _ = writeln!(
            out,
            "  {} = {};",
            escape_dot_quoted(k),
            escape_dot_quoted(v)
        );
    }
    for node in &doc.nodes {
        write_dot_topology_node(&mut out, node);
    }
    for edge in &doc.edges {
        write_dot_topology_edge(&mut out, edge, connector);
    }
    out.push_str("}\n");
    out
}

#[verified_engine::verified]
fn write_dot_topology_node(out: &mut String, node: &TopologyNode) {
    let mut attrs = vec![format!("label={}", escape_dot_quoted(&node.label))];
    if let Some([x, y]) = node.position {
        attrs.push(format!(
            "pos=\"{},{}\"",
            format_compact_f64(x),
            format_compact_f64(y)
        ));
    }
    if node.shape != TopologyNodeShape::Circle {
        attrs.push(format!(
            "shape={}",
            escape_dot_quoted(shape_to_str(node.shape))
        ));
    }
    if let Some(fill) = &node.fill_hex {
        attrs.push(format!("fillcolor={}", escape_dot_quoted(fill)));
    }
    if let Some(stroke) = &node.stroke_hex {
        attrs.push(format!("color={}", escape_dot_quoted(stroke)));
    }
    for (k, v) in &node.attributes {
        attrs.push(format!("{}={}", escape_dot_quoted(k), escape_dot_quoted(v)));
    }
    let _ = writeln!(
        out,
        "  {} [{}];",
        escape_dot_quoted(&node.id),
        attrs.join(", ")
    );
}

#[verified_engine::verified]
fn write_dot_topology_edge(out: &mut String, edge: &TopologyEdge, connector: &str) {
    let mut attrs = vec![format!("weight={}", format_compact_f64(edge.weight))];
    if let Some(lbl) = &edge.label {
        attrs.push(format!("label={}", escape_dot_quoted(lbl)));
    }
    if edge.style != TopologyEdgeStyle::Solid {
        attrs.push(format!(
            "style={}",
            escape_dot_quoted(edge_style_to_str(edge.style))
        ));
    }
    if let Some(col) = &edge.color_hex {
        attrs.push(format!("color={}", escape_dot_quoted(col)));
    }
    for (k, v) in &edge.attributes {
        attrs.push(format!("{}={}", escape_dot_quoted(k), escape_dot_quoted(v)));
    }
    let _ = writeln!(
        out,
        "  {} {connector} {} [{}];",
        escape_dot_quoted(&edge.source),
        escape_dot_quoted(&edge.target),
        attrs.join(", ")
    );
}

/// Parses a Graphviz `.dot` (`graph` or `digraph`) specification into a [`GraphTopologyDocument`].
#[verified_engine::verified]
pub fn parse_dot_topology(raw: &str) -> Result<GraphTopologyDocument, GraphTheoryError> {
    let stripped = strip_utf8_bom_str(raw);
    check_import_byte_limit(stripped)?;
    let cleaned = strip_dot_comments(stripped);
    let Some(open_brace) = cleaned.find('{') else {
        return Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: "missing opening '{'".to_string(),
        });
    };
    let Some(close_brace) = cleaned.rfind('}') else {
        return Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: "missing closing '}'".to_string(),
        });
    };
    if close_brace < open_brace {
        return Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: "malformed brace delimiters".to_string(),
        });
    }
    let (directed, name_opt) = parse_dot_header_and_name(&cleaned[..open_brace])?;
    let mut doc = GraphTopologyDocument::new(name_opt.unwrap_or_else(|| "G".to_string()), directed);
    let mut index_map = HashMap::new();
    for stmt in split_dot_statements(&cleaned[(open_brace + 1)..close_brace]) {
        apply_dot_doc_stmt(&stmt, &mut doc, &mut index_map)?;
    }
    doc.validate()?;
    Ok(doc)
}

#[verified_engine::verified]
fn apply_dot_doc_stmt(
    stmt: &str,
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
) -> Result<(), GraphTheoryError> {
    let (head, attrs_raw) = split_head_and_attrs(stmt);
    let head = head.trim();
    if head.is_empty() {
        return Ok(());
    }
    if head.contains('=') && !head.contains("->") && !head.contains("--") && attrs_raw.is_none() {
        if let Some((k, v)) = head.split_once('=') {
            doc.attributes
                .insert(unquote_dot(k.trim()), unquote_dot(v.trim()));
        }
        return Ok(());
    }
    if is_global_dot_attr(head, attrs_raw.is_some()) {
        return Ok(());
    }
    let attrs = attrs_raw.map(parse_dot_attrs).unwrap_or_default();
    let edge_op = if doc.directed { "->" } else { "--" };
    let alt_op = if doc.directed { "--" } else { "->" };
    if head.contains(alt_op) {
        return Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: format!("edge operator '{alt_op}' does not match graph header"),
        });
    }
    if head.contains(edge_op) {
        return apply_dot_doc_edge_chain(head, edge_op, &attrs, doc, index_map);
    }
    let id = unquote_dot(head);
    if !id.is_empty() {
        let idx = ensure_node_index(&mut doc.nodes, index_map, &id)?;
        apply_dot_doc_node_attrs(idx, &mut doc.nodes[idx], &attrs)?;
    }
    Ok(())
}

#[verified_engine::verified]
fn apply_dot_doc_node_attrs(
    idx: usize,
    node: &mut TopologyNode,
    attrs: &BTreeMap<String, String>,
) -> Result<(), GraphTheoryError> {
    for (k, v) in attrs {
        match k.as_str() {
            "label" => node.label.clone_from(v),
            "pos" => {
                let (x, y) = parse_dot_pos(idx, v)?;
                node.position = Some([x, y]);
            }
            "shape" => node.shape = str_to_shape(v),
            "fillcolor" => node.fill_hex = Some(v.clone()),
            "color" => node.stroke_hex = Some(v.clone()),
            other => {
                node.attributes.insert(other.to_string(), v.clone());
            }
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn apply_dot_doc_edge_chain(
    head: &str,
    edge_op: &str,
    attrs: &BTreeMap<String, String>,
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
) -> Result<(), GraphTheoryError> {
    let weight = extract_dot_edge_weight(attrs)?;
    let endpoints: Vec<String> = head.split(edge_op).map(unquote_dot).collect();
    if endpoints.len() < 2 || endpoints.iter().any(String::is_empty) {
        return Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: "incomplete or empty endpoint in edge statement".to_string(),
        });
    }
    for ep in &endpoints {
        ensure_node_index(&mut doc.nodes, index_map, ep)?;
    }
    for pair in endpoints.windows(2) {
        if doc.edges.len() >= MAX_TOPOLOGY_EDGES {
            return Err(GraphTheoryError::EdgeLimitExceeded {
                count: doc.edges.len() + 1,
                max: MAX_TOPOLOGY_EDGES,
            });
        }
        let mut edge = TopologyEdge::new(pair[0].clone(), pair[1].clone(), weight);
        for (k, v) in attrs {
            match k.as_str() {
                "weight" => {}
                "label" => edge.label = Some(v.clone()),
                "style" => edge.style = str_to_edge_style(v),
                "color" => edge.color_hex = Some(v.clone()),
                other => {
                    edge.attributes.insert(other.to_string(), v.clone());
                }
            }
        }
        doc.edges.push(edge);
    }
    Ok(())
}

// [cite:graph_parameters_rust]
