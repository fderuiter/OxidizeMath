//! Rich `.graphml` XML serializer and parser for [`super::GraphTopologyDocument`],
//! reusing the shared XML tag and escaping primitives in `graphml` and `escape`.

use super::document::{check_import_byte_limit, strip_utf8_bom_str};
use super::escape::{
    edge_style_to_str, ensure_node_index, escape_xml, format_compact_f64, shape_to_str,
    str_to_edge_style, str_to_shape, unescape_xml,
};
use super::graphml::{
    detect_graphml_header, extract_element_body, parse_data_entries, parse_xml_attrs,
    reject_xml_dtd,
};
use super::{
    GraphTheoryError, GraphTopologyDocument, MAX_TOPOLOGY_EDGES, TopologyEdge, TopologyEdgeStyle,
    TopologyNode, TopologyNodeShape,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

/// Serializes a [`GraphTopologyDocument`] deterministically into Gephi/Cytoscape `.graphml` XML.
#[must_use]
#[verified_engine::verified]
pub fn to_graphml_topology(doc: &GraphTopologyDocument) -> String {
    let mut out = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n\
           <key id=\"d_label\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>\n\
           <key id=\"d_x\" for=\"node\" attr.name=\"x\" attr.type=\"double\"/>\n\
           <key id=\"d_y\" for=\"node\" attr.name=\"y\" attr.type=\"double\"/>\n\
           <key id=\"d_fill\" for=\"node\" attr.name=\"fill\" attr.type=\"string\"/>\n\
           <key id=\"d_stroke\" for=\"node\" attr.name=\"stroke\" attr.type=\"string\"/>\n\
           <key id=\"d_shape\" for=\"node\" attr.name=\"shape\" attr.type=\"string\"/>\n\
           <key id=\"e_weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n\
           <key id=\"e_label\" for=\"edge\" attr.name=\"label\" attr.type=\"string\"/>\n\
           <key id=\"e_style\" for=\"edge\" attr.name=\"style\" attr.type=\"string\"/>\n\
           <key id=\"e_color\" for=\"edge\" attr.name=\"color\" attr.type=\"string\"/>\n",
    );
    let node_keys = collect_sorted_keys(doc.nodes.iter().map(|n| &n.attributes));
    let edge_keys = collect_sorted_keys(doc.edges.iter().map(|e| &e.attributes));
    let graph_keys: Vec<&String> = doc.attributes.keys().collect();
    for (i, k) in graph_keys.iter().enumerate() {
        write_graphml_key(&mut out, &format!("gk{i}"), "graph", k);
    }
    for (i, k) in node_keys.iter().enumerate() {
        write_graphml_key(&mut out, &format!("nk{i}"), "node", k);
    }
    for (i, k) in edge_keys.iter().enumerate() {
        write_graphml_key(&mut out, &format!("ek{i}"), "edge", k);
    }
    let edgedef = if doc.directed {
        "directed"
    } else {
        "undirected"
    };
    let gid = if doc.name.trim().is_empty() {
        "G"
    } else {
        doc.name.trim()
    };
    let _ = writeln!(
        out,
        "  <graph id=\"{}\" edgedefault=\"{edgedef}\">",
        escape_xml(gid)
    );
    for (i, k) in graph_keys.iter().enumerate() {
        if let Some(v) = doc.attributes.get(*k) {
            let _ = writeln!(out, "    <data key=\"gk{i}\">{}</data>", escape_xml(v));
        }
    }
    for node in &doc.nodes {
        write_graphml_doc_node(&mut out, node, &node_keys);
    }
    for (eidx, edge) in doc.edges.iter().enumerate() {
        write_graphml_doc_edge(&mut out, eidx, edge, &edge_keys);
    }
    out.push_str("  </graph>\n</graphml>\n");
    out
}

#[verified_engine::verified]
fn collect_sorted_keys<'a>(
    maps: impl Iterator<Item = &'a BTreeMap<String, String>>,
) -> Vec<String> {
    let mut set = BTreeSet::new();
    for map in maps {
        for k in map.keys() {
            set.insert(k.clone());
        }
    }
    set.into_iter().collect()
}

#[verified_engine::verified]
fn write_graphml_key(out: &mut String, id: &str, domain: &str, name: &str) {
    let _ = writeln!(
        out,
        "  <key id=\"{id}\" for=\"{domain}\" attr.name=\"{}\" attr.type=\"string\"/>",
        escape_xml(name)
    );
}

#[verified_engine::verified]
fn write_graphml_doc_node(out: &mut String, node: &TopologyNode, custom_keys: &[String]) {
    let _ = writeln!(out, "    <node id=\"{}\">", escape_xml(&node.id));
    let _ = writeln!(
        out,
        "      <data key=\"d_label\">{}</data>",
        escape_xml(&node.label)
    );
    if let Some([x, y]) = node.position {
        let _ = writeln!(
            out,
            "      <data key=\"d_x\">{}</data>",
            format_compact_f64(x)
        );
        let _ = writeln!(
            out,
            "      <data key=\"d_y\">{}</data>",
            format_compact_f64(y)
        );
    }
    if let Some(fill) = &node.fill_hex {
        let _ = writeln!(
            out,
            "      <data key=\"d_fill\">{}</data>",
            escape_xml(fill)
        );
    }
    if let Some(stroke) = &node.stroke_hex {
        let _ = writeln!(
            out,
            "      <data key=\"d_stroke\">{}</data>",
            escape_xml(stroke)
        );
    }
    if node.shape != TopologyNodeShape::Circle {
        let _ = writeln!(
            out,
            "      <data key=\"d_shape\">{}</data>",
            shape_to_str(node.shape)
        );
    }
    for (i, k) in custom_keys.iter().enumerate() {
        if let Some(v) = node.attributes.get(k) {
            let _ = writeln!(out, "      <data key=\"nk{i}\">{}</data>", escape_xml(v));
        }
    }
    out.push_str("    </node>\n");
}

#[verified_engine::verified]
fn write_graphml_doc_edge(
    out: &mut String,
    eidx: usize,
    edge: &TopologyEdge,
    custom_keys: &[String],
) {
    let _ = writeln!(
        out,
        "    <edge id=\"e{eidx}\" source=\"{}\" target=\"{}\">",
        escape_xml(&edge.source),
        escape_xml(&edge.target)
    );
    let _ = writeln!(
        out,
        "      <data key=\"e_weight\">{}</data>",
        format_compact_f64(edge.weight)
    );
    if let Some(lbl) = &edge.label {
        let _ = writeln!(
            out,
            "      <data key=\"e_label\">{}</data>",
            escape_xml(lbl)
        );
    }
    if edge.style != TopologyEdgeStyle::Solid {
        let _ = writeln!(
            out,
            "      <data key=\"e_style\">{}</data>",
            edge_style_to_str(edge.style)
        );
    }
    if let Some(col) = &edge.color_hex {
        let _ = writeln!(
            out,
            "      <data key=\"e_color\">{}</data>",
            escape_xml(col)
        );
    }
    for (i, k) in custom_keys.iter().enumerate() {
        if let Some(v) = edge.attributes.get(k) {
            let _ = writeln!(out, "      <data key=\"ek{i}\">{}</data>", escape_xml(v));
        }
    }
    out.push_str("    </edge>\n");
}

/// Parses a DTD-free `.graphml` XML document into a [`GraphTopologyDocument`].
#[verified_engine::verified]
pub fn parse_graphml_topology(raw: &str) -> Result<GraphTopologyDocument, GraphTheoryError> {
    let input = strip_utf8_bom_str(raw);
    check_import_byte_limit(input)?;
    reject_xml_dtd(input)?;
    if !input.contains("<graph") {
        return Err(GraphTheoryError::ParseError {
            format: "GraphML",
            message: "missing <graph> element".to_string(),
        });
    }
    let key_names = collect_key_attr_names(input);
    let (directed, gid) = detect_graphml_header(input);
    let mut doc = GraphTopologyDocument::new(gid.unwrap_or_else(|| "G".to_string()), directed);
    let mut index_map = HashMap::new();
    parse_graphml_doc_nodes(input, &key_names, &mut doc, &mut index_map)?;
    parse_graphml_doc_edges(input, &key_names, &mut doc, &mut index_map)?;
    doc.validate()?;
    Ok(doc)
}

#[verified_engine::verified]
fn collect_key_attr_names(input: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut rest = input;
    while let Some(pos) = rest.find("<key") {
        let after = &rest[(pos + 4)..];
        let Some(end) = after.find('>') else {
            break;
        };
        let attrs = parse_xml_attrs(after[..end].trim_end_matches('/'));
        if let (Some(id), Some(name)) = (attrs.get("id"), attrs.get("attr.name")) {
            map.insert(id.clone(), name.clone());
        }
        rest = &after[(end + 1)..];
    }
    map
}

#[verified_engine::verified]
fn parse_graphml_doc_nodes(
    input: &str,
    key_names: &HashMap<String, String>,
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
) -> Result<(), GraphTheoryError> {
    let mut rest = input;
    while let Some(pos) = rest.find("<node") {
        let after = &rest[(pos + 5)..];
        if !after
            .chars()
            .next()
            .is_some_and(|c| c.is_whitespace() || c == '/' || c == '>')
        {
            rest = after;
            continue;
        }
        let Some(tag_end) = after.find('>') else {
            return Err(GraphTheoryError::ParseError {
                format: "GraphML",
                message: "unclosed <node> tag".to_string(),
            });
        };
        let header = &after[..tag_end];
        let self_closing = header.trim_end().ends_with('/');
        let attrs = parse_xml_attrs(header.trim_end_matches('/'));
        let id = attrs
            .get("id")
            .ok_or_else(|| GraphTheoryError::ParseError {
                format: "GraphML",
                message: "<node> missing 'id'".to_string(),
            })?;
        let (inner, next_slice) =
            extract_element_body(&after[(tag_end + 1)..], "</node>", self_closing)?;
        let idx = ensure_node_index(&mut doc.nodes, index_map, id)?;
        populate_graphml_doc_node(idx, &mut doc.nodes[idx], inner, key_names)?;
        rest = next_slice;
    }
    Ok(())
}

#[verified_engine::verified]
fn populate_graphml_doc_node(
    idx: usize,
    node: &mut TopologyNode,
    inner: &str,
    key_names: &HashMap<String, String>,
) -> Result<(), GraphTheoryError> {
    let mut x_val: Option<f64> = None;
    let mut y_val: Option<f64> = None;
    for (k, raw_v) in parse_data_entries(inner) {
        let attr_name = key_names.get(&k).map_or(k.as_str(), String::as_str);
        let v = unescape_xml(&raw_v);
        match attr_name {
            "label" | "d_label" => node.label = v,
            "x" | "d_x" => x_val = Some(parse_graphml_coord(idx, &v)?),
            "y" | "d_y" => y_val = Some(parse_graphml_coord(idx, &v)?),
            "fill" | "d_fill" => node.fill_hex = Some(v),
            "stroke" | "d_stroke" => node.stroke_hex = Some(v),
            "shape" | "d_shape" => node.shape = str_to_shape(&v),
            other => {
                node.attributes.insert(other.to_string(), v);
            }
        }
    }
    if let (Some(x), Some(y)) = (x_val, y_val) {
        node.position = Some([x, y]);
    }
    Ok(())
}

#[verified_engine::verified]
fn parse_graphml_coord(node: usize, raw: &str) -> Result<f64, GraphTheoryError> {
    let val = raw
        .trim()
        .parse::<f64>()
        .map_err(|_| GraphTheoryError::ParseError {
            format: "GraphML",
            message: format!("invalid coordinate '{raw}'"),
        })?;
    if !val.is_finite() {
        return Err(GraphTheoryError::InvalidCoordinate {
            node,
            x: val,
            y: val,
        });
    }
    Ok(val)
}

#[verified_engine::verified]
fn parse_graphml_doc_edges(
    input: &str,
    key_names: &HashMap<String, String>,
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
) -> Result<(), GraphTheoryError> {
    let mut rest = input;
    while let Some(pos) = rest.find("<edge") {
        let after = &rest[(pos + 5)..];
        if !after
            .chars()
            .next()
            .is_some_and(|c| c.is_whitespace() || c == '/' || c == '>')
        {
            rest = after;
            continue;
        }
        let Some(tag_end) = after.find('>') else {
            return Err(GraphTheoryError::ParseError {
                format: "GraphML",
                message: "unclosed <edge> tag".to_string(),
            });
        };
        let header = &after[..tag_end];
        let self_closing = header.trim_end().ends_with('/');
        let attrs = parse_xml_attrs(header.trim_end_matches('/'));
        let src = attrs
            .get("source")
            .ok_or_else(|| GraphTheoryError::ParseError {
                format: "GraphML",
                message: "<edge> missing 'source'".to_string(),
            })?;
        let dst = attrs
            .get("target")
            .ok_or_else(|| GraphTheoryError::ParseError {
                format: "GraphML",
                message: "<edge> missing 'target'".to_string(),
            })?;
        ensure_node_index(&mut doc.nodes, index_map, src)?;
        ensure_node_index(&mut doc.nodes, index_map, dst)?;
        let (inner, next_slice) =
            extract_element_body(&after[(tag_end + 1)..], "</edge>", self_closing)?;
        if doc.edges.len() >= MAX_TOPOLOGY_EDGES {
            return Err(GraphTheoryError::EdgeLimitExceeded {
                count: doc.edges.len() + 1,
                max: MAX_TOPOLOGY_EDGES,
            });
        }
        doc.edges
            .push(build_graphml_doc_edge(src, dst, &attrs, inner, key_names)?);
        rest = next_slice;
    }
    Ok(())
}

#[verified_engine::verified]
fn build_graphml_doc_edge(
    src: &str,
    dst: &str,
    tag_attrs: &BTreeMap<String, String>,
    inner: &str,
    key_names: &HashMap<String, String>,
) -> Result<TopologyEdge, GraphTheoryError> {
    let mut edge = TopologyEdge::new(src, dst, 1.0);
    if let Some(attr_w) = tag_attrs.get("weight") {
        edge.weight = parse_graphml_weight_str(attr_w)?;
    }
    for (k, raw_v) in parse_data_entries(inner) {
        let attr_name = key_names.get(&k).map_or(k.as_str(), String::as_str);
        let v = unescape_xml(&raw_v);
        match attr_name {
            "weight" | "d_weight" | "e_weight" => edge.weight = parse_graphml_weight_str(&v)?,
            "label" | "e_label" => edge.label = Some(v),
            "style" | "e_style" => edge.style = str_to_edge_style(&v),
            "color" | "e_color" => edge.color_hex = Some(v),
            other => {
                edge.attributes.insert(other.to_string(), v);
            }
        }
    }
    Ok(edge)
}

#[verified_engine::verified]
fn parse_graphml_weight_str(raw: &str) -> Result<f64, GraphTheoryError> {
    let w = raw
        .trim()
        .parse::<f64>()
        .map_err(|_| GraphTheoryError::ParseError {
            format: "GraphML",
            message: format!("invalid edge weight '{raw}'"),
        })?;
    if !w.is_finite() {
        return Err(GraphTheoryError::InvalidWeight { value: w });
    }
    Ok(w)
}

// [cite:graph_parameters_rust]
