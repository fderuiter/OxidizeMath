//! Deterministic `.graphml` XML parser and serializer for weighted graphs.

use super::{GraphTheoryError, InterchangeGraph, InterchangeNode, format_f64, resolve_token_id};
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

#[verified_engine::verified]
fn escape_xml(raw: &str) -> String {
    raw.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[verified_engine::verified]
fn unescape_xml(raw: &str) -> String {
    raw.replace("&quot;", "\"")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
}

#[derive(Default)]
struct GraphMlKeyMap {
    weight_key: Option<String>,
    x_key: Option<String>,
    y_key: Option<String>,
    label_key: Option<String>,
}

/// Parses a `.graphml` XML document into a validated [`InterchangeGraph`].
#[verified_engine::verified]
pub fn parse_graphml(input: &str) -> Result<InterchangeGraph, GraphTheoryError> {
    if !input.contains("<graph") {
        return Err(GraphTheoryError::ParseError {
            format: "GraphML",
            message: "missing <graph> element".to_string(),
        });
    }
    let keys = collect_graphml_keys(input);
    let directed = detect_graphml_direction(input);
    let mut graph = InterchangeGraph::new(directed);
    let mut sym_table = BTreeMap::new();
    let mut next_sym_id = 0usize;

    parse_all_graphml_nodes(input, &keys, &mut graph, &mut sym_table, &mut next_sym_id)?;
    parse_all_graphml_edges(input, &keys, &mut graph, &mut sym_table, &mut next_sym_id)?;
    graph.validate()?;
    Ok(graph)
}

#[verified_engine::verified]
fn collect_graphml_keys(input: &str) -> GraphMlKeyMap {
    let mut map = GraphMlKeyMap::default();
    let mut rest = input;
    while let Some(pos) = rest.find("<key") {
        let after = &rest[(pos + 4)..];
        let Some(end) = after.find('>') else {
            break;
        };
        let attrs = parse_xml_attrs(&after[..end]);
        if let (Some(id), Some(name)) = (attrs.get("id"), attrs.get("attr.name")) {
            match name.to_ascii_lowercase().as_str() {
                "weight" => map.weight_key = Some(id.clone()),
                "x" => map.x_key = Some(id.clone()),
                "y" => map.y_key = Some(id.clone()),
                "label" => map.label_key = Some(id.clone()),
                _ => {}
            }
        }
        rest = &after[(end + 1)..];
    }
    map
}

#[verified_engine::verified]
fn detect_graphml_direction(input: &str) -> bool {
    let mut search = input;
    while let Some(pos) = search.find("<graph") {
        let after = &search[(pos + 6)..];
        let next_ch = after.chars().next();
        if matches!(next_ch, Some(ch) if ch.is_whitespace() || ch == '>') {
            if let Some(end) = after.find('>') {
                let attrs = parse_xml_attrs(&after[..end]);
                return attrs
                    .get("edgedefault")
                    .is_some_and(|v| v.eq_ignore_ascii_case("directed"));
            }
            break;
        }
        search = after;
    }
    false
}

#[verified_engine::verified]
fn parse_all_graphml_nodes(
    input: &str,
    keys: &GraphMlKeyMap,
    graph: &mut InterchangeGraph,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
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
        let tag_header = &after[..tag_end];
        let self_closing = tag_header.trim_end().ends_with('/');
        let attrs = parse_xml_attrs(tag_header.trim_end_matches('/'));
        let raw_id = attrs
            .get("id")
            .ok_or_else(|| GraphTheoryError::ParseError {
                format: "GraphML",
                message: "<node> missing 'id' attribute".to_string(),
            })?;
        let id = resolve_token_id(raw_id, sym_table, next_sym_id);
        let (inner, next_slice) =
            extract_element_body(&after[(tag_end + 1)..], "</node>", self_closing)?;
        let node = build_graphml_node(id, inner, keys)?;
        graph.insert_node(node)?;
        rest = next_slice;
    }
    Ok(())
}

#[verified_engine::verified]
fn build_graphml_node(
    id: usize,
    inner: &str,
    keys: &GraphMlKeyMap,
) -> Result<InterchangeNode, GraphTheoryError> {
    let data = parse_data_entries(inner);
    let x_opt = lookup_data(&data, keys.x_key.as_deref(), &["d_x", "x"]);
    let y_opt = lookup_data(&data, keys.y_key.as_deref(), &["d_y", "y"]);
    let pos = match (x_opt, y_opt) {
        (Some(xs), Some(ys)) => {
            let x = xs
                .parse::<f64>()
                .map_err(|_| GraphTheoryError::ParseError {
                    format: "GraphML",
                    message: format!("invalid node x '{xs}'"),
                })?;
            let y = ys
                .parse::<f64>()
                .map_err(|_| GraphTheoryError::ParseError {
                    format: "GraphML",
                    message: format!("invalid node y '{ys}'"),
                })?;
            if !x.is_finite() || !y.is_finite() {
                return Err(GraphTheoryError::InvalidCoordinate { node: id, x, y });
            }
            Some((x, y))
        }
        _ => None,
    };
    let label =
        lookup_data(&data, keys.label_key.as_deref(), &["d_label", "label"]).map(unescape_xml);
    Ok(InterchangeNode { id, pos, label })
}

#[verified_engine::verified]
fn parse_all_graphml_edges(
    input: &str,
    keys: &GraphMlKeyMap,
    graph: &mut InterchangeGraph,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
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
        let tag_header = &after[..tag_end];
        let self_closing = tag_header.trim_end().ends_with('/');
        let attrs = parse_xml_attrs(tag_header.trim_end_matches('/'));
        let src_raw = attrs
            .get("source")
            .ok_or_else(|| GraphTheoryError::ParseError {
                format: "GraphML",
                message: "<edge> missing 'source'".to_string(),
            })?;
        let dst_raw = attrs
            .get("target")
            .ok_or_else(|| GraphTheoryError::ParseError {
                format: "GraphML",
                message: "<edge> missing 'target'".to_string(),
            })?;
        let u = resolve_token_id(src_raw, sym_table, next_sym_id);
        let v = resolve_token_id(dst_raw, sym_table, next_sym_id);
        let (inner, next_slice) =
            extract_element_body(&after[(tag_end + 1)..], "</edge>", self_closing)?;
        let weight = extract_graphml_weight(&attrs, inner, keys)?;
        graph.push_edge(u, v, weight)?;
        rest = next_slice;
    }
    Ok(())
}

#[verified_engine::verified]
fn extract_graphml_weight(
    attrs: &BTreeMap<String, String>,
    inner: &str,
    keys: &GraphMlKeyMap,
) -> Result<f64, GraphTheoryError> {
    let data = parse_data_entries(inner);
    let raw = lookup_data(&data, keys.weight_key.as_deref(), &["d_weight", "weight"])
        .or_else(|| attrs.get("weight").map(String::as_str));
    let Some(w_str) = raw else {
        return Ok(1.0);
    };
    let w = w_str
        .trim()
        .parse::<f64>()
        .map_err(|_| GraphTheoryError::ParseError {
            format: "GraphML",
            message: format!("invalid edge weight '{w_str}'"),
        })?;
    if !w.is_finite() {
        return Err(GraphTheoryError::InvalidWeight { value: w });
    }
    Ok(w)
}

#[verified_engine::verified]
fn extract_element_body<'a>(
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
fn parse_data_entries(inner: &str) -> BTreeMap<String, String> {
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
fn lookup_data<'a>(
    data: &'a BTreeMap<String, String>,
    primary: Option<&str>,
    fallbacks: &[&str],
) -> Option<&'a str> {
    if let Some(pk) = primary
        && let Some(val) = data.get(pk)
    {
        return Some(val.as_str());
    }
    for &fb in fallbacks {
        if let Some(val) = data.get(fb) {
            return Some(val.as_str());
        }
    }
    None
}

#[verified_engine::verified]
fn parse_xml_attrs(raw: &str) -> BTreeMap<String, String> {
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
