//! Deterministic Graphviz `.dot` serializer and bounded iterative parser.

use super::escape::{
    edge_style_to_str, ensure_node_index, escape_dot_quoted, format_compact_f64, shape_to_str,
    str_to_edge_style, str_to_shape,
};
use super::{
    GraphTopologyDocument, MAX_TOPOLOGY_EDGES, TopologyCodecError, TopologyEdge, TopologyEdgeStyle,
    TopologyNode, TopologyNodeShape, check_import_byte_limit, strip_utf8_bom_str,
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
        write_dot_node(&mut out, node);
    }
    for edge in &doc.edges {
        write_dot_edge(&mut out, edge, connector);
    }
    out.push_str("}\n");
    out
}

#[verified_engine::verified]
fn write_dot_node(out: &mut String, node: &TopologyNode) {
    let mut attrs = Vec::new();
    attrs.push(format!("label={}", escape_dot_quoted(&node.label)));
    if node.shape != TopologyNodeShape::Circle {
        attrs.push(format!(
            "shape={}",
            escape_dot_quoted(shape_to_str(node.shape))
        ));
    }
    if let Some([x, y]) = node.position {
        attrs.push(format!(
            "pos={}",
            escape_dot_quoted(&format!(
                "{},{}!",
                format_compact_f64(x),
                format_compact_f64(y)
            ))
        ));
    }
    if let Some(fill) = &node.fill_hex {
        attrs.push("style=\"filled\"".to_string());
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
fn write_dot_edge(out: &mut String, edge: &TopologyEdge, connector: &str) {
    let mut attrs = Vec::new();
    attrs.push(format!("weight={}", format_compact_f64(edge.weight)));
    if let Some(lbl) = &edge.label {
        attrs.push(format!("label={}", escape_dot_quoted(lbl)));
    }
    if edge.style != TopologyEdgeStyle::Solid {
        attrs.push(format!(
            "style={}",
            escape_dot_quoted(edge_style_to_str(edge.style))
        ));
    }
    if let Some(color) = &edge.color_hex {
        attrs.push(format!("color={}", escape_dot_quoted(color)));
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum DotToken {
    Word(String),
    ArrowDir,
    ArrowUndir,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Equals,
    Comma,
    Semi,
}

/// Parses a Graphviz `.dot` / `.gv` string into a validated [`GraphTopologyDocument`].
#[verified_engine::verified]
pub fn parse_dot_topology(raw: &str) -> Result<GraphTopologyDocument, TopologyCodecError> {
    let cleaned = strip_utf8_bom_str(raw);
    check_import_byte_limit(cleaned)?;
    let tokens = tokenize_dot(cleaned)?;
    let mut pos = 0usize;
    if matches!(tokens.get(pos), Some(DotToken::Word(w)) if w.eq_ignore_ascii_case("strict")) {
        pos += 1;
    }
    let directed = match tokens.get(pos) {
        Some(DotToken::Word(w)) if w.eq_ignore_ascii_case("digraph") => true,
        Some(DotToken::Word(w)) if w.eq_ignore_ascii_case("graph") => false,
        _ => {
            return Err(TopologyCodecError::Parse(
                "Expected 'graph' or 'digraph' header in DOT input".into(),
            ));
        }
    };
    pos += 1;
    let mut name = "G".to_string();
    if let Some(DotToken::Word(w)) = tokens.get(pos) {
        name.clone_from(w);
        pos += 1;
    }
    if tokens.get(pos) != Some(&DotToken::LBrace) {
        return Err(TopologyCodecError::Parse(
            "Expected '{' opening DOT graph body".into(),
        ));
    }
    pos += 1;

    let mut doc = GraphTopologyDocument::new(name, directed);
    let mut index_map = HashMap::new();
    parse_dot_statements(&tokens, &mut pos, &mut doc, &mut index_map)?;
    if doc.nodes.is_empty() {
        return Err(TopologyCodecError::EmptyTopology(
            "DOT graph defines zero nodes".into(),
        ));
    }
    doc.validate()?;
    Ok(doc)
}

const MAX_DOT_BRACE_DEPTH: usize = 32;

#[verified_engine::verified]
fn parse_dot_statements(
    tokens: &[DotToken],
    pos: &mut usize,
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
) -> Result<(), TopologyCodecError> {
    let mut brace_depth = 1usize;
    while *pos < tokens.len() && brace_depth > 0 {
        match &tokens[*pos] {
            DotToken::RBrace => {
                *pos += 1;
                brace_depth -= 1;
            }
            DotToken::Semi | DotToken::Comma => *pos += 1,
            DotToken::LBrace => {
                brace_depth += 1;
                if brace_depth > MAX_DOT_BRACE_DEPTH {
                    return Err(TopologyCodecError::Parse(
                        "DOT subgraph nesting exceeds maximum depth of 32".into(),
                    ));
                }
                *pos += 1;
            }
            DotToken::Word(w) if w.eq_ignore_ascii_case("subgraph") => {
                *pos += 1;
                if matches!(tokens.get(*pos), Some(DotToken::Word(_))) {
                    *pos += 1;
                }
            }
            DotToken::Word(w)
                if matches!(w.to_ascii_lowercase().as_str(), "graph" | "node" | "edge")
                    && tokens.get(*pos + 1) == Some(&DotToken::LBracket) =>
            {
                let kw = w.to_ascii_lowercase();
                *pos += 1;
                let attrs = parse_dot_attr_block(tokens, pos)?;
                if kw == "graph" {
                    doc.attributes.extend(attrs);
                }
            }
            DotToken::Word(first) => {
                let first_id = skip_port_if_present(tokens, pos, first.clone());
                if tokens.get(*pos) == Some(&DotToken::Equals) {
                    *pos += 1;
                    if let Some(DotToken::Word(val)) = tokens.get(*pos) {
                        doc.attributes.insert(first_id, val.clone());
                        *pos += 1;
                    }
                } else if matches!(
                    tokens.get(*pos),
                    Some(DotToken::ArrowDir | DotToken::ArrowUndir)
                ) {
                    parse_dot_edge_chain(tokens, pos, first_id, doc, index_map)?;
                } else {
                    let attrs = if tokens.get(*pos) == Some(&DotToken::LBracket) {
                        parse_dot_attr_block(tokens, pos)?
                    } else {
                        BTreeMap::new()
                    };
                    let idx = ensure_node_index(&mut doc.nodes, index_map, &first_id)?;
                    apply_dot_node_attrs(&mut doc.nodes[idx], attrs);
                }
            }
            _ => *pos += 1,
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn skip_port_if_present(tokens: &[DotToken], pos: &mut usize, base: String) -> String {
    *pos += 1;
    let _ = tokens;
    base
}

#[verified_engine::verified]
fn parse_dot_edge_chain(
    tokens: &[DotToken],
    pos: &mut usize,
    first_id: String,
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
) -> Result<(), TopologyCodecError> {
    let mut chain = vec![first_id];
    while matches!(
        tokens.get(*pos),
        Some(DotToken::ArrowDir | DotToken::ArrowUndir)
    ) {
        *pos += 1;
        let Some(DotToken::Word(next_id)) = tokens.get(*pos) else {
            return Err(TopologyCodecError::Parse(
                "Expected target node ID after DOT edge operator".into(),
            ));
        };
        chain.push(next_id.clone());
        *pos += 1;
    }
    let attrs = if tokens.get(*pos) == Some(&DotToken::LBracket) {
        parse_dot_attr_block(tokens, pos)?
    } else {
        BTreeMap::new()
    };
    for id in &chain {
        ensure_node_index(&mut doc.nodes, index_map, id)?;
    }
    for pair in chain.windows(2) {
        if doc.edges.len() >= MAX_TOPOLOGY_EDGES {
            return Err(TopologyCodecError::TooManyEdges {
                actual: doc.edges.len() + 1,
                max: MAX_TOPOLOGY_EDGES,
            });
        }
        let mut edge = TopologyEdge::new(&pair[0], &pair[1], 1.0);
        apply_dot_edge_attrs(&mut edge, &attrs)?;
        doc.edges.push(edge);
    }
    Ok(())
}

#[verified_engine::verified]
fn parse_dot_attr_block(
    tokens: &[DotToken],
    pos: &mut usize,
) -> Result<BTreeMap<String, String>, TopologyCodecError> {
    let mut map = BTreeMap::new();
    if tokens.get(*pos) != Some(&DotToken::LBracket) {
        return Ok(map);
    }
    *pos += 1;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            DotToken::RBracket => {
                *pos += 1;
                break;
            }
            DotToken::Comma | DotToken::Semi => *pos += 1,
            DotToken::Word(key) => {
                let k = key.clone();
                *pos += 1;
                if tokens.get(*pos) == Some(&DotToken::Equals) {
                    *pos += 1;
                    if let Some(DotToken::Word(val)) = tokens.get(*pos) {
                        map.insert(k, val.clone());
                        *pos += 1;
                    }
                }
            }
            _ => *pos += 1,
        }
    }
    Ok(map)
}

#[verified_engine::verified]
fn apply_dot_node_attrs(node: &mut TopologyNode, mut attrs: BTreeMap<String, String>) {
    if let Some(lbl) = attrs.remove("label") {
        node.label = lbl;
    }
    if let Some(shape) = attrs.remove("shape") {
        node.shape = str_to_shape(&shape);
    }
    if let Some(pos_str) = attrs.remove("pos") {
        let trimmed = pos_str.trim().trim_end_matches('!');
        let mut parts = trimmed.split(',');
        if let (Some(xs), Some(ys)) = (parts.next(), parts.next())
            && let (Ok(x), Ok(y)) = (xs.trim().parse::<f64>(), ys.trim().parse::<f64>())
            && x.is_finite()
            && y.is_finite()
        {
            node.position = Some([x, y]);
        }
    } else if let (Some(xs), Some(ys)) = (attrs.remove("x"), attrs.remove("y"))
        && let (Ok(x), Ok(y)) = (xs.trim().parse::<f64>(), ys.trim().parse::<f64>())
        && x.is_finite()
        && y.is_finite()
    {
        node.position = Some([x, y]);
    }
    if let Some(fill) = attrs.remove("fillcolor") {
        node.fill_hex = Some(fill);
    }
    if let Some(stroke) = attrs.remove("color") {
        node.stroke_hex = Some(stroke);
    }
    let _ = attrs.remove("style");
    node.attributes.extend(attrs);
}

#[verified_engine::verified]
fn apply_dot_edge_attrs(
    edge: &mut TopologyEdge,
    attrs: &BTreeMap<String, String>,
) -> Result<(), TopologyCodecError> {
    for (k, v) in attrs {
        match k.as_str() {
            "weight" => {
                let w = v.trim().parse::<f64>().map_err(|_| {
                    TopologyCodecError::Parse(format!("Invalid DOT edge weight '{v}'"))
                })?;
                if !w.is_finite() {
                    return Err(TopologyCodecError::Parse(
                        "Non-finite DOT edge weight".into(),
                    ));
                }
                edge.weight = w;
            }
            "label" => edge.label = Some(v.clone()),
            "style" => edge.style = str_to_edge_style(v),
            "color" => edge.color_hex = Some(v.clone()),
            _ => {
                edge.attributes.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn tokenize_dot(src: &str) -> Result<Vec<DotToken>, TopologyCodecError> {
    let mut tokens = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
        } else if b == b'#' || (b == b'/' && bytes.get(i + 1) == Some(&b'/')) {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else if b == b'-' && bytes.get(i + 1) == Some(&b'>') {
            tokens.push(DotToken::ArrowDir);
            i += 2;
        } else if b == b'-' && bytes.get(i + 1) == Some(&b'-') {
            tokens.push(DotToken::ArrowUndir);
            i += 2;
        } else if matches!(b, b'{' | b'}' | b'[' | b']' | b'=' | b',' | b';') {
            tokens.push(match b {
                b'{' => DotToken::LBrace,
                b'}' => DotToken::RBrace,
                b'[' => DotToken::LBracket,
                b']' => DotToken::RBracket,
                b'=' => DotToken::Equals,
                b',' => DotToken::Comma,
                _ => DotToken::Semi,
            });
            i += 1;
        } else if b == b'"' {
            tokens.push(DotToken::Word(read_dot_quoted(src, &mut i)?));
        } else if b == b':' {
            // Skip compass/port suffix `:port` attached to node identifiers.
            i += 1;
            while i < bytes.len() && is_bare_word_byte(bytes[i]) {
                i += 1;
            }
        } else {
            let start = i;
            while i < bytes.len() && is_bare_word_byte(bytes[i]) && !starts_dot_arrow(bytes, i) {
                i += 1;
            }
            if i == start {
                i += 1;
            } else {
                tokens.push(DotToken::Word(src[start..i].to_string()));
            }
        }
    }
    Ok(tokens)
}

#[inline]
#[verified_engine::verified]
fn starts_dot_arrow(bytes: &[u8], i: usize) -> bool {
    bytes.get(i) == Some(&b'-') && matches!(bytes.get(i + 1), Some(&b'>' | &b'-'))
}

#[inline]
#[verified_engine::verified]
fn is_bare_word_byte(b: u8) -> bool {
    !b.is_ascii_whitespace()
        && !matches!(
            b,
            b'{' | b'}' | b'[' | b']' | b'=' | b',' | b';' | b'"' | b':' | b'/'
        )
}

#[verified_engine::verified]
fn read_dot_quoted(src: &str, idx: &mut usize) -> Result<String, TopologyCodecError> {
    let mut out = String::new();
    let mut chars = src[*idx + 1..].char_indices();
    while let Some((offset, ch)) = chars.next() {
        if ch == '\\' {
            if let Some((_, next_ch)) = chars.next() {
                match next_ch {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    other => out.push(other),
                }
            }
        } else if ch == '"' {
            *idx += 1 + offset + 1;
            return Ok(out);
        } else {
            out.push(ch);
        }
    }
    Err(TopologyCodecError::Parse(
        "Unterminated quoted string in DOT input".into(),
    ))
}

// [cite:graph_parameters_rust]
