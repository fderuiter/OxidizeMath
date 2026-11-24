//! Deterministic weighted edge-list `.csv`/`.tsv` parsers and serializers for
//! [`super::InterchangeGraph`] and [`super::GraphTopologyDocument`].

use super::document::{check_import_byte_limit, strip_utf8_bom_str};
use super::escape::{ensure_node_index, escape_csv_field, format_compact_f64};
use super::{
    GraphTheoryError, GraphTopologyDocument, InterchangeGraph, MAX_TOPOLOGY_EDGES, TopologyEdge,
    format_f64, resolve_token_id,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

/// Serializes `graph` into a deterministic `source,target,weight` `.csv` table.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::interchange::{
///     InterchangeGraph, parse_edge_csv, to_edge_csv,
/// };
///
/// let mut g = InterchangeGraph::new(false);
/// g.push_edge(0, 1, 1.5)?;
/// g.push_edge(1, 2, 3.25)?;
/// let csv = to_edge_csv(&g)?;
/// let round_trip = parse_edge_csv(&csv)?;
/// assert_eq!(round_trip.edges.len(), 2);
/// assert!((round_trip.edges[1].weight - 3.25).abs() < 1e-9);
/// # Ok::<(), math_explorer::error::Error>(())
/// ```
#[verified_engine::verified]
pub fn to_edge_csv(graph: &InterchangeGraph) -> Result<String, GraphTheoryError> {
    graph.validate()?;
    let mut out = String::new();
    if graph.directed {
        out.push_str("# directed=true\n");
    }
    for (&id, node) in &graph.nodes {
        if let Some((x, y)) = node.pos {
            out.push_str(&format!(
                "# node,{id},{},{}\n",
                format_f64(x),
                format_f64(y)
            ));
        }
    }
    out.push_str("source,target,weight\n");
    for edge in &graph.edges {
        out.push_str(&format!(
            "{},{},{}\n",
            edge.source,
            edge.target,
            format_f64(edge.weight)
        ));
    }
    Ok(out)
}

/// Parses a `source,target[,weight]` `.csv` or `.tsv` edge list into a validated [`InterchangeGraph`].
#[verified_engine::verified]
pub fn parse_edge_csv(raw: &str) -> Result<InterchangeGraph, GraphTheoryError> {
    let input = strip_utf8_bom_str(raw);
    check_import_byte_limit(input)?;
    let mut graph = InterchangeGraph::new(false);
    let mut sym_table = BTreeMap::new();
    let mut next_sym_id = 0usize;
    let mut saw_data_row = false;

    for (line_idx, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            apply_csv_metadata_comment(
                comment.trim(),
                &mut graph,
                &mut sym_table,
                &mut next_sym_id,
            )?;
            continue;
        }
        let delim = detect_line_delimiter(line);
        if !saw_data_row && is_csv_header(line, delim) {
            saw_data_row = true;
            continue;
        }
        saw_data_row = true;
        parse_csv_edge_row(
            line_idx + 1,
            line,
            delim,
            &mut graph,
            &mut sym_table,
            &mut next_sym_id,
        )?;
    }
    graph.validate()?;
    Ok(graph)
}

#[verified_engine::verified]
fn detect_line_delimiter(line: &str) -> char {
    if line.contains('\t') && !line.contains(',') {
        '\t'
    } else {
        ','
    }
}

#[verified_engine::verified]
fn apply_csv_metadata_comment(
    comment: &str,
    graph: &mut InterchangeGraph,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
) -> Result<(), GraphTheoryError> {
    if comment.eq_ignore_ascii_case("directed=true") {
        graph.directed = true;
        return Ok(());
    }
    let cols: Vec<&str> = comment.split(',').map(str::trim).collect();
    if cols.len() == 4 && cols[0].eq_ignore_ascii_case("node") {
        let id = resolve_token_id(cols[1], sym_table, next_sym_id);
        let x = cols[2]
            .parse::<f64>()
            .map_err(|_| GraphTheoryError::ParseError {
                format: "CSV",
                message: format!("invalid node x coordinate '{}'", cols[2]),
            })?;
        let y = cols[3]
            .parse::<f64>()
            .map_err(|_| GraphTheoryError::ParseError {
                format: "CSV",
                message: format!("invalid node y coordinate '{}'", cols[3]),
            })?;
        graph.insert_node(super::InterchangeNode {
            id,
            pos: Some((x, y)),
            label: None,
        })?;
    }
    Ok(())
}

#[verified_engine::verified]
fn is_csv_header(line: &str, delim: char) -> bool {
    let first = line
        .split(delim)
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('"');
    first.eq_ignore_ascii_case("source")
        || first.eq_ignore_ascii_case("src")
        || first.eq_ignore_ascii_case("u")
        || first.eq_ignore_ascii_case("from")
}

#[verified_engine::verified]
fn parse_csv_edge_row(
    line_num: usize,
    line: &str,
    delim: char,
    graph: &mut InterchangeGraph,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
) -> Result<(), GraphTheoryError> {
    let cols = split_delimited_row(line, delim);
    if cols.len() < 2 || cols[0].is_empty() {
        return Err(GraphTheoryError::ParseError {
            format: "CSV",
            message: format!("line {line_num}: expected 'source,target[,weight]'"),
        });
    }
    let u = resolve_token_id(&cols[0], sym_table, next_sym_id);
    if cols[1].is_empty() {
        return graph.ensure_node(u);
    }
    let v = resolve_token_id(&cols[1], sym_table, next_sym_id);
    let weight = if cols.len() >= 3 && !cols[2].is_empty() {
        parse_csv_weight(line_num, &cols[2])?
    } else {
        1.0
    };
    graph.push_edge(u, v, weight)
}

#[verified_engine::verified]
fn parse_csv_weight(line_num: usize, raw: &str) -> Result<f64, GraphTheoryError> {
    let w = raw
        .trim()
        .parse::<f64>()
        .map_err(|_| GraphTheoryError::ParseError {
            format: "CSV",
            message: format!("line {line_num}: invalid weight '{raw}'"),
        })?;
    if !w.is_finite() {
        return Err(GraphTheoryError::InvalidWeight { value: w });
    }
    Ok(w)
}

/// Serializes a [`GraphTopologyDocument`] deterministically into a canonical CSV topology table.
#[must_use]
#[verified_engine::verified]
pub fn to_csv_topology(doc: &GraphTopologyDocument) -> String {
    let mut out =
        String::from("source,target,weight,directed,label,source_x,source_y,target_x,target_y\n");
    let pos_lookup: HashMap<&str, [f64; 2]> = doc
        .nodes
        .iter()
        .filter_map(|n| n.position.map(|p| (n.id.as_str(), p)))
        .collect();
    let mut covered_nodes: HashSet<&str> = HashSet::with_capacity(doc.nodes.len());
    let dir_str = if doc.directed { "true" } else { "false" };

    for edge in &doc.edges {
        covered_nodes.insert(edge.source.as_str());
        covered_nodes.insert(edge.target.as_str());
        let (sx, sy) = fmt_opt_pos(pos_lookup.get(edge.source.as_str()).copied());
        let (tx, ty) = fmt_opt_pos(pos_lookup.get(edge.target.as_str()).copied());
        let lbl = edge.label.as_deref().unwrap_or("");
        let _ = writeln!(
            out,
            "{},{},{},{dir_str},{},{sx},{sy},{tx},{ty}",
            escape_csv_field(&edge.source),
            escape_csv_field(&edge.target),
            format_compact_f64(edge.weight),
            escape_csv_field(lbl),
        );
    }

    for node in &doc.nodes {
        if !covered_nodes.contains(node.id.as_str()) {
            let (sx, sy) = fmt_opt_pos(node.position);
            let _ = writeln!(
                out,
                "{},,,{dir_str},{},{sx},{sy},,",
                escape_csv_field(&node.id),
                escape_csv_field(&node.label),
            );
        }
    }
    out
}

#[verified_engine::verified]
fn fmt_opt_pos(pos: Option<[f64; 2]>) -> (String, String) {
    match pos {
        Some([x, y]) if x.is_finite() && y.is_finite() => {
            (format_compact_f64(x), format_compact_f64(y))
        }
        _ => (String::new(), String::new()),
    }
}

/// Parses a comma- or tab-delimited `.csv`/`.tsv` topology table into a [`GraphTopologyDocument`].
#[verified_engine::verified]
pub fn parse_csv_topology(raw: &str) -> Result<GraphTopologyDocument, GraphTheoryError> {
    let cleaned = strip_utf8_bom_str(raw);
    check_import_byte_limit(cleaned)?;
    let mut doc = GraphTopologyDocument::new("CSVGraph", false);
    let mut index_map = HashMap::new();
    let mut saw_header = false;

    for (line_idx, raw_line) in cleaned.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let delim = detect_line_delimiter(line);
        if !saw_header && is_csv_header(line, delim) {
            saw_header = true;
            continue;
        }
        saw_header = true;
        apply_csv_doc_row(line_idx + 1, line, delim, &mut doc, &mut index_map)?;
    }
    if doc.nodes.is_empty() {
        return Err(GraphTheoryError::ParseError {
            format: "CSV",
            message: "CSV/TSV topology file contains no valid nodes or edges".to_string(),
        });
    }
    doc.validate()?;
    Ok(doc)
}

#[verified_engine::verified]
fn apply_csv_doc_row(
    line_num: usize,
    line: &str,
    delim: char,
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
) -> Result<(), GraphTheoryError> {
    let cols = split_delimited_row(line, delim);
    if cols.len() < 2 || cols[0].is_empty() {
        return Err(GraphTheoryError::ParseError {
            format: "CSV",
            message: format!("line {line_num}: expected 'source,target[,weight]'"),
        });
    }
    if let Some(dir) = cols.get(3)
        && dir.eq_ignore_ascii_case("true")
    {
        doc.directed = true;
    }
    let u_idx = ensure_node_index(&mut doc.nodes, index_map, &cols[0])?;
    apply_opt_csv_pos(&mut doc.nodes[u_idx], cols.get(5), cols.get(6));
    if cols[1].is_empty() {
        if let Some(lbl) = cols.get(4).filter(|s| !s.is_empty()) {
            doc.nodes[u_idx].label.clone_from(lbl);
        }
        return Ok(());
    }
    let v_idx = ensure_node_index(&mut doc.nodes, index_map, &cols[1])?;
    apply_opt_csv_pos(&mut doc.nodes[v_idx], cols.get(7), cols.get(8));
    let weight = match cols.get(2).filter(|s| !s.is_empty()) {
        Some(w_str) => parse_csv_weight(line_num, w_str)?,
        None => 1.0,
    };
    if doc.edges.len() >= MAX_TOPOLOGY_EDGES {
        return Err(GraphTheoryError::EdgeLimitExceeded {
            count: doc.edges.len() + 1,
            max: MAX_TOPOLOGY_EDGES,
        });
    }
    let mut edge = TopologyEdge::new(cols[0].clone(), cols[1].clone(), weight);
    if let Some(lbl) = cols.get(4).filter(|s| !s.is_empty()) {
        edge.label = Some(lbl.clone());
    }
    doc.edges.push(edge);
    Ok(())
}

#[verified_engine::verified]
fn apply_opt_csv_pos(node: &mut super::TopologyNode, xs: Option<&String>, ys: Option<&String>) {
    if let (Some(x_str), Some(y_str)) = (xs, ys)
        && let (Ok(x), Ok(y)) = (x_str.parse::<f64>(), y_str.parse::<f64>())
        && x.is_finite()
        && y.is_finite()
    {
        node.position = Some([x, y]);
    }
}

#[verified_engine::verified]
fn split_delimited_row(line: &str, delim: char) -> Vec<String> {
    let mut cols = Vec::new();
    let mut cur = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                cur.push('"');
                let _ = chars.next();
            } else {
                in_quotes = !in_quotes;
            }
        } else if ch == delim && !in_quotes {
            cols.push(std::mem::take(&mut cur).trim().to_string());
        } else {
            cur.push(ch);
        }
    }
    cols.push(cur.trim().to_string());
    cols
}

// [cite:graph_parameters_rust]
