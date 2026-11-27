//! Deterministic weighted edge-list `.csv` parser and serializer.

use super::{GraphTheoryError, InterchangeGraph, format_f64, resolve_token_id};
use std::collections::BTreeMap;

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

/// Parses a `source,target[,weight]` `.csv` edge list into a validated [`InterchangeGraph`].
#[verified_engine::verified]
pub fn parse_edge_csv(input: &str) -> Result<InterchangeGraph, GraphTheoryError> {
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
        if !saw_data_row && is_csv_header(line) {
            saw_data_row = true;
            continue;
        }
        saw_data_row = true;
        parse_csv_edge_row(
            line_idx + 1,
            line,
            &mut graph,
            &mut sym_table,
            &mut next_sym_id,
        )?;
    }
    graph.validate()?;
    Ok(graph)
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
fn is_csv_header(line: &str) -> bool {
    let first = line.split(',').next().unwrap_or("").trim();
    first.eq_ignore_ascii_case("source")
        || first.eq_ignore_ascii_case("src")
        || first.eq_ignore_ascii_case("u")
        || first.eq_ignore_ascii_case("from")
}

#[verified_engine::verified]
fn parse_csv_edge_row(
    line_num: usize,
    line: &str,
    graph: &mut InterchangeGraph,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
) -> Result<(), GraphTheoryError> {
    let cols: Vec<&str> = line.split(',').map(str::trim).collect();
    if cols.len() < 2 || cols.len() > 4 || cols[0].is_empty() || cols[1].is_empty() {
        return Err(GraphTheoryError::ParseError {
            format: "CSV",
            message: format!("line {line_num}: expected 'source,target[,weight]'"),
        });
    }
    let u = resolve_token_id(cols[0], sym_table, next_sym_id);
    let v = resolve_token_id(cols[1], sym_table, next_sym_id);
    let weight = if cols.len() >= 3 && !cols[2].is_empty() {
        let w = cols[2]
            .parse::<f64>()
            .map_err(|_| GraphTheoryError::ParseError {
                format: "CSV",
                message: format!("line {line_num}: invalid weight '{}'", cols[2]),
            })?;
        if !w.is_finite() {
            return Err(GraphTheoryError::InvalidWeight { value: w });
        }
        w
    } else {
        1.0
    };
    graph.push_edge(u, v, weight)
}

// [cite:graph_parameters_rust]
