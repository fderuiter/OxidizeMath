//! Deterministic `.csv` / `.tsv` topology table serializer and bounded edge-list parser.

use super::escape::{ensure_node_index, escape_csv_field, format_compact_f64};
use super::{
    GraphTopologyDocument, MAX_TOPOLOGY_EDGES, TopologyCodecError, TopologyEdge,
    check_import_byte_limit, strip_utf8_bom_str,
};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// Serializes a [`GraphTopologyDocument`] deterministically into a canonical CSV topology table.
///
/// Emits header `source,target,weight,directed,label,source_x,source_y,target_x,target_y` with one
/// row per edge, followed by isolated-vertex rows (`target` empty) when degree-0 nodes exist so
/// node counts, positions, edge weights, and graph parameters roundtrip losslessly.
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

#[derive(Clone, Copy)]
struct CsvColumnMap {
    source: usize,
    target: usize,
    weight: Option<usize>,
    directed: Option<usize>,
    label: Option<usize>,
    source_x: Option<usize>,
    source_y: Option<usize>,
    target_x: Option<usize>,
    target_y: Option<usize>,
}

/// Parses a comma- or tab-delimited edge-list `.csv`/`.tsv` into a [`GraphTopologyDocument`].
#[verified_engine::verified]
pub fn parse_csv_topology(raw: &str) -> Result<GraphTopologyDocument, TopologyCodecError> {
    let cleaned = strip_utf8_bom_str(raw);
    check_import_byte_limit(cleaned)?;

    let lines: Vec<&str> = cleaned
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if lines.is_empty() {
        return Err(TopologyCodecError::EmptyTopology(
            "CSV/TSV topology file is empty".into(),
        ));
    }

    let delimiter = detect_delimiter(lines[0]);
    let first_fields = split_delimited_row(lines[0], delimiter)?;
    let (col_map, start_row) = resolve_column_map(&first_fields)?;

    let mut doc = GraphTopologyDocument::new("ImportedCSV", false);
    let mut index_map = HashMap::new();

    for &line in &lines[start_row..] {
        let fields = split_delimited_row(line, delimiter)?;
        ingest_csv_row(&fields, col_map, &mut doc, &mut index_map)?;
    }

    if doc.nodes.is_empty() {
        return Err(TopologyCodecError::EmptyTopology(
            "CSV/TSV topology file contains no valid nodes or edges".into(),
        ));
    }
    doc.validate()?;
    Ok(doc)
}

#[verified_engine::verified]
fn detect_delimiter(header_line: &str) -> char {
    let tabs = header_line.chars().filter(|&c| c == '\t').count();
    let commas = header_line.chars().filter(|&c| c == ',').count();
    if tabs > commas { '\t' } else { ',' }
}

#[verified_engine::verified]
fn resolve_column_map(
    first_fields: &[String],
) -> Result<(CsvColumnMap, usize), TopologyCodecError> {
    if first_fields.len() < 2 {
        return Err(TopologyCodecError::Parse(
            "CSV/TSV edge list requires at least 2 columns (source, target)".into(),
        ));
    }
    let lowered: Vec<String> = first_fields
        .iter()
        .map(|f| f.trim().to_ascii_lowercase())
        .collect();
    let src_col = lowered
        .iter()
        .position(|h| matches!(h.as_str(), "source" | "src" | "u" | "from" | "node_a"));
    let dst_col = lowered
        .iter()
        .position(|h| matches!(h.as_str(), "target" | "dst" | "v" | "to" | "node_b"));

    if let (Some(source), Some(target)) = (src_col, dst_col) {
        let map = CsvColumnMap {
            source,
            target,
            weight: lowered
                .iter()
                .position(|h| matches!(h.as_str(), "weight" | "w" | "value")),
            directed: lowered.iter().position(|h| h == "directed"),
            label: lowered.iter().position(|h| h == "label"),
            source_x: lowered
                .iter()
                .position(|h| matches!(h.as_str(), "source_x" | "x")),
            source_y: lowered
                .iter()
                .position(|h| matches!(h.as_str(), "source_y" | "y")),
            target_x: lowered.iter().position(|h| h == "target_x"),
            target_y: lowered.iter().position(|h| h == "target_y"),
        };
        Ok((map, 1))
    } else {
        let map = CsvColumnMap {
            source: 0,
            target: 1,
            weight: (first_fields.len() >= 3).then_some(2),
            directed: None,
            label: None,
            source_x: None,
            source_y: None,
            target_x: None,
            target_y: None,
        };
        Ok((map, 0))
    }
}

#[verified_engine::verified]
fn ingest_csv_row(
    fields: &[String],
    cols: CsvColumnMap,
    doc: &mut GraphTopologyDocument,
    index_map: &mut HashMap<String, usize>,
) -> Result<(), TopologyCodecError> {
    let src = fields.get(cols.source).map_or("", |s| s.trim());
    if src.is_empty() {
        return Ok(());
    }
    if let Some(dir_col) = cols.directed
        && let Some(dval) = fields.get(dir_col)
        && matches!(
            dval.trim().to_ascii_lowercase().as_str(),
            "true" | "1" | "directed"
        )
    {
        doc.directed = true;
    }

    let s_idx = ensure_node_index(&mut doc.nodes, index_map, src)?;
    apply_optional_xy(
        fields,
        cols.source_x,
        cols.source_y,
        &mut doc.nodes[s_idx].position,
    );

    let dst = fields.get(cols.target).map_or("", |s| s.trim());
    if dst.is_empty() {
        if let Some(lbl_col) = cols.label
            && let Some(lbl) = fields.get(lbl_col)
            && !lbl.trim().is_empty()
        {
            doc.nodes[s_idx].label = lbl.trim().to_string();
        }
        return Ok(());
    }

    let t_idx = ensure_node_index(&mut doc.nodes, index_map, dst)?;
    apply_optional_xy(
        fields,
        cols.target_x,
        cols.target_y,
        &mut doc.nodes[t_idx].position,
    );

    if doc.edges.len() >= MAX_TOPOLOGY_EDGES {
        return Err(TopologyCodecError::TooManyEdges {
            actual: doc.edges.len() + 1,
            max: MAX_TOPOLOGY_EDGES,
        });
    }
    let weight = parse_optional_weight(fields, cols.weight)?;
    let mut edge = TopologyEdge::new(src, dst, weight);
    if let Some(lbl_col) = cols.label
        && let Some(lbl) = fields.get(lbl_col)
        && !lbl.trim().is_empty()
    {
        edge.label = Some(lbl.trim().to_string());
    }
    doc.edges.push(edge);
    Ok(())
}

#[verified_engine::verified]
fn apply_optional_xy(
    fields: &[String],
    x_col: Option<usize>,
    y_col: Option<usize>,
    slot: &mut Option<[f64; 2]>,
) {
    if slot.is_some() {
        return;
    }
    if let (Some(xc), Some(yc)) = (x_col, y_col)
        && let (Some(xs), Some(ys)) = (fields.get(xc), fields.get(yc))
        && let (Ok(x), Ok(y)) = (xs.trim().parse::<f64>(), ys.trim().parse::<f64>())
        && x.is_finite()
        && y.is_finite()
    {
        *slot = Some([x, y]);
    }
}

#[verified_engine::verified]
fn parse_optional_weight(
    fields: &[String],
    weight_col: Option<usize>,
) -> Result<f64, TopologyCodecError> {
    let Some(wc) = weight_col else {
        return Ok(1.0);
    };
    let Some(raw) = fields.get(wc).map(|s| s.trim()).filter(|s| !s.is_empty()) else {
        return Ok(1.0);
    };
    let val = raw
        .parse::<f64>()
        .map_err(|_| TopologyCodecError::Parse(format!("Invalid CSV edge weight '{raw}'")))?;
    if !val.is_finite() {
        return Err(TopologyCodecError::Parse(
            "Non-finite CSV edge weight".into(),
        ));
    }
    Ok(val)
}

#[verified_engine::verified]
fn split_delimited_row(row: &str, delimiter: char) -> Result<Vec<String>, TopologyCodecError> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = row.chars().peekable();
    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    let _ = chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == delimiter {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    if in_quotes {
        return Err(TopologyCodecError::Parse(
            "Unterminated quoted field in CSV/TSV row".into(),
        ));
    }
    fields.push(current);
    Ok(fields)
}

// [cite:graph_parameters_rust]
