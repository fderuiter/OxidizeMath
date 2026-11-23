//! Deterministic Graphviz `.dot` parser and styled serializer.

use super::{GraphTheoryError, InterchangeGraph, InterchangeNode, resolve_token_id};
use std::collections::BTreeMap;

#[path = "dot_emit.rs"]
mod dot_emit;
pub use dot_emit::{to_dot, to_styled_dot};

/// Parses a Graphviz `.dot` (`graph` or `digraph`) specification into a validated [`InterchangeGraph`].
#[verified_engine::verified]
pub fn parse_dot(input: &str) -> Result<InterchangeGraph, GraphTheoryError> {
    let cleaned = strip_dot_comments(input);
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
    let directed = parse_dot_header(&cleaned[..open_brace])?;
    let body = &cleaned[(open_brace + 1)..close_brace];
    let mut graph = InterchangeGraph::new(directed);
    let mut sym_table = BTreeMap::new();
    let mut next_sym_id = 0usize;
    for stmt in split_dot_statements(body) {
        parse_dot_statement(&stmt, &mut graph, &mut sym_table, &mut next_sym_id)?;
    }
    graph.validate()?;
    Ok(graph)
}

#[verified_engine::verified]
fn strip_dot_comments(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;
    let mut in_quotes = false;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        if ch == '"' && (i == 0 || bytes[i - 1] != b'\\') {
            in_quotes = !in_quotes;
            out.push(ch);
            i += 1;
        } else if !in_quotes && i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if !in_quotes && i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else if !in_quotes && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else {
            out.push(ch);
            i += 1;
        }
    }
    out
}

#[verified_engine::verified]
fn parse_dot_header(prefix: &str) -> Result<bool, GraphTheoryError> {
    let mut words = prefix.split_whitespace();
    let first = words.next().ok_or_else(|| GraphTheoryError::ParseError {
        format: "DOT",
        message: "expected 'graph' or 'digraph'".to_string(),
    })?;
    let kw = if first.eq_ignore_ascii_case("strict") {
        words.next().ok_or_else(|| GraphTheoryError::ParseError {
            format: "DOT",
            message: "expected 'graph' or 'digraph' after 'strict'".to_string(),
        })?
    } else {
        first
    };
    if kw.eq_ignore_ascii_case("digraph") {
        Ok(true)
    } else if kw.eq_ignore_ascii_case("graph") {
        Ok(false)
    } else {
        Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: format!("unrecognized graph keyword '{kw}'"),
        })
    }
}

#[verified_engine::verified]
fn split_dot_statements(body: &str) -> Vec<String> {
    let mut stmts = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut bracket_depth = 0usize;
    let mut prev_backslash = false;
    for ch in body.chars() {
        if ch == '"' && !prev_backslash {
            in_quotes = !in_quotes;
        }
        prev_backslash = ch == '\\' && !prev_backslash;
        if !in_quotes {
            if ch == '[' {
                bracket_depth += 1;
            } else if ch == ']' {
                bracket_depth = bracket_depth.saturating_sub(1);
            } else if (ch == ';' || ch == '\n') && bracket_depth == 0 {
                if !cur.trim().is_empty() {
                    stmts.push(cur.trim().to_string());
                }
                cur.clear();
                continue;
            }
        }
        cur.push(ch);
    }
    if !cur.trim().is_empty() {
        stmts.push(cur.trim().to_string());
    }
    stmts
}

#[verified_engine::verified]
fn parse_dot_statement(
    stmt: &str,
    graph: &mut InterchangeGraph,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
) -> Result<(), GraphTheoryError> {
    let (head, attrs_raw) = split_head_and_attrs(stmt);
    let head = head.trim();
    if head.is_empty() || is_global_dot_attr(head, attrs_raw.is_some()) {
        return Ok(());
    }
    let attrs = attrs_raw.map(parse_dot_attrs).unwrap_or_default();
    let edge_op = if graph.directed { "->" } else { "--" };
    let alt_op = if graph.directed { "--" } else { "->" };
    if head.contains(alt_op) {
        return Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: format!("edge operator '{alt_op}' does not match graph header"),
        });
    }
    if head.contains(edge_op) {
        return apply_dot_edge_chain(head, edge_op, &attrs, graph, sym_table, next_sym_id);
    }
    apply_dot_node(head, &attrs, graph, sym_table, next_sym_id)
}

#[verified_engine::verified]
fn is_global_dot_attr(head: &str, has_brackets: bool) -> bool {
    if head.contains('=') && !head.contains("->") && !head.contains("--") {
        return true;
    }
    has_brackets
        && (head.eq_ignore_ascii_case("graph")
            || head.eq_ignore_ascii_case("node")
            || head.eq_ignore_ascii_case("edge"))
}

#[verified_engine::verified]
fn split_head_and_attrs(stmt: &str) -> (&str, Option<&str>) {
    let mut in_quotes = false;
    let mut prev_bs = false;
    let mut open_idx = None;
    for (idx, ch) in stmt.char_indices() {
        if ch == '"' && !prev_bs {
            in_quotes = !in_quotes;
        }
        prev_bs = ch == '\\' && !prev_bs;
        if !in_quotes && ch == '[' {
            open_idx = Some(idx);
            break;
        }
    }
    if let Some(start) = open_idx
        && let Some(end) = stmt.rfind(']')
        && end > start
    {
        return (&stmt[..start], Some(&stmt[(start + 1)..end]));
    }
    (stmt, None)
}

#[verified_engine::verified]
fn parse_dot_attrs(raw: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut prev_bs = false;
    for ch in raw.chars() {
        if ch == '"' && !prev_bs {
            in_quotes = !in_quotes;
        }
        prev_bs = ch == '\\' && !prev_bs;
        if !in_quotes && (ch == ',' || ch == ';') {
            parts.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
    }
    parts.push(cur);
    for item in parts {
        if let Some((k, v)) = item.split_once('=') {
            let key = k.trim().to_ascii_lowercase();
            let val = unquote_dot(v.trim());
            map.insert(key, val);
        }
    }
    map
}

#[verified_engine::verified]
fn unquote_dot(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        trimmed[1..(trimmed.len() - 1)]
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    } else {
        trimmed.to_string()
    }
}

#[verified_engine::verified]
fn apply_dot_node(
    head: &str,
    attrs: &BTreeMap<String, String>,
    graph: &mut InterchangeGraph,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
) -> Result<(), GraphTheoryError> {
    let clean_name = unquote_dot(head);
    if clean_name.is_empty() {
        return Ok(());
    }
    let id = resolve_token_id(&clean_name, sym_table, next_sym_id);
    let pos = match attrs.get("pos") {
        Some(pos_str) => Some(parse_dot_pos(id, pos_str)?),
        None => graph.nodes.get(&id).and_then(|n| n.pos),
    };
    let label = attrs
        .get("label")
        .cloned()
        .or_else(|| graph.nodes.get(&id).and_then(|n| n.label.clone()));
    graph.insert_node(InterchangeNode { id, pos, label })
}

#[verified_engine::verified]
fn parse_dot_pos(node: usize, raw: &str) -> Result<(f64, f64), GraphTheoryError> {
    let stripped = raw.trim().trim_end_matches('!');
    let Some((xs, ys)) = stripped.split_once(',') else {
        return Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: format!("invalid pos attribute '{raw}' on vertex {node}"),
        });
    };
    let x = xs
        .trim()
        .parse::<f64>()
        .map_err(|_| GraphTheoryError::ParseError {
            format: "DOT",
            message: format!("invalid x coordinate '{xs}'"),
        })?;
    let y = ys
        .trim()
        .parse::<f64>()
        .map_err(|_| GraphTheoryError::ParseError {
            format: "DOT",
            message: format!("invalid y coordinate '{ys}'"),
        })?;
    if !x.is_finite() || !y.is_finite() {
        return Err(GraphTheoryError::InvalidCoordinate { node, x, y });
    }
    Ok((x, y))
}

#[verified_engine::verified]
fn apply_dot_edge_chain(
    head: &str,
    edge_op: &str,
    attrs: &BTreeMap<String, String>,
    graph: &mut InterchangeGraph,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
) -> Result<(), GraphTheoryError> {
    let weight = extract_dot_edge_weight(attrs)?;
    let endpoints: Vec<&str> = head.split(edge_op).collect();
    if endpoints.len() < 2 {
        return Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: "incomplete edge statement".to_string(),
        });
    }
    let mut ids = Vec::with_capacity(endpoints.len());
    for raw_ep in endpoints {
        let ep = unquote_dot(raw_ep);
        if ep.is_empty() {
            return Err(GraphTheoryError::ParseError {
                format: "DOT",
                message: "empty endpoint in edge chain".to_string(),
            });
        }
        ids.push(resolve_token_id(&ep, sym_table, next_sym_id));
    }
    let mut i = 0usize;
    while i + 1 < ids.len() {
        graph.push_edge(ids[i], ids[i + 1], weight)?;
        i += 1;
    }
    Ok(())
}

#[verified_engine::verified]
fn extract_dot_edge_weight(attrs: &BTreeMap<String, String>) -> Result<f64, GraphTheoryError> {
    if let Some(w_str) = attrs.get("weight") {
        let w = w_str
            .trim()
            .parse::<f64>()
            .map_err(|_| GraphTheoryError::ParseError {
                format: "DOT",
                message: format!("non-numeric edge weight '{w_str}'"),
            })?;
        if !w.is_finite() {
            return Err(GraphTheoryError::InvalidWeight { value: w });
        }
        return Ok(w);
    }
    if let Some(lbl) = attrs.get("label")
        && let Ok(w) = lbl.trim().parse::<f64>()
    {
        if !w.is_finite() {
            return Err(GraphTheoryError::InvalidWeight { value: w });
        }
        return Ok(w);
    }
    Ok(1.0)
}

// [cite:graph_parameters_rust]
