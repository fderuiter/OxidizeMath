//! Deterministic Graphviz `.dot` parser and styled serializer.

use super::{GraphTheoryError, InterchangeGraph, clamp_interchange_limit, parse_dot_topology};
use std::collections::BTreeMap;

#[path = "dot_emit.rs"]
mod dot_emit;
pub use dot_emit::{to_dot, to_styled_dot};

/// Parses a Graphviz `.dot` (`graph` or `digraph`) specification into a validated [`InterchangeGraph`].
#[verified_engine::verified]
pub fn parse_dot(input: &str) -> Result<InterchangeGraph, GraphTheoryError> {
    parse_dot_topology(input)
        .and_then(|doc| doc.to_interchange())
        .map_err(clamp_interchange_limit)
}

#[verified_engine::verified]
pub(super) fn strip_dot_comments(input: &str) -> String {
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
pub(super) fn parse_dot_header_and_name(
    prefix: &str,
) -> Result<(bool, Option<String>), GraphTheoryError> {
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
    let name = words.next().map(unquote_dot).filter(|s| !s.is_empty());
    if kw.eq_ignore_ascii_case("digraph") {
        Ok((true, name))
    } else if kw.eq_ignore_ascii_case("graph") {
        Ok((false, name))
    } else {
        Err(GraphTheoryError::ParseError {
            format: "DOT",
            message: format!("unrecognized graph keyword '{kw}'"),
        })
    }
}

#[verified_engine::verified]
pub(super) fn split_dot_statements(body: &str) -> Vec<String> {
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
pub(super) fn is_global_dot_attr(head: &str, has_brackets: bool) -> bool {
    if head.contains('=') && !head.contains("->") && !head.contains("--") {
        return true;
    }
    has_brackets
        && (head.eq_ignore_ascii_case("graph")
            || head.eq_ignore_ascii_case("node")
            || head.eq_ignore_ascii_case("edge"))
}

#[verified_engine::verified]
pub(super) fn split_head_and_attrs(stmt: &str) -> (&str, Option<&str>) {
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
pub(super) fn parse_dot_attrs(raw: &str) -> BTreeMap<String, String> {
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
pub(super) fn unquote_dot(raw: &str) -> String {
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
pub(super) fn parse_dot_pos(node: usize, raw: &str) -> Result<(f64, f64), GraphTheoryError> {
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
pub(super) fn extract_dot_edge_weight(
    attrs: &BTreeMap<String, String>,
) -> Result<f64, GraphTheoryError> {
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
