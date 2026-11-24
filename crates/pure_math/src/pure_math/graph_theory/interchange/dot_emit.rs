//! Graphviz `.dot` deterministic and styled serialization helpers.

use super::super::{
    DotEdgeStyle, DotNodeStyle, DotStyleOverlay, GraphTheoryError, InterchangeGraph,
    InterchangeNode, format_f64,
};

/// Serializes `graph` into deterministic Graphviz `.dot` text without extra style overlays.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::interchange::{
///     InterchangeGraph, parse_dot, to_dot,
/// };
///
/// let mut g = InterchangeGraph::new(false);
/// g.push_edge(0, 1, 2.5)?;
/// let dot = to_dot(&g)?;
/// let parsed = parse_dot(&dot)?;
/// assert_eq!(parsed.edges.len(), 1);
/// assert!((parsed.edges[0].weight - 2.5).abs() < 1e-9);
/// # Ok::<(), math_explorer::error::Error>(())
/// ```
#[verified_engine::verified]
pub fn to_dot(graph: &InterchangeGraph) -> Result<String, GraphTheoryError> {
    to_styled_dot(graph, &DotStyleOverlay::default())
}

/// Serializes `graph` into deterministic Graphviz `.dot` text with `color`, `penwidth`, and `xlabel` overlays.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::graph_theory::interchange::{
///     DotEdgeStyle, DotStyleOverlay, InterchangeGraph, to_styled_dot,
/// };
///
/// let mut g = InterchangeGraph::new(true);
/// g.push_edge(0, 1, 5.0)?;
/// let mut overlay = DotStyleOverlay::default();
/// overlay.edge_styles.insert(
///     0,
///     DotEdgeStyle {
///         color: Some("#2ca02c".to_string()),
///         penwidth: Some(3.0),
///         xlabel: Some("f=5.0".to_string()),
///     },
/// );
/// let dot = to_styled_dot(&g, &overlay)?;
/// assert!(dot.contains("color=\"#2ca02c\""));
/// assert!(dot.contains("penwidth=3.0"));
/// assert!(dot.contains("xlabel=\"f=5.0\""));
/// # Ok::<(), math_explorer::error::Error>(())
/// ```
#[verified_engine::verified]
pub fn to_styled_dot(
    graph: &InterchangeGraph,
    overlay: &DotStyleOverlay,
) -> Result<String, GraphTheoryError> {
    graph.validate()?;
    validate_overlay_numbers(overlay)?;
    let header = if graph.directed {
        "digraph G {\n"
    } else {
        "graph G {\n"
    };
    let mut out = String::from(header);
    if let Some(caption) = &overlay.graph_label {
        out.push_str(&format!(
            "  label=\"{}\";\n  labelloc=\"t\";\n",
            escape_dot(caption)
        ));
    }
    for (&id, node) in &graph.nodes {
        append_dot_node(&mut out, id, node, overlay.node_styles.get(&id));
    }
    let op = if graph.directed { "->" } else { "--" };
    for (idx, edge) in graph.edges.iter().enumerate() {
        append_dot_edge(
            &mut out,
            op,
            edge.source,
            edge.target,
            edge.weight,
            overlay.edge_styles.get(&idx),
        );
    }
    out.push_str("}\n");
    Ok(out)
}

#[verified_engine::verified]
fn validate_overlay_numbers(overlay: &DotStyleOverlay) -> Result<(), GraphTheoryError> {
    for style in overlay.node_styles.values() {
        if let Some(pw) = style.penwidth
            && (!pw.is_finite() || pw < 0.0)
        {
            return Err(GraphTheoryError::InvalidWeight { value: pw });
        }
    }
    for style in overlay.edge_styles.values() {
        if let Some(pw) = style.penwidth
            && (!pw.is_finite() || pw < 0.0)
        {
            return Err(GraphTheoryError::InvalidWeight { value: pw });
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn escape_dot(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out
}

#[verified_engine::verified]
fn append_dot_node(
    out: &mut String,
    id: usize,
    node: &InterchangeNode,
    style: Option<&DotNodeStyle>,
) {
    let mut attrs = Vec::new();
    if let Some(lbl) = &node.label {
        attrs.push(format!("label=\"{}\"", escape_dot(lbl)));
    }
    if let Some((x, y)) = node.pos {
        attrs.push(format!("pos=\"{},{}!\"", format_f64(x), format_f64(y)));
    }
    if let Some(st) = style {
        if let Some(col) = &st.color {
            attrs.push(format!("color=\"{}\"", escape_dot(col)));
        }
        if let Some(fill) = &st.fillcolor {
            attrs.push("style=\"filled\"".to_string());
            attrs.push(format!("fillcolor=\"{}\"", escape_dot(fill)));
        }
        if let Some(pw) = st.penwidth {
            attrs.push(format!("penwidth={}", format_f64(pw)));
        }
        if let Some(xl) = &st.xlabel {
            attrs.push(format!("xlabel=\"{}\"", escape_dot(xl)));
        }
    }
    if attrs.is_empty() {
        out.push_str(&format!("  {id};\n"));
    } else {
        out.push_str(&format!("  {id} [{}];\n", attrs.join(", ")));
    }
}

#[verified_engine::verified]
fn append_dot_edge(
    out: &mut String,
    op: &str,
    u: usize,
    v: usize,
    weight: f64,
    style: Option<&DotEdgeStyle>,
) {
    let w_str = format_f64(weight);
    let mut attrs = vec![format!("weight={w_str}"), format!("label=\"{w_str}\"")];
    if let Some(st) = style {
        if let Some(col) = &st.color {
            attrs.push(format!("color=\"{}\"", escape_dot(col)));
        }
        if let Some(pw) = st.penwidth {
            attrs.push(format!("penwidth={}", format_f64(pw)));
        }
        if let Some(xl) = &st.xlabel {
            attrs.push(format!("xlabel=\"{}\"", escape_dot(xl)));
        }
    }
    out.push_str(&format!("  {u} {op} {v} [{}];\n", attrs.join(", ")));
}

// [cite:graph_parameters_rust]
