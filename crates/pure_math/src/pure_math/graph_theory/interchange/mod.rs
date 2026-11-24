//! Deterministic Graphviz `.dot`, `.graphml`, and weighted `.csv` topology interchange codecs.

mod csv;
mod dot;
mod graphml;

pub use csv::{parse_edge_csv, to_edge_csv};
pub use dot::{parse_dot, to_dot, to_styled_dot};
pub use graphml::{parse_graphml, to_graphml};

use super::error::GraphTheoryError;
use std::collections::BTreeMap;

/// Maximum vertex count permitted when importing or serializing an [`InterchangeGraph`].
pub const MAX_INTERCHANGE_VERTICES: usize = 256;

/// Maximum edge count permitted when importing or serializing an [`InterchangeGraph`].
pub const MAX_INTERCHANGE_EDGES: usize = 2048;

/// Canonical vertex record with optional finite 2D layout coordinates and display label.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InterchangeNode {
    /// Non-negative integer vertex identifier.
    pub id: usize,
    /// Optional finite `(x, y)` layout position.
    pub pos: Option<(f64, f64)>,
    /// Optional human-readable vertex label.
    pub label: Option<String>,
}

/// Canonical weighted edge record connecting `source` to `target`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InterchangeEdge {
    /// Source endpoint vertex identifier.
    pub source: usize,
    /// Target endpoint vertex identifier.
    pub target: usize,
    /// Finite scalar edge weight or capacity.
    pub weight: f64,
}

/// Canonical bounded graph representation shared across `.dot`, `.graphml`, and `.csv` codecs.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InterchangeGraph {
    /// `true` for directed (`digraph`), `false` for undirected (`graph`).
    pub directed: bool,
    /// Deterministically ordered vertex table keyed by vertex `id`.
    pub nodes: BTreeMap<usize, InterchangeNode>,
    /// Ordered edge list.
    pub edges: Vec<InterchangeEdge>,
}

/// Optional Graphviz `.dot` styling attributes for an individual vertex.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DotNodeStyle {
    /// Stroke color (e.g., `"#1f77b4"`).
    pub color: Option<String>,
    /// Interior fill color (e.g., `"#ff7f0e"`).
    pub fillcolor: Option<String>,
    /// Border stroke width in points.
    pub penwidth: Option<f64>,
    /// External annotation label (`xlabel`).
    pub xlabel: Option<String>,
}

/// Optional Graphviz `.dot` styling attributes for an individual edge.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DotEdgeStyle {
    /// Edge stroke color (e.g., `"#2ca02c"`).
    pub color: Option<String>,
    /// Edge stroke width in points.
    pub penwidth: Option<f64>,
    /// External edge annotation label (`xlabel`, e.g., flow or cut status).
    pub xlabel: Option<String>,
}

/// Active solver or invariant visual overlay serialized into styled `.dot` exports.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DotStyleOverlay {
    /// Optional top-level graph caption (`label`).
    pub graph_label: Option<String>,
    /// Per-vertex style overrides keyed by vertex `id`.
    pub node_styles: BTreeMap<usize, DotNodeStyle>,
    /// Per-edge style overrides keyed by index in [`InterchangeGraph::edges`].
    pub edge_styles: BTreeMap<usize, DotEdgeStyle>,
}

impl InterchangeGraph {
    /// Creates an empty undirected (`directed == false`) or directed (`directed == true`) topology.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(directed: bool) -> Self {
        Self {
            directed,
            nodes: BTreeMap::new(),
            edges: Vec::new(),
        }
    }

    /// Inserts or updates a vertex after checking finite coordinates and vertex count bounds.
    #[verified_engine::verified]
    pub fn insert_node(&mut self, node: InterchangeNode) -> Result<(), GraphTheoryError> {
        if let Some((x, y)) = node.pos
            && (!x.is_finite() || !y.is_finite())
        {
            return Err(GraphTheoryError::InvalidCoordinate {
                node: node.id,
                x,
                y,
            });
        }
        if !self.nodes.contains_key(&node.id) && self.nodes.len() >= MAX_INTERCHANGE_VERTICES {
            return Err(GraphTheoryError::VertexLimitExceeded {
                count: self.nodes.len() + 1,
                max: MAX_INTERCHANGE_VERTICES,
            });
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    /// Ensures vertex `id` exists in `nodes`, inserting a default coordinate-free record if absent.
    #[verified_engine::verified]
    pub fn ensure_node(&mut self, id: usize) -> Result<(), GraphTheoryError> {
        if self.nodes.contains_key(&id) {
            return Ok(());
        }
        self.insert_node(InterchangeNode {
            id,
            pos: None,
            label: None,
        })
    }

    /// Appends a weighted edge, auto-registering missing endpoints and enforcing size/finiteness bounds.
    #[verified_engine::verified]
    pub fn push_edge(
        &mut self,
        source: usize,
        target: usize,
        weight: f64,
    ) -> Result<(), GraphTheoryError> {
        if !weight.is_finite() {
            return Err(GraphTheoryError::InvalidWeight { value: weight });
        }
        if self.edges.len() >= MAX_INTERCHANGE_EDGES {
            return Err(GraphTheoryError::EdgeLimitExceeded {
                count: self.edges.len() + 1,
                max: MAX_INTERCHANGE_EDGES,
            });
        }
        self.ensure_node(source)?;
        self.ensure_node(target)?;
        self.edges.push(InterchangeEdge {
            source,
            target,
            weight,
        });
        Ok(())
    }

    /// Validates vertex/edge caps, endpoint existence, and coordinate/weight finiteness.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), GraphTheoryError> {
        if self.nodes.len() > MAX_INTERCHANGE_VERTICES {
            return Err(GraphTheoryError::VertexLimitExceeded {
                count: self.nodes.len(),
                max: MAX_INTERCHANGE_VERTICES,
            });
        }
        if self.edges.len() > MAX_INTERCHANGE_EDGES {
            return Err(GraphTheoryError::EdgeLimitExceeded {
                count: self.edges.len(),
                max: MAX_INTERCHANGE_EDGES,
            });
        }
        for (&id, node) in &self.nodes {
            if node.id != id {
                return Err(GraphTheoryError::UnknownEndpoint { node: node.id });
            }
            if let Some((x, y)) = node.pos
                && (!x.is_finite() || !y.is_finite())
            {
                return Err(GraphTheoryError::InvalidCoordinate { node: id, x, y });
            }
        }
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.source) {
                return Err(GraphTheoryError::UnknownEndpoint { node: edge.source });
            }
            if !self.nodes.contains_key(&edge.target) {
                return Err(GraphTheoryError::UnknownEndpoint { node: edge.target });
            }
            if !edge.weight.is_finite() {
                return Err(GraphTheoryError::InvalidWeight { value: edge.weight });
            }
        }
        Ok(())
    }
}

#[verified_engine::verified]
pub(super) fn resolve_token_id(
    raw: &str,
    sym_table: &mut BTreeMap<String, usize>,
    next_sym_id: &mut usize,
) -> usize {
    let trimmed = raw.trim();
    if let Ok(id) = trimmed.parse::<usize>() {
        if id >= *next_sym_id {
            *next_sym_id = id.saturating_add(1);
        }
        return id;
    }
    if let Some(rest) = trimmed.strip_prefix('n')
        && !rest.is_empty()
        && let Ok(id) = rest.parse::<usize>()
    {
        if id >= *next_sym_id {
            *next_sym_id = id.saturating_add(1);
        }
        return id;
    }
    if let Some(&existing) = sym_table.get(trimmed) {
        return existing;
    }
    let assigned = *next_sym_id;
    *next_sym_id = next_sym_id.saturating_add(1);
    sym_table.insert(trimmed.to_string(), assigned);
    assigned
}

#[verified_engine::verified]
pub(super) fn format_f64(val: f64) -> String {
    let rounded = (val * 1e6).round() / 1e6;
    if (rounded - rounded.round()).abs() < 1e-9 {
        format!("{:.1}", rounded)
    } else {
        let s = format!("{:.6}", rounded);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

// [cite:graph_parameters_rust]
