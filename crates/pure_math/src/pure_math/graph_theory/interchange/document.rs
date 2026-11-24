//! Rich cross-domain [`GraphTopologyDocument`] interchange representation and layout helpers.

use super::escape::looks_like_dot;
use super::{
    GraphTheoryError, InterchangeGraph, InterchangeNode, parse_csv_topology, parse_dot_topology,
    parse_graphml_topology, resolve_token_id, to_csv_topology, to_dot_topology,
    to_graphml_topology, to_svg_topology, to_tikz_topology,
};
use crate::pure_math::graph_theory::graph::Graph;
use crate::pure_math::graph_theory::layout::{
    LayoutBounds, circular_layout, force_directed_layout,
};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Maximum untrusted UTF-8 byte payload accepted by topology parsers (`2 MiB`).
pub const MAX_TOPOLOGY_IMPORT_BYTES: usize = 2 * 1024 * 1024;

/// Maximum vertex count permitted in an imported or validated [`GraphTopologyDocument`].
pub const MAX_TOPOLOGY_NODES: usize = 500;

/// Maximum edge count permitted in an imported or validated [`GraphTopologyDocument`].
pub const MAX_TOPOLOGY_EDGES: usize = 5_000;

/// Default 2D layout bounding box `(min_x, min_y, max_x, max_y)` for unpositioned imports.
pub const DEFAULT_TOPOLOGY_LAYOUT_BOUNDS: LayoutBounds = (80.0, 80.0, 560.0, 420.0);

/// Supported interchange serialization and import formats for [`GraphTopologyDocument`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopologyFormat {
    /// Graphviz DOT language (`.dot`, `.gv`).
    Dot,
    /// XML-based Graph Markup Language (`.graphml`).
    GraphMl,
    /// Standalone compilable LaTeX `TikZ` document (`.tex`).
    Tikz,
    /// Standalone Scalable Vector Graphics diagram (`.svg`).
    Svg,
    /// Delimited node/edge table (`.csv`, `.tsv`).
    Csv,
}

/// Visual node shape hint carried across `.dot`, `TikZ`, and `.svg` exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum TopologyNodeShape {
    /// Standard circular vertex glyph.
    #[default]
    Circle,
    /// Highlighted double-ring vertex glyph (e.g. active automaton state or vertex cover member).
    DoubleCircle,
    /// Rounded rectangular box glyph.
    Box,
}

/// Visual edge stroke hint carried across `.dot`, `TikZ`, and `.svg` exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum TopologyEdgeStyle {
    /// Standard continuous edge stroke.
    #[default]
    Solid,
    /// Dashed edge stroke (e.g. quarantined epidemic contact or disabled shortcut).
    Dashed,
    /// Emphasized thick edge stroke (e.g. shortest-path tree edge or high-flow arc).
    Thick,
}

/// Canonical vertex record inside a [`GraphTopologyDocument`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyNode {
    /// Unique identifier within the parent [`GraphTopologyDocument`].
    pub id: String,
    /// Human-readable display label.
    pub label: String,
    /// Optional finite 2D canvas coordinates `[x, y]`.
    pub position: Option<[f64; 2]>,
    /// Optional CSS hex fill color (`#RRGGBB`).
    pub fill_hex: Option<String>,
    /// Optional CSS hex stroke color (`#RRGGBB`).
    pub stroke_hex: Option<String>,
    /// Shape hint for vector diagram renderers.
    pub shape: TopologyNodeShape,
    /// Deterministic sorted key-value metadata attributes.
    pub attributes: BTreeMap<String, String>,
}

impl TopologyNode {
    /// Constructs a [`TopologyNode`] with `id`, `label`, and default styling.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self::new_owned(id.into(), label.into())
    }

    #[verified_engine::verified]
    fn new_owned(id: String, label: String) -> Self {
        Self {
            id,
            label,
            position: None,
            fill_hex: None,
            stroke_hex: None,
            shape: TopologyNodeShape::Circle,
            attributes: BTreeMap::new(),
        }
    }

    /// Sets finite 2D `[x, y]` coordinates on this vertex.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        if x.is_finite() && y.is_finite() {
            self.position = Some([x, y]);
        }
        self
    }

    /// Sets the visual shape hint on this vertex.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_shape(mut self, shape: TopologyNodeShape) -> Self {
        self.shape = shape;
        self
    }

    /// Sets the CSS hex fill color (`#RRGGBB`) on this vertex.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_color(mut self, hex: impl Into<String>) -> Self {
        self.fill_hex = Some(hex.into());
        self
    }

    /// Sets the CSS hex stroke color (`#RRGGBB`) on this vertex.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_stroke(mut self, hex: impl Into<String>) -> Self {
        self.stroke_hex = Some(hex.into());
        self
    }

    /// Inserts a metadata key-value attribute on this vertex.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        insert_topology_attr(&mut self.attributes, key.into(), value.into());
        self
    }
}

/// Canonical directed or undirected edge record inside a [`GraphTopologyDocument`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyEdge {
    /// Source vertex `id`.
    pub source: String,
    /// Target vertex `id`.
    pub target: String,
    /// Numeric edge weight (defaults to `1.0`).
    pub weight: f64,
    /// Optional human-readable edge annotation (e.g. rule fraction or latency formula).
    pub label: Option<String>,
    /// Stroke style hint for vector diagram renderers.
    pub style: TopologyEdgeStyle,
    /// Optional CSS hex stroke color (`#RRGGBB`).
    pub color_hex: Option<String>,
    /// Deterministic sorted key-value metadata attributes.
    pub attributes: BTreeMap<String, String>,
}

impl TopologyEdge {
    /// Constructs a [`TopologyEdge`] between `source` and `target` with finite `weight`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(source: impl Into<String>, target: impl Into<String>, weight: f64) -> Self {
        Self::new_owned(source.into(), target.into(), weight)
    }

    #[verified_engine::verified]
    fn new_owned(source: String, target: String, weight: f64) -> Self {
        Self {
            source,
            target,
            weight: if weight.is_finite() { weight } else { 1.0 },
            label: None,
            style: TopologyEdgeStyle::Solid,
            color_hex: None,
            attributes: BTreeMap::new(),
        }
    }

    /// Sets a human-readable edge label.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the visual line style hint on this edge.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_style(mut self, style: TopologyEdgeStyle) -> Self {
        self.style = style;
        self
    }

    /// Sets the CSS hex stroke color (`#RRGGBB`) on this edge.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_color(mut self, hex: impl Into<String>) -> Self {
        self.color_hex = Some(hex.into());
        self
    }

    /// Inserts a metadata key-value attribute on this edge.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        insert_topology_attr(&mut self.attributes, key.into(), value.into());
        self
    }
}

#[verified_engine::verified]
fn insert_topology_attr(map: &mut BTreeMap<String, String>, key: String, value: String) {
    map.insert(key, value);
}

/// Shared interchange representation for directed and undirected graphs, automata, and flow networks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphTopologyDocument {
    /// Identifier or title of the graph.
    pub name: String,
    /// `true` when edges are directed (`->`), `false` when undirected (`--`).
    pub directed: bool,
    /// Ordered vertex list (`<= MAX_TOPOLOGY_NODES`).
    pub nodes: Vec<TopologyNode>,
    /// Ordered edge list (`<= MAX_TOPOLOGY_EDGES`).
    pub edges: Vec<TopologyEdge>,
    /// Deterministic sorted graph-level metadata attributes.
    pub attributes: BTreeMap<String, String>,
}

impl GraphTopologyDocument {
    /// Creates an empty [`GraphTopologyDocument`] with `name` and directionality.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(name: impl Into<String>, directed: bool) -> Self {
        Self {
            name: name.into(),
            directed,
            nodes: Vec::new(),
            edges: Vec::new(),
            attributes: BTreeMap::new(),
        }
    }

    /// Builds a [`GraphTopologyDocument`] from a solver [`InterchangeGraph`].
    #[must_use]
    #[verified_engine::verified]
    pub fn from_interchange(name: impl Into<String>, graph: &InterchangeGraph) -> Self {
        let mut doc = Self::new(name, graph.directed);
        for (&id, node) in &graph.nodes {
            let label = node.label.clone().unwrap_or_else(|| id.to_string());
            let mut tn = TopologyNode::new(id.to_string(), label);
            if let Some((x, y)) = node.pos {
                tn = tn.with_position(x, y);
            }
            doc.nodes.push(tn);
        }
        for edge in &graph.edges {
            doc.edges.push(TopologyEdge::new(
                edge.source.to_string(),
                edge.target.to_string(),
                edge.weight,
            ));
        }
        doc
    }

    /// Converts this document into a bounded integer-indexed [`InterchangeGraph`].
    #[verified_engine::verified]
    pub fn to_interchange(&self) -> Result<InterchangeGraph, GraphTheoryError> {
        self.validate()?;
        let mut g = InterchangeGraph::new(self.directed);
        let mut sym_table = BTreeMap::new();
        let mut next_sym_id = 0usize;
        for node in &self.nodes {
            let id = resolve_token_id(&node.id, &mut sym_table, &mut next_sym_id);
            let pos = node.position.map(|[x, y]| (x, y));
            let label = (node.label != node.id).then(|| node.label.clone());
            g.insert_node(InterchangeNode { id, pos, label })?;
        }
        for edge in &self.edges {
            let u = resolve_token_id(&edge.source, &mut sym_table, &mut next_sym_id);
            let v = resolve_token_id(&edge.target, &mut sym_table, &mut next_sym_id);
            g.push_edge(u, v, edge.weight)?;
        }
        Ok(g)
    }

    /// Validates structural caps, finite numeric fields, unique node IDs, and endpoint references.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), GraphTheoryError> {
        if self.nodes.len() > MAX_TOPOLOGY_NODES {
            return Err(GraphTheoryError::VertexLimitExceeded {
                count: self.nodes.len(),
                max: MAX_TOPOLOGY_NODES,
            });
        }
        if self.edges.len() > MAX_TOPOLOGY_EDGES {
            return Err(GraphTheoryError::EdgeLimitExceeded {
                count: self.edges.len(),
                max: MAX_TOPOLOGY_EDGES,
            });
        }
        let mut seen = HashSet::with_capacity(self.nodes.len());
        for (idx, node) in self.nodes.iter().enumerate() {
            validate_topology_node(idx, node, &mut seen)?;
        }
        for edge in &self.edges {
            validate_topology_edge(edge, &seen)?;
        }
        Ok(())
    }

    /// Assigns deterministic 2D fallback coordinates inside `bounds` whenever any node lacks `position`.
    #[verified_engine::verified]
    pub fn ensure_positions(&mut self, bounds: LayoutBounds) {
        let all_positioned = self.nodes.iter().all(|n| {
            n.position
                .is_some_and(|[x, y]| x.is_finite() && y.is_finite())
        });
        if all_positioned || self.nodes.is_empty() {
            return;
        }
        let Ok((pg, indices)) = self.to_undirected_petgraph() else {
            return;
        };
        let layout_map = if !self.edges.is_empty() && self.nodes.len() <= 120 {
            force_directed_layout(&pg, bounds, 60)
        } else {
            circular_layout(&pg, bounds)
        };
        for (node, idx) in self.nodes.iter_mut().zip(indices) {
            if node
                .position
                .is_none_or(|[x, y]| !x.is_finite() || !y.is_finite())
                && let Some(&(x, y)) = layout_map.get(&idx)
            {
                node.position = Some([f64::from(x), f64::from(y)]);
            }
        }
    }

    /// Converts this document into a simple undirected [`Graph<String, f64>`] for parameter analysis.
    #[verified_engine::verified]
    pub fn to_undirected_petgraph(
        &self,
    ) -> Result<(Graph<String, f64>, Vec<NodeIndex>), GraphTheoryError> {
        self.validate()?;
        let mut g = Graph::new();
        let mut id_to_idx = HashMap::with_capacity(self.nodes.len());
        let mut indices = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let idx = g.add_node(node.label.clone());
            id_to_idx.insert(node.id.as_str(), idx);
            indices.push(idx);
        }
        let mut added_pairs = HashSet::with_capacity(self.edges.len());
        for edge in &self.edges {
            let u = id_to_idx[edge.source.as_str()];
            let v = id_to_idx[edge.target.as_str()];
            if u == v {
                continue;
            }
            let pair = if u.index() <= v.index() {
                (u, v)
            } else {
                (v, u)
            };
            if added_pairs.insert(pair) {
                g.add_edge(u, v, edge.weight);
            }
        }
        Ok((g, indices))
    }

    /// Serializes this topology document into the requested [`TopologyFormat`].
    #[verified_engine::verified]
    pub fn serialize(&self, format: TopologyFormat) -> Result<String, GraphTheoryError> {
        self.validate()?;
        match format {
            TopologyFormat::Dot => Ok(to_dot_topology(self)),
            TopologyFormat::GraphMl => Ok(to_graphml_topology(self)),
            TopologyFormat::Tikz => Ok(to_tikz_topology(self)),
            TopologyFormat::Svg => Ok(to_svg_topology(self)),
            TopologyFormat::Csv => Ok(to_csv_topology(self)),
        }
    }

    /// Parses `.dot`/`.gv`, `.graphml`, or `.csv`/`.tsv` text with BOM stripping, size caps, and layout fallback.
    #[verified_engine::verified]
    pub fn parse_with_filename(filename: &str, raw: &str) -> Result<Self, GraphTheoryError> {
        let cleaned = strip_utf8_bom_str(raw);
        check_import_byte_limit(cleaned)?;
        let lower = filename.trim().to_ascii_lowercase();
        let mut doc = if lower.ends_with(".graphml") || cleaned.trim_start().starts_with('<') {
            parse_graphml_topology(cleaned)?
        } else if lower.ends_with(".dot") || lower.ends_with(".gv") || looks_like_dot(cleaned) {
            parse_dot_topology(cleaned)?
        } else {
            parse_csv_topology(cleaned)?
        };
        doc.ensure_positions(DEFAULT_TOPOLOGY_LAYOUT_BOUNDS);
        Ok(doc)
    }
}

#[verified_engine::verified]
fn validate_topology_node<'a>(
    idx: usize,
    node: &'a TopologyNode,
    seen: &mut HashSet<&'a str>,
) -> Result<(), GraphTheoryError> {
    if node.id.trim().is_empty() {
        return Err(GraphTheoryError::ParseError {
            format: "Topology",
            message: "node ID cannot be empty".into(),
        });
    }
    if !seen.insert(node.id.as_str()) {
        return Err(GraphTheoryError::ParseError {
            format: "Topology",
            message: format!("duplicate node ID '{}'", node.id),
        });
    }
    if let Some([x, y]) = node.position
        && (!x.is_finite() || !y.is_finite())
    {
        return Err(GraphTheoryError::InvalidCoordinate { node: idx, x, y });
    }
    Ok(())
}

#[verified_engine::verified]
fn validate_topology_edge(
    edge: &TopologyEdge,
    seen: &HashSet<&str>,
) -> Result<(), GraphTheoryError> {
    if !edge.weight.is_finite() {
        return Err(GraphTheoryError::InvalidWeight { value: edge.weight });
    }
    if !seen.contains(edge.source.as_str()) || !seen.contains(edge.target.as_str()) {
        return Err(GraphTheoryError::ParseError {
            format: "Topology",
            message: format!(
                "edge '{} -> {}' references undeclared node",
                edge.source, edge.target
            ),
        });
    }
    Ok(())
}

/// Strips a leading UTF-8 byte-order mark (`U+FEFF`) when present.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn strip_utf8_bom_str(raw: &str) -> &str {
    raw.strip_prefix('\u{feff}').unwrap_or(raw)
}

#[inline]
#[verified_engine::verified]
pub(super) fn check_import_byte_limit(raw: &str) -> Result<(), GraphTheoryError> {
    if raw.len() > MAX_TOPOLOGY_IMPORT_BYTES {
        return Err(GraphTheoryError::PayloadTooLarge {
            actual: raw.len(),
            max: MAX_TOPOLOGY_IMPORT_BYTES,
        });
    }
    Ok(())
}

// [cite:graph_parameters_rust]
