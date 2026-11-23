//! Root-owned frame-staged `PaneDataflowBus` and cycle-checked `PanePipelineDag`.
//!
//! Routes typed graph topologies, 3D chaotic landmark clouds, and 2D scalar fields
//! across multi-pane viewports without holding any mutex across `ExplorerTab::show_inside`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "dataflow_bus_staging.rs"]
mod staging;

#[cfg(any(test, feature = "epidemiology", feature = "biology"))]
pub use staging::poll_pane_graph;
#[cfg(any(test, feature = "physics", feature = "biology"))]
pub use staging::publish_pane_landmarks;
#[cfg(any(test, feature = "biology"))]
pub use staging::publish_pane_scalar_field;
pub use staging::{PaneDataflowBus, commit_pane_outputs, stage_pane_inputs};
#[cfg(any(test, feature = "pure_math"))]
pub use staging::{poll_pane_landmarks, poll_pane_scalar_field, publish_pane_graph};

use serde::{Deserialize, Serialize};

/// Maximum number of viewport panes supported by `ViewportLayout::QuadGrid`.
pub const MAX_DATAFLOW_PANES: usize = 4;
/// Upper bound on graph vertices routed over `PaneDataflowBus` and pipeline URLs.
pub const MAX_DATAFLOW_GRAPH_NODES: usize = 200;
/// Upper bound on graph edges routed over `PaneDataflowBus`.
pub const MAX_DATAFLOW_GRAPH_EDGES: usize = 2_000;
/// Upper bound on 3D landmarks routed over `PaneDataflowBus`.
pub const MAX_DATAFLOW_LANDMARKS: usize = 64;
/// Upper bound per axis on 2D scalar field grids routed over `PaneDataflowBus`.
pub const MAX_DATAFLOW_FIELD_DIM: usize = 64;

/// Typed cross-pane stream families supported by [`PaneDataflowBus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataflowStreamKind {
    /// Compact weighted graph topology (`<= 200` vertices).
    GraphTopology,
    /// Subsampled 3D attractor landmark cloud (`<= 64` points) and Lorenz parameters.
    Landmarks3D,
    /// Row-major 2D scalar field grid (`<= 64x64` samples) and domain bounds.
    ScalarField2D,
}

impl DataflowStreamKind {
    /// Human-readable label for UI menus and ARIA live announcements.
    pub const fn label(self) -> &'static str {
        match self {
            Self::GraphTopology => "graph topology",
            Self::Landmarks3D => "3D landmarks",
            Self::ScalarField2D => "2D scalar field",
        }
    }
}

/// Sanitized graph topology snapshot published by `GraphEditorTool`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphTopologyPayload {
    /// Monotonic or content-addressed revision token from upstream producer.
    pub revision: u64,
    /// Compact `0..N` 2D canvas positions (`N <= MAX_DATAFLOW_GRAPH_NODES`).
    pub node_positions: Vec<[f32; 2]>,
    /// Compact `(u, v, weight)` edges with `u, v < node_positions.len()`.
    pub edges: Vec<(usize, usize, f64)>,
    /// Whether edges are directed arcs (`true`) or undirected couplings (`false`).
    pub directed: bool,
    /// Original vertex count when truncated to `MAX_DATAFLOW_GRAPH_NODES`.
    pub truncated_from: Option<usize>,
}

impl GraphTopologyPayload {
    /// Builds a sanitized payload enforcing the `200`-node cap and finite coordinates/weights.
    pub fn sanitized(
        revision: u64,
        mut node_positions: Vec<[f32; 2]>,
        edges: Vec<(usize, usize, f64)>,
        directed: bool,
    ) -> Self {
        let truncated_from =
            (node_positions.len() > MAX_DATAFLOW_GRAPH_NODES).then_some(node_positions.len());
        node_positions.truncate(MAX_DATAFLOW_GRAPH_NODES);
        for pos in &mut node_positions {
            if !pos[0].is_finite() {
                pos[0] = 0.0;
            }
            if !pos[1].is_finite() {
                pos[1] = 0.0;
            }
        }
        let n = node_positions.len();
        let mut clean_edges = Vec::with_capacity(edges.len().min(MAX_DATAFLOW_GRAPH_EDGES));
        for (u, v, w) in edges {
            if clean_edges.len() >= MAX_DATAFLOW_GRAPH_EDGES {
                break;
            }
            if u < n && v < n && u != v && w.is_finite() {
                clean_edges.push((u, v, w));
            }
        }
        Self {
            revision: revision.max(1),
            node_positions,
            edges: clean_edges,
            directed,
            truncated_from,
        }
    }

    /// Centers and scales up to `limit` vertex coordinates into `[-half_span, half_span]^2`.
    pub fn normalized_positions(&self, limit: usize, half_span: f32) -> Vec<[f32; 2]> {
        let count = self.node_positions.len().min(limit);
        let raw = &self.node_positions[..count];
        if raw.is_empty() {
            return Vec::new();
        }
        let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        for &[x, y] in raw {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let cx = 0.5 * (min_x + max_x);
        let cy = 0.5 * (min_y + max_y);
        let max_extent = ((max_x - min_x).max(max_y - min_y) * 0.5).max(1.0);
        let scale = half_span / max_extent;
        raw.iter()
            .map(|&[x, y]| [(x - cx) * scale, (y - cy) * scale])
            .collect()
    }
}

/// Sanitized 3D chaotic landmark payload published by `AttractorPlotter`.
#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkCloud3DPayload {
    /// Monotonic or state revision token from upstream producer.
    pub revision: u64,
    /// Finite 3D coordinates (`<= MAX_DATAFLOW_LANDMARKS`).
    pub points: Vec<[f64; 3]>,
    /// Lorenz Prandtl parameter `σ`.
    pub sigma: f64,
    /// Lorenz Rayleigh parameter `ρ`.
    pub rho: f64,
    /// Lorenz geometric parameter `β`.
    pub beta: f64,
    /// Integration step `dt`.
    pub dt: f64,
}

impl LandmarkCloud3DPayload {
    /// Builds a sanitized 3D landmark payload capped at `MAX_DATAFLOW_LANDMARKS`.
    pub fn sanitized(
        revision: u64,
        points: Vec<[f64; 3]>,
        sigma: f64,
        rho: f64,
        beta: f64,
        dt: f64,
    ) -> Self {
        let clean_points: Vec<[f64; 3]> = points
            .into_iter()
            .filter(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite())
            .take(MAX_DATAFLOW_LANDMARKS)
            .collect();
        Self {
            revision: revision.max(1),
            points: clean_points,
            sigma: if sigma.is_finite() { sigma } else { 10.0 },
            rho: if rho.is_finite() { rho } else { 28.0 },
            beta: if beta.is_finite() { beta } else { 8.0 / 3.0 },
            dt: if dt.is_finite() && dt > 0.0 { dt } else { 0.01 },
        }
    }
}

/// Sanitized 2D scalar field payload published by `MorphogenesisTool`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField2DPayload {
    /// Monotonic or state revision token from upstream producer.
    pub revision: u64,
    /// Downsampled grid width (`1..=MAX_DATAFLOW_FIELD_DIM`).
    pub width: usize,
    /// Downsampled grid height (`1..=MAX_DATAFLOW_FIELD_DIM`).
    pub height: usize,
    /// Row-major finite scalar field values (`width * height`).
    pub values: Vec<f64>,
    /// Spatial `u` (`x`) domain bounds `[min, max]`.
    pub u_bounds: [f64; 2],
    /// Spatial `v` (`y`) domain bounds `[min, max]`.
    pub v_bounds: [f64; 2],
}

impl ScalarField2DPayload {
    /// Downsamples a row-major `src_width * src_height` grid to `<= MAX_DATAFLOW_FIELD_DIM`.
    pub fn from_grid_downsampled(
        revision: u64,
        src_width: usize,
        src_height: usize,
        src_values: &[f64],
        u_bounds: [f64; 2],
        v_bounds: [f64; 2],
    ) -> Option<Self> {
        if src_width == 0
            || src_height == 0
            || src_values.len() < src_width.saturating_mul(src_height)
        {
            return None;
        }
        let dst_w = src_width.min(MAX_DATAFLOW_FIELD_DIM);
        let dst_h = src_height.min(MAX_DATAFLOW_FIELD_DIM);
        let mut values = Vec::with_capacity(dst_w * dst_h);
        for dy in 0..dst_h {
            let sy = (dy * src_height / dst_h).min(src_height - 1);
            for dx in 0..dst_w {
                let sx = (dx * src_width / dst_w).min(src_width - 1);
                let raw = src_values
                    [math_commons::math_kernel::types::flatten_2d_index(sx, sy, src_width)];
                values.push(if raw.is_finite() { raw } else { 0.0 });
            }
        }
        Some(Self {
            revision: revision.max(1),
            width: dst_w,
            height: dst_h,
            values,
            u_bounds,
            v_bounds,
        })
    }
}

/// Returns the stream kind emitted by `tool_id`, if it acts as a bus producer.
pub fn tool_produced_stream(tool_id: Option<&str>) -> Option<DataflowStreamKind> {
    match tool_id {
        Some("GraphEditorTool") => Some(DataflowStreamKind::GraphTopology),
        Some("AttractorPlotter") => Some(DataflowStreamKind::Landmarks3D),
        Some("MorphogenesisTool") => Some(DataflowStreamKind::ScalarField2D),
        _ => None,
    }
}

/// Returns stream kinds consumed by `tool_id`, if it acts as a bus sink.
pub fn tool_accepted_streams(tool_id: Option<&str>) -> &'static [DataflowStreamKind] {
    match tool_id {
        Some("NetworkPropagationTool" | "NeuralNetworkVizTool") => {
            &[DataflowStreamKind::GraphTopology]
        }
        Some("VietorisRipsTool") => &[
            DataflowStreamKind::Landmarks3D,
            DataflowStreamKind::ScalarField2D,
        ],
        Some("SurfaceViewer") => &[
            DataflowStreamKind::ScalarField2D,
            DataflowStreamKind::Landmarks3D,
        ],
        _ => &[],
    }
}

/// Resolves the compatible [`DataflowStreamKind`] between a producer tool and consumer tool.
pub fn compatible_stream_kind(
    producer_tool_id: Option<&str>,
    consumer_tool_id: Option<&str>,
) -> Option<DataflowStreamKind> {
    let produced = tool_produced_stream(producer_tool_id)?;
    tool_accepted_streams(consumer_tool_id)
        .contains(&produced)
        .then_some(produced)
}

/// Directed dataflow edge connecting an upstream source pane to a downstream sink pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PaneLink {
    /// Upstream producer pane index (`0..MAX_DATAFLOW_PANES`).
    pub from_pane: usize,
    /// Downstream consumer pane index (`0..MAX_DATAFLOW_PANES`).
    pub to_pane: usize,
    /// Routed payload stream kind.
    pub kind: DataflowStreamKind,
}

/// Validation failure returned when mutating [`PanePipelineDag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DagLinkError {
    /// Pane index exceeds `MAX_DATAFLOW_PANES - 1` or active pane count.
    OutOfBounds,
    /// Source and destination pane indices are identical (`from_pane == to_pane`).
    SelfLoop,
    /// Adding the edge would introduce a directed cycle (`A -> ... -> A`).
    CycleDetected,
    /// Source and destination tools do not share a compatible stream type.
    IncompatibleStream,
}

impl DagLinkError {
    /// Accessible diagnostic message suitable for `TwinSlotLiveRegion` assertive announcements.
    pub const fn message(self) -> &'static str {
        match self {
            Self::OutOfBounds => "Pipeline link rejected: pane index is inactive.",
            Self::SelfLoop => "Pipeline link rejected: a pane cannot stream into itself.",
            Self::CycleDetected => "Pipeline link rejected: connection would create a cycle.",
            Self::IncompatibleStream => {
                "Pipeline link rejected: upstream and downstream tools are incompatible."
            }
        }
    }
}

/// Cycle-checked directed acyclic graph of active cross-pane dataflow links.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PanePipelineDag {
    links: Vec<PaneLink>,
}

impl PanePipelineDag {
    /// Returns `true` when no cross-pane pipeline links are active.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Returns active links sorted deterministically by `(to_pane, from_pane)`.
    pub fn links(&self) -> &[PaneLink] {
        &self.links
    }

    /// Returns the active upstream link feeding `to_pane`, if connected.
    pub fn upstream_for(&self, to_pane: usize) -> Option<PaneLink> {
        self.links.iter().copied().find(|l| l.to_pane == to_pane)
    }

    /// Returns `true` if `from_pane` feeds at least one downstream pane.
    pub fn has_downstream(&self, from_pane: usize) -> bool {
        self.links.iter().any(|l| l.from_pane == from_pane)
    }

    /// Checks whether routing `from_pane -> to_pane` (replacing any prior input on `to_pane`) forms a cycle.
    pub fn would_create_cycle(&self, from_pane: usize, to_pane: usize) -> bool {
        if from_pane == to_pane {
            return true;
        }
        let mut visited = [false; MAX_DATAFLOW_PANES];
        let mut stack = [0usize; MAX_DATAFLOW_PANES];
        let mut top = 0usize;
        if to_pane < MAX_DATAFLOW_PANES {
            visited[to_pane] = true;
            stack[top] = to_pane;
            top += 1;
        }
        while top > 0 {
            top -= 1;
            let curr = stack[top];
            if curr == from_pane {
                return true;
            }
            for link in &self.links {
                if link.to_pane == to_pane {
                    continue;
                }
                if link.from_pane == curr
                    && link.to_pane < MAX_DATAFLOW_PANES
                    && !visited[link.to_pane]
                {
                    visited[link.to_pane] = true;
                    stack[top] = link.to_pane;
                    top += 1;
                }
            }
        }
        false
    }

    /// Connects `from_pane -> to_pane` after validating bounds, self-loops, and DAG acyclicity.
    pub fn connect(
        &mut self,
        from_pane: usize,
        to_pane: usize,
        kind: DataflowStreamKind,
    ) -> Result<(), DagLinkError> {
        if from_pane >= MAX_DATAFLOW_PANES || to_pane >= MAX_DATAFLOW_PANES {
            return Err(DagLinkError::OutOfBounds);
        }
        if from_pane == to_pane {
            return Err(DagLinkError::SelfLoop);
        }
        if self.would_create_cycle(from_pane, to_pane) {
            return Err(DagLinkError::CycleDetected);
        }
        self.links.retain(|l| l.to_pane != to_pane);
        self.links.push(PaneLink {
            from_pane,
            to_pane,
            kind,
        });
        self.sort_canonical();
        Ok(())
    }

    /// Disconnects any upstream feed targeting `to_pane`.
    pub fn disconnect_target(&mut self, to_pane: usize) -> Option<PaneLink> {
        let pos = self.links.iter().position(|l| l.to_pane == to_pane)?;
        Some(self.links.remove(pos))
    }

    /// Clears all pipeline links.
    pub fn clear(&mut self) {
        self.links.clear();
    }

    /// Remaps endpoint pane indices when `pane_a` and `pane_b` swap slots.
    pub fn swap_panes(&mut self, pane_a: usize, pane_b: usize) {
        if pane_a == pane_b {
            return;
        }
        let remap = |idx: usize| {
            if idx == pane_a {
                pane_b
            } else if idx == pane_b {
                pane_a
            } else {
                idx
            }
        };
        for link in &mut self.links {
            link.from_pane = remap(link.from_pane);
            link.to_pane = remap(link.to_pane);
        }
        self.sort_canonical();
    }

    /// Mirrors an upstream feed when cloning `src_pane` into `dst_pane` if valid and acyclic.
    pub fn on_clone_pane(&mut self, src_pane: usize, dst_pane: usize) {
        self.disconnect_target(dst_pane);
        if let Some(up) = self.upstream_for(src_pane)
            && up.from_pane != dst_pane
        {
            let _ = self.connect(up.from_pane, dst_pane, up.kind);
        }
    }

    /// Prunes links whose source or target exceeds `active_pane_count - 1`.
    pub fn prune_inactive(&mut self, active_pane_count: usize) {
        self.links
            .retain(|l| l.from_pane < active_pane_count && l.to_pane < active_pane_count);
    }

    /// Prunes links whose current pane tools no longer match `link.kind`.
    pub fn prune_incompatible<'a, F>(&mut self, active_pane_count: usize, tool_id_at: F)
    where
        F: Fn(usize) -> Option<&'a str>,
    {
        self.prune_inactive(active_pane_count);
        self.links.retain(|l| {
            compatible_stream_kind(tool_id_at(l.from_pane), tool_id_at(l.to_pane)) == Some(l.kind)
        });
    }

    /// Atomically validates and restores a slice of links for `active_pane_count` panes.
    pub fn restore_links(
        &mut self,
        links: &[PaneLink],
        active_pane_count: usize,
    ) -> Result<(), DagLinkError> {
        let mut next = Self::default();
        for link in links {
            if link.from_pane >= active_pane_count || link.to_pane >= active_pane_count {
                return Err(DagLinkError::OutOfBounds);
            }
            next.connect(link.from_pane, link.to_pane, link.kind)?;
        }
        *self = next;
        Ok(())
    }

    fn sort_canonical(&mut self) {
        self.links
            .sort_by_key(|l| (l.to_pane, l.from_pane, l.kind as u8));
    }
}
