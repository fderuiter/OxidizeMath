//! Live [`crate::app::SharedGraph`] publishing and hydration for [`super::GraphEditorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use crate::app::data_bus::{publish_shared_graph, read_shared_graph, SharedGraph};
use eframe::egui::{self, Pos2};

impl GraphEditorTool {
    /// Builds a normalized [`SharedGraph`] snapshot from the editor's current nodes and signed edges.
    #[must_use]
    pub fn to_shared_graph(&self, revision: u64) -> SharedGraph {
        let sorted_nodes: Vec<(usize, Pos2)> = self
            .node_positions
            .iter()
            .map(|(&id, &pos)| (id, pos))
            .collect();
        SharedGraph::from_canvas_positions(revision, &sorted_nodes, &self.edges)
    }

    /// Hydrates `GraphEditorTool` from an existing [`SharedGraph`] when opened empty on a live bus.
    pub fn hydrate_from_shared_graph(&mut self, shared: &SharedGraph) {
        self.node_positions.clear();
        self.edges.clear();

        let mut max_id = 0usize;
        for node in &shared.nodes {
            let x = 100.0 + node.norm_pos[0].clamp(0.0, 1.0) * 420.0;
            let y = 80.0 + node.norm_pos[1].clamp(0.0, 1.0) * 320.0;
            self.node_positions.insert(node.id, Pos2::new(x, y));
            max_id = max_id.max(node.id.saturating_add(1));
        }

        for &(u_idx, v_idx, w) in &shared.edges {
            if let (Some(nu), Some(nv)) = (shared.nodes.get(u_idx), shared.nodes.get(v_idx)) {
                self.edges.push((nu.id, nv.id, w));
            }
        }

        self.next_node_id = max_id;
        self.selected_node = None;
        self.rebuild_graph();
        self.bump_revision();
        self.last_published_revision = self.revision;
    }

    /// Synchronizes local graph edits to [`crate::app::WorkspaceDataBus`] on `ctx`.
    ///
    /// Publishes a new `Arc<SharedGraph>` revision whenever `self.revision` advances due to
    /// node edits, drags, `+w`/`-w` brush strokes, hub stamps, sever strokes, layouts, clear,
    /// undo, redo, preset imports, or session loads.
    pub(super) fn sync_workspace_bus(&mut self, ctx: &egui::Context) {
        if self.revision == 0
            && self.node_positions.is_empty()
            && self.edges.is_empty()
            && self.last_published_revision == 0
        {
            if let Some(existing) = read_shared_graph(ctx) {
                if !existing.nodes.is_empty() {
                    self.hydrate_from_shared_graph(&existing);
                }
            }
            return;
        }

        if self.revision != self.last_published_revision {
            let snapshot = self.to_shared_graph(1);
            let _ = publish_shared_graph(ctx, snapshot);
            self.last_published_revision = self.revision;
        }
    }
}
