//! Frame-staged storage and `egui::Context` helpers for [`super::PaneDataflowBus`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    DataflowStreamKind, GraphTopologyPayload, LandmarkCloud3DPayload, MAX_DATAFLOW_PANES,
    PanePipelineDag, ScalarField2DPayload,
};
use crate::app::viewport::focus::CURRENT_RENDERING_PANE_KEY;
use eframe::egui;
use std::sync::Arc;

#[derive(Clone, Default)]
struct StagedPaneInput {
    graph: Option<Arc<GraphTopologyPayload>>,
    landmarks: Option<Arc<LandmarkCloud3DPayload>>,
    field: Option<Arc<ScalarField2DPayload>>,
}

#[derive(Clone, Default)]
struct StagedPaneOutput {
    graph: Option<Arc<GraphTopologyPayload>>,
    landmarks: Option<Arc<LandmarkCloud3DPayload>>,
    field: Option<Arc<ScalarField2DPayload>>,
}

/// Root-owned typed payload store for active panes (`MathExplorerApp::dataflow_bus`).
#[derive(Default)]
pub struct PaneDataflowBus {
    graphs: [Option<Arc<GraphTopologyPayload>>; MAX_DATAFLOW_PANES],
    landmarks: [Option<Arc<LandmarkCloud3DPayload>>; MAX_DATAFLOW_PANES],
    fields: [Option<Arc<ScalarField2DPayload>>; MAX_DATAFLOW_PANES],
}

impl PaneDataflowBus {
    /// Swaps cached stream slots when two panes exchange positions.
    pub fn swap_panes(&mut self, pane_a: usize, pane_b: usize) {
        if pane_a < MAX_DATAFLOW_PANES && pane_b < MAX_DATAFLOW_PANES && pane_a != pane_b {
            self.graphs.swap(pane_a, pane_b);
            self.landmarks.swap(pane_a, pane_b);
            self.fields.swap(pane_a, pane_b);
        }
    }

    /// Drops cached stream payloads on inactive panes (`>= active_pane_count`).
    pub fn prune_inactive(&mut self, active_pane_count: usize) {
        for idx in active_pane_count..MAX_DATAFLOW_PANES {
            self.clear_pane(idx);
        }
    }

    /// Clears all published payloads for `pane_idx`.
    pub fn clear_pane(&mut self, pane_idx: usize) {
        if pane_idx < MAX_DATAFLOW_PANES {
            self.graphs[pane_idx] = None;
            self.landmarks[pane_idx] = None;
            self.fields[pane_idx] = None;
        }
    }

    /// Returns the latest graph payload published by `pane_idx`.
    pub fn latest_graph(&self, pane_idx: usize) -> Option<&Arc<GraphTopologyPayload>> {
        self.graphs.get(pane_idx).and_then(Option::as_ref)
    }

    /// Returns the latest 3D landmark payload published by `pane_idx`.
    pub fn latest_landmarks(&self, pane_idx: usize) -> Option<&Arc<LandmarkCloud3DPayload>> {
        self.landmarks.get(pane_idx).and_then(Option::as_ref)
    }

    /// Returns the latest 2D scalar field payload published by `pane_idx`.
    pub fn latest_field(&self, pane_idx: usize) -> Option<&Arc<ScalarField2DPayload>> {
        self.fields.get(pane_idx).and_then(Option::as_ref)
    }
}

fn input_slot_id(pane_idx: usize) -> egui::Id {
    egui::Id::new(("pane_dataflow_in", pane_idx))
}

fn output_slot_id(pane_idx: usize) -> egui::Id {
    egui::Id::new(("pane_dataflow_out", pane_idx))
}

fn active_pane_index(ctx: &egui::Context) -> Option<usize> {
    ctx.data(|d| d.get_temp::<usize>(egui::Id::new(CURRENT_RENDERING_PANE_KEY)))
}

/// Stages upstream `Arc` payloads into `ctx` temp storage immediately before `tab.show_inside`.
pub fn stage_pane_inputs(
    ctx: &egui::Context,
    pane_idx: usize,
    dag: &PanePipelineDag,
    bus: &PaneDataflowBus,
) {
    let staged = match dag.upstream_for(pane_idx) {
        Some(link) => match link.kind {
            DataflowStreamKind::GraphTopology => StagedPaneInput {
                graph: bus.latest_graph(link.from_pane).cloned(),
                ..Default::default()
            },
            DataflowStreamKind::Landmarks3D => StagedPaneInput {
                landmarks: bus.latest_landmarks(link.from_pane).cloned(),
                ..Default::default()
            },
            DataflowStreamKind::ScalarField2D => StagedPaneInput {
                field: bus.latest_field(link.from_pane).cloned(),
                ..Default::default()
            },
        },
        None => StagedPaneInput::default(),
    };
    ctx.data_mut(|d| {
        d.insert_temp(input_slot_id(pane_idx), staged);
        d.remove::<StagedPaneOutput>(output_slot_id(pane_idx));
    });
}

/// Commits any outgoing payloads emitted during `tab.show_inside` back into `PaneDataflowBus`.
pub fn commit_pane_outputs(
    ctx: &egui::Context,
    pane_idx: usize,
    dag: &PanePipelineDag,
    bus: &mut PaneDataflowBus,
) {
    let out = ctx.data_mut(|d| {
        d.remove::<StagedPaneInput>(input_slot_id(pane_idx));
        d.remove_temp::<StagedPaneOutput>(output_slot_id(pane_idx))
    });
    let Some(out) = out else {
        return;
    };
    if pane_idx >= MAX_DATAFLOW_PANES {
        return;
    }
    let mut advanced = false;
    if let Some(graph) = out.graph {
        advanced |= bus.graphs[pane_idx]
            .as_ref()
            .is_none_or(|prev| prev.revision != graph.revision);
        bus.graphs[pane_idx] = Some(graph);
    }
    if let Some(landmarks) = out.landmarks {
        advanced |= bus.landmarks[pane_idx]
            .as_ref()
            .is_none_or(|prev| prev.revision != landmarks.revision);
        bus.landmarks[pane_idx] = Some(landmarks);
    }
    if let Some(field) = out.field {
        advanced |= bus.fields[pane_idx]
            .as_ref()
            .is_none_or(|prev| prev.revision != field.revision);
        bus.fields[pane_idx] = Some(field);
    }
    if advanced
        && dag
            .links()
            .iter()
            .any(|l| l.from_pane == pane_idx && l.to_pane < pane_idx)
    {
        ctx.request_repaint();
    }
}

/// Publishes a sanitized graph topology from the currently rendering pane.
#[cfg(any(test, feature = "pure_math"))]
pub fn publish_pane_graph(ctx: &egui::Context, payload: GraphTopologyPayload) {
    let Some(pane_idx) = active_pane_index(ctx) else {
        return;
    };
    let arc = Arc::new(payload);
    ctx.data_mut(|d| {
        let mut entry = d
            .get_temp::<StagedPaneOutput>(output_slot_id(pane_idx))
            .unwrap_or_default();
        entry.graph = Some(arc);
        d.insert_temp(output_slot_id(pane_idx), entry);
    });
}

/// Publishes a sanitized 3D landmark cloud from the currently rendering pane.
#[cfg(any(test, feature = "physics", feature = "biology"))]
pub fn publish_pane_landmarks(ctx: &egui::Context, payload: LandmarkCloud3DPayload) {
    let Some(pane_idx) = active_pane_index(ctx) else {
        return;
    };
    let arc = Arc::new(payload);
    ctx.data_mut(|d| {
        let mut entry = d
            .get_temp::<StagedPaneOutput>(output_slot_id(pane_idx))
            .unwrap_or_default();
        entry.landmarks = Some(arc);
        d.insert_temp(output_slot_id(pane_idx), entry);
    });
}

/// Publishes a sanitized 2D scalar field from the currently rendering pane.
#[cfg(any(test, feature = "biology"))]
pub fn publish_pane_scalar_field(ctx: &egui::Context, payload: ScalarField2DPayload) {
    let Some(pane_idx) = active_pane_index(ctx) else {
        return;
    };
    let arc = Arc::new(payload);
    ctx.data_mut(|d| {
        let mut entry = d
            .get_temp::<StagedPaneOutput>(output_slot_id(pane_idx))
            .unwrap_or_default();
        entry.field = Some(arc);
        d.insert_temp(output_slot_id(pane_idx), entry);
    });
}

/// Polls any staged upstream graph topology routed to the currently rendering pane.
#[cfg(any(test, feature = "epidemiology", feature = "biology"))]
pub fn poll_pane_graph(ctx: &egui::Context) -> Option<Arc<GraphTopologyPayload>> {
    let pane_idx = active_pane_index(ctx)?;
    ctx.data(|d| {
        d.get_temp::<StagedPaneInput>(input_slot_id(pane_idx))
            .and_then(|s| s.graph)
    })
}

/// Polls any staged upstream 3D landmark cloud routed to the currently rendering pane.
#[cfg(any(test, feature = "pure_math"))]
pub fn poll_pane_landmarks(ctx: &egui::Context) -> Option<Arc<LandmarkCloud3DPayload>> {
    let pane_idx = active_pane_index(ctx)?;
    ctx.data(|d| {
        d.get_temp::<StagedPaneInput>(input_slot_id(pane_idx))
            .and_then(|s| s.landmarks)
    })
}

/// Polls any staged upstream 2D scalar field routed to the currently rendering pane.
#[cfg(any(test, feature = "pure_math"))]
pub fn poll_pane_scalar_field(ctx: &egui::Context) -> Option<Arc<ScalarField2DPayload>> {
    let pane_idx = active_pane_index(ctx)?;
    ctx.data(|d| {
        d.get_temp::<StagedPaneInput>(input_slot_id(pane_idx))
            .and_then(|s| s.field)
    })
}
