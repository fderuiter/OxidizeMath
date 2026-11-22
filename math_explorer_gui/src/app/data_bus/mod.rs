//! Typed workspace signal bus projected onto [`eframe::egui::Context`] across viewport panes.
//!
//! Follows the `WorkspaceHistoryBridge` pattern so publishers and subscribed consumers exchange
//! immutable [`Arc<SharedGraph>`] topologies and [`Arc<EpidemicTraceSignal>`] trajectories
//! without altering [`crate::tabs::ExplorerTab`] or [`crate::framework::InteractiveTool`] traits.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod types;
pub mod ui;

#[cfg(test)]
mod tests;

pub use types::{
    EpidemicTraceSignal, SharedGraph, SharedNode, SESSION_MAX_BUS_EDGES, SESSION_MAX_BUS_NODES,
    SESSION_MAX_TRACE_POINTS, URL_MAX_BUS_EDGES, URL_MAX_BUS_NODES, URL_MAX_TRACE_POINTS,
};

#[cfg(any(test, feature = "applied"))]
pub use types::MAX_GAME_NODE_BUDGET;
#[cfg(any(test, feature = "biology"))]
pub use types::MAX_HH_NEURON_BUDGET;
#[cfg(any(test, feature = "physics"))]
pub use types::MAX_SPIN_NODE_BUDGET;
#[cfg(any(test, feature = "epidemiology"))]
pub use types::{adjacency_degree_moments, MAX_EPIDEMIC_NODE_BUDGET};

use crate::framework::ToolStateBounds;
use crate::url_state::URL_JSON_BOUNDS;
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const WORKSPACE_DATA_BUS_ID: &str = "oxidize_workspace_data_bus";
const PANE_SYNC_META_ID: &str = "oxidize_workspace_pane_sync_meta";

/// Per-pane header sync state `[pane_idx -> (assigned_tab_idx, follow_enabled)]` and pending clicks.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PaneGraphSyncMeta {
    pub active_status: [Option<(usize, bool)>; 4],
    pub pending_toggle: [Option<bool>; 4],
}

mod serde_opt_arc {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub(super) fn serialize<S, T>(val: &Option<Arc<T>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        match val {
            Some(arc) => serializer.serialize_some(arc.as_ref()),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D, T>(deserializer: D) -> Result<Option<Arc<T>>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        Option::<T>::deserialize(deserializer).map(|opt| opt.map(Arc::new))
    }
}

/// Typed cross-pane signal bus owned by [`super::MathExplorerApp`] and projected onto `egui::Context`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceDataBus {
    /// Latest shared signed-weight graph snapshot (`None` until published or loaded).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_opt_arc"
    )]
    pub shared_graph: Option<Arc<SharedGraph>>,
    /// Latest empirical network epidemic trace (`None` until `NetworkPropagationTool` publishes).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_opt_arc"
    )]
    pub epidemic_trace: Option<Arc<EpidemicTraceSignal>>,
}

impl WorkspaceDataBus {
    /// Returns `true` when neither a shared graph nor an epidemic trace is present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shared_graph.is_none() && self.epidemic_trace.is_none()
    }

    /// Returns the monotonic revision of the current [`SharedGraph`], or `0` if empty.
    #[must_use]
    pub fn graph_revision(&self) -> u64 {
        self.shared_graph.as_ref().map_or(0, |g| g.revision)
    }

    /// Returns the monotonic revision of the current [`EpidemicTraceSignal`], or `0` if empty.
    #[must_use]
    pub fn trace_revision(&self) -> u64 {
        self.epidemic_trace.as_ref().map_or(0, |t| t.revision)
    }

    /// Validates `.mws` session payload limits on all carried signals.
    pub fn validate_session_bounds(&self) -> Result<(), String> {
        if let Some(ref graph) = self.shared_graph {
            graph.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
        }
        if let Some(ref trace) = self.epidemic_trace {
            trace.validate(SESSION_MAX_TRACE_POINTS)?;
        }
        Ok(())
    }

    /// Validates `#state=` URL payload limits on all carried signals.
    pub fn validate_url_bounds(&self) -> Result<(), String> {
        if let Some(ref graph) = self.shared_graph {
            graph.validate(URL_MAX_BUS_NODES, URL_MAX_BUS_EDGES)?;
        }
        if let Some(ref trace) = self.epidemic_trace {
            trace.validate(URL_MAX_TRACE_POINTS)?;
        }
        Ok(())
    }

    /// Parses a standalone serialized `WorkspaceDataBus` JSON string with `.mws` `ToolStateBounds`.
    #[cfg(test)]
    pub fn from_session_json(raw: &str) -> Result<Self, String> {
        let cleaned = egui_plot::export::strip_utf8_bom(raw);
        ToolStateBounds::SESSION.validate_json(cleaned)?;
        let bus: Self =
            serde_json::from_str(cleaned).map_err(|e| format!("Invalid WorkspaceDataBus: {e}"))?;
        bus.validate_session_bounds()?;
        Ok(bus)
    }

    /// Parses a standalone serialized `WorkspaceDataBus` JSON string with `URL_JSON_BOUNDS`.
    #[cfg(test)]
    pub fn from_url_json(raw: &str) -> Result<Self, String> {
        let cleaned = egui_plot::export::strip_utf8_bom(raw);
        URL_JSON_BOUNDS.validate_json(cleaned)?;
        let bus: Self = serde_json::from_str(cleaned)
            .map_err(|e| format!("Invalid URL WorkspaceDataBus: {e}"))?;
        bus.validate_url_bounds()?;
        Ok(bus)
    }
}

/// Validates any embedded `data_bus` or `seeded_graph` payloads inside a `#state=` JSON document.
pub fn validate_embedded_url_bus_fields(parsed: &serde_json::Value) -> Result<(), String> {
    let _ = (ToolStateBounds::DEFAULT, URL_JSON_BOUNDS);
    let Some(obj) = parsed.as_object() else {
        return Ok(());
    };
    if let Some(bus_val) = obj.get("data_bus") {
        let bus: WorkspaceDataBus = serde_json::from_value(bus_val.clone())
            .map_err(|e| format!("Invalid URL data_bus payload: {e}"))?;
        bus.validate_url_bounds()?;
    }
    if let Some(sg_val) = obj
        .get("config")
        .and_then(serde_json::Value::as_object)
        .and_then(|cfg| cfg.get("seeded_graph"))
        .filter(|v| !v.is_null())
    {
        let graph: SharedGraph = serde_json::from_value(sg_val.clone())
            .map_err(|e| format!("Invalid URL seeded_graph payload: {e}"))?;
        graph.validate(URL_MAX_BUS_NODES, URL_MAX_BUS_EDGES)?;
    }
    if let Some(sg_root) = obj.get("seeded_graph").filter(|v| !v.is_null()) {
        let graph: SharedGraph = serde_json::from_value(sg_root.clone())
            .map_err(|e| format!("Invalid URL seeded_graph payload: {e}"))?;
        graph.validate(URL_MAX_BUS_NODES, URL_MAX_BUS_EDGES)?;
    }
    Ok(())
}

/// Projects `app.data_bus` onto `egui::Context` before single- or multi-pane viewport rendering.
pub(crate) fn sync_app_bus_before_viewport(app: &super::MathExplorerApp, ctx: &egui::Context) {
    let bus_clone = app.data_bus.clone();
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(WORKSPACE_DATA_BUS_ID), bus_clone);
    });
}

/// Commits any newer revisions published during viewport rendering back to `app.data_bus`.
///
/// Requests an immediate repaint if a revision advanced mid-frame in a multi-pane layout so
/// earlier-rendered panes apply the new snapshot on the very next frame without idle polling.
pub(crate) fn commit_app_bus_after_viewport(app: &mut super::MathExplorerApp, ctx: &egui::Context) {
    let prev_graph_rev = app.data_bus.graph_revision();
    let prev_trace_rev = app.data_bus.trace_revision();

    let updated =
        ctx.data_mut(|d| d.get_temp::<WorkspaceDataBus>(egui::Id::new(WORKSPACE_DATA_BUS_ID)));
    if let Some(bus) = updated {
        let advanced =
            bus.graph_revision() > prev_graph_rev || bus.trace_revision() > prev_trace_rev;
        app.data_bus = bus;
        if advanced && app.viewport_layout.pane_count() > 1 {
            ctx.request_repaint();
        }
    }
}

/// Reads the current [`Arc<SharedGraph>`] from `egui::Context` temporary storage in $O(1)$.
#[cfg(any(
    test,
    feature = "pure_math",
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
#[must_use]
pub fn read_shared_graph(ctx: &egui::Context) -> Option<Arc<SharedGraph>> {
    ctx.data(|d| {
        d.get_temp::<WorkspaceDataBus>(egui::Id::new(WORKSPACE_DATA_BUS_ID))
            .and_then(|bus| bus.shared_graph)
    })
}

/// Publishes a new [`SharedGraph`] snapshot onto `egui::Context`, advancing the revision monotonically.
#[cfg(any(
    test,
    feature = "pure_math",
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
pub fn publish_shared_graph(ctx: &egui::Context, mut graph: SharedGraph) -> Arc<SharedGraph> {
    ctx.data_mut(|d| {
        let id = egui::Id::new(WORKSPACE_DATA_BUS_ID);
        let mut bus = d.get_temp::<WorkspaceDataBus>(id).unwrap_or_default();
        let next_rev = bus
            .graph_revision()
            .saturating_add(1)
            .max(graph.revision.max(1));
        graph.revision = next_rev;
        let arc = Arc::new(graph);
        bus.shared_graph = Some(Arc::clone(&arc));
        d.insert_temp(id, bus);
        arc
    })
}

/// Seeds `egui::Context` with `graph` only if no shared graph is currently published on the bus.
#[cfg(any(
    test,
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
pub fn seed_shared_graph_if_empty(ctx: &egui::Context, graph: &SharedGraph) -> Arc<SharedGraph> {
    if let Some(existing) = read_shared_graph(ctx) {
        return existing;
    }
    publish_shared_graph(ctx, graph.clone())
}

/// Reads the current [`Arc<EpidemicTraceSignal>`] from `egui::Context` temporary storage.
#[cfg(any(test, feature = "epidemiology"))]
#[must_use]
pub fn read_epidemic_trace(ctx: &egui::Context) -> Option<Arc<EpidemicTraceSignal>> {
    ctx.data(|d| {
        d.get_temp::<WorkspaceDataBus>(egui::Id::new(WORKSPACE_DATA_BUS_ID))
            .and_then(|bus| bus.epidemic_trace)
    })
}

/// Publishes a new [`EpidemicTraceSignal`] onto `egui::Context`, advancing its revision monotonically.
#[cfg(any(test, feature = "epidemiology"))]
pub fn publish_epidemic_trace(
    ctx: &egui::Context,
    mut trace: EpidemicTraceSignal,
) -> Arc<EpidemicTraceSignal> {
    ctx.data_mut(|d| {
        let id = egui::Id::new(WORKSPACE_DATA_BUS_ID);
        let mut bus = d.get_temp::<WorkspaceDataBus>(id).unwrap_or_default();
        let next_rev = bus
            .trace_revision()
            .saturating_add(1)
            .max(trace.revision.max(1));
        trace.revision = next_rev;
        let arc = Arc::new(trace);
        bus.epidemic_trace = Some(Arc::clone(&arc));
        d.insert_temp(id, bus);
        arc
    })
}

pub(crate) fn read_pane_sync_meta(ctx: &egui::Context) -> PaneGraphSyncMeta {
    ctx.data(|d| {
        d.get_temp::<PaneGraphSyncMeta>(egui::Id::new(PANE_SYNC_META_ID))
            .unwrap_or_default()
    })
}

pub(crate) fn write_pane_sync_meta(ctx: &egui::Context, meta: PaneGraphSyncMeta) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new(PANE_SYNC_META_ID), meta));
}
