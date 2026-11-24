//! Workspace-scoped `SharedSimulationBus` streaming immutable `Arc` copy-on-write CSR graphs
//! and 3D chaotic landmark point clouds across split panes through `egui::Context`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "sim_bus/topology.rs"]
mod topology;

#[cfg(any(feature = "biology", test))]
pub use topology::MAX_NEURAL_SHARED_NODES;
#[cfg(any(feature = "physics", test))]
pub use topology::MAX_SHARED_LANDMARKS;
pub use topology::{
    SharedGraphTopology, SharedPointCloud3D, TopologySource, MAX_SHARED_GRAPH_EDGES,
    MAX_SHARED_GRAPH_NODES,
};

use crate::accessibility::announce_live_status;
use crate::app::theme::{contrasting_text_color, SemanticPalette};
use eframe::egui;
use std::sync::{Arc, RwLock};

const SIM_BUS_CONTEXT_KEY: &str = "SHARED_SIMULATION_BUS_HANDLE";
const SIM_BUS_LINK_TOPOLOGY_KEY: &str = "SHARED_SIMULATION_LINK_TOPOLOGY";

#[derive(Debug)]
struct SimBusState {
    next_revision: u64,
    graph_fingerprint: Option<u64>,
    graph: Option<Arc<SharedGraphTopology>>,
    point_cloud_3d: Option<Arc<SharedPointCloud3D>>,
    pending_vr_handoff: Option<Arc<SharedPointCloud3D>>,
    pending_open_vr: bool,
}

impl Default for SimBusState {
    fn default() -> Self {
        Self {
            next_revision: 1,
            graph_fingerprint: None,
            graph: None,
            point_cloud_3d: None,
            pending_vr_handoff: None,
            pending_open_vr: false,
        }
    }
}

/// Workspace-scoped simulation topology and point-cloud bus owned by `MathExplorerApp`.
#[derive(Clone, Debug, Default)]
pub struct SharedSimulationBus {
    inner: Arc<RwLock<SimBusState>>,
}

impl SharedSimulationBus {
    /// Creates an isolated workspace-scoped [`SharedSimulationBus`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when both handles point to the same workspace bus allocation.
    #[must_use]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Installs this workspace bus handle and the active `link_topology` toggle into `egui::Context`.
    pub fn install_in_context(&self, ctx: &egui::Context, link_topology: bool) {
        let prior = ctx.data_mut(|d| {
            let prev = d.get_temp::<Self>(egui::Id::new(SIM_BUS_CONTEXT_KEY));
            d.insert_temp(egui::Id::new(SIM_BUS_CONTEXT_KEY), self.clone());
            d.insert_temp(egui::Id::new(SIM_BUS_LINK_TOPOLOGY_KEY), link_topology);
            prev
        });
        if let Some(prev_bus) = prior {
            if !self.ptr_eq(&prev_bus) {
                self.absorb_ephemeral_bus(&prev_bus);
            }
        }
    }

    fn absorb_ephemeral_bus(&self, other: &Self) {
        let Ok(mut other_guard) = other.inner.write() else {
            return;
        };
        let Ok(mut self_guard) = self.inner.write() else {
            return;
        };
        if self_guard.graph.is_none() && other_guard.graph.is_some() {
            self_guard.graph = other_guard.graph.take();
            self_guard.graph_fingerprint = other_guard.graph_fingerprint.take();
        }
        if other_guard.point_cloud_3d.is_some() {
            self_guard.point_cloud_3d = other_guard.point_cloud_3d.take();
        }
        if other_guard.pending_vr_handoff.is_some() {
            self_guard.pending_vr_handoff = other_guard.pending_vr_handoff.take();
        }
        if other_guard.pending_open_vr {
            self_guard.pending_open_vr = true;
            other_guard.pending_open_vr = false;
        }
        self_guard.next_revision = self_guard.next_revision.max(other_guard.next_revision);
    }

    /// Retrieves the workspace [`SharedSimulationBus`] bound to `ctx`, if installed.
    #[must_use]
    pub fn from_context(ctx: &egui::Context) -> Option<Self> {
        ctx.data(|d| d.get_temp::<Self>(egui::Id::new(SIM_BUS_CONTEXT_KEY)))
    }

    /// Retrieves the workspace [`SharedSimulationBus`] from `ctx`, inserting a context-local bus if absent.
    #[must_use]
    pub fn from_context_or_insert(ctx: &egui::Context) -> Self {
        ctx.data_mut(|d| {
            if let Some(existing) = d.get_temp::<Self>(egui::Id::new(SIM_BUS_CONTEXT_KEY)) {
                existing
            } else {
                let created = Self::new();
                d.insert_temp(egui::Id::new(SIM_BUS_CONTEXT_KEY), created.clone());
                created
            }
        })
    }

    /// Returns `true` when workspace `Link Topology` synchronization is enabled on `ctx`.
    #[must_use]
    pub fn is_link_topology_enabled(ctx: &egui::Context) -> bool {
        ctx.data(|d| {
            d.get_temp::<bool>(egui::Id::new(SIM_BUS_LINK_TOPOLOGY_KEY))
                .unwrap_or(false)
        })
    }

    /// Resolves the active [`SharedGraphTopology`] for a consumer pane given its [`TopologySource`].
    #[must_use]
    pub fn active_graph_for(
        ctx: &egui::Context,
        source: TopologySource,
    ) -> Option<Arc<SharedGraphTopology>> {
        if source != TopologySource::SharedBus || !Self::is_link_topology_enabled(ctx) {
            return None;
        }
        Self::from_context(ctx)?.latest_graph()
    }

    /// Publishes a copy-on-write CSR graph snapshot when `fingerprint` differs from the current revision.
    pub fn publish_graph_if_changed(
        &self,
        fingerprint: u64,
        directed: bool,
        raw_nodes: &[(usize, [f32; 2])],
        raw_edges: &[(usize, usize, f64)],
    ) -> Option<(Arc<SharedGraphTopology>, bool)> {
        let mut guard = self.inner.write().ok()?;
        if guard.graph_fingerprint == Some(fingerprint) {
            if let Some(ref existing) = guard.graph {
                return Some((Arc::clone(existing), false));
            }
        }
        let rev = guard.next_revision;
        guard.next_revision = guard.next_revision.saturating_add(1);
        let snap = Arc::new(SharedGraphTopology::from_raw(
            rev, directed, raw_nodes, raw_edges,
        ));
        guard.graph_fingerprint = Some(fingerprint);
        guard.graph = Some(Arc::clone(&snap));
        Some((snap, true))
    }

    /// Returns the latest published [`SharedGraphTopology`] snapshot, if any.
    #[must_use]
    pub fn latest_graph(&self) -> Option<Arc<SharedGraphTopology>> {
        self.inner.read().ok()?.graph.clone()
    }

    /// Publishes a live 3D landmark point-cloud frame into this workspace bus.
    #[cfg(any(feature = "physics", test))]
    pub fn publish_point_cloud_3d(
        &self,
        mut points: Vec<[f64; 3]>,
        sigma: f64,
        rho: f64,
        beta: f64,
        dt: f64,
    ) -> Option<Arc<SharedPointCloud3D>> {
        points.truncate(MAX_SHARED_LANDMARKS);
        let mut guard = self.inner.write().ok()?;
        let rev = guard.next_revision;
        guard.next_revision = guard.next_revision.saturating_add(1);
        let cloud = Arc::new(SharedPointCloud3D {
            revision: rev,
            points,
            sigma,
            rho,
            beta,
            dt,
        });
        guard.point_cloud_3d = Some(Arc::clone(&cloud));
        Some(cloud)
    }

    /// Returns the latest live 3D landmark point-cloud frame if newer than `after_revision`.
    #[must_use]
    pub fn latest_point_cloud_3d_since(
        &self,
        after_revision: u64,
    ) -> Option<Arc<SharedPointCloud3D>> {
        let guard = self.inner.read().ok()?;
        let cloud = guard.point_cloud_3d.as_ref()?;
        (cloud.revision > after_revision).then(|| Arc::clone(cloud))
    }

    /// Submits a one-click 3D landmark handoff payload and optionally requests opening `VietorisRipsTool`.
    #[cfg(any(feature = "physics", test))]
    pub fn submit_vr_handoff(
        &self,
        mut points: Vec<[f64; 3]>,
        sigma: f64,
        rho: f64,
        beta: f64,
        dt: f64,
        request_open: bool,
    ) -> Option<Arc<SharedPointCloud3D>> {
        points.truncate(MAX_SHARED_LANDMARKS);
        let mut guard = self.inner.write().ok()?;
        let rev = guard.next_revision;
        guard.next_revision = guard.next_revision.saturating_add(1);
        let cloud = Arc::new(SharedPointCloud3D {
            revision: rev,
            points,
            sigma,
            rho,
            beta,
            dt,
        });
        guard.pending_vr_handoff = Some(Arc::clone(&cloud));
        guard.point_cloud_3d = Some(Arc::clone(&cloud));
        if request_open {
            guard.pending_open_vr = true;
        }
        Some(cloud)
    }

    /// Consumes a pending one-click `VietorisRipsTool` landmark handoff payload.
    #[must_use]
    pub fn take_vr_handoff(&self) -> Option<Arc<SharedPointCloud3D>> {
        self.inner.write().ok()?.pending_vr_handoff.take()
    }

    /// Marks a pending cross-tab navigation request to open `VietorisRipsTool`.
    #[cfg(test)]
    pub fn request_open_vietoris_rips(&self) {
        if let Ok(mut guard) = self.inner.write() {
            guard.pending_open_vr = true;
        }
    }

    /// Consumes any pending request to open `VietorisRipsTool`.
    #[must_use]
    pub fn take_open_vietoris_rips_request(&self) -> bool {
        let Ok(mut guard) = self.inner.write() else {
            return false;
        };
        let requested = guard.pending_open_vr;
        guard.pending_open_vr = false;
        requested
    }
}

/// Renders an accessible `Topology Source` selector with WCAG 2.2 AA status and clamping badges.
pub fn render_topology_source_bar(
    ui: &mut egui::Ui,
    id_salt: &'static str,
    source: &mut TopologySource,
    synced_rev: Option<u64>,
    clamp_badge: Option<&str>,
) -> bool {
    let ctx = ui.ctx().clone();
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let linked = SharedSimulationBus::is_link_topology_enabled(&ctx);
    let mut changed = false;

    ui.horizontal_wrapped(|ui| {
        ui.label("Topology Source:");
        let prev = *source;
        egui::ComboBox::from_id_salt((id_salt, "topology_source_combo"))
            .selected_text(source.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(source, TopologySource::SharedBus, "Shared Bus");
                ui.selectable_value(source, TopologySource::LocalPreset, "Local Preset");
            });
        if *source != prev {
            changed = true;
            let msg = match *source {
                TopologySource::SharedBus => "Topology source set to Shared Bus",
                TopologySource::LocalPreset => {
                    "Topology source set to Local Preset (opted out of shared bus)"
                }
            };
            announce_live_status(&ctx, msg);
        }

        let (badge_bg, badge_txt) = match (*source, linked, synced_rev) {
            (TopologySource::LocalPreset, _, _) => {
                (palette.surface_fill, "Local Preset".to_string())
            }
            (TopologySource::SharedBus, true, Some(rev)) => {
                (palette.info, format!("Bus rev #{rev}"))
            }
            (TopologySource::SharedBus, true, None) => {
                (palette.surface_fill, "Awaiting Graph Editor".to_string())
            }
            (TopologySource::SharedBus, false, _) => {
                (palette.surface_fill, "Link Topology Off".to_string())
            }
        };
        paint_accessible_pill(ui, badge_bg, &badge_txt, false);
        if let Some(clamp_msg) = clamp_badge {
            paint_accessible_pill(ui, palette.warning, clamp_msg, true);
        }
    });

    changed
}

fn paint_accessible_pill(ui: &mut egui::Ui, bg: egui::Color32, text: &str, strong: bool) {
    let fg = contrasting_text_color(bg);
    egui::Frame::NONE
        .fill(bg)
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(6, 2))
        .show(ui, |ui| {
            let mut rt = egui::RichText::new(text).color(fg).small();
            if strong {
                rt = rt.strong();
            }
            ui.label(rt);
        });
}
