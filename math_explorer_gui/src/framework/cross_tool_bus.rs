#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Instance-scoped typed cross-tool message bus for multi-pane and cross-tab handoffs.
//!
//! Replaces process-global `OnceLock` singletons with per-[`super::SimulationFramework`] queues
//! routed explicitly by `MathExplorerApp` across active viewport panes.

use std::collections::VecDeque;

use eframe::egui;

use super::SimulationFramework;

/// Maximum queued one-shot envelopes retained per [`CrossToolBus`] direction.
pub const MAX_CROSS_TOOL_QUEUE: usize = 16;

/// Typed payload variants transferred between cooperative tools across panes or tabs.
#[derive(Clone, Debug, PartialEq)]
pub enum CrossToolPayload {
    /// Bounded 3D point cloud / landmark coordinates paired with named scalar model parameters.
    PointCloud3D {
        /// World-space 3D coordinates `[x, y, z]`.
        points: Vec<[f64; 3]>,
        /// Associated named scalar parameters (e.g. `sigma`, `rho`, `beta`, `dt`).
        parameters: Vec<(String, f64)>,
    },
    /// Named 2D `[x, y]` trajectory or time-series curve.
    TimeSeries2D {
        /// Human-readable series identifier.
        series_name: String,
        /// Ordered `[x, y]` sample coordinates.
        points: Vec<[f64; 2]>,
    },
    /// Named scalar parameter dictionary for cross-tool synchronization.
    ParameterSet {
        /// Key-value parameter pairs.
        parameters: Vec<(String, f64)>,
    },
}

/// Routed cross-tool message envelope carrying addressing metadata and a [`CrossToolPayload`].
#[derive(Clone, Debug, PartialEq)]
pub struct CrossToolEnvelope {
    /// Canonical ID or display name of the originating tool.
    pub source_tool: &'static str,
    /// Optional target tab display name (e.g. `"Geometry & Topology"`).
    pub target_tab: Option<&'static str>,
    /// Optional target tool ID or display name (e.g. `"VietorisRipsTool"`).
    pub target_tool: Option<&'static str>,
    /// Whether routing should switch/activate the target tab and tool if not already open.
    pub focus_target: bool,
    /// Monotonic sequence number assigned by the sender.
    pub seq: u64,
    /// Typed scientific payload.
    pub payload: CrossToolPayload,
}

impl CrossToolEnvelope {
    /// Returns `true` when this envelope targets `(tool_id, tool_display_name)`.
    #[must_use]
    pub fn matches_tool(&self, tool_id: Option<&str>, tool_name: &str) -> bool {
        match self.target_tool {
            None => true,
            Some(target) => {
                target.eq_ignore_ascii_case(tool_name)
                    || tool_id.is_some_and(|id| target.eq_ignore_ascii_case(id))
            }
        }
    }
}

/// Instance-scoped cross-tool message bus owned by each [`SimulationFramework`].
#[derive(Clone, Debug, Default)]
pub struct CrossToolBus {
    outgoing_actions: VecDeque<CrossToolEnvelope>,
    incoming_actions: VecDeque<CrossToolEnvelope>,
    outgoing_live: Option<CrossToolEnvelope>,
    incoming_live: Option<CrossToolEnvelope>,
    last_polled_source_seq: u64,
    last_delivered_live_seq: u64,
    next_seq: u64,
}

impl CrossToolBus {
    /// Creates an empty instance-scoped bus.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn assign_seq(&mut self, seq: u64) -> u64 {
        if seq != 0 {
            self.next_seq = self.next_seq.max(seq);
            seq
        } else {
            self.next_seq = self.next_seq.wrapping_add(1).max(1);
            self.next_seq
        }
    }

    /// Queues an outgoing one-shot action envelope from the active tool.
    pub fn publish_action(&mut self, mut envelope: CrossToolEnvelope) {
        envelope.seq = self.assign_seq(envelope.seq);
        if self.outgoing_actions.len() >= MAX_CROSS_TOOL_QUEUE {
            self.outgoing_actions.pop_front();
        }
        self.outgoing_actions.push_back(envelope);
    }

    /// Publishes the latest live-stream envelope from the active tool.
    pub fn publish_live(&mut self, mut envelope: CrossToolEnvelope) {
        envelope.seq = self.assign_seq(envelope.seq);
        self.outgoing_live = Some(envelope);
    }

    /// Drains all pending outgoing one-shot envelopes queued on this framework.
    pub fn drain_outgoing_actions(&mut self) -> Vec<CrossToolEnvelope> {
        self.outgoing_actions.drain(..).collect()
    }

    /// Takes the latest outgoing live-stream envelope if present.
    pub fn take_outgoing_live(&mut self) -> Option<CrossToolEnvelope> {
        self.outgoing_live.take()
    }

    /// Enqueues an incoming one-shot action envelope routed from another pane or tab.
    pub fn enqueue_incoming_action(&mut self, mut envelope: CrossToolEnvelope) {
        envelope.seq = self.assign_seq(envelope.seq);
        if self.incoming_actions.len() >= MAX_CROSS_TOOL_QUEUE {
            self.incoming_actions.pop_front();
        }
        self.incoming_actions.push_back(envelope);
    }

    /// Stores the latest incoming live-stream envelope routed from another pane or tab.
    pub fn set_incoming_live(&mut self, mut envelope: CrossToolEnvelope) {
        envelope.seq = self.assign_seq(envelope.seq);
        let replace = self
            .incoming_live
            .as_ref()
            .is_none_or(|prev| envelope.seq >= prev.seq);
        if replace {
            self.incoming_live = Some(envelope);
        }
    }

    /// Returns the number of undelivered incoming action envelopes.
    #[must_use]
    pub fn pending_incoming_count(&self) -> usize {
        self.incoming_actions.len()
    }
}

impl SimulationFramework {
    /// Harvests any outgoing cross-tool envelopes queued inside the active tool into `self.cross_tool_bus`.
    pub fn collect_tool_bus_outbox(&mut self, ctx: Option<&egui::Context>) {
        let Some(tool) = &mut self.active_tool else {
            return;
        };
        let actions = tool.drain_cross_tool_envelopes();
        let had_actions = !actions.is_empty();
        for env in actions {
            self.cross_tool_bus.publish_action(env);
        }
        if let Some(live) =
            tool.poll_live_cross_tool_envelope(self.cross_tool_bus.last_polled_source_seq)
        {
            self.cross_tool_bus.last_polled_source_seq =
                self.cross_tool_bus.last_polled_source_seq.max(live.seq);
            self.cross_tool_bus.publish_live(live);
        }
        if had_actions {
            if let Some(c) = ctx {
                c.request_repaint();
            }
        }
    }

    /// Delivers matching incoming envelopes from `self.cross_tool_bus` into the active tool.
    pub fn deliver_tool_bus_inbox(&mut self, ctx: Option<&egui::Context>) -> bool {
        let active_id = self
            .selected_tool_index
            .and_then(|idx| self.available_tools.get(idx).map(|m| m.id));
        let Some(tool) = &mut self.active_tool else {
            return false;
        };
        let tool_name = tool.name();
        let mut delivered = false;
        let mut retained = VecDeque::with_capacity(self.cross_tool_bus.incoming_actions.len());
        while let Some(env) = self.cross_tool_bus.incoming_actions.pop_front() {
            if env.matches_tool(active_id, tool_name) {
                if tool.receive_cross_tool_envelope(env).unwrap_or(false) {
                    delivered = true;
                }
            } else {
                retained.push_back(env);
            }
        }
        self.cross_tool_bus.incoming_actions = retained;

        if let Some(live) = self.cross_tool_bus.incoming_live.clone() {
            if live.seq > self.cross_tool_bus.last_delivered_live_seq
                && live.matches_tool(active_id, tool_name)
            {
                self.cross_tool_bus.last_delivered_live_seq = live.seq;
                if tool.receive_live_cross_tool_envelope(live).unwrap_or(false) {
                    delivered = true;
                }
            }
        }

        if delivered {
            self.bump_framework_revision();
            if let Some(c) = ctx {
                c.request_repaint();
            }
        }
        delivered
    }
}
