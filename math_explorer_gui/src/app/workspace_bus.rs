//! Bounded, revision-counted cross-pane workspace bus (`SharedWorkspaceBus`).
//!
//! Provides opt-in per-pane coupling via [`GraphTopologyChannel`] (256-node / 4,096-edge cap)
//! and [`ScalarSignalChannel`] (1,024-sample ring buffer per series) with zero mutexes and
//! strict isolation for unlinked panes.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};

/// Maximum number of vertices carried by [`GraphTopologyChannel`].
pub const MAX_TOPOLOGY_NODES: usize = 256;

/// Maximum number of finite undirected edges carried by [`GraphTopologyChannel`].
pub const MAX_TOPOLOGY_EDGES: usize = 4_096;

/// Maximum number of `[t, value]` samples retained per series in [`ScalarSignalChannel`].
pub const MAX_SCALAR_SAMPLES: usize = 1_024;

/// Maximum number of distinct scalar series retained in [`ScalarSignalChannel`].
pub const MAX_SCALAR_SERIES: usize = 16;

/// Canonical [`ScalarSignalChannel`] series key for live Kuramoto order-parameter coherence `R(t)`.
#[cfg(any(feature = "physics", test))]
pub const KURAMOTO_COHERENCE_SERIES: &str = "kuramoto_coherence_r";

/// Revision-counted channel streaming normalized 2D graph coordinates and weighted edges.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GraphTopologyChannel {
    revision: u64,
    positions: Vec<[f32; 2]>,
    edges: Vec<(usize, usize, f64)>,
}

impl GraphTopologyChannel {
    /// Monotonic revision counter (`0` before any topology has been published).
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns `true` when the channel currently holds zero vertices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Normalized 2D node coordinates in `[-1.0, 1.0]` (`len() <= 256`).
    #[must_use]
    pub fn positions(&self) -> &[[f32; 2]] {
        &self.positions
    }

    /// Undirected finite weighted edges `(u, v, weight)` with `u < v < positions().len()` (`len() <= 4,096`).
    #[must_use]
    pub fn edges(&self) -> &[(usize, usize, f64)] {
        &self.edges
    }

    /// Sanitizes, caps (`<= 256` nodes, `<= 4,096` finite edges), and publishes a topology snapshot.
    ///
    /// Advances `revision` monotonically if and only if the sanitized topology differs from the
    /// currently stored snapshot (or on the first non-empty publish). Returns the active revision.
    pub fn publish(
        &mut self,
        raw_positions: &[[f32; 2]],
        raw_edges: &[(usize, usize, f64)],
    ) -> u64 {
        let mut clean_pos = Vec::with_capacity(raw_positions.len().min(MAX_TOPOLOGY_NODES));
        for &[x, y] in raw_positions.iter().take(MAX_TOPOLOGY_NODES) {
            let nx = if x.is_finite() {
                x.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let ny = if y.is_finite() {
                y.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            clean_pos.push([nx, ny]);
        }

        let n = clean_pos.len();
        let mut clean_edges = Vec::with_capacity(raw_edges.len().min(MAX_TOPOLOGY_EDGES));
        for &(u, v, w) in raw_edges {
            if u < n && v < n && u != v && w.is_finite() {
                let (lo, hi) = if u < v { (u, v) } else { (v, u) };
                clean_edges.push((lo, hi, w));
            }
        }
        clean_edges.sort_unstable_by_key(|a| (a.0, a.1));
        clean_edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        clean_edges.truncate(MAX_TOPOLOGY_EDGES);

        let unchanged = self.revision > 0
            && self.positions == clean_pos
            && edges_bitwise_eq(&self.edges, &clean_edges);
        if unchanged || (self.revision == 0 && clean_pos.is_empty()) {
            return self.revision;
        }

        self.positions = clean_pos;
        self.edges = clean_edges;
        self.revision = self.revision.saturating_add(1);
        self.revision
    }

    /// Resets the topology channel to its initial empty `revision == 0` state.
    pub fn reset(&mut self) {
        self.revision = 0;
        self.positions.clear();
        self.edges.clear();
    }
}

fn edges_bitwise_eq(lhs: &[(usize, usize, f64)], rhs: &[(usize, usize, f64)]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(a, b)| a.0 == b.0 && a.1 == b.1 && a.2.to_bits() == b.2.to_bits())
}

/// Ring-buffered time series entry inside [`ScalarSignalChannel`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScalarSeries {
    revision: u64,
    samples: VecDeque<[f64; 2]>,
}

impl ScalarSeries {
    /// Monotonic series revision counter.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Ring-buffered `[t, value]` samples (`len() <= 1,024`).
    #[must_use]
    pub fn samples(&self) -> &VecDeque<[f64; 2]> {
        &self.samples
    }

    /// Most recently pushed `[t, value]` sample, if any.
    #[must_use]
    pub fn latest(&self) -> Option<[f64; 2]> {
        self.samples.back().copied()
    }
}

/// Bounded ring-buffered scalar signal bus (`<= 1,024` finite samples per series).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScalarSignalChannel {
    revision: u64,
    series: BTreeMap<String, ScalarSeries>,
}

impl ScalarSignalChannel {
    /// Channel-wide monotonic revision counter.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Appends a finite `[t, value]` sample to `series_key`, evicting the oldest sample above `1,024`.
    pub fn push_sample(&mut self, series_key: &str, t: f64, value: f64) -> bool {
        if series_key.is_empty() || !t.is_finite() || !value.is_finite() {
            return false;
        }
        if !self.series.contains_key(series_key) && self.series.len() >= MAX_SCALAR_SERIES {
            return false;
        }
        let entry = self.series.entry(series_key.to_string()).or_default();
        if entry.samples.len() >= MAX_SCALAR_SAMPLES {
            entry.samples.pop_front();
        }
        entry.samples.push_back([t, value]);
        entry.revision = entry.revision.saturating_add(1);
        self.revision = self.revision.saturating_add(1);
        true
    }

    /// Clears all buffered samples for `series_key` while advancing revision counters if non-empty.
    pub fn clear_series(&mut self, series_key: &str) {
        if let Some(entry) = self.series.get_mut(series_key) {
            if !entry.samples.is_empty() {
                entry.samples.clear();
                entry.revision = entry.revision.saturating_add(1);
                self.revision = self.revision.saturating_add(1);
            }
        }
    }

    /// Looks up a ring-buffered [`ScalarSeries`] by key.
    #[must_use]
    pub fn series(&self, series_key: &str) -> Option<&ScalarSeries> {
        self.series.get(series_key)
    }

    /// Resets all scalar signal series and revision counters.
    pub fn reset(&mut self) {
        self.revision = 0;
        self.series.clear();
    }
}

/// Typed, revision-counted cross-pane workspace bus owned by [`crate::app::MathExplorerApp`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SharedWorkspaceBus {
    /// Live graph topology stream (`<= 256` nodes, `<= 4,096` finite edges).
    pub topology: GraphTopologyChannel,
    /// Live ring-buffered scalar signal stream (`<= 1,024` samples per series).
    pub signals: ScalarSignalChannel,
}

impl SharedWorkspaceBus {
    /// Combined revision tuple `(topology_rev, signals_rev)` for frame-change detection.
    #[must_use]
    pub fn revisions(&self) -> (u64, u64) {
        (self.topology.revision(), self.signals.revision())
    }

    /// Resets both channels to empty `revision == 0` state.
    pub fn reset(&mut self) {
        self.topology.reset();
        self.signals.reset();
    }
}

thread_local! {
    static ACTIVE_LINKED_BUS: RefCell<Option<(usize, SharedWorkspaceBus)>> = const { RefCell::new(None) };
}

struct BusScopeRestoreGuard<'a> {
    target: Option<&'a mut SharedWorkspaceBus>,
    previous: Option<(usize, SharedWorkspaceBus)>,
}

impl Drop for BusScopeRestoreGuard<'_> {
    fn drop(&mut self) {
        let finished = ACTIVE_LINKED_BUS.with(|slot| slot.replace(self.previous.take()));
        if let (Some(dst), Some((_, updated_bus))) = (self.target.take(), finished) {
            *dst = updated_bus;
        }
    }
}

/// Scopes `SharedWorkspaceBus` access to `pane_idx` for the duration of `f`.
///
/// Pass `Some(&mut app.workspace_bus)` when `pane_idx` is explicitly linked, or `None` when
/// unlinked so `with_linked_bus` strictly returns `None`.
pub(crate) fn with_pane_bus_scope<R>(
    bus: Option<&mut SharedWorkspaceBus>,
    pane_idx: usize,
    f: impl FnOnce() -> R,
) -> R {
    let (staged, target) = match bus {
        Some(bus_ref) => {
            let taken = std::mem::take(bus_ref);
            (Some((pane_idx, taken)), Some(bus_ref))
        }
        None => (None, None),
    };
    let previous = ACTIVE_LINKED_BUS.with(|slot| slot.replace(staged));
    let _guard = BusScopeRestoreGuard { target, previous };
    f()
}

/// Invokes `f` with mutable access to the active pane's linked [`SharedWorkspaceBus`],
/// or returns `None` when the current pane is unlinked (or running standalone).
#[cfg(any(feature = "pure_math", test))]
pub(crate) fn with_linked_bus<R>(f: impl FnOnce(&mut SharedWorkspaceBus) -> R) -> Option<R> {
    ACTIVE_LINKED_BUS.with(|slot| {
        let mut borrow = slot.borrow_mut();
        borrow.as_mut().map(|(_, bus)| f(bus))
    })
}

/// Returns `true` when the currently rendering pane is linked to [`SharedWorkspaceBus`].
#[cfg(any(feature = "pure_math", test))]
#[must_use]
pub(crate) fn is_current_pane_linked() -> bool {
    ACTIVE_LINKED_BUS.with(|slot| slot.borrow().is_some())
}

impl super::MathExplorerApp {
    /// Returns `true` if `pane_idx` (`0..4`) has opted into [`SharedWorkspaceBus`] coupling.
    #[must_use]
    pub fn is_pane_bus_linked(&self, pane_idx: usize) -> bool {
        self.pane_bus_links.get(pane_idx).copied().unwrap_or(false)
    }

    /// Explicitly links or unlinks `pane_idx` (`0..4`) on [`SharedWorkspaceBus`].
    pub fn set_pane_bus_linked(&mut self, pane_idx: usize, linked: bool) {
        if let Some(slot) = self.pane_bus_links.get_mut(pane_idx) {
            *slot = linked;
        }
        if !self.pane_bus_links.iter().any(|&b| b) {
            self.workspace_bus.reset();
        }
    }

    /// Toggles the [`SharedWorkspaceBus`] link binding on `pane_idx` (`0..4`), returning the new state.
    pub fn toggle_pane_bus_link(&mut self, pane_idx: usize) -> bool {
        let next = !self.is_pane_bus_linked(pane_idx);
        self.set_pane_bus_linked(pane_idx, next);
        next
    }

    /// Returns `true` when at least one active multi-pane slot is linked to [`SharedWorkspaceBus`].
    #[must_use]
    pub fn has_active_multi_pane_bus_links(&self) -> bool {
        let count = self.viewport_layout.pane_count();
        count > 1 && self.pane_bus_links.iter().take(count).any(|&b| b)
    }

    /// Runs a two-pass synchronization across active linked panes so publishers populate
    /// [`SharedWorkspaceBus`] in Pass 1 and subscribers ingest updated revisions in Pass 2.
    pub fn sync_linked_panes_bus(&mut self) {
        let pane_count = self.viewport_layout.pane_count();
        if pane_count <= 1 || !self.pane_bus_links.iter().take(pane_count).any(|&b| b) {
            return;
        }
        for _pass in 0..2 {
            for pane_idx in 0..pane_count {
                if !self.is_pane_bus_linked(pane_idx) {
                    continue;
                }
                let mut bus_snapshot = std::mem::take(&mut self.workspace_bus);
                if let Some(tab) = self.pane_tab_mut(pane_idx) {
                    tab.sync_workspace_bus(&mut bus_snapshot);
                }
                self.workspace_bus = bus_snapshot;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_topology_channel_enforces_node_and_edge_caps() {
        let mut ch = GraphTopologyChannel::default();
        assert_eq!(ch.revision(), 0);

        let raw_nodes: Vec<[f32; 2]> = (0..300)
            .map(|i| [(i as f32) / 150.0 - 1.0, f32::NAN])
            .collect();
        let mut raw_edges = Vec::new();
        for u in 0..MAX_TOPOLOGY_NODES {
            for v in (u + 1)..MAX_TOPOLOGY_NODES {
                raw_edges.push((u, v, 1.0));
                if raw_edges.len() >= 5_000 {
                    break;
                }
            }
            if raw_edges.len() >= 5_000 {
                break;
            }
        }
        raw_edges.push((0, 999, 1.0));
        raw_edges.push((1, 2, f64::INFINITY));

        let rev1 = ch.publish(&raw_nodes, &raw_edges);
        assert_eq!(rev1, 1);
        assert_eq!(ch.positions().len(), MAX_TOPOLOGY_NODES);
        assert!(ch
            .positions()
            .iter()
            .all(|p| p[0].is_finite() && p[1] == 0.0));
        assert_eq!(ch.edges().len(), MAX_TOPOLOGY_EDGES);

        // Identical re-publish does not advance revision.
        let prev_pos = ch.positions().to_vec();
        let prev_edges = ch.edges().to_vec();
        let rev2 = ch.publish(&prev_pos, &prev_edges);
        assert_eq!(rev2, 1);
    }

    #[test]
    fn test_scalar_signal_channel_ring_buffer_cap_and_finiteness() {
        let mut sig = ScalarSignalChannel::default();
        assert!(!sig.push_sample(KURAMOTO_COHERENCE_SERIES, f64::NAN, 0.5));
        assert!(!sig.push_sample(KURAMOTO_COHERENCE_SERIES, 0.0, f64::INFINITY));

        for step in 0..1_200 {
            assert!(sig.push_sample(KURAMOTO_COHERENCE_SERIES, step as f64 * 0.01, 0.75));
        }
        let series = sig.series(KURAMOTO_COHERENCE_SERIES).unwrap();
        assert_eq!(series.samples().len(), MAX_SCALAR_SAMPLES);
        assert_eq!(
            series.samples().front().copied(),
            Some([176.0 * 0.01, 0.75])
        );
        assert_eq!(series.latest(), Some([1_199.0 * 0.01, 0.75]));
    }
}
