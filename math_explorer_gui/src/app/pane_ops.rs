#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MathExplorerApp, SplitLayoutMode, ViewportLayout};
use eframe::egui;

impl MathExplorerApp {
    /// Swaps the primary and secondary pane active tabs and their per-pane states immediately.
    pub fn swap_panes(&mut self) {
        self.swap_pane_pair(0, 1);
    }

    /// Sets the assigned tab index for a specific pane.
    pub fn set_pane_tab(&mut self, pane_idx: usize, tab_idx: usize) {
        if pane_idx < self.pane_tabs.len() && tab_idx < self.tabs.len() {
            self.pane_tabs[pane_idx] = tab_idx;
            if pane_idx == 0 {
                self.selected_tab = tab_idx;
            } else if pane_idx == 1 {
                self.secondary_tab = tab_idx;
            }
            self.on_pane_tab_assigned(pane_idx, tab_idx);
        }
    }

    /// Sets the authoritative viewport layout preset and synchronizes legacy layout mode.
    pub fn set_layout(&mut self, layout: ViewportLayout) {
        self.seed_uninitialized_panes_on_initial_split(layout);
        self.viewport_layout = layout;
        self.sync_layout();
        if !self.pane_tabs.is_empty() {
            self.pane_tabs[0] = self.selected_tab;
        }
        if layout != ViewportLayout::Single && self.pane_tabs.len() > 1 {
            self.pane_tabs[1] = self.secondary_tab;
        }
        self.prune_inactive_pane_slots(layout.pane_count());
    }

    /// Synchronizes legacy `split_layout_mode` with current `viewport_layout`.
    pub fn sync_layout(&mut self) {
        self.split_layout_mode = match self.viewport_layout {
            ViewportLayout::Single => SplitLayoutMode::Single,
            ViewportLayout::SplitHorizontal => SplitLayoutMode::Horizontal,
            ViewportLayout::SplitVertical => SplitLayoutMode::Vertical,
            ViewportLayout::QuadGrid => SplitLayoutMode::Single,
        };
    }

    /// Swaps the assigned simulation modules and per-pane runtime states between `pane_a` and `pane_b`.
    pub fn swap_pane_pair(&mut self, pane_a: usize, pane_b: usize) {
        if pane_a >= 4 || pane_b >= 4 || pane_a == pane_b || self.tabs.is_empty() {
            return;
        }
        self.clear_active_pane_caches();
        self.ensure_four_pane_tabs();
        self.pane_tabs[0] = self.selected_tab;
        if self.viewport_layout != ViewportLayout::QuadGrid {
            self.pane_tabs[1] = self.secondary_tab;
        }

        let (lo, hi) = if pane_a < pane_b {
            (pane_a, pane_b)
        } else {
            (pane_b, pane_a)
        };

        self.pane_tabs.swap(lo, hi);
        self.pane_bus_links.swap(lo, hi);
        self.selected_tab = self.pane_tabs[0];
        self.secondary_tab = self.pane_tabs[1];
        self.pending_focus_prune = true;

        if lo == 0 {
            self.swap_primary_and_secondary_slot_states(hi);
        } else {
            self.tab_state_cache.clear();
            self.pane_slots.swap_secondary_slots(lo, hi);
        }
        self.pipeline_dag.swap_panes(lo, hi);
        self.dataflow_bus.swap_panes(lo, hi);
    }

    /// Queries the active `InteractiveTool` ID of `pane_idx` without mutating lazy secondary slots.
    pub fn pane_active_tool_id(&self, pane_idx: usize) -> Option<String> {
        if let Some(tab) = self.pane_tab(pane_idx) {
            return tab.active_tool_id().map(str::to_string);
        }
        let tab_idx = self.pane_tabs.get(pane_idx).copied()?;
        let template_tab = self.tabs.get(tab_idx)?;
        if let Some(saved) = self
            .pane_slots
            .slot(pane_idx)
            .and_then(|s| s.saved_states.get(template_tab.name()))
        {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(saved) {
                for key in ["active_tool_id", "tool_id", "selected_tool"] {
                    if let Some(id) = val.get(key).and_then(serde_json::Value::as_str) {
                        return Some(id.to_string());
                    }
                }
            }
            if let Some(mut temp) = crate::tabs::instantiate_tab(tab_idx) {
                temp.load_state(saved);
                return temp.active_tool_id().map(str::to_string);
            }
        }
        crate::tabs::instantiate_tab(tab_idx).and_then(|t| t.active_tool_id().map(str::to_string))
    }

    /// Prunes any pipeline links whose active tools are no longer stream-compatible.
    pub fn prune_incompatible_pipeline_links(&mut self) {
        let count = self.viewport_layout.pane_count();
        if count <= 1 {
            self.pipeline_dag.clear();
            self.dataflow_bus.prune_inactive(1);
            return;
        }
        let tool_ids: [Option<String>; 4] = std::array::from_fn(|idx| {
            (idx < count)
                .then(|| self.pane_active_tool_id(idx))
                .flatten()
        });
        self.pipeline_dag
            .prune_incompatible(count, |idx| tool_ids.get(idx).and_then(Option::as_deref));
    }

    /// Connects `from_pane -> to_pane` on `pipeline_dag` and emits WCAG 2.2 AA ARIA announcements.
    pub fn connect_pane_pipeline_interactive(
        &mut self,
        ctx: Option<&egui::Context>,
        from_pane: usize,
        to_pane: usize,
    ) -> Result<super::DataflowStreamKind, super::dataflow_bus::DagLinkError> {
        use super::dataflow_bus::{compatible_stream_kind, DagLinkError};
        let count = self.viewport_layout.pane_count();
        if from_pane >= count || to_pane >= count {
            Self::announce_pipeline_error(ctx, DagLinkError::OutOfBounds);
            return Err(DagLinkError::OutOfBounds);
        }
        if from_pane == to_pane {
            Self::announce_pipeline_error(ctx, DagLinkError::SelfLoop);
            return Err(DagLinkError::SelfLoop);
        }
        if self.pipeline_dag.would_create_cycle(from_pane, to_pane) {
            Self::announce_pipeline_error(ctx, DagLinkError::CycleDetected);
            return Err(DagLinkError::CycleDetected);
        }
        let src_tool = self.pane_active_tool_id(from_pane);
        let dst_tool = self.pane_active_tool_id(to_pane);
        let Some(kind) = compatible_stream_kind(src_tool.as_deref(), dst_tool.as_deref()) else {
            Self::announce_pipeline_error(ctx, DagLinkError::IncompatibleStream);
            return Err(DagLinkError::IncompatibleStream);
        };
        if let Err(err) = self.pipeline_dag.connect(from_pane, to_pane, kind) {
            Self::announce_pipeline_error(ctx, err);
            return Err(err);
        }
        let labels = self.viewport_layout.pane_labels();
        let src_lbl = labels.get(from_pane).copied().unwrap_or("Upstream");
        let dst_lbl = labels.get(to_pane).copied().unwrap_or("Downstream");
        let msg = format!(
            "Connected {} {} stream into {}",
            src_lbl,
            kind.label(),
            dst_lbl
        );
        if let Some(c) = ctx {
            crate::accessibility::announce_live_status(c, &msg);
        } else {
            crate::accessibility::announce_status(&msg);
        }
        Ok(kind)
    }

    /// Disconnects any upstream pipeline feeding `to_pane` and announces the change politely.
    pub fn disconnect_pane_pipeline_interactive(
        &mut self,
        ctx: Option<&egui::Context>,
        to_pane: usize,
    ) -> bool {
        let Some(removed) = self.pipeline_dag.disconnect_target(to_pane) else {
            return false;
        };
        let labels = self.viewport_layout.pane_labels();
        let dst_lbl = labels.get(to_pane).copied().unwrap_or("Pane");
        let msg = format!(
            "Disconnected {} stream from {}",
            removed.kind.label(),
            dst_lbl
        );
        if let Some(c) = ctx {
            crate::accessibility::announce_live_status(c, &msg);
        } else {
            crate::accessibility::announce_status(&msg);
        }
        true
    }

    fn announce_pipeline_error(
        ctx: Option<&egui::Context>,
        err: super::dataflow_bus::DagLinkError,
    ) {
        if let Some(c) = ctx {
            crate::accessibility::announce_live_assertive_status(c, err.message());
        } else {
            crate::accessibility::announce_status_with_priority(err.message(), "assertive");
        }
    }

    /// Rotates active viewport panes clockwise (`true`) or counter-clockwise (`false`).
    ///
    /// In `QuadGrid` (`0 = Top-Left`, `1 = Top-Right`, `2 = Bottom-Left`, `3 = Bottom-Right`),
    /// clockwise rotation moves `Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left -> Top-Left`.
    pub fn rotate_panes(&mut self, clockwise: bool) {
        match self.viewport_layout.pane_count() {
            4 => {
                if clockwise {
                    self.swap_pane_pair(0, 2);
                    self.swap_pane_pair(2, 3);
                    self.swap_pane_pair(3, 1);
                    self.focused_pane = match self.focused_pane {
                        0 => 1,
                        1 => 3,
                        3 => 2,
                        2 => 0,
                        other => other,
                    };
                } else {
                    self.swap_pane_pair(3, 1);
                    self.swap_pane_pair(2, 3);
                    self.swap_pane_pair(0, 2);
                    self.focused_pane = match self.focused_pane {
                        0 => 2,
                        2 => 3,
                        3 => 1,
                        1 => 0,
                        other => other,
                    };
                }
            }
            2 => {
                self.swap_pane_pair(0, 1);
                if self.focused_pane < 2 {
                    self.focused_pane = 1 - self.focused_pane;
                }
            }
            _ => {}
        }
    }

    /// Rotates all active viewport panes clockwise by one step.
    pub fn rotate_panes_clockwise(&mut self) {
        self.rotate_panes(true);
    }

    /// Rotates all active viewport panes counter-clockwise by one step.
    pub fn rotate_panes_counter_clockwise(&mut self) {
        self.rotate_panes(false);
    }

    fn ensure_four_pane_tabs(&mut self) {
        while self.pane_tabs.len() < 4 {
            let fallback = self.pane_tabs.len().min(self.tabs.len().saturating_sub(1));
            self.pane_tabs.push(fallback);
        }
    }

    /// Exchanges runtime tab states between Pane 0 and secondary `sec_pane` (`1..=3`) after index swap.
    pub(super) fn swap_primary_and_secondary_slot_states(&mut self, sec_pane: usize) {
        self.tab_state_cache.clear();
        if let Some(slot) = self.pane_slots.slot_mut(sec_pane) {
            slot.initialized = true;
        }
        let new_p0 = self.selected_tab;
        let Some(new_sec) = self.pane_tabs.get(sec_pane).copied() else {
            return;
        };
        if new_p0 >= self.tabs.len() || new_sec >= self.tabs.len() {
            return;
        }
        if self.viewport_layout == ViewportLayout::Single {
            self.swap_serialized_slot_state(sec_pane, new_p0, new_sec);
            return;
        }
        if new_p0 == new_sec {
            if let Some(slot) = self.pane_slots.slot_mut(sec_pane) {
                if let Some(sec_tab) = slot.ensure_tab(new_p0) {
                    std::mem::swap(&mut self.tabs[new_p0], sec_tab);
                }
                slot.snapshot_active();
            }
            return;
        }
        self.swap_distinct_slot_instances(sec_pane, new_p0, new_sec);
    }

    fn swap_serialized_slot_state(&mut self, sec_pane: usize, new_p0: usize, new_sec: usize) {
        let p0_name = self.tabs[new_p0].name().to_string();
        let sec_name = self.tabs[new_sec].name().to_string();
        let s_old_p0 = self.tabs[new_sec].save_state();
        let mut s_old_sec = None;
        if let Some(slot) = self.pane_slots.slot_mut(sec_pane) {
            s_old_sec = slot.saved_states.get(&p0_name).cloned();
            if let Some(s0) = s_old_p0 {
                slot.saved_states.insert(sec_name, s0.into());
            }
        }
        if let Some(saved) = s_old_sec {
            self.tabs[new_p0].load_state(&saved);
        } else if let Some(fresh) = (self.viewport_layout != ViewportLayout::Single)
            .then(|| crate::tabs::instantiate_tab(new_p0))
            .flatten()
        {
            self.tabs[new_p0] = fresh;
        }
    }

    fn swap_distinct_slot_instances(&mut self, sec_pane: usize, new_p0: usize, new_sec: usize) {
        let old_slot = self
            .pane_slots
            .slot_mut(sec_pane)
            .and_then(|s| s.active.take());
        let Some((idx, box_sec)) = old_slot else {
            self.swap_serialized_slot_state(sec_pane, new_p0, new_sec);
            return;
        };
        if idx == new_p0 {
            self.tabs[new_p0] = box_sec;
        } else if let Some(slot) = self.pane_slots.slot_mut(sec_pane) {
            if let Some(st) = box_sec.save_state() {
                slot.saved_states
                    .insert(box_sec.name().to_string(), st.into());
            }
            let p0_name = self.tabs[new_p0].name().to_string();
            if let Some(saved) = slot.saved_states.get(&p0_name).cloned() {
                self.tabs[new_p0].load_state(&saved);
            }
        }
        if let Some(mut fresh_primary) = crate::tabs::instantiate_tab(new_sec) {
            if let Some(s) = self.tabs[new_sec].save_state() {
                fresh_primary.load_state(&s);
            }
            let live_old_p0 = std::mem::replace(&mut self.tabs[new_sec], fresh_primary);
            if let Some(slot) = self.pane_slots.slot_mut(sec_pane) {
                slot.active = Some((new_sec, live_old_p0));
                slot.snapshot_active();
            }
        }
    }
}
