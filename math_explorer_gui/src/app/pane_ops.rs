#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MathExplorerApp, ViewportLayout};

impl MathExplorerApp {
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
                slot.saved_states.insert(sec_name, s0);
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
                slot.saved_states.insert(box_sec.name().to_string(), st);
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
