#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MathExplorerApp, SharedTabJson, ViewportLayout};
use crate::tabs::ExplorerTab;
use eframe::egui;
use std::collections::HashMap;

/// Maximum number of secondary panes supported (Panes 1, 2, and 3 in `QuadGrid`).
pub const MAX_SECONDARY_PANES: usize = 3;

/// Per-pane lazy runtime container for a secondary viewport slot.
#[derive(Default)]
pub(super) struct SecondarySlot {
    /// Whether this secondary slot has been assigned or cloned since startup/reset.
    pub(super) initialized: bool,
    /// Currently instantiated `(tab_index, tab_instance)` for this secondary pane.
    pub(super) active: Option<(usize, Box<dyn ExplorerTab>)>,
    /// Cached serialized state per tab name (`tab.name()`) within this pane.
    pub(super) saved_states: HashMap<String, SharedTabJson>,
    /// Interior cache `(tab_name, state_revision, json)` for clean active secondary pane tabs.
    active_cache: std::cell::RefCell<Option<(String, u64, Option<SharedTabJson>)>>,
}

impl SecondarySlot {
    pub(super) fn clear_active_cache(&self) {
        *self.active_cache.borrow_mut() = None;
    }

    pub(super) fn reset_active_without_snapshot(&mut self) {
        *self.active_cache.borrow_mut() = None;
        self.active = None;
    }

    pub(super) fn snapshot_active(&mut self) {
        if let Some((_, ref tab)) = self.active {
            let rev = tab.state_revision();
            if let Some(state_json) = tab.save_state().map(SharedTabJson::from) {
                let name = tab.name().to_string();
                *self.active_cache.borrow_mut() =
                    Some((name.clone(), rev, Some(state_json.clone())));
                self.saved_states.insert(name, state_json);
            } else {
                *self.active_cache.borrow_mut() = Some((tab.name().to_string(), rev, None));
            }
        }
    }

    fn drop_active(&mut self) {
        self.snapshot_active();
        *self.active_cache.borrow_mut() = None;
        self.active = None;
    }

    pub(super) fn ensure_tab(&mut self, tab_idx: usize) -> Option<&mut Box<dyn ExplorerTab>> {
        self.initialized = true;
        let needs_rebuild = match &self.active {
            Some((current_idx, _)) => *current_idx != tab_idx,
            None => true,
        };
        if needs_rebuild {
            self.drop_active();
            let mut new_tab = crate::tabs::instantiate_tab(tab_idx)?;
            if let Some(saved) = self.saved_states.get(new_tab.name()) {
                new_tab.load_state(saved);
            }
            *self.active_cache.borrow_mut() = None;
            self.active = Some((tab_idx, new_tab));
        }
        self.active.as_mut().map(|(_, tab)| tab)
    }

    fn extract_states(&self, had_input: bool) -> HashMap<String, SharedTabJson> {
        let mut states = self.saved_states.clone();
        if let Some((_, ref tab)) = self.active {
            let rev = tab.state_revision();
            let name = tab.name();
            let cached_hit = self.active_cache.borrow().as_ref().and_then(
                |(cached_name, cached_rev, cached_json)| {
                    let matches = cached_name == name
                        && if rev > 0 {
                            *cached_rev == rev
                        } else {
                            !had_input
                        };
                    matches.then(|| cached_json.clone())
                },
            );
            let maybe_json = if let Some(cached_opt) = cached_hit {
                cached_opt
            } else {
                let fresh = tab.save_state().map(SharedTabJson::from);
                *self.active_cache.borrow_mut() = Some((name.to_string(), rev, fresh.clone()));
                fresh
            };
            if let Some(state_json) = maybe_json {
                states.insert(name.to_string(), state_json);
            }
        }
        states
    }
}

/// Manages lazy `ExplorerTab` instances for secondary split/quad panes (indices `1..=3`).
#[derive(Default)]
pub(super) struct SecondaryPaneSlots {
    slots: [SecondarySlot; MAX_SECONDARY_PANES],
}

impl SecondaryPaneSlots {
    pub(super) fn slot_mut(&mut self, pane_idx: usize) -> Option<&mut SecondarySlot> {
        pane_idx
            .checked_sub(1)
            .and_then(|idx| self.slots.get_mut(idx))
    }

    pub(super) fn slot(&self, pane_idx: usize) -> Option<&SecondarySlot> {
        pane_idx.checked_sub(1).and_then(|idx| self.slots.get(idx))
    }

    pub(super) fn swap_secondary_slots(&mut self, pane_a: usize, pane_b: usize) {
        if let (Some(idx_a), Some(idx_b)) = (pane_a.checked_sub(1), pane_b.checked_sub(1)) {
            if idx_a < self.slots.len() && idx_b < self.slots.len() {
                self.slots.swap(idx_a, idx_b);
            }
        }
    }
}

impl MathExplorerApp {
    /// Returns the clamped focused pane index for the active viewport layout.
    pub fn active_focused_pane(&self) -> usize {
        let count = self.viewport_layout.pane_count();
        if count <= 1 {
            0
        } else {
            self.focused_pane.min(count - 1)
        }
    }

    /// Returns the assigned tab index of the currently focused pane.
    pub fn focused_tab_index(&self) -> usize {
        let pane = self.active_focused_pane();
        if pane == 0 {
            self.selected_tab
        } else {
            self.pane_tabs.get(pane).copied().unwrap_or(0)
        }
    }

    /// Updates `focused_pane` when `pane_idx` is within the active layout's pane count.
    pub fn focus_pane(&mut self, pane_idx: usize) {
        let count = self.viewport_layout.pane_count();
        if pane_idx < count {
            self.focused_pane = pane_idx;
        }
    }

    /// Drops instantiated secondary pane tabs whose pane index is inactive under `active_pane_count`.
    pub fn prune_inactive_pane_slots(&mut self, active_pane_count: usize) {
        let keep_secondary = active_pane_count.saturating_sub(1);
        let mut dropped_active = false;
        for (slot_idx, slot) in self.pane_slots.slots.iter_mut().enumerate() {
            if slot_idx >= keep_secondary {
                dropped_active |= slot.active.is_some();
                slot.drop_active();
            }
        }
        if dropped_active || self.focused_pane >= active_pane_count {
            self.pending_focus_prune = true;
        }
        if active_pane_count <= 1 {
            self.focused_pane = 0;
        } else if self.focused_pane >= active_pane_count {
            self.focused_pane = active_pane_count - 1;
        }
    }

    /// Surrenders focus on the root `egui::Context` when the focused widget belonged to a pruned viewport pane.
    pub(super) fn flush_stale_pane_focus(&mut self, ctx: &egui::Context) {
        if !std::mem::take(&mut self.pending_focus_prune) {
            return;
        }
        let Some(fid) = ctx.memory(|m| m.focused()) else {
            return;
        };
        let keep = self.viewport_layout.pane_count();
        let in_kept_pane = ctx.read_response(fid).is_some_and(|resp| {
            ctx.data(|d| {
                d.get_temp::<Vec<egui::Rect>>(egui::Id::new("LAST_PANE_RECTS"))
                    .is_none_or(|rects| {
                        rects
                            .iter()
                            .take(keep)
                            .any(|r| r.contains(resp.rect.center()))
                    })
            })
        });
        if !in_kept_pane {
            ctx.memory_mut(|m| m.surrender_focus(fid));
        }
    }

    /// Releases a secondary pane's active instance if its assigned domain index changed.
    pub(super) fn on_pane_tab_assigned(&mut self, pane_idx: usize, tab_idx: usize) {
        if let Some(slot) = self.pane_slots.slot_mut(pane_idx) {
            slot.initialized = true;
            if slot.active.as_ref().is_some_and(|(idx, _)| *idx != tab_idx) {
                slot.drop_active();
            }
        }
    }

    /// Returns a mutable reference to the `ExplorerTab` assigned to `pane_idx`, instantiating lazily for secondary panes.
    pub fn pane_tab_mut(&mut self, pane_idx: usize) -> Option<&mut Box<dyn ExplorerTab>> {
        if pane_idx == 0 {
            let tab_idx = self.active_primary_tab_index();
            return self.tabs.get_mut(tab_idx);
        }
        let tab_idx = self.pane_tabs.get(pane_idx).copied()?;
        if tab_idx >= self.tabs.len() {
            return None;
        }
        self.pane_slots.slot_mut(pane_idx)?.ensure_tab(tab_idx)
    }

    /// Clears cached serialized JSON for active primary and secondary viewport panes.
    pub(super) fn clear_active_pane_caches(&self) {
        let primary_idx = self.active_primary_tab_index();
        if let Some(tab) = self.tabs.get(primary_idx) {
            self.tab_state_cache.invalidate_tab(tab.name());
        }
        for slot in &self.pane_slots.slots {
            if slot.active.is_some() {
                *slot.active_cache.borrow_mut() = None;
            }
        }
    }

    /// Returns a mutable reference to the `ExplorerTab` of the currently focused pane.
    pub fn focused_pane_tab_mut(&mut self) -> Option<&mut Box<dyn ExplorerTab>> {
        let target = self.active_focused_pane();
        self.pane_tab_mut(target)
    }

    /// Returns the primary `app.tabs` index rendered in Pane 0.
    pub(super) fn active_primary_tab_index(&self) -> usize {
        if self.viewport_layout == ViewportLayout::Single {
            self.selected_tab
        } else {
            self.pane_tabs.first().copied().unwrap_or(self.selected_tab)
        }
    }

    /// Computes a deterministic hash and tracking flag across secondary pane slots.
    pub(super) fn secondary_slots_revision_hash(&self) -> (u64, bool) {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let mut all_tracked = true;
        for (slot_idx, slot) in self.pane_slots.slots.iter().enumerate() {
            slot_idx.hash(&mut h);
            slot.initialized.hash(&mut h);
            slot.saved_states.len().hash(&mut h);
            if let Some((tab_idx, ref tab)) = slot.active {
                tab_idx.hash(&mut h);
                let rev = tab.state_revision();
                if rev == 0 {
                    all_tracked = false;
                }
                rev.hash(&mut h);
            } else {
                let mut saved_xor = 0_u64;
                for (k, v) in &slot.saved_states {
                    let mut entry_hasher = std::collections::hash_map::DefaultHasher::new();
                    k.hash(&mut entry_hasher);
                    v.hash(&mut entry_hasher);
                    saved_xor ^= entry_hasher.finish();
                }
                saved_xor.hash(&mut h);
            }
        }
        (h.finish(), all_tracked)
    }

    /// Extracts serialized per-pane state maps for `AppState`.
    pub(super) fn extract_pane_tab_states(&self) -> Vec<HashMap<String, SharedTabJson>> {
        let mut all = Vec::with_capacity(4);
        all.push(HashMap::new());
        let mut any_non_empty = false;
        for slot in &self.pane_slots.slots {
            let map = slot.extract_states(self.last_frame_had_input);
            if !map.is_empty() {
                any_non_empty = true;
            }
            all.push(map);
        }
        if any_non_empty || self.viewport_layout != ViewportLayout::Single {
            all
        } else {
            Vec::new()
        }
    }

    /// Restores serialized per-pane state maps from `AppState`, falling back to `fallback_tab_states` for legacy sessions.
    pub(super) fn restore_pane_tab_states(
        &mut self,
        pane_tab_states: Vec<HashMap<String, SharedTabJson>>,
        fallback_tab_states: &HashMap<String, SharedTabJson>,
    ) {
        let pane_count = self.viewport_layout.pane_count();
        self.prune_inactive_pane_slots(pane_count);
        let is_legacy = pane_tab_states.is_empty();

        for pane_idx in 1..=MAX_SECONDARY_PANES {
            let assigned = self.pane_tabs.get(pane_idx).copied().unwrap_or(pane_idx);
            let tab_name = self.tabs.get(assigned).map(|t| t.name().to_string());
            let Some(slot) = self.pane_slots.slot_mut(pane_idx) else {
                continue;
            };
            *slot.active_cache.borrow_mut() = None;
            if let Some(map) = pane_tab_states.get(pane_idx) {
                slot.saved_states = map.clone();
            } else if is_legacy {
                slot.saved_states.clear();
                if pane_idx < pane_count {
                    if let Some(ref name) = tab_name {
                        if let Some(fallback) = fallback_tab_states.get(name) {
                            slot.saved_states.insert(name.clone(), fallback.clone());
                        }
                    }
                }
            }
            slot.initialized = !slot.saved_states.is_empty() || pane_idx < pane_count;
            if pane_idx < pane_count {
                if let Some((active_idx, ref mut active_tab)) = slot.active {
                    if active_idx == assigned {
                        if let Some(saved) = slot.saved_states.get(active_tab.name()) {
                            active_tab.load_state(saved);
                        }
                    } else {
                        slot.active = None;
                    }
                }
            }
        }
    }

    /// Updates `focused_pane` strictly when raw input contains a pointer press or touch start inside an active pane.
    pub(super) fn update_focused_pane_from_input(
        &mut self,
        ctx: &egui::Context,
        raw_input: &egui::RawInput,
        pane_rects: &[egui::Rect],
    ) {
        if self.is_dragging_divider
            || self.dragging_quad_axes != (false, false)
            || self.command_palette.is_open
            || self.show_starter_gallery
            || self.show_help_menu
            || egui::Popup::is_any_open(ctx)
            || ctx.memory(|m| m.top_modal_layer().is_some())
        {
            return;
        }
        let pane_count = self.viewport_layout.pane_count().min(pane_rects.len());
        for event in &raw_input.events {
            let press_pos = match event {
                egui::Event::PointerButton {
                    pos, pressed: true, ..
                }
                | egui::Event::Touch {
                    pos,
                    phase: egui::TouchPhase::Start,
                    ..
                } => Some(*pos),
                _ => None,
            };
            if let Some(idx) =
                press_pos.and_then(|p| (0..pane_count).find(|&i| pane_rects[i].contains(p)))
            {
                self.focus_pane(idx);
                Self::surrender_focus_outside_pane(ctx, pane_rects[idx]);
            }
        }
    }

    fn surrender_focus_outside_pane(ctx: &egui::Context, target_pane_rect: egui::Rect) {
        let Some(fid) = ctx.memory(|m| m.focused()) else {
            return;
        };
        if super::viewport::sash::is_any_sash_id(fid) {
            return;
        }
        let inside_target = ctx.read_response(fid).is_some_and(|resp| {
            super::viewport::focus::response_belongs_to_pane(&resp, target_pane_rect)
        });
        if !inside_target {
            ctx.memory_mut(|m| m.surrender_focus(fid));
        }
    }

    /// Returns the number of currently instantiated secondary pane tab instances (for test verification).
    #[cfg(test)]
    pub fn active_secondary_slot_count(&self) -> usize {
        self.pane_slots
            .slots
            .iter()
            .filter(|s| s.active.is_some())
            .count()
    }

    /// Returns an immutable reference to a pane's instantiated `ExplorerTab` if currently allocated.
    pub fn pane_tab(&self, pane_idx: usize) -> Option<&dyn ExplorerTab> {
        if pane_idx == 0 {
            let tab_idx = self.active_primary_tab_index();
            self.tabs.get(tab_idx).map(|b| b.as_ref())
        } else {
            self.pane_slots
                .slot(pane_idx)
                .and_then(|s| s.active.as_ref().map(|(_, b)| b.as_ref()))
        }
    }
}
