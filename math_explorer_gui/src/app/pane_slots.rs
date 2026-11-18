#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MathExplorerApp, ViewportLayout};
use crate::tabs::ExplorerTab;
use eframe::egui;
use std::collections::HashMap;

/// Maximum number of secondary panes supported (Panes 1, 2, and 3 in `QuadGrid`).
pub const MAX_SECONDARY_PANES: usize = 3;

/// Per-pane lazy runtime container for a secondary viewport slot.
#[derive(Default)]
struct SecondarySlot {
    /// Currently instantiated `(tab_index, tab_instance)` for this secondary pane.
    active: Option<(usize, Box<dyn ExplorerTab>)>,
    /// Cached serialized state per tab name (`tab.name()`) within this pane.
    saved_states: HashMap<String, String>,
}

impl SecondarySlot {
    fn snapshot_active(&mut self) {
        if let Some((_, ref tab)) = self.active {
            if let Some(state_json) = tab.save_state() {
                self.saved_states.insert(tab.name().to_string(), state_json);
            }
        }
    }

    fn drop_active(&mut self) {
        self.snapshot_active();
        self.active = None;
    }

    fn ensure_tab(&mut self, tab_idx: usize) -> Option<&mut Box<dyn ExplorerTab>> {
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
            self.active = Some((tab_idx, new_tab));
        }
        self.active.as_mut().map(|(_, tab)| tab)
    }

    fn extract_states(&self) -> HashMap<String, String> {
        let mut states = self.saved_states.clone();
        if let Some((_, ref tab)) = self.active {
            if let Some(state_json) = tab.save_state() {
                states.insert(tab.name().to_string(), state_json);
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
    fn slot_mut(&mut self, pane_idx: usize) -> Option<&mut SecondarySlot> {
        pane_idx
            .checked_sub(1)
            .and_then(|idx| self.slots.get_mut(idx))
    }

    #[cfg(test)]
    fn slot(&self, pane_idx: usize) -> Option<&SecondarySlot> {
        pane_idx.checked_sub(1).and_then(|idx| self.slots.get(idx))
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
        for (slot_idx, slot) in self.pane_slots.slots.iter_mut().enumerate() {
            if slot_idx >= keep_secondary {
                slot.drop_active();
            }
        }
        if active_pane_count <= 1 {
            self.focused_pane = 0;
        } else if self.focused_pane >= active_pane_count {
            self.focused_pane = active_pane_count - 1;
        }
    }

    /// Releases a secondary pane's active instance if its assigned domain index changed.
    pub(super) fn on_pane_tab_assigned(&mut self, pane_idx: usize, tab_idx: usize) {
        if let Some(slot) = self.pane_slots.slot_mut(pane_idx) {
            if slot.active.as_ref().is_some_and(|(idx, _)| *idx != tab_idx) {
                slot.drop_active();
            }
        }
    }

    /// Returns a mutable reference to the `ExplorerTab` assigned to `pane_idx`, instantiating lazily for secondary panes.
    pub fn pane_tab_mut(&mut self, pane_idx: usize) -> Option<&mut Box<dyn ExplorerTab>> {
        if pane_idx == 0 {
            let tab_idx = self.pane_tabs.first().copied().unwrap_or(self.selected_tab);
            return self.tabs.get_mut(tab_idx);
        }
        let tab_idx = self.pane_tabs.get(pane_idx).copied()?;
        if tab_idx >= self.tabs.len() {
            return None;
        }
        self.pane_slots.slot_mut(pane_idx)?.ensure_tab(tab_idx)
    }

    /// Returns a mutable reference to the `ExplorerTab` of the currently focused pane.
    pub fn focused_pane_tab_mut(&mut self) -> Option<&mut Box<dyn ExplorerTab>> {
        let target = self.active_focused_pane();
        self.pane_tab_mut(target)
    }

    /// Split-borrows the pane's `ExplorerTab`, sub-`egui::Context`, and texture map for rendering.
    #[allow(clippy::type_complexity)]
    pub(super) fn pane_render_parts_mut(
        &mut self,
        pane_idx: usize,
    ) -> Option<(
        &mut dyn ExplorerTab,
        &egui::Context,
        &mut HashMap<egui::TextureId, egui::TextureId>,
    )> {
        if pane_idx >= 4 {
            return None;
        }
        let tab_idx = self.pane_tabs.get(pane_idx).copied().unwrap_or(0);
        if tab_idx >= self.tabs.len() {
            return None;
        }
        let pane_ctx = &self.pane_contexts[pane_idx];
        let tex_map = &mut self.pane_textures[pane_idx];
        if pane_idx == 0 {
            let tab = self.tabs.get_mut(tab_idx)?.as_mut();
            Some((tab, pane_ctx, tex_map))
        } else {
            let slot = self.pane_slots.slot_mut(pane_idx)?;
            let tab = slot.ensure_tab(tab_idx)?.as_mut();
            Some((tab, pane_ctx, tex_map))
        }
    }

    /// Exchanges runtime tab states between Pane 0 and Pane 1 after their assigned indices are swapped.
    pub(super) fn swap_primary_and_secondary_states(&mut self) {
        let new_p0 = self.selected_tab;
        let new_p1 = self.secondary_tab;
        if new_p0 >= self.tabs.len() || new_p1 >= self.tabs.len() {
            return;
        }
        if self.viewport_layout == ViewportLayout::Single {
            self.swap_serialized_slot1_state(new_p0, new_p1);
            return;
        }
        if new_p0 == new_p1 {
            if let Some(slot1) = self.pane_slots.slot_mut(1) {
                if let Some(tab1) = slot1.ensure_tab(new_p0) {
                    std::mem::swap(&mut self.tabs[new_p0], tab1);
                }
                slot1.snapshot_active();
            }
            return;
        }
        self.swap_distinct_slot1_instances(new_p0, new_p1);
    }

    fn swap_serialized_slot1_state(&mut self, new_p0: usize, new_p1: usize) {
        let p0_name = self.tabs[new_p0].name().to_string();
        let p1_name = self.tabs[new_p1].name().to_string();
        let s_old_p0 = self.tabs[new_p1].save_state();
        let mut s_old_p1 = None;
        if let Some(slot1) = self.pane_slots.slot_mut(1) {
            s_old_p1 = slot1.saved_states.get(&p0_name).cloned();
            if let Some(s0) = s_old_p0 {
                slot1.saved_states.insert(p1_name, s0);
            }
        }
        if let Some(saved) = s_old_p1 {
            self.tabs[new_p0].load_state(&saved);
        } else if self.viewport_layout != ViewportLayout::Single {
            if let Some(fresh) = crate::tabs::instantiate_tab(new_p0) {
                self.tabs[new_p0] = fresh;
            }
        }
    }

    fn swap_distinct_slot1_instances(&mut self, new_p0: usize, new_p1: usize) {
        let old_slot1 = self.pane_slots.slot_mut(1).and_then(|s| s.active.take());
        let Some((idx, box1)) = old_slot1 else {
            self.swap_serialized_slot1_state(new_p0, new_p1);
            return;
        };
        if idx == new_p0 {
            self.tabs[new_p0] = box1;
        } else if let Some(slot1) = self.pane_slots.slot_mut(1) {
            if let Some(st) = box1.save_state() {
                slot1.saved_states.insert(box1.name().to_string(), st);
            }
            let p0_name = self.tabs[new_p0].name().to_string();
            if let Some(saved) = slot1.saved_states.get(&p0_name).cloned() {
                self.tabs[new_p0].load_state(&saved);
            }
        }
        if let Some(mut fresh_primary) = crate::tabs::instantiate_tab(new_p1) {
            if let Some(s) = self.tabs[new_p1].save_state() {
                fresh_primary.load_state(&s);
            }
            let live_old_p0 = std::mem::replace(&mut self.tabs[new_p1], fresh_primary);
            if let Some(slot1) = self.pane_slots.slot_mut(1) {
                slot1.active = Some((new_p1, live_old_p0));
                slot1.snapshot_active();
            }
        }
    }

    /// Extracts serialized per-pane state maps for `AppState`.
    pub(super) fn extract_pane_tab_states(&self) -> Vec<HashMap<String, String>> {
        let mut all = Vec::with_capacity(4);
        all.push(HashMap::new());
        let mut any_non_empty = false;
        for slot in &self.pane_slots.slots {
            let map = slot.extract_states();
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
        pane_tab_states: Vec<HashMap<String, String>>,
        fallback_tab_states: &HashMap<String, String>,
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

    /// Updates `focused_pane` when raw input contains a click or typing event inside an active pane.
    pub(super) fn update_focused_pane_from_input(
        &mut self,
        ctx: &egui::Context,
        raw_input: &egui::RawInput,
        pane_rects: &[egui::Rect],
        pointer_pos: Option<egui::Pos2>,
    ) {
        if self.is_dragging_divider
            || self.command_palette.is_open
            || self.show_help_menu
            || egui::Popup::is_any_open(ctx)
        {
            return;
        }
        let pane_count = self.viewport_layout.pane_count().min(pane_rects.len());
        for event in &raw_input.events {
            match event {
                egui::Event::PointerButton {
                    pos, pressed: true, ..
                }
                | egui::Event::Touch {
                    pos,
                    phase: egui::TouchPhase::Start,
                    ..
                } => {
                    if let Some(idx) = (0..pane_count).find(|&i| pane_rects[i].contains(*pos)) {
                        self.focus_pane(idx);
                    }
                }
                egui::Event::Text(_) => {
                    if let Some(pos) = pointer_pos {
                        if let Some(idx) = (0..pane_count).find(|&i| pane_rects[i].contains(pos)) {
                            self.focus_pane(idx);
                        }
                    }
                }
                egui::Event::Key {
                    pressed: true,
                    modifiers,
                    ..
                } if !modifiers.ctrl && !modifiers.mac_cmd && !modifiers.command => {
                    if let Some(pos) = pointer_pos {
                        if let Some(idx) = (0..pane_count).find(|&i| pane_rects[i].contains(pos)) {
                            self.focus_pane(idx);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Applies any selection returned by the global command palette to `focused_pane`.
    pub(super) fn apply_command_palette_selection(&mut self, ctx: &egui::Context) {
        let Some(selection) = self.command_palette.show(ctx, &self.tabs) else {
            return;
        };
        if selection.tab_index >= self.tabs.len() {
            return;
        }
        self.select_tab(ctx, selection.tab_index);
        let Some(tool_name) = &selection.tool_name else {
            return;
        };
        if let Some(tab) = self.focused_pane_tab_mut() {
            tab.select_tool_by_name(tool_name);
            let msg = format!("Switched to tool {} in tab {}", tool_name, tab.name());
            crate::accessibility::announce_status(&msg);
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("aria_live_message"), msg);
            });
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
    #[cfg(test)]
    pub fn pane_tab(&self, pane_idx: usize) -> Option<&dyn ExplorerTab> {
        if pane_idx == 0 {
            let tab_idx = self.pane_tabs.first().copied().unwrap_or(self.selected_tab);
            self.tabs.get(tab_idx).map(|b| b.as_ref())
        } else {
            self.pane_slots
                .slot(pane_idx)
                .and_then(|s| s.active.as_ref().map(|(_, b)| b.as_ref()))
        }
    }
}
