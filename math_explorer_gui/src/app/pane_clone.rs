#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MathExplorerApp, SharedTabJson, ViewportLayout};
use eframe::egui;

impl MathExplorerApp {
    /// Clones the domain assignment, active `ToolMetadata` selection, and serialized parameters
    /// from `src_pane` into `dst_pane`, invalidating caches so undo/redo captures one step.
    pub fn clone_pane_to(&mut self, src_pane: usize, dst_pane: usize) -> bool {
        if src_pane >= 4 || dst_pane >= 4 || src_pane == dst_pane || self.tabs.is_empty() {
            return false;
        }
        let Some((src_tab_idx, tab_name, saved_json)) = self.snapshot_pane_for_clone(src_pane)
        else {
            return false;
        };

        self.clear_active_pane_caches();
        while self.pane_tabs.len() < 4 {
            let next_idx = self.pane_tabs.len().min(self.tabs.len().saturating_sub(1));
            self.pane_tabs.push(next_idx);
        }
        self.pane_tabs[dst_pane] = src_tab_idx;
        if dst_pane == 0 {
            self.selected_tab = src_tab_idx;
        } else if dst_pane == 1 {
            self.secondary_tab = src_tab_idx;
        }

        self.apply_cloned_state_to_pane(dst_pane, src_tab_idx, &tab_name, saved_json.as_ref());
        self.tab_state_cache.invalidate_tab(&tab_name);
        self.clear_active_pane_caches();
        true
    }

    fn snapshot_pane_for_clone(
        &self,
        src_pane: usize,
    ) -> Option<(usize, String, Option<SharedTabJson>)> {
        let src_tab_idx = if src_pane == 0 {
            self.active_primary_tab_index()
        } else {
            self.pane_tabs.get(src_pane).copied()?
        };
        let primary_tab = self.tabs.get(src_tab_idx)?;
        let tab_name = primary_tab.name().to_string();

        let saved_json = if src_pane == 0 {
            primary_tab.save_state().map(SharedTabJson::from)
        } else if let Some(slot) = self.pane_slots.slot(src_pane) {
            if let Some((active_idx, ref active_tab)) = slot.active {
                if active_idx == src_tab_idx {
                    active_tab.save_state().map(SharedTabJson::from)
                } else {
                    slot.saved_states.get(&tab_name).cloned()
                }
            } else {
                slot.saved_states
                    .get(&tab_name)
                    .cloned()
                    .or_else(|| primary_tab.save_state().map(SharedTabJson::from))
            }
        } else {
            None
        };
        Some((src_tab_idx, tab_name, saved_json))
    }

    fn apply_cloned_state_to_pane(
        &mut self,
        dst_pane: usize,
        src_tab_idx: usize,
        tab_name: &str,
        saved_json: Option<&SharedTabJson>,
    ) {
        if dst_pane == 0 {
            if let Some(json) = saved_json {
                if let Some(tab) = self.tabs.get_mut(src_tab_idx) {
                    tab.load_state(json);
                }
            } else if let Some(fresh) = crate::tabs::instantiate_tab(src_tab_idx) {
                self.tabs[src_tab_idx] = fresh;
            }
            return;
        }

        let is_active_pane = dst_pane < self.viewport_layout.pane_count();
        let Some(slot) = self.pane_slots.slot_mut(dst_pane) else {
            return;
        };
        slot.initialized = true;
        if let Some(json) = saved_json {
            slot.saved_states.insert(tab_name.to_string(), json.clone());
        } else {
            slot.saved_states.remove(tab_name);
        }
        slot.clear_active_cache();

        if is_active_pane {
            if let Some(dst_tab) = slot.ensure_tab(src_tab_idx) {
                if let Some(json) = saved_json {
                    dst_tab.load_state(json);
                }
            }
            slot.snapshot_active();
        } else if slot
            .active
            .as_ref()
            .is_some_and(|(idx, _)| *idx != src_tab_idx)
        {
            slot.reset_active_without_snapshot();
        }
    }

    /// Duplicates the currently focused pane (splitting into `SplitHorizontal` when in `Single`),
    /// returning the destination pane index when successful.
    pub fn duplicate_focused_pane(&mut self, ctx: Option<&egui::Context>) -> Option<usize> {
        if self.tabs.is_empty() {
            return None;
        }
        let (src_pane, dst_pane) = match self.viewport_layout {
            ViewportLayout::Single => {
                self.viewport_layout = ViewportLayout::SplitHorizontal;
                self.sync_layout();
                if !self.pane_tabs.is_empty() {
                    self.pane_tabs[0] = self.selected_tab;
                }
                (0, 1)
            }
            ViewportLayout::SplitHorizontal | ViewportLayout::SplitVertical => {
                let src = self.active_focused_pane();
                (src, 1 - src)
            }
            ViewportLayout::QuadGrid => {
                let src = self.active_focused_pane();
                (src, (src + 1) % 4)
            }
        };

        if !self.clone_pane_to(src_pane, dst_pane) {
            return None;
        }
        self.focus_pane(dst_pane);
        self.announce_pane_clone(ctx, src_pane, dst_pane);
        Some(dst_pane)
    }

    /// Clones `src_pane` into `dst_pane` and emits a polite ARIA live announcement.
    pub fn clone_pane_interactive(
        &mut self,
        ctx: &egui::Context,
        src_pane: usize,
        dst_pane: usize,
    ) -> bool {
        if !self.clone_pane_to(src_pane, dst_pane) {
            return false;
        }
        self.announce_pane_clone(Some(ctx), src_pane, dst_pane);
        ctx.request_repaint();
        true
    }

    fn announce_pane_clone(&self, ctx: Option<&egui::Context>, src_pane: usize, dst_pane: usize) {
        let Some(c) = ctx else {
            return;
        };
        let labels = self.viewport_layout.pane_labels();
        let src_lbl = labels.get(src_pane).copied().unwrap_or("Source Pane");
        let dst_lbl = labels.get(dst_pane).copied().unwrap_or("Target Pane");
        let tab_idx = self.pane_tabs.get(dst_pane).copied().unwrap_or(0);
        let tab_name = self.tabs.get(tab_idx).map_or("module", |t| t.name());
        let msg = format!("Duplicated {tab_name} from {src_lbl} to {dst_lbl}");
        crate::accessibility::announce_live_status(c, msg);
        c.request_repaint();
    }

    /// Seeds uninitialized secondary slots with Pane 0's live state when entering multi-pane mode
    /// for the first time, unless the caller explicitly pre-assigned `secondary_tab`.
    pub(super) fn seed_uninitialized_panes_on_initial_split(
        &mut self,
        target_layout: ViewportLayout,
    ) {
        if target_layout == ViewportLayout::Single || self.tabs.is_empty() {
            return;
        }
        let secondary_explicitly_preassigned =
            self.secondary_tab != self.pane_tabs.get(1).copied().unwrap_or(1);
        let Some((src_tab_idx, tab_name, saved_json)) = self.snapshot_pane_for_clone(0) else {
            return;
        };
        let pane_count = target_layout.pane_count();
        let mut cloned_any = false;
        for pane_idx in 1..pane_count {
            if pane_idx == 1 && secondary_explicitly_preassigned {
                if let Some(slot) = self.pane_slots.slot_mut(1) {
                    slot.initialized = true;
                }
                continue;
            }
            let already_init = self
                .pane_slots
                .slot(pane_idx)
                .is_some_and(|s| s.initialized);
            if already_init {
                continue;
            }
            if pane_idx < self.pane_tabs.len() {
                self.pane_tabs[pane_idx] = src_tab_idx;
            }
            if pane_idx == 1 {
                self.secondary_tab = src_tab_idx;
            }
            if let Some(slot) = self.pane_slots.slot_mut(pane_idx) {
                slot.initialized = true;
                if let Some(ref json) = saved_json {
                    slot.saved_states.insert(tab_name.clone(), json.clone());
                } else {
                    slot.saved_states.remove(&tab_name);
                }
                slot.clear_active_cache();
            }
            cloned_any = true;
        }
        if cloned_any {
            self.tab_state_cache.invalidate_tab(&tab_name);
            self.clear_active_pane_caches();
        }
    }

    /// Registers and handles `Ctrl+Shift+2` (`Cmd+Shift+2` on macOS) to duplicate the focused pane.
    pub(super) fn poll_duplicate_pane_shortcut(&mut self, ctx: &egui::Context) {
        let base_mod = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };
        let mods = base_mod | egui::Modifiers::SHIFT;
        let shortcut = egui_plot::commands::CommandTrigger::Shortcut(mods, egui::Key::Num2);
        let triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Duplicate Focused Pane",
            "Split or clone the focused simulation pane and its parameters into a comparison pane",
            shortcut,
            true,
            "Viewport",
            None,
            None,
        ) || consume_physical_shift_num2(ctx, mods);
        if triggered {
            let _ = self.duplicate_focused_pane(Some(ctx));
        }
    }
}

fn consume_physical_shift_num2(ctx: &egui::Context, mods: egui::Modifiers) -> bool {
    if ctx.wants_keyboard_input() {
        return false;
    }
    ctx.input_mut(|i| {
        let mut matched = false;
        i.events.retain(|ev| {
            let hit = matches!(
                ev,
                egui::Event::Key {
                    physical_key: Some(egui::Key::Num2),
                    modifiers,
                    pressed: true,
                    ..
                } if modifiers.matches_logically(mods)
            );
            matched |= hit;
            !hit
        });
        matched
    })
}
