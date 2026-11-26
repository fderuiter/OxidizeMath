//! Viewport visibility synchronization keeping only active visible panes unsuspended.

use super::pane_slots::MAX_SECONDARY_PANES;
use super::{MathExplorerApp, SplitLayoutMode, ViewportLayout};

impl MathExplorerApp {
    /// Resolves the effective [`ViewportLayout`], honoring legacy `split_layout_mode` overrides.
    #[must_use]
    pub(super) fn effective_viewport_layout(&self) -> ViewportLayout {
        if self.viewport_layout != ViewportLayout::Single {
            self.viewport_layout
        } else {
            match self.split_layout_mode {
                SplitLayoutMode::Horizontal => ViewportLayout::SplitHorizontal,
                SplitLayoutMode::Vertical => ViewportLayout::SplitVertical,
                SplitLayoutMode::Single => ViewportLayout::Single,
            }
        }
    }

    /// Synchronizes background worker suspension across `self.tabs` and `SecondaryPaneSlots` so
    /// only `active_primary_tab_index` in Pane 0 and instantiated `1..pane_count` secondary slots
    /// remain unsuspended.
    pub fn sync_viewport_visibility(&mut self) {
        let pane_count = self.effective_viewport_layout().pane_count();
        let active_primary = self.active_primary_tab_index();
        for (idx, tab) in self.tabs.iter_mut().enumerate() {
            tab.set_suspended(idx != active_primary);
        }
        for pane_idx in 1..=MAX_SECONDARY_PANES {
            let is_visible_pane = pane_idx < pane_count;
            if let Some(slot) = self.pane_slots.slot_mut(pane_idx) {
                if let Some((_, ref mut active_tab)) = slot.active {
                    active_tab.set_suspended(!is_visible_pane);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "tests_visibility_suspension.rs"]
mod tests;
