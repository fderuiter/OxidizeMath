#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AccessibleHoverText, InputMode, SimulationFramework};
use crate::app::session::shortcuts::{
    is_workspace_history_active, read_workspace_history_bridge, request_workspace_redo,
    request_workspace_undo,
};
use eframe::egui;
use std::hash::{Hash, Hasher};

/// Renders accessible workspace `Undo` and `Redo` toolbar buttons synchronized with
/// [`crate::app::session::shortcuts::WorkspaceHistoryBridge`].
pub fn render_workspace_history_buttons(ui: &mut egui::Ui) {
    let ctx = ui.ctx();
    let (saw_touch, saw_mouse_move) = ctx.input(|i| {
        let touch = i.any_touches()
            || i.multi_touch().is_some()
            || i.events
                .iter()
                .any(|e| matches!(e, egui::Event::Touch { .. }));
        (touch, !touch && i.pointer.is_moving())
    });
    if saw_touch {
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("INPUT_MODE"), InputMode::Touch);
            d.insert_temp(egui::Id::new("INPUT_MODE_TOUCH"), true);
        });
    } else if saw_mouse_move {
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("INPUT_MODE"), InputMode::Mouse);
            d.insert_temp(egui::Id::new("INPUT_MODE_TOUCH"), false);
        });
    }
    let active = is_workspace_history_active(ctx);
    let bridge = read_workspace_history_bridge(ctx);
    let touch_mode = ctx.data(|d| {
        d.get_temp::<InputMode>(egui::Id::new("INPUT_MODE")) == Some(InputMode::Touch)
            || d.get_temp::<bool>(egui::Id::new("INPUT_MODE_TOUCH"))
                .unwrap_or(false)
    });
    let min_size = if touch_mode {
        egui::vec2(44.0, 44.0)
    } else {
        egui::Vec2::ZERO
    };

    let undo_btn = egui::Button::new("↩ Undo").min_size(min_size);
    if ui
        .add_enabled(active && bridge.can_undo, undo_btn)
        .accessible_hover_text("Undo the last workspace change (Ctrl+Z / Cmd+Z)")
        .clicked()
    {
        request_workspace_undo(ui.ctx());
    }

    let redo_btn = egui::Button::new("↪ Redo").min_size(min_size);
    if ui
        .add_enabled(active && bridge.can_redo, redo_btn)
        .accessible_hover_text("Redo the last undone workspace change (Ctrl+Y / Cmd+Shift+Z)")
        .clicked()
    {
        request_workspace_redo(ui.ctx());
    }
}

impl SimulationFramework {
    pub(super) fn render_starter_gallery_toggle(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        if ui
            .button("🎨 Starter Gallery")
            .accessible_hover_text("Toggle starter template gallery overlay")
            .clicked()
        {
            self.show_starter_gallery = !self.show_starter_gallery;
            ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("SHOW_STARTER_GALLERY"),
                    self.show_starter_gallery,
                );
            });
        }
    }

    pub(super) fn render_theory_portal_toggle(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let btn = ui
            .button("📖 Theory")
            .accessible_hover_text("Toggle Theory Context Portal");
        super::overlay_sync::record_default_theory_portal_toggle(ctx, btn.id);
        if btn.clicked() {
            self.show_theory_portal = !self.show_theory_portal;
            super::overlay_sync::record_theory_portal_toggle(ctx, btn.id, self.show_theory_portal);
        }
    }

    /// Computes a deterministic O(1) revision token for this framework's persisted state.
    pub fn state_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash_history_fields(&mut hasher);
        self.show_theory_portal.hash(&mut hasher);
        self.show_starter_gallery.hash(&mut hasher);
        hasher.finish().max(1)
    }

    /// Computes a deterministic O(1) revision token excluding transient overlay chrome.
    pub fn history_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash_history_fields(&mut hasher);
        hasher.finish().max(1)
    }

    fn hash_history_fields(&self, hasher: &mut impl Hasher) {
        self.rev_cache.framework_revision.hash(hasher);
        self.selected_tool_index.hash(hasher);
        self.saved_tool_states.len().hash(hasher);
        self.preset_tool_states.len().hash(hasher);
        self.hash_saved_plot_states(hasher);
        if let Some(tool) = &self.active_tool {
            tool.name().hash(hasher);
            let tool_rev = tool.state_revision();
            if tool_rev > 0 {
                1_u8.hash(hasher);
                tool_rev.hash(hasher);
            } else {
                0_u8.hash(hasher);
                self.rev_cache.untracked_tool_revision.hash(hasher);
            }
        }
    }

    /// Restores live `show_theory_portal` and `show_starter_gallery` flags without disturbing tool state.
    pub(crate) fn restore_overlay_chrome(&mut self, portal: bool, gallery: bool) {
        self.show_theory_portal = portal;
        self.show_starter_gallery = gallery;
        self.sync_state.theory_portal = Some(portal);
        self.sync_state.starter_gallery = Some(gallery);
    }
}
