#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AccessibleHoverText, InputMode, SimulationFramework};
use crate::app::session::shortcuts::{
    is_workspace_history_active, read_workspace_history_bridge, request_workspace_redo,
    request_workspace_undo,
};
use eframe::egui;

/// Renders accessible workspace `Undo` and `Redo` toolbar buttons synchronized with
/// [`crate::app::session::shortcuts::WorkspaceHistoryBridge`].
pub fn render_workspace_history_buttons(ui: &mut egui::Ui) {
    let active = is_workspace_history_active(ui.ctx());
    let bridge = read_workspace_history_bridge(ui.ctx());
    let touch_mode = ui.ctx().data(|d| {
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
        if ui
            .button("📖 Theory")
            .accessible_hover_text("Toggle Theory Context Portal")
            .clicked()
        {
            self.show_theory_portal = !self.show_theory_portal;
            let next = self.show_theory_portal;
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), next);
                if let Some(inst_key) =
                    d.get_temp::<egui::Id>(egui::Id::new("CURRENT_THEORY_PORTAL_INST_ID"))
                {
                    d.insert_temp(inst_key, next);
                }
            });
            ctx.request_repaint();
        }
    }
}
