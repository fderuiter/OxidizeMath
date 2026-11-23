#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export_ui::{export_action_button, wrapped_colored_label};
use super::*;
use eframe::egui;

impl SimulationFramework {
    fn export_active_preset_json(&mut self, file_slug: &str) {
        if let Some(tool) = &self.active_tool {
            self.preset_dialog
                .export_preset_result(tool.export_preset_json(), file_slug);
        }
    }

    fn render_revert_to_preset_button(&mut self, ui: &mut egui::Ui, close_on_click: bool) {
        let has_preset = self.has_preset_baseline();
        let tip = if has_preset {
            "Revert parameter edits to the active loaded preset"
        } else {
            "Revert parameter edits to the active loaded preset (no preset loaded)"
        };
        if export_action_button(ui, has_preset, "↺ Revert to Preset", tip, close_on_click) {
            if self.reset_to_preset() {
                crate::accessibility::announce_status("Parameters reverted to active preset");
            }
            ui.ctx().request_repaint();
        }
    }

    fn render_factory_reset_button(&mut self, ui: &mut egui::Ui, close_on_click: bool) {
        let has_preset = self.has_preset_baseline();
        let (label, tip, announcement) = if has_preset {
            (
                "↻ Factory Defaults",
                "Restore initial factory default parameters",
                "Parameters reset to factory defaults",
            )
        } else {
            (
                "↻ Reset Parameters",
                "Restore default baseline parameters",
                "Parameters reset to baseline defaults",
            )
        };
        if export_action_button(ui, true, label, tip, close_on_click) {
            self.reset_to_factory_defaults();
            crate::accessibility::announce_status(announcement);
            ui.ctx().request_repaint();
        }
    }

    fn render_dual_reset_buttons(&mut self, ui: &mut egui::Ui, close_on_click: bool) {
        if self.has_preset_baseline() {
            self.render_revert_to_preset_button(ui, close_on_click);
        }
        self.render_factory_reset_button(ui, close_on_click);
    }

    fn render_preset_toasts(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        if let Some(ref msg) = self.preset_dialog.toast_info {
            ui.separator();
            wrapped_colored_label(ui, palette.success, format!("ℹ️ {msg}"));
            if ui
                .small_button("✖")
                .accessible_hover_text("Dismiss notification")
                .clicked()
            {
                self.preset_dialog.toast_info = None;
            }
        }
        if let Some(ref err) = self.preset_dialog.toast_error {
            ui.separator();
            wrapped_colored_label(ui, palette.error, format!("⚠️ {err}"));
            if ui
                .small_button("✖")
                .accessible_hover_text("Dismiss warning")
                .clicked()
            {
                self.preset_dialog.toast_error = None;
            }
        }
    }

    fn render_preset_json_buttons(
        &mut self,
        ui: &mut egui::Ui,
        file_slug: &str,
        close_on_click: bool,
    ) {
        if export_action_button(
            ui,
            true,
            "Import Preset JSON",
            "Import parameter preset from a JSON file",
            close_on_click,
        ) {
            self.preset_dialog.trigger_import();
        }
        if export_action_button(
            ui,
            true,
            "Export Preset JSON",
            "Export current parameter preset to a JSON file",
            close_on_click,
        ) {
            self.export_active_preset_json(file_slug);
        }
        if export_action_button(
            ui,
            true,
            "🔗 Copy Share Link",
            "Copy a shareable simulation link to the clipboard",
            close_on_click,
        ) {
            let ctx = ui.ctx().clone();
            self.copy_active_share_link(&ctx);
        }
    }

    pub(super) fn render_compact_preset_controls(
        &mut self,
        ui: &mut egui::Ui,
        file_slug: &str,
        inline_reset: bool,
    ) {
        let resp = ui.menu_button("⚙ Presets ▾", |menu_ui| {
            self.render_preset_json_buttons(menu_ui, file_slug, true);
            menu_ui.separator();
            self.render_revert_to_preset_button(menu_ui, true);
            self.render_factory_reset_button(menu_ui, true);
        });
        resp.response
            .accessible_hover_text("Open parameter presets menu");
        if inline_reset {
            self.render_dual_reset_buttons(ui, false);
        }
        self.render_preset_toasts(ui);
    }

    pub(super) fn render_preset_action_bar_controls(&mut self, ui: &mut egui::Ui, file_slug: &str) {
        ui.separator();
        ui.label(egui::RichText::new("⚙ Presets:").strong());
        self.render_preset_json_buttons(ui, file_slug, false);
        self.render_revert_to_preset_button(ui, false);
        self.render_factory_reset_button(ui, false);
        self.render_preset_toasts(ui);
    }
}
