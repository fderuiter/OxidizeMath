//! Global keyboard shortcut registration and handlers for [`super::MathExplorerApp`] overlays.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MathExplorerApp;
use eframe::egui;

impl MathExplorerApp {
    /// Registers and handles global overlay shortcuts (`Command Palette`, `Keyboard Shortcuts`,
    /// and `Toggle Issues & Diagnostics`).
    pub(super) fn poll_global_overlay_shortcuts(&mut self, ctx: &egui::Context) {
        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };

        self.poll_command_palette_shortcut(ctx, modifiers);
        self.poll_keyboard_shortcuts_modal(ctx, modifiers);
        self.poll_diagnostics_panel_shortcut(ctx, modifiers);
    }

    fn poll_command_palette_shortcut(&mut self, ctx: &egui::Context, modifiers: egui::Modifiers) {
        let palette_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Command Palette",
            "Open command palette to quickly switch simulation tabs",
            egui_plot::commands::CommandTrigger::Shortcut(modifiers, egui::Key::K),
            true,
            "Global",
            None,
            None,
        );

        if palette_triggered {
            self.command_palette.toggle();
            let status_msg = if self.command_palette.is_open {
                "Command palette opened"
            } else {
                "Command palette closed"
            };
            crate::accessibility::announce_live_status(ctx, status_msg);
        }
    }

    fn poll_keyboard_shortcuts_modal(&mut self, ctx: &egui::Context, modifiers: egui::Modifiers) {
        let slash_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Keyboard Shortcuts",
            "Toggle keyboard shortcuts modal",
            egui_plot::commands::CommandTrigger::Shortcut(modifiers, egui::Key::Slash),
            true,
            "Global",
            None,
            None,
        );

        egui_plot::commands::CommandRegistryData::register(
            ctx,
            "Keyboard Shortcuts",
            "Toggle keyboard shortcuts modal",
            egui_plot::commands::CommandTrigger::Key(egui::Key::Questionmark),
            true,
            "Global",
        );

        let question_triggered = !ctx.wants_keyboard_input()
            && ctx.input_mut(|i| {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Questionmark)
                    || i.consume_key(egui::Modifiers::SHIFT, egui::Key::Questionmark)
            });

        if slash_triggered || question_triggered {
            self.show_help_menu = !self.show_help_menu;
            let status_msg = if self.show_help_menu {
                "Keyboard shortcuts opened"
            } else {
                "Keyboard shortcuts closed"
            };
            crate::accessibility::announce_live_status(ctx, status_msg);
        }
    }

    fn poll_diagnostics_panel_shortcut(&mut self, ctx: &egui::Context, modifiers: egui::Modifiers) {
        let shift_modifiers = modifiers | egui::Modifiers::SHIFT;
        let toggle_diagnostics_triggered =
            egui_plot::commands::CommandRegistryData::register_and_check(
                ctx,
                "Toggle Issues & Diagnostics",
                "Toggle visibility of the Issues & Diagnostics panel",
                egui_plot::commands::CommandTrigger::Shortcut(shift_modifiers, egui::Key::D),
                true,
                "Global",
                None,
                None,
            );

        if toggle_diagnostics_triggered {
            self.show_diagnostics_panel = !self.show_diagnostics_panel;
            let status_msg = if self.show_diagnostics_panel {
                "Issues & Diagnostics panel opened"
            } else {
                "Issues & Diagnostics panel closed"
            };
            crate::accessibility::announce_live_status(ctx, status_msg);
        }
    }
}
