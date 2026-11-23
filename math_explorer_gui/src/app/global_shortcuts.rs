#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ui_panels, viewport, MathExplorerApp, ViewportLayout};
use eframe::egui;
use egui_plot::commands::{CommandRegistryData, CommandTrigger};
use federated_registry::Severity;

pub(crate) fn is_context_modal_or_popup_open(ctx: &egui::Context) -> bool {
    egui::Popup::is_any_open(ctx) || egui_plot::active_top_modal_layer(ctx).is_some()
}

pub(crate) struct DesktopShortcutSpec<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub trigger: CommandTrigger,
    pub context_name: &'a str,
}

pub(crate) fn register_and_check_shortcut_if_allowed(
    ctx: &egui::Context,
    spec: DesktopShortcutSpec<'_>,
    ui: Option<&egui::Ui>,
    allowed: bool,
) -> bool {
    if !allowed {
        CommandRegistryData::register(
            ctx,
            spec.name,
            spec.description,
            spec.trigger,
            true,
            spec.context_name,
        );
        return false;
    }
    CommandRegistryData::register_and_check(
        ctx,
        spec.name,
        spec.description,
        spec.trigger,
        true,
        spec.context_name,
        ui,
        None,
    )
}

fn cmd_modifiers() -> egui::Modifiers {
    if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    }
}

impl MathExplorerApp {
    /// Returns `true` when any foreground modal dialog or popup menu currently holds focus.
    pub(crate) fn has_active_modal_or_popup(&self, ctx: &egui::Context) -> bool {
        is_context_modal_or_popup_open(ctx) || self.command_palette.is_open || self.show_help_menu
    }

    pub(crate) fn can_trigger_command_palette_shortcut(&self, ctx: &egui::Context) -> bool {
        if egui::Popup::is_any_open(ctx) || self.show_help_menu {
            return false;
        }
        let palette_layer = egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("command_palette_modal"),
        );
        match egui_plot::active_top_modal_layer(ctx) {
            None => true,
            Some(layer) => self.command_palette.is_open && layer == palette_layer,
        }
    }

    pub(crate) fn can_trigger_help_shortcut(&self, ctx: &egui::Context) -> bool {
        if egui::Popup::is_any_open(ctx) || self.command_palette.is_open {
            return false;
        }
        let help_layer = egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("keyboard_shortcuts_modal"),
        );
        match egui_plot::active_top_modal_layer(ctx) {
            None => true,
            Some(layer) => self.show_help_menu && layer == help_layer,
        }
    }

    pub(super) fn register_and_handle_global_shortcuts(&mut self, ctx: &egui::Context) {
        viewport::focus::register_and_handle_pane_focus_shortcuts(self, ctx);
        let modifiers = cmd_modifiers();

        let palette_allowed = self.can_trigger_command_palette_shortcut(ctx);
        if register_and_check_shortcut_if_allowed(
            ctx,
            DesktopShortcutSpec {
                name: "Command Palette",
                description: "Open command palette to quickly switch simulation tabs",
                trigger: CommandTrigger::Shortcut(modifiers, egui::Key::K),
                context_name: "Global",
            },
            None,
            palette_allowed,
        ) {
            self.command_palette.toggle();
            let msg = if self.command_palette.is_open {
                "Command palette opened"
            } else {
                "Command palette closed"
            };
            crate::accessibility::announce_live_status(ctx, msg);
        }

        self.register_and_handle_help_and_diagnostics_shortcuts(ctx, modifiers);
        self.poll_share_shortcut_and_clipboard(ctx);
        self.poll_duplicate_pane_shortcut(ctx);
    }

    fn register_and_handle_help_and_diagnostics_shortcuts(
        &mut self,
        ctx: &egui::Context,
        modifiers: egui::Modifiers,
    ) {
        let help_allowed = self.can_trigger_help_shortcut(ctx);
        let slash_triggered = register_and_check_shortcut_if_allowed(
            ctx,
            DesktopShortcutSpec {
                name: "Keyboard Shortcuts",
                description: "Toggle keyboard shortcuts modal",
                trigger: CommandTrigger::Shortcut(modifiers, egui::Key::Slash),
                context_name: "Global",
            },
            None,
            help_allowed,
        );
        CommandRegistryData::register(
            ctx,
            "Keyboard Shortcuts",
            "Toggle keyboard shortcuts modal",
            CommandTrigger::Key(egui::Key::Questionmark),
            true,
            "Global",
        );
        let question_triggered = help_allowed
            && !ctx.wants_keyboard_input()
            && ctx.input_mut(|i| {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Questionmark)
                    || i.consume_key(egui::Modifiers::SHIFT, egui::Key::Questionmark)
            });
        if slash_triggered || question_triggered {
            self.show_help_menu = !self.show_help_menu;
            let msg = if self.show_help_menu {
                "Keyboard shortcuts opened"
            } else {
                "Keyboard shortcuts closed"
            };
            crate::accessibility::announce_live_status(ctx, msg);
        }

        let no_modal_or_popup = !self.has_active_modal_or_popup(ctx);
        if register_and_check_shortcut_if_allowed(
            ctx,
            DesktopShortcutSpec {
                name: "Toggle Issues & Diagnostics",
                description: "Toggle visibility of the Issues & Diagnostics panel",
                trigger: CommandTrigger::Shortcut(modifiers | egui::Modifiers::SHIFT, egui::Key::D),
                context_name: "Global",
            },
            None,
            no_modal_or_popup,
        ) {
            self.show_diagnostics_panel = !self.show_diagnostics_panel;
            let msg = if self.show_diagnostics_panel {
                "Issues & Diagnostics panel opened"
            } else {
                "Issues & Diagnostics panel closed"
            };
            crate::accessibility::announce_live_status(ctx, msg);
        }
    }

    pub(super) fn poll_and_render_diagnostics(&mut self, ctx: &egui::Context) {
        let new_events = crate::telemetry::poll_app_telemetry();
        let mut has_error_or_fatal = false;
        for event in &new_events {
            if matches!(event.severity, Severity::Error | Severity::Fatal) {
                has_error_or_fatal = true;
                crate::accessibility::announce_status_with_priority(
                    &format!("{}: {}", event.severity, event.message),
                    "assertive",
                );
            }
        }
        if has_error_or_fatal {
            self.show_diagnostics_panel = true;
        }
        self.diagnostic_events.extend(new_events);

        if self.show_diagnostics_panel {
            ui_panels::render_issues_panel(
                ctx,
                &mut self.diagnostic_events,
                &mut self.show_errors,
                &mut self.show_warnings,
                &mut self.show_info,
                &mut self.show_diagnostics_panel,
            );
        }
    }

    pub(super) fn resolve_active_viewport_layout(&mut self) -> ViewportLayout {
        if self.viewport_layout != ViewportLayout::Single {
            self.sync_layout();
            self.viewport_layout
        } else if self.split_layout_mode != super::SplitLayoutMode::Single {
            let layout = match self.split_layout_mode {
                super::SplitLayoutMode::Horizontal => ViewportLayout::SplitHorizontal,
                super::SplitLayoutMode::Vertical => ViewportLayout::SplitVertical,
                super::SplitLayoutMode::Single => ViewportLayout::Single,
            };
            self.viewport_layout = layout;
            if !self.pane_tabs.is_empty() {
                self.pane_tabs[0] = self.selected_tab;
            }
            if self.pane_tabs.len() > 1 {
                self.pane_tabs[1] = self.secondary_tab;
            }
            layout
        } else {
            ViewportLayout::Single
        }
    }
}
