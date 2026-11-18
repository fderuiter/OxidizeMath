use crate::app::{MathExplorerApp, ThemeMode, ViewportLayout};
use eframe::egui;
use federated_registry::{Severity, TelemetryEvent};

/// Renders the Issues & Diagnostics bottom panel in the UI.
#[allow(clippy::too_many_lines)]
pub fn render_issues_panel(
    ctx: &egui::Context,
    diagnostic_events: &mut Vec<TelemetryEvent>,
    show_errors: &mut bool,
    show_warnings: &mut bool,
    show_info: &mut bool,
    show_panel: &mut bool,
) {
    egui::TopBottomPanel::bottom("issues_panel")
        .resizable(true)
        .min_height(100.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Issues & Diagnostics");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Hide").clicked() {
                        *show_panel = false;
                    }
                    if ui.button("Clear").clicked() {
                        diagnostic_events.clear();
                    }
                    ui.checkbox(show_errors, "Errors/Fatal");
                    ui.checkbox(show_warnings, "Warnings");
                    ui.checkbox(show_info, "Info");
                });
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                if diagnostic_events.is_empty() {
                    ui.label(egui::RichText::new("No issues or diagnostics logged.").italics());
                    return;
                }

                let mut visible_count = 0;
                for event in &*diagnostic_events {
                    let show = match event.severity {
                        Severity::Info => *show_info,
                        Severity::Warning => *show_warnings,
                        Severity::Error | Severity::Fatal => *show_errors,
                    };
                    if !show {
                        continue;
                    }
                    visible_count += 1;

                    let color = match event.severity {
                        Severity::Info => egui::Color32::LIGHT_BLUE,
                        Severity::Warning => egui::Color32::YELLOW,
                        Severity::Error => egui::Color32::RED,
                        Severity::Fatal => egui::Color32::DARK_RED,
                    };

                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "[{} - {}]",
                                    event.source, event.severity
                                ))
                                .color(color)
                                .strong(),
                            );
                            if let Some(thread) = &event.thread_name {
                                ui.label(
                                    egui::RichText::new(format!("(Thread: {})", thread)).italics(),
                                );
                            }
                            ui.label(&event.message);
                        });
                        if !event.metadata.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                for (k, v) in &event.metadata {
                                    ui.label(
                                        egui::RichText::new(format!("{}: {}", k, v))
                                            .monospace()
                                            .size(10.0),
                                    );
                                }
                            });
                        }
                    });
                }

                if visible_count == 0 {
                    ui.label(
                        egui::RichText::new("No events match the current filter settings.")
                            .italics(),
                    );
                }
            });
        });
}

/// Renders the global command cheatsheet modal window.
pub fn render_help_window(ctx: &egui::Context, show_help_menu: &mut bool) {
    if *show_help_menu {
        egui::Window::new("Help Menu")
            .open(show_help_menu)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Available Commands");
                ui.separator();

                let registry_data = ctx.data(|d| {
                    d.get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new(
                        "CMD_REGISTRY",
                    ))
                    .unwrap_or_default()
                });

                if registry_data.commands.is_empty() {
                    ui.label("No commands available for the current context.");
                } else {
                    for cmd in registry_data.commands {
                        ui.group(|ui| {
                            ui.label(egui::RichText::new(&cmd.name).strong());
                            ui.label(&cmd.description);

                            let trigger_str = match &cmd.trigger {
                                egui_plot::commands::CommandTrigger::Key(k) => {
                                    format!("Key: {:?}", k)
                                }
                                egui_plot::commands::CommandTrigger::Shortcut(m, k) => {
                                    format!("Shortcut: {:?} + {:?}", m, k)
                                }
                                egui_plot::commands::CommandTrigger::AltClick => {
                                    "Alt-Click".to_string()
                                }
                            };
                            ui.label(egui::RichText::new(trigger_str).code());

                            if cmd.desktop_only {
                                ui.label(egui::RichText::new("Desktop Only").italics());
                            }
                        });
                    }
                }
            });
    }
}

#[cfg(any(not(target_os = "macos"), test))]
fn render_pane_assignments_submenu(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
) {
    ui.menu_button("Pane Assignments", |ui| {
        let pane_count = app.viewport_layout.pane_count();
        let pane_prefixes = app.viewport_layout.pane_label_prefixes();
        let mut pending_assignment = None;

        for pane_idx in 0..pane_count {
            let prefix = pane_prefixes.get(pane_idx).copied().unwrap_or("Pane:");
            let assigned = app.pane_tabs.get(pane_idx).copied().unwrap_or(0);
            let assigned_name = app
                .tabs
                .get(assigned)
                .map(|t| t.name())
                .unwrap_or("Unknown");

            ui.push_id(("pane_assign", pane_idx), |ui| {
                ui.menu_button((prefix, assigned_name), |pane_ui| {
                    if let Some(tab_idx) = crate::app::domain_bar::render_domain_module_submenus(
                        pane_ui, &app.tabs, assigned,
                    ) {
                        pending_assignment = Some((pane_idx, tab_idx));
                    }
                });
            });
        }

        if let Some((pane_idx, tab_idx)) = pending_assignment {
            app.select_pane_tab(ctx, pane_idx, tab_idx);
            ui.close();
        }
    });
}

/// Renders the top menu bar in the UI.
#[allow(clippy::too_many_lines, dead_code)]
pub fn render_main_menu(app: &mut MathExplorerApp, ctx: &egui::Context, ui: &mut egui::Ui) {
    let modifiers = if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    };

    let quit_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
        ctx,
        "Quit",
        "Quit the application",
        egui_plot::commands::CommandTrigger::Shortcut(modifiers, egui::Key::Q),
        true,
        "Global",
        Some(ui),
        None,
    );

    if quit_triggered {
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    egui::MenuBar::new().ui(ui, |ui| {
        let cmd_str = if cfg!(target_os = "macos") {
            "Cmd+"
        } else {
            "Ctrl+"
        };
        let shift_cmd_str = if cfg!(target_os = "macos") {
            "Cmd+Shift+"
        } else {
            "Ctrl+Shift+"
        };

        ui.menu_button("File", |ui| {
            if ui
                .add(
                    egui::Button::new("Open Workspace Session...")
                        .shortcut_text(format!("{}O", cmd_str)),
                )
                .clicked()
            {
                app.session_manager.trigger_open();
                ui.close();
            }
            if ui
                .add(
                    egui::Button::new("Save Workspace Session")
                        .shortcut_text(format!("{}S", cmd_str)),
                )
                .clicked()
            {
                let state = app.get_app_state();
                let env_json = app.session_manager.create_envelope(state);
                app.session_manager.trigger_save(env_json);
                ui.close();
            }
            if ui
                .add(
                    egui::Button::new("Save Workspace Session As...")
                        .shortcut_text(format!("{}S", shift_cmd_str)),
                )
                .clicked()
            {
                let state = app.get_app_state();
                let env_json = app.session_manager.create_envelope(state);
                app.session_manager.trigger_save_as(env_json);
                ui.close();
            }
            ui.separator();
            if ui
                .add(egui::Button::new("Quit").shortcut_text(format!("{}Q", cmd_str)))
                .clicked()
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });

        ui.menu_button("Edit", |ui| {
            let undo_btn = egui::Button::new("Undo").shortcut_text(format!("{}Z", cmd_str));
            if ui
                .add_enabled(app.session_manager.can_undo(), undo_btn)
                .clicked()
            {
                app.undo();
                ui.close();
            }

            let redo_shortcut = if cfg!(target_os = "macos") {
                format!("{}Z", shift_cmd_str)
            } else {
                format!("{}Y", cmd_str)
            };
            let redo_btn = egui::Button::new("Redo").shortcut_text(redo_shortcut);
            if ui
                .add_enabled(app.session_manager.can_redo(), redo_btn)
                .clicked()
            {
                app.redo();
                ui.close();
            }
        });

        ui.menu_button("View", |ui| {
            let cmd_shortcut = if cfg!(target_os = "macos") {
                "Cmd+K"
            } else {
                "Ctrl+K"
            };
            if ui
                .add(egui::Button::new("🔍 Command Palette...").shortcut_text(cmd_shortcut))
                .clicked()
            {
                app.command_palette.toggle();
                let status_msg = if app.command_palette.is_open {
                    "Command palette opened"
                } else {
                    "Command palette closed"
                };
                ctx.data_mut(|d| {
                    d.insert_temp(egui::Id::new("aria_live_message"), status_msg.to_string());
                });
                ui.close();
            }
            ui.separator();
            ui.menu_button("Viewport Layout", |ui| {
                for layout in [
                    ViewportLayout::Single,
                    ViewportLayout::SplitHorizontal,
                    ViewportLayout::SplitVertical,
                    ViewportLayout::QuadGrid,
                ] {
                    let mut current = app.viewport_layout;
                    if ui
                        .radio_value(&mut current, layout, layout.name())
                        .clicked()
                    {
                        app.set_layout(layout);
                        ui.close();
                    }
                }

                if app.viewport_layout != ViewportLayout::Single {
                    ui.separator();
                    if ui.button("⇄ Swap Panes").clicked() {
                        app.swap_panes();
                        ui.close();
                    }
                }
            });

            if app.viewport_layout != ViewportLayout::Single {
                render_pane_assignments_submenu(app, ctx, ui);
            }

            ui.separator();
            ui.menu_button("Primary Module", |ui| {
                let selected = app.selected_tab;
                if let Some(tab_idx) =
                    crate::app::domain_bar::render_domain_module_submenus(ui, &app.tabs, selected)
                {
                    app.select_tab(ctx, tab_idx);
                    ui.close();
                }
            });
            ui.separator();
            if ui
                .checkbox(
                    &mut app.show_diagnostics_panel,
                    "Issues & Diagnostics Panel",
                )
                .clicked()
            {
                ui.close();
            }
        });

        ui.menu_button("Appearance", |ui| {
            if ui
                .radio_value(&mut app.theme, ThemeMode::Dark, ThemeMode::Dark.name())
                .clicked()
            {
                ctx.set_visuals(app.theme.visuals());
                ui.close();
            }
            if ui
                .radio_value(&mut app.theme, ThemeMode::Light, ThemeMode::Light.name())
                .clicked()
            {
                ctx.set_visuals(app.theme.visuals());
                ui.close();
            }
            if ui
                .radio_value(
                    &mut app.theme,
                    ThemeMode::HighContrast,
                    ThemeMode::HighContrast.name(),
                )
                .clicked()
            {
                ctx.set_visuals(app.theme.visuals());
                ui.close();
            }
        });

        ui.menu_button("Help", |ui| {
            if ui
                .checkbox(&mut app.show_help_menu, "Command Framework")
                .clicked()
            {
                ui.close();
            }
        });
    });

    app.session_manager.render_toast(ui);
}
