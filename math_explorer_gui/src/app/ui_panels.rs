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
) {
    egui::TopBottomPanel::bottom("issues_panel")
        .resizable(true)
        .min_height(100.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Issues & Diagnostics");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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
                for event in diagnostic_events {
                    let show = match event.severity {
                        Severity::Info => *show_info,
                        Severity::Warning => *show_warnings,
                        Severity::Error | Severity::Fatal => *show_errors,
                    };
                    if !show {
                        continue;
                    }

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
