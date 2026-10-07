use eframe::egui;

use super::MathExplorerApp;

impl MathExplorerApp {
    pub(super) fn render_help_menu(&mut self, ctx: &egui::Context) {
        if !self.show_help_menu {
            return;
        }

        egui::Window::new("Help Menu")
            .open(&mut self.show_help_menu)
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
