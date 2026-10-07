mod theme;

#[cfg(test)]
mod tests;

pub use theme::ThemeMode;

use crate::tabs::ExplorerTab;
use eframe::egui;
use federated_registry::{global_registry, Severity, TelemetryEvent};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AppState {
    pub selected_tab: usize,
    pub selected_tab_name: Option<String>,
    pub show_info: bool,
    pub show_warnings: bool,
    pub show_errors: bool,
    #[serde(default)]
    pub theme: ThemeMode,
    pub tab_states: std::collections::HashMap<String, String>,
}

pub struct MathExplorerApp {
    tabs: Vec<Box<dyn ExplorerTab>>,
    selected_tab: usize,
    diagnostic_events: Vec<TelemetryEvent>,
    show_info: bool,
    show_warnings: bool,
    show_errors: bool,
    show_help_menu: bool,
    pub theme: ThemeMode,
}

impl Default for MathExplorerApp {
    #[allow(clippy::vec_init_then_push)]
    fn default() -> Self {
        let tabs: Vec<Box<dyn ExplorerTab>> = crate::tabs::instantiate_tabs();

        Self {
            tabs,
            selected_tab: 0,
            diagnostic_events: Vec::new(),
            show_info: true,
            show_warnings: true,
            show_errors: true,
            show_help_menu: false,
            theme: ThemeMode::Dark,
        }
    }
}

impl MathExplorerApp {
    #[allow(clippy::vec_init_then_push)]
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();

        if let Some(storage) = cc.storage {
            if let Some(state_str) = storage.get_string(eframe::APP_KEY) {
                app.load_state_from_json(&state_str);
            }
        }

        #[cfg(target_os = "macos")]
        {
            use muda::{Menu, MenuId, MenuItem, PredefinedMenuItem, Submenu};
            let menu = Menu::new();

            // App Menu
            let app_menu = Submenu::new("Math Explorer", true);
            app_menu.append(&PredefinedMenuItem::quit(None)).unwrap();
            menu.append(&app_menu).unwrap();

            // View Menu
            let view_menu = Submenu::new("View", true);
            for (i, tab) in app.tabs.iter().enumerate() {
                let item =
                    MenuItem::with_id(MenuId::new(format!("tab_{}", i)), tab.name(), true, None);
                view_menu.append(&item).unwrap();
            }
            menu.append(&view_menu).unwrap();

            // Appearance Menu
            let appearance_menu = Submenu::new("Appearance", true);
            let dark_item = MenuItem::with_id(MenuId::new("theme_dark"), "Dark", true, None);
            let light_item = MenuItem::with_id(MenuId::new("theme_light"), "Light", true, None);
            let hc_item = MenuItem::with_id(
                MenuId::new("theme_high_contrast"),
                "High Contrast",
                true,
                None,
            );
            appearance_menu.append(&dark_item).unwrap();
            appearance_menu.append(&light_item).unwrap();
            appearance_menu.append(&hc_item).unwrap();
            menu.append(&appearance_menu).unwrap();

            menu.init_for_nsapp();
        }

        app
    }

    /// Serializes the entire application state (selected tab, panel toggles, individual tab states) to JSON.
    pub fn save_state_to_json(&self) -> String {
        let mut tab_states = std::collections::HashMap::new();
        for tab in &self.tabs {
            if let Some(tab_str) = tab.save_state() {
                tab_states.insert(tab.name().to_string(), tab_str);
            }
        }
        let selected_tab_name = self
            .tabs
            .get(self.selected_tab)
            .map(|t| t.name().to_string());
        let app_state = AppState {
            selected_tab: self.selected_tab,
            selected_tab_name,
            show_info: self.show_info,
            show_warnings: self.show_warnings,
            show_errors: self.show_errors,
            theme: self.theme,
            tab_states,
        };
        serde_json::to_string(&app_state).unwrap_or_default()
    }

    /// Deserializes and restores the application state from JSON.
    pub fn load_state_from_json(&mut self, json_str: &str) {
        let app_state: AppState = match serde_json::from_str(json_str) {
            Ok(state) => state,
            Err(err) => {
                global_registry().emit(TelemetryEvent {
                    source: "MathExplorerApp".to_string(),
                    severity: Severity::Warning,
                    message: format!(
                        "Failed to parse saved session state, resetting to defaults: {}",
                        err
                    ),
                    metadata: std::collections::HashMap::new(),
                    thread_name: None,
                });
                return;
            }
        };

        self.show_info = app_state.show_info;
        self.show_warnings = app_state.show_warnings;
        self.show_errors = app_state.show_errors;
        self.theme = app_state.theme;

        for tab in &mut self.tabs {
            if let Some(tab_str) = app_state.tab_states.get(tab.name()) {
                tab.load_state(tab_str);
            }
        }

        if let Some(tab_name) = &app_state.selected_tab_name {
            if let Some(idx) = self.tabs.iter().position(|t| t.name() == tab_name) {
                self.selected_tab = idx;
            } else if app_state.selected_tab < self.tabs.len() {
                self.selected_tab = app_state.selected_tab;
            }
        } else if app_state.selected_tab < self.tabs.len() {
            self.selected_tab = app_state.selected_tab;
        }
    }
}

impl eframe::App for MathExplorerApp {
    #[allow(clippy::too_many_lines)]
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if ctx.style().visuals != self.theme.visuals() {
            ctx.set_visuals(self.theme.visuals());
        }

        if let Some(msg) = ctx.data_mut(|d| {
            let msg = d.get_temp::<String>(egui::Id::new("aria_live_message"));
            d.remove::<String>(egui::Id::new("aria_live_message"));
            msg
        }) {
            crate::accessibility::announce_status(&msg);
        }

        if let Some(msg) = ctx.data_mut(|d| {
            let msg = d.get_temp::<String>(egui::Id::new("aria_live_assertive_message"));
            d.remove::<String>(egui::Id::new("aria_live_assertive_message"));
            msg
        }) {
            crate::accessibility::announce_status_with_priority(&msg, "assertive");
        }

        // Global Keyboard Shortcut Listener for Cheatsheet Modal
        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };

        let slash_triggered = egui_plot::commands::CommandRegistryData::register_and_check(
            ctx,
            "Command Cheatsheet",
            "Toggle global command cheatsheet modal",
            egui_plot::commands::CommandTrigger::Shortcut(modifiers, egui::Key::Slash),
            true,
            "Global",
            None,
            None,
        );

        let question_triggered = !ctx.wants_keyboard_input()
            && ctx.input_mut(|i| {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Questionmark)
                    || i.consume_key(egui::Modifiers::SHIFT, egui::Key::Questionmark)
            });

        if slash_triggered || question_triggered {
            self.show_help_menu = !self.show_help_menu;
            let status_msg = if self.show_help_menu {
                "Hotkey overlay opened"
            } else {
                "Hotkey overlay closed"
            };
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("aria_live_message"), status_msg.to_string());
            });
        }

        // Fetch new events
        let new_events = global_registry().try_recv_all();
        for event in &new_events {
            if matches!(event.severity, Severity::Error | Severity::Fatal) {
                crate::accessibility::announce_status_with_priority(
                    &format!("{}: {}", event.severity, event.message),
                    "assertive",
                );
            }
        }
        self.diagnostic_events.extend(new_events);

        // Issues & Diagnostics Panel
        egui::TopBottomPanel::bottom("issues_panel")
            .resizable(true)
            .min_height(100.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Issues & Diagnostics");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Clear").clicked() {
                            self.diagnostic_events.clear();
                        }
                        ui.checkbox(&mut self.show_errors, "Errors/Fatal");
                        ui.checkbox(&mut self.show_warnings, "Warnings");
                        ui.checkbox(&mut self.show_info, "Info");
                    });
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for event in &self.diagnostic_events {
                        let show = match event.severity {
                            Severity::Info => self.show_info,
                            Severity::Warning => self.show_warnings,
                            Severity::Error | Severity::Fatal => self.show_errors,
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
                                        egui::RichText::new(format!("(Thread: {})", thread))
                                            .italics(),
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

        #[cfg(target_os = "macos")]
        {
            if let Ok(event) = muda::MenuEvent::receiver().try_recv() {
                if event.id.0.starts_with("tab_") {
                    if let Ok(idx) = event.id.0[4..].parse::<usize>() {
                        self.selected_tab = idx;
                    }
                } else if event.id.0 == "help_commands" {
                    self.show_help_menu = !self.show_help_menu;
                } else if event.id.0 == "theme_dark" {
                    self.theme = ThemeMode::Dark;
                    ctx.set_visuals(self.theme.visuals());
                } else if event.id.0 == "theme_light" {
                    self.theme = ThemeMode::Light;
                    ctx.set_visuals(self.theme.visuals());
                } else if event.id.0 == "theme_high_contrast" {
                    self.theme = ThemeMode::HighContrast;
                    ctx.set_visuals(self.theme.visuals());
                }
            }
        }

        // Render Menu Bar (Native fallback for non-macOS or inside the app)
        #[cfg(not(target_os = "macos"))]
        egui::TopBottomPanel::top("main_menu").show(ctx, |ui| {
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
                ui.menu_button("File", |ui| {
                    let shortcut_text = if cfg!(target_os = "macos") {
                        "Cmd+Q"
                    } else {
                        "Ctrl+Q"
                    };
                    if ui
                        .add(egui::Button::new("Quit").shortcut_text(shortcut_text))
                        .clicked()
                    {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    for (i, tab) in self.tabs.iter().enumerate() {
                        if ui
                            .radio_value(&mut self.selected_tab, i, tab.name())
                            .clicked()
                        {
                            ui.close();
                        }
                    }
                });
                ui.menu_button("Appearance", |ui| {
                    if ui
                        .radio_value(&mut self.theme, ThemeMode::Dark, ThemeMode::Dark.name())
                        .clicked()
                    {
                        ctx.set_visuals(self.theme.visuals());
                        ui.close();
                    }
                    if ui
                        .radio_value(&mut self.theme, ThemeMode::Light, ThemeMode::Light.name())
                        .clicked()
                    {
                        ctx.set_visuals(self.theme.visuals());
                        ui.close();
                    }
                    if ui
                        .radio_value(
                            &mut self.theme,
                            ThemeMode::HighContrast,
                            ThemeMode::HighContrast.name(),
                        )
                        .clicked()
                    {
                        ctx.set_visuals(self.theme.visuals());
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui
                        .checkbox(&mut self.show_help_menu, "Command Framework")
                        .clicked()
                    {
                        ui.close();
                    }
                });
            });
        });

        // Render Active Tab
        if let Some(tab) = self.tabs.get_mut(self.selected_tab) {
            tab.show(ctx, frame);
        } else {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label("No module selected.");
                });
            });
        }

        if self.show_help_menu {
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

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let json_str = self.save_state_to_json();
        storage.set_string(eframe::APP_KEY, json_str);
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }
}
}
