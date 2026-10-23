mod session;
mod theme;
mod ui_panels;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use session::WorkspaceEnvelope;
pub use session::WorkspaceSessionManager;
pub use theme::ThemeMode;

use crate::tabs::ExplorerTab;
use eframe::egui;
use federated_registry::{global_registry, Severity, TelemetryEvent};

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
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
    pub session_manager: WorkspaceSessionManager,
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
            session_manager: WorkspaceSessionManager::new(),
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

            // File Menu
            let file_menu = Submenu::new("File", true);
            let open_item = MenuItem::with_id(
                MenuId::new("file_open"),
                "Open Workspace Session...",
                true,
                None,
            );
            let save_item = MenuItem::with_id(
                MenuId::new("file_save"),
                "Save Workspace Session",
                true,
                None,
            );
            let save_as_item = MenuItem::with_id(
                MenuId::new("file_save_as"),
                "Save Workspace Session As...",
                true,
                None,
            );
            file_menu.append(&open_item).unwrap();
            file_menu.append(&save_item).unwrap();
            file_menu.append(&save_as_item).unwrap();
            menu.append(&file_menu).unwrap();

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

    /// Extracts current application state into an `AppState` struct.
    pub fn get_app_state(&self) -> AppState {
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
        AppState {
            selected_tab: self.selected_tab,
            selected_tab_name,
            show_info: self.show_info,
            show_warnings: self.show_warnings,
            show_errors: self.show_errors,
            theme: self.theme,
            tab_states,
        }
    }

    /// Restores application state from an `AppState` struct.
    pub fn apply_app_state(&mut self, app_state: AppState) {
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

    /// Serializes the entire application state (selected tab, panel toggles, individual tab states) to JSON.
    pub fn save_state_to_json(&self) -> String {
        serde_json::to_string(&self.get_app_state()).unwrap_or_default()
    }

    /// Deserializes and restores the application state from JSON.
    pub fn load_state_from_json(&mut self, json_str: &str) {
        match self.session_manager.parse_envelope(json_str) {
            Ok(app_state) => self.apply_app_state(app_state),
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
            }
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

        let mut session_mgr = std::mem::take(&mut self.session_manager);
        session_mgr.register_and_check_shortcuts(ctx, self);
        session_mgr.poll(self, ctx);
        self.session_manager = session_mgr;

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
        ui_panels::render_issues_panel(
            ctx,
            &mut self.diagnostic_events,
            &mut self.show_errors,
            &mut self.show_warnings,
            &mut self.show_info,
        );

        #[cfg(target_os = "macos")]
        {
            if let Ok(event) = muda::MenuEvent::receiver().try_recv() {
                if event.id.0 == "file_open" {
                    self.session_manager.trigger_open();
                } else if event.id.0 == "file_save" {
                    let state = self.get_app_state();
                    let env_json = self.session_manager.create_envelope(state);
                    self.session_manager.trigger_save(env_json);
                } else if event.id.0 == "file_save_as" {
                    let state = self.get_app_state();
                    let env_json = self.session_manager.create_envelope(state);
                    self.session_manager.trigger_save_as(env_json);
                } else if event.id.0.starts_with("tab_") {
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

                    if ui
                        .add(
                            egui::Button::new("Open Workspace Session...")
                                .shortcut_text(format!("{}O", cmd_str)),
                        )
                        .clicked()
                    {
                        self.session_manager.trigger_open();
                        ui.close();
                    }
                    if ui
                        .add(
                            egui::Button::new("Save Workspace Session")
                                .shortcut_text(format!("{}S", cmd_str)),
                        )
                        .clicked()
                    {
                        let state = self.get_app_state();
                        let env_json = self.session_manager.create_envelope(state);
                        self.session_manager.trigger_save(env_json);
                        ui.close();
                    }
                    if ui
                        .add(
                            egui::Button::new("Save Workspace Session As...")
                                .shortcut_text(format!("{}S", shift_cmd_str)),
                        )
                        .clicked()
                    {
                        let state = self.get_app_state();
                        let env_json = self.session_manager.create_envelope(state);
                        self.session_manager.trigger_save_as(env_json);
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

            self.session_manager.render_toast(ui);
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

        ui_panels::render_help_window(ctx, &mut self.show_help_menu);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let json_str = self.save_state_to_json();
        storage.set_string(eframe::APP_KEY, json_str);
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }
}
