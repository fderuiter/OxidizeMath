#[cfg(target_os = "macos")]
mod macos_menu;
mod session;
mod split_view;
mod theme;
mod ui_panels;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_diagnostics;
#[cfg(test)]
mod tests_split_view;

#[cfg(target_os = "macos")]
use macos_menu::setup_macos_menu;
#[allow(unused_imports)]
pub use session::WorkspaceEnvelope;
pub use session::WorkspaceSessionManager;
pub use split_view::{SplitLayoutMode, SplitViewState};
pub use theme::ThemeMode;

use crate::tabs::ExplorerTab;
use eframe::egui;
use federated_registry::{global_registry, Severity, TelemetryEvent};

fn default_true() -> bool {
    true
}

fn default_secondary_tab() -> usize {
    1
}

fn default_split_ratio() -> f32 {
    0.5
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct AppState {
    pub selected_tab: usize,
    pub selected_tab_name: Option<String>,
    pub show_info: bool,
    pub show_warnings: bool,
    pub show_errors: bool,
    #[serde(default = "default_true")]
    pub show_diagnostics_panel: bool,
    #[serde(default)]
    pub theme: ThemeMode,
    pub tab_states: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub split_layout_mode: SplitLayoutMode,
    #[serde(default = "default_secondary_tab")]
    pub secondary_tab: usize,
    #[serde(default)]
    pub secondary_tab_name: Option<String>,
    #[serde(default = "default_split_ratio")]
    pub split_ratio: f32,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: 0,
            selected_tab_name: None,
            show_info: true,
            show_warnings: true,
            show_errors: true,
            show_diagnostics_panel: true,
            theme: ThemeMode::Dark,
            tab_states: std::collections::HashMap::new(),
            split_layout_mode: SplitLayoutMode::Single,
            secondary_tab: 1,
            secondary_tab_name: None,
            split_ratio: 0.5,
        }
    }
}

pub struct MathExplorerApp {
    tabs: Vec<Box<dyn ExplorerTab>>,
    selected_tab: usize,
    pub split_layout_mode: SplitLayoutMode,
    pub secondary_tab: usize,
    pub split_ratio: f32,
    split_view_state: SplitViewState,
    diagnostic_events: Vec<TelemetryEvent>,
    pub show_diagnostics_panel: bool,
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
            split_layout_mode: SplitLayoutMode::Single,
            secondary_tab: 1,
            split_ratio: 0.5,
            split_view_state: SplitViewState::new(),
            diagnostic_events: Vec::new(),
            show_diagnostics_panel: true,
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
    /// Swaps the primary and secondary pane active tabs immediately.
    pub fn swap_panes(&mut self) {
        if self.tabs.len() > 1 {
            std::mem::swap(&mut self.selected_tab, &mut self.secondary_tab);
        }
    }
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();

        if let Some(storage) = cc.storage {
            if let Some(state_str) = storage.get_string(eframe::APP_KEY) {
                app.load_state_from_json(&state_str);
            }
        }

        #[cfg(target_os = "macos")]
        setup_macos_menu(&app);

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
        let secondary_tab_name = self
            .tabs
            .get(self.secondary_tab)
            .map(|t| t.name().to_string());
        AppState {
            selected_tab: self.selected_tab,
            selected_tab_name,
            show_info: self.show_info,
            show_warnings: self.show_warnings,
            show_errors: self.show_errors,
            show_diagnostics_panel: self.show_diagnostics_panel,
            theme: self.theme,
            tab_states,
            split_layout_mode: self.split_layout_mode,
            secondary_tab: self.secondary_tab,
            secondary_tab_name,
            split_ratio: self.split_ratio.clamp(0.20, 0.80),
        }
    }

    /// Restores application state from an `AppState` struct.
    pub fn apply_app_state(&mut self, app_state: AppState) {
        self.show_info = app_state.show_info;
        self.show_warnings = app_state.show_warnings;
        self.show_errors = app_state.show_errors;
        self.show_diagnostics_panel = app_state.show_diagnostics_panel;
        self.theme = app_state.theme;
        self.split_layout_mode = app_state.split_layout_mode;
        self.split_ratio = app_state.split_ratio.clamp(0.20, 0.80);

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

        if let Some(tab_name) = &app_state.secondary_tab_name {
            if let Some(idx) = self.tabs.iter().position(|t| t.name() == tab_name) {
                self.secondary_tab = idx;
            } else if app_state.secondary_tab < self.tabs.len() {
                self.secondary_tab = app_state.secondary_tab;
            }
        } else if app_state.secondary_tab < self.tabs.len() {
            self.secondary_tab = app_state.secondary_tab;
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

    /// Returns the names of all registered tabs.
    #[allow(dead_code)]
    pub fn tab_names(&self) -> Vec<&'static str> {
        self.tabs.iter().map(|t| t.name()).collect()
    }

    /// Restores the previous application state from the undo history stack.
    pub fn undo(&mut self) {
        if !self.session_manager.can_undo() {
            return;
        }
        if let Some(prev) = self.session_manager.pop_undo() {
            let current = self.get_app_state();
            self.session_manager.push_redo(current);
            self.apply_app_state(prev);
            self.session_manager.set_is_restoring(true);
        }
    }

    /// Re-applies an undone application state from the redo history stack.
    pub fn redo(&mut self) {
        if !self.session_manager.can_redo() {
            return;
        }
        if let Some(next) = self.session_manager.pop_redo() {
            let current = self.get_app_state();
            self.session_manager.push_undo(current);
            self.apply_app_state(next);
            self.session_manager.set_is_restoring(true);
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
                } else if event.id.0 == "edit_undo" {
                    self.undo();
                } else if event.id.0 == "edit_redo" {
                    self.redo();
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
                } else if event.id.0 == "view_toggle_diagnostics" {
                    self.show_diagnostics_panel = !self.show_diagnostics_panel;
                } else if event.id.0 == "view_split_single" {
                    self.split_layout_mode = SplitLayoutMode::Single;
                } else if event.id.0 == "view_split_horizontal" {
                    self.split_layout_mode = SplitLayoutMode::Horizontal;
                } else if event.id.0 == "view_split_vertical" {
                    self.split_layout_mode = SplitLayoutMode::Vertical;
                } else if event.id.0 == "view_split_swap" {
                    self.swap_panes();
                }
            }
        }

        #[cfg(target_os = "macos")]
        if self.session_manager.toast().is_some() {
            egui::TopBottomPanel::top("toast_panel").show(ctx, |ui| {
                self.session_manager.render_toast(ui);
            });
        }

        // Render Menu Bar (Native fallback for non-macOS or inside the app)
        #[cfg(not(target_os = "macos"))]
        egui::TopBottomPanel::top("main_menu").show(ctx, |ui| {
            ui_panels::render_main_menu(self, ctx, ui);
        });

        // Render Active Tab(s)
        match self.split_layout_mode {
            SplitLayoutMode::Single => {
                if let Some(tab) = self.tabs.get_mut(self.selected_tab) {
                    tab.show(ctx, frame);
                } else {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        ui.centered_and_justified(|ui| {
                            ui.label("No module selected.");
                        });
                    });
                }
            }
            SplitLayoutMode::Horizontal | SplitLayoutMode::Vertical => {
                let mut split_state = std::mem::take(&mut self.split_view_state);
                split_state.render_split(self, ctx, frame);
                self.split_view_state = split_state;
            }
        }

        ui_panels::render_help_window(ctx, &mut self.show_help_menu);

        let current_app_state = self.get_app_state();
        self.session_manager
            .track_state_changes(ctx, current_app_state);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let json_str = self.save_state_to_json();
        storage.set_string(eframe::APP_KEY, json_str);
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }
}
