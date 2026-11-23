pub mod domain_bar;
#[cfg(target_os = "macos")]
mod macos_menu;
#[cfg(target_os = "macos")]
mod macos_menu_view;
pub mod palette;
mod pane_clone;
mod pane_ops;
mod pane_slots;
pub(crate) mod session;
mod share_link;
pub mod shared_json;
mod state;
pub mod theme;
pub(crate) mod ui_panels;
pub(crate) mod viewport;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_adaptive_chrome;
#[cfg(test)]
mod tests_command_palette;
#[cfg(test)]
mod tests_diagnostics;
#[cfg(test)]
mod tests_domain_bar;
#[cfg(test)]
mod tests_focus_routing;
#[cfg(test)]
mod tests_multi_pane;
#[cfg(test)]
mod tests_multi_pane_a11y;
#[cfg(test)]
mod tests_multi_pane_plot_live;
#[cfg(test)]
mod tests_pane_clone_sync;
#[cfg(test)]
mod tests_pane_isolation;
#[cfg(test)]
mod tests_secondary_menus;
#[cfg(test)]
mod tests_session_lifecycle;
#[cfg(test)]
mod tests_shortcuts;
#[cfg(test)]
mod tests_split_view;
#[cfg(test)]
mod tests_tab_cache_perf;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

#[allow(unused_imports)]
pub use domain_bar::AcademicDomain;
#[allow(unused_imports)]
pub use session::{SessionLifecycleState, WorkspaceEnvelope, WorkspaceSessionManager};
pub use shared_json::SharedTabJson;
pub use state::{AppState, SplitLayoutMode};
#[allow(unused_imports)]
pub use theme::{
    contrast_ratio, contrasting_text_color, relative_luminance, SemanticPalette, ThemeMode,
};
pub use viewport::compare_sync::ComparisonSyncConfig;
pub use viewport::ViewportLayout;

use crate::command_palette::CommandPalette;
use crate::tabs::ExplorerTab;
use eframe::egui;
#[cfg(test)]
use federated_registry::global_registry;
use federated_registry::{Severity, TelemetryEvent};

pub struct MathExplorerApp {
    pub(super) tabs: Vec<Box<dyn ExplorerTab>>,
    pub selected_tab: usize,
    pub split_layout_mode: SplitLayoutMode,
    pub secondary_tab: usize,
    pub split_ratio: f32,
    diagnostic_events: Vec<TelemetryEvent>,
    pub show_diagnostics_panel: bool,
    pub(super) show_info: bool,
    pub(super) show_warnings: bool,
    pub(super) show_errors: bool,
    pub(super) show_help_menu: bool,
    pub theme: ThemeMode,
    pub(super) last_synced_theme: Option<ThemeMode>,
    pub command_palette: CommandPalette,
    pub session_manager: WorkspaceSessionManager,
    pub viewport_layout: ViewportLayout,
    pub pane_tabs: Vec<usize>,
    pub focused_pane: usize,
    pane_slots: pane_slots::SecondaryPaneSlots,
    pub is_dragging_divider: bool,
    /// Horizontal and vertical split ratios `(x, y)` for `ViewportLayout::QuadGrid`.
    pub quad_split_ratios: (f32, f32),
    pub(super) dragging_quad_axes: (bool, bool),
    tab_state_cache: state::TabStateCache,
    last_frame_had_input: bool,
    pending_focus_prune: bool,
    pub compare_sync: ComparisonSyncConfig,
    pub(super) url_sync: crate::url_state::UrlSyncController,
}

impl Default for MathExplorerApp {
    #[allow(clippy::vec_init_then_push)]
    fn default() -> Self {
        crate::accessibility::ensure_plot_live_bridge();
        crate::framework::ensure_custom_fit_bridge();
        let tabs: Vec<Box<dyn ExplorerTab>> = crate::tabs::instantiate_tabs();

        Self {
            tabs,
            selected_tab: 0,
            split_layout_mode: SplitLayoutMode::Single,
            secondary_tab: 1,
            split_ratio: 0.5,
            diagnostic_events: Vec::new(),
            show_diagnostics_panel: false,
            show_info: true,
            show_warnings: true,
            show_errors: true,
            show_help_menu: false,
            theme: ThemeMode::Dark,
            last_synced_theme: Some(ThemeMode::Dark),
            command_palette: CommandPalette::default(),
            session_manager: WorkspaceSessionManager::new(),
            viewport_layout: ViewportLayout::Single,
            pane_tabs: vec![0, 1, 2, 3],
            focused_pane: 0,
            pane_slots: pane_slots::SecondaryPaneSlots::default(),
            is_dragging_divider: false,
            quad_split_ratios: (0.5, 0.5),
            dragging_quad_axes: (false, false),
            tab_state_cache: state::TabStateCache::default(),
            last_frame_had_input: false,
            pending_focus_prune: false,
            compare_sync: ComparisonSyncConfig::default(),
            url_sync: crate::url_state::UrlSyncController::default(),
        }
    }
}

impl MathExplorerApp {
    /// Swaps the primary and secondary pane active tabs and their per-pane states immediately.
    pub fn swap_panes(&mut self) {
        self.swap_pane_pair(0, 1);
    }

    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();
        theme::init_startup_theme(&mut app, cc);
        app.hydrate_from_browser_url(&cc.egui_ctx);

        #[cfg(target_os = "macos")]
        macos_menu::setup_macos_menu(&app);

        app
    }

    /// Sets the assigned tab index for a specific pane.
    pub fn set_pane_tab(&mut self, pane_idx: usize, tab_idx: usize) {
        if pane_idx < self.pane_tabs.len() && tab_idx < self.tabs.len() {
            self.pane_tabs[pane_idx] = tab_idx;
            if pane_idx == 0 {
                self.selected_tab = tab_idx;
            } else if pane_idx == 1 {
                self.secondary_tab = tab_idx;
            }
            self.on_pane_tab_assigned(pane_idx, tab_idx);
        }
    }

    /// Sets the authoritative viewport layout preset and synchronizes legacy layout mode.
    pub fn set_layout(&mut self, layout: ViewportLayout) {
        self.seed_uninitialized_panes_on_initial_split(layout);
        self.viewport_layout = layout;
        self.sync_layout();
        if !self.pane_tabs.is_empty() {
            self.pane_tabs[0] = self.selected_tab;
        }
        if layout != ViewportLayout::Single && self.pane_tabs.len() > 1 {
            self.pane_tabs[1] = self.secondary_tab;
        }
        self.prune_inactive_pane_slots(layout.pane_count());
    }

    /// Synchronizes legacy `split_layout_mode` with current `viewport_layout`.
    pub fn sync_layout(&mut self) {
        self.split_layout_mode = match self.viewport_layout {
            ViewportLayout::Single => SplitLayoutMode::Single,
            ViewportLayout::SplitHorizontal => SplitLayoutMode::Horizontal,
            ViewportLayout::SplitVertical => SplitLayoutMode::Vertical,
            ViewportLayout::QuadGrid => SplitLayoutMode::Single,
        };
    }

    /// Extracts current application state into an `AppState` struct.
    pub fn get_app_state(&self) -> AppState {
        AppState::extract(self)
    }

    /// Restores application state from an `AppState` struct.
    pub fn apply_app_state(&mut self, app_state: AppState) {
        app_state.apply(self);
    }

    /// Serializes the entire application state (selected tab, panel toggles, individual tab states) to JSON.
    pub fn save_state_to_json(&self) -> String {
        if self.session_manager.lifecycle_state == SessionLifecycleState::DragEditing {
            self.clear_active_pane_caches();
        }
        serde_json::to_string(&self.get_app_state()).unwrap_or_default()
    }

    /// Deserializes and restores the application state from JSON.
    pub fn load_state_from_json(&mut self, json_str: &str) {
        match self.session_manager.parse_envelope(json_str) {
            Ok(app_state) => self.apply_app_state(app_state),
            Err(err) => {
                crate::telemetry::emit_telemetry(
                    "MathExplorerApp",
                    Severity::Warning,
                    format!(
                        "Failed to parse saved session state, resetting to defaults: {}",
                        err
                    ),
                    std::collections::HashMap::new(),
                );
            }
        }
    }

    /// Selects an active tab by index, synchronizing primary pane and emitting accessibility announcements.
    pub fn select_tab(&mut self, ctx: &egui::Context, tab_idx: usize) {
        domain_bar::select_tab(self, ctx, tab_idx);
    }

    /// Assigns a simulation module to a specific viewport pane, synchronizing state and announcing the change.
    pub fn select_pane_tab(&mut self, ctx: &egui::Context, pane_idx: usize, tab_idx: usize) {
        domain_bar::select_pane_tab(self, ctx, pane_idx, tab_idx);
    }

    /// Restores the previous application state from the workspace undo history stack.
    pub fn undo(&mut self) {
        let mut sm = std::mem::take(&mut self.session_manager);
        sm.execute_undo(self);
        self.session_manager = sm;
    }

    /// Re-applies an undone application state from the workspace redo history stack.
    pub fn redo(&mut self) {
        let mut sm = std::mem::take(&mut self.session_manager);
        sm.execute_redo(self);
        self.session_manager = sm;
    }
}

impl eframe::App for MathExplorerApp {
    #[allow(clippy::too_many_lines)]
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        crate::reflective_ui::slider::reset_frame_sweep_flags(ctx);
        ctx.data_mut(|d| {
            d.insert_temp(
                egui::Id::new("CMD_REGISTRY"),
                egui_plot::commands::CommandRegistryData::default(),
            );
        });

        theme::sync_frame_theme(self, ctx, frame);

        crate::accessibility::ensure_plot_live_bridge();
        crate::accessibility::sync_context_live_messages(ctx, true);

        self.flush_stale_pane_focus(ctx);
        let mut session_mgr = std::mem::take(&mut self.session_manager);
        session_mgr.register_and_check_shortcuts(ctx, self);
        session_mgr.poll(self, ctx);
        self.session_manager = session_mgr;
        theme::sync_frame_theme(self, ctx, frame);

        viewport::focus::register_and_handle_pane_focus_shortcuts(self, ctx);

        // Global Keyboard Shortcut Listener for Cheatsheet Modal
        let modifiers = if cfg!(target_os = "macos") {
            egui::Modifiers::MAC_CMD
        } else {
            egui::Modifiers::CTRL
        };

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

        self.poll_share_shortcut_and_clipboard(ctx);
        self.poll_duplicate_pane_shortcut(ctx);

        // Fetch new events
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
        macos_menu::handle_macos_menu_events(self, ctx);

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

        // Render Persistent Categorized Domain Switcher Bar
        egui::TopBottomPanel::top("domain_switcher_bar")
            .exact_height(34.0)
            .show(ctx, |ui| {
                domain_bar::render_domain_switcher_bar(self, ctx, ui);
            });

        // Render Active Tab(s) or Multi-Pane Viewports
        let active_layout = if self.viewport_layout != ViewportLayout::Single {
            self.sync_layout();
            self.viewport_layout
        } else if self.split_layout_mode != SplitLayoutMode::Single {
            let layout = match self.split_layout_mode {
                SplitLayoutMode::Horizontal => ViewportLayout::SplitHorizontal,
                SplitLayoutMode::Vertical => ViewportLayout::SplitVertical,
                SplitLayoutMode::Single => ViewportLayout::Single,
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
        };

        self.prune_inactive_pane_slots(active_layout.pane_count());
        self.sync_workspace_history_bridge(ctx);

        if active_layout != ViewportLayout::Single {
            viewport::render_multi_pane_manager(self, ctx, frame);
        } else {
            viewport::render_single_pane_viewport(self, ctx, frame);
        }
        self.drain_workspace_history_bridge(ctx);

        ui_panels::render_help_window(ctx, &mut self.show_help_menu);

        // Render Command Palette Overlay
        self.apply_command_palette_selection(ctx);
        crate::accessibility::sync_context_live_messages(ctx, false);

        self.last_frame_had_input =
            state::has_frame_input(ctx) || crate::reflective_ui::slider::has_sweep_mutation(ctx);
        let mut session_manager = std::mem::take(&mut self.session_manager);
        session_manager.track_frame_changes(ctx, self);
        self.session_manager = session_manager;
        theme::sync_frame_theme(self, ctx, frame);
        self.tick_url_sync(ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.clear_active_pane_caches();
        let json_str = self.save_state_to_json();
        storage.set_string(eframe::APP_KEY, json_str);
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(5)
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}
