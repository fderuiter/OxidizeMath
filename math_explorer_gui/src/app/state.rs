#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::split_view::SplitLayoutMode;
use super::theme::ThemeMode;
use super::viewport::ViewportLayout;

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
    #[serde(default)]
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
    #[serde(default)]
    pub viewport_layout: ViewportLayout,
    #[serde(default)]
    pub pane_tabs: Vec<usize>,
    #[serde(default)]
    pub focused_pane: usize,
    #[serde(default)]
    pub pane_tab_states: Vec<std::collections::HashMap<String, String>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            selected_tab: 0,
            selected_tab_name: None,
            show_info: true,
            show_warnings: true,
            show_errors: true,
            show_diagnostics_panel: false,
            theme: ThemeMode::Dark,
            tab_states: std::collections::HashMap::new(),
            split_layout_mode: SplitLayoutMode::Single,
            secondary_tab: 1,
            secondary_tab_name: None,
            split_ratio: 0.5,
            viewport_layout: ViewportLayout::Single,
            pane_tabs: vec![0, 1, 2, 3],
            focused_pane: 0,
            pane_tab_states: Vec::new(),
        }
    }
}

impl AppState {
    /// Extracts the serialized state snapshot from a live `MathExplorerApp` instance.
    pub fn extract(app: &super::MathExplorerApp) -> Self {
        let mut tab_states = std::collections::HashMap::new();
        for tab in &app.tabs {
            if let Some(tab_str) = tab.save_state() {
                tab_states.insert(tab.name().to_string(), tab_str);
            }
        }
        let selected_tab_name = app.tabs.get(app.selected_tab).map(|t| t.name().to_string());
        let secondary_tab_name = app
            .tabs
            .get(app.secondary_tab)
            .map(|t| t.name().to_string());
        let mut pane_tabs = app.pane_tabs.clone();
        if !pane_tabs.is_empty() {
            pane_tabs[0] = app.selected_tab;
        }
        if pane_tabs.len() > 1 && app.viewport_layout != ViewportLayout::QuadGrid {
            pane_tabs[1] = app.secondary_tab;
        }
        let split_layout_mode = if app.viewport_layout != ViewportLayout::Single {
            match app.viewport_layout {
                ViewportLayout::Single => SplitLayoutMode::Single,
                ViewportLayout::SplitHorizontal => SplitLayoutMode::Horizontal,
                ViewportLayout::SplitVertical => SplitLayoutMode::Vertical,
                ViewportLayout::QuadGrid => SplitLayoutMode::Single,
            }
        } else {
            app.split_layout_mode
        };
        let viewport_layout = if app.viewport_layout != ViewportLayout::Single {
            app.viewport_layout
        } else {
            match split_layout_mode {
                SplitLayoutMode::Horizontal => ViewportLayout::SplitHorizontal,
                SplitLayoutMode::Vertical => ViewportLayout::SplitVertical,
                SplitLayoutMode::Single => ViewportLayout::Single,
            }
        };
        Self {
            selected_tab: app.selected_tab,
            selected_tab_name,
            show_info: app.show_info,
            show_warnings: app.show_warnings,
            show_errors: app.show_errors,
            show_diagnostics_panel: app.show_diagnostics_panel,
            theme: app.theme,
            tab_states,
            split_layout_mode,
            secondary_tab: app.secondary_tab,
            secondary_tab_name,
            split_ratio: app.split_ratio.clamp(0.20, 0.80),
            viewport_layout,
            pane_tabs,
            focused_pane: app.active_focused_pane(),
            pane_tab_states: app.extract_pane_tab_states(),
        }
    }

    /// Restores serialized state into the provided `MathExplorerApp` instance.
    #[allow(clippy::too_many_lines)]
    pub fn apply(self, app: &mut super::MathExplorerApp) {
        app.show_info = self.show_info;
        app.show_warnings = self.show_warnings;
        app.show_errors = self.show_errors;
        app.show_diagnostics_panel = self.show_diagnostics_panel;
        app.theme = self.theme;
        app.split_layout_mode = self.split_layout_mode;
        app.split_ratio = self.split_ratio.clamp(0.20, 0.80);
        app.viewport_layout = self.viewport_layout;

        if app.viewport_layout == ViewportLayout::Single
            && app.split_layout_mode != SplitLayoutMode::Single
        {
            app.viewport_layout = match app.split_layout_mode {
                SplitLayoutMode::Horizontal => ViewportLayout::SplitHorizontal,
                SplitLayoutMode::Vertical => ViewportLayout::SplitVertical,
                SplitLayoutMode::Single => ViewportLayout::Single,
            };
        } else {
            app.sync_layout();
        }

        for tab in &mut app.tabs {
            if let Some(tab_str) = self.tab_states.get(tab.name()) {
                tab.load_state(tab_str);
            }
        }

        if let Some(tab_name) = &self.selected_tab_name {
            if let Some(idx) = app.tabs.iter().position(|t| t.name() == tab_name) {
                app.selected_tab = idx;
            } else if self.selected_tab < app.tabs.len() {
                app.selected_tab = self.selected_tab;
            }
        } else if self.selected_tab < app.tabs.len() {
            app.selected_tab = self.selected_tab;
        }

        if let Some(tab_name) = &self.secondary_tab_name {
            if let Some(idx) = app.tabs.iter().position(|t| t.name() == tab_name) {
                app.secondary_tab = idx;
            } else if self.secondary_tab < app.tabs.len() {
                app.secondary_tab = self.secondary_tab;
            }
        } else if self.secondary_tab < app.tabs.len() {
            app.secondary_tab = self.secondary_tab;
        }

        if self.pane_tabs.is_empty() {
            app.pane_tabs = vec![app.selected_tab, app.secondary_tab, 2, 3];
        } else {
            app.pane_tabs = self.pane_tabs;
        }
        while app.pane_tabs.len() < 4 {
            let next_idx = app.pane_tabs.len().min(app.tabs.len().saturating_sub(1));
            app.pane_tabs.push(next_idx);
        }
        for idx in &mut app.pane_tabs {
            if *idx >= app.tabs.len() {
                *idx = 0;
            }
        }
        if app.viewport_layout == ViewportLayout::Single {
            app.pane_tabs[0] = app.selected_tab;
            app.focused_pane = 0;
        } else {
            if let Some(&first) = app.pane_tabs.first() {
                app.selected_tab = first;
            }
            if let Some(&second) = app.pane_tabs.get(1) {
                app.secondary_tab = second;
            }
            let max_pane = app.viewport_layout.pane_count().saturating_sub(1);
            app.focused_pane = self.focused_pane.min(max_pane);
        }

        app.restore_pane_tab_states(self.pane_tab_states, &self.tab_states);
    }
}
