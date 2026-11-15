#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::split_view::SplitLayoutMode;
use super::theme::ThemeMode;
use super::viewport::ViewportLayout;

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
    #[serde(default)]
    pub viewport_layout: ViewportLayout,
    #[serde(default)]
    pub pane_tabs: Vec<usize>,
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
            viewport_layout: ViewportLayout::Single,
            pane_tabs: vec![0, 1, 2, 3],
        }
    }
}
