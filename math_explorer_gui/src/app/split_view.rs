#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::MathExplorerApp;
use eframe::egui;

/// Layout mode for the main simulation viewport container.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitLayoutMode {
    /// Single viewport rendering one active module.
    #[default]
    Single,
    /// Dual viewport split horizontally (left and right panes).
    Horizontal,
    /// Dual viewport split vertically (top and bottom panes).
    Vertical,
}

impl SplitLayoutMode {
    /// Returns the human-readable display label for the layout mode.
    #[cfg(test)]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Single => "Single View",
            Self::Horizontal => "Horizontal Split",
            Self::Vertical => "Vertical Split",
        }
    }
}

/// Persistent state manager for dual-pane split view layout composition.
///
/// Multi-pane rendering delegates to the authoritative viewport manager to
/// prevent allocating duplicate egui contexts and ensure synchronized layouts.
#[derive(Default)]
pub struct SplitViewState {
    pub is_dragging_divider: bool,
}

impl SplitViewState {
    /// Creates a new split view state instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Renders dual simulation viewports delegating to the unified viewport manager.
    pub fn render_split(
        &mut self,
        app: &mut MathExplorerApp,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
    ) {
        crate::app::viewport::render_multi_pane_manager(app, ctx, frame);
    }
}
