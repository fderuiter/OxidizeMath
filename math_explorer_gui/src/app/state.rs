#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::theme::ThemeMode;
use super::viewport::ViewportLayout;
use crate::tabs::ExplorerTab;
use eframe::egui;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

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

fn default_secondary_tab() -> usize {
    1
}

fn default_split_ratio() -> f32 {
    0.5
}

fn default_quad_split_ratios() -> (f32, f32) {
    (0.5, 0.5)
}

fn is_all_false_links(links: &[bool]) -> bool {
    links.iter().all(|&linked| !linked)
}

pub(super) fn has_frame_input(ctx: &egui::Context) -> bool {
    ctx.input(|i| {
        i.pointer.any_down()
            || i.pointer.any_pressed()
            || i.pointer.any_released()
            || i.any_touches()
            || i.raw_scroll_delta != egui::Vec2::ZERO
            || i.smooth_scroll_delta != egui::Vec2::ZERO
            || (i.zoom_delta() - 1.0).abs() > f32::EPSILON
            || i.events.iter().any(|e| {
                !matches!(
                    e,
                    egui::Event::PointerMoved(_)
                        | egui::Event::PointerGone
                        | egui::Event::WindowFocused(_)
                )
            })
    })
}

/// Interior-mutable per-tab JSON cache avoiding redundant `ExplorerTab::save_state` calls.
#[derive(Default)]
pub(super) struct TabStateCache {
    entries: RefCell<HashMap<String, (u64, Option<String>)>>,
}

impl TabStateCache {
    pub(super) fn clear(&self) {
        self.entries.borrow_mut().clear();
    }

    pub(super) fn invalidate_tab(&self, name: &str) {
        self.entries.borrow_mut().remove(name);
    }

    pub(super) fn extract_tab_json(
        &self,
        tab: &dyn ExplorerTab,
        is_active_pane: bool,
        had_input: bool,
    ) -> Option<String> {
        let rev = tab.state_revision();
        let name = tab.name();
        {
            let guard = self.entries.borrow();
            if let Some((cached_rev, cached_json)) = guard.get(name) {
                let can_reuse = if !is_active_pane || rev > 0 {
                    *cached_rev == rev
                } else {
                    !had_input
                };
                if can_reuse {
                    return cached_json.clone();
                }
            }
        }
        let fresh = tab.save_state();
        self.entries
            .borrow_mut()
            .insert(name.to_string(), (rev, fresh.clone()));
        fresh
    }
}

/// Zero-allocation O(1) state fingerprint used to skip idle and `DragEditing` workspace extractions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AppStateFingerprint {
    selected_tab: usize,
    secondary_tab: usize,
    flags: u8,
    theme_discriminant: u8,
    split_mode_discriminant: u8,
    viewport_discriminant: u8,
    split_ratio_bits: u32,
    quad_split_ratio_bits: (u32, u32),
    focused_pane: usize,
    pane_tabs: [usize; 4],
    pane_bus_links_mask: u8,
    primary_revisions_hash: u64,
    secondary_revisions_hash: u64,
    pub(crate) all_revisions_tracked: bool,
}

impl AppStateFingerprint {
    pub(crate) fn capture(app: &super::MathExplorerApp) -> Self {
        let mut primary_hasher = std::collections::hash_map::DefaultHasher::new();
        let mut all_tracked = true;
        let active_primary = app.active_primary_tab_index();
        for (idx, tab) in app.tabs.iter().enumerate() {
            let rev = tab.state_revision();
            if idx == active_primary && rev == 0 {
                all_tracked = false;
            }
            idx.hash(&mut primary_hasher);
            rev.hash(&mut primary_hasher);
        }
        let (secondary_hash, secondary_tracked) = app.secondary_slots_revision_hash();
        all_tracked &= secondary_tracked;

        let mut pane_tabs = [0_usize; 4];
        for (i, slot) in pane_tabs.iter_mut().enumerate() {
            *slot = app.pane_tabs.get(i).copied().unwrap_or(i);
        }
        pane_tabs[0] = app.selected_tab;
        if app.viewport_layout != ViewportLayout::QuadGrid {
            pane_tabs[1] = app.secondary_tab;
        }

        let flags = u8::from(app.show_info)
            | (u8::from(app.show_warnings) << 1)
            | (u8::from(app.show_errors) << 2)
            | (u8::from(app.show_diagnostics_panel) << 3);

        Self {
            selected_tab: app.selected_tab,
            secondary_tab: app.secondary_tab,
            flags,
            theme_discriminant: match app.theme {
                ThemeMode::Dark => 0,
                ThemeMode::Light => 1,
                ThemeMode::HighContrast => 2,
            },
            split_mode_discriminant: match app.split_layout_mode {
                SplitLayoutMode::Single => 0,
                SplitLayoutMode::Horizontal => 1,
                SplitLayoutMode::Vertical => 2,
            },
            viewport_discriminant: match app.viewport_layout {
                ViewportLayout::Single => 0,
                ViewportLayout::SplitHorizontal => 1,
                ViewportLayout::SplitVertical => 2,
                ViewportLayout::QuadGrid => 3,
            },
            split_ratio_bits: app.split_ratio.clamp(0.20, 0.80).to_bits(),
            quad_split_ratio_bits: (
                app.quad_split_ratios.0.clamp(0.20, 0.80).to_bits(),
                app.quad_split_ratios.1.clamp(0.20, 0.80).to_bits(),
            ),
            focused_pane: app.active_focused_pane(),
            pane_tabs,
            pane_bus_links_mask: pack_pane_links_mask(app.pane_bus_links),
            primary_revisions_hash: primary_hasher.finish(),
            secondary_revisions_hash: secondary_hash,
            all_revisions_tracked: all_tracked,
        }
    }
}

const fn pack_pane_links_mask(links: [bool; 4]) -> u8 {
    (links[0] as u8) | ((links[1] as u8) << 1) | ((links[2] as u8) << 2) | ((links[3] as u8) << 3)
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
    pub tab_states: HashMap<String, String>,
    #[serde(default)]
    pub split_layout_mode: SplitLayoutMode,
    #[serde(default = "default_secondary_tab")]
    pub secondary_tab: usize,
    #[serde(default)]
    pub secondary_tab_name: Option<String>,
    #[serde(default = "default_split_ratio")]
    pub split_ratio: f32,
    /// Horizontal and vertical split ratios `(x, y)` for `ViewportLayout::QuadGrid`.
    #[serde(default = "default_quad_split_ratios")]
    pub quad_split_ratios: (f32, f32),
    #[serde(default)]
    pub viewport_layout: ViewportLayout,
    #[serde(default)]
    pub pane_tabs: Vec<usize>,
    #[serde(default)]
    pub focused_pane: usize,
    #[serde(default)]
    pub pane_tab_states: Vec<HashMap<String, String>>,
    /// Per-pane `SharedWorkspaceBus` link flags (`0..4`).
    #[serde(default, skip_serializing_if = "is_all_false_links")]
    pub pane_bus_links: Vec<bool>,
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
            tab_states: HashMap::new(),
            split_layout_mode: SplitLayoutMode::Single,
            secondary_tab: 1,
            secondary_tab_name: None,
            split_ratio: 0.5,
            quad_split_ratios: (0.5, 0.5),
            viewport_layout: ViewportLayout::Single,
            pane_tabs: vec![0, 1, 2, 3],
            focused_pane: 0,
            pane_tab_states: Vec::new(),
            pane_bus_links: Vec::new(),
        }
    }
}

impl AppState {
    /// Extracts the serialized state snapshot from a live `MathExplorerApp` instance,
    /// reusing cached JSON for tabs outside active panes and clean visible tabs.
    pub fn extract(app: &super::MathExplorerApp) -> Self {
        let active_primary = app.active_primary_tab_index();
        let mut tab_states = HashMap::with_capacity(app.tabs.len());
        for (idx, tab) in app.tabs.iter().enumerate() {
            let is_active = idx == active_primary;
            if let Some(tab_str) = app.tab_state_cache.extract_tab_json(
                tab.as_ref(),
                is_active,
                app.last_frame_had_input,
            ) {
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
        let (split_layout_mode, viewport_layout) = Self::resolve_extract_layouts(app);
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
            quad_split_ratios: (
                app.quad_split_ratios.0.clamp(0.20, 0.80),
                app.quad_split_ratios.1.clamp(0.20, 0.80),
            ),
            viewport_layout,
            pane_tabs,
            focused_pane: app.active_focused_pane(),
            pane_tab_states: app.extract_pane_tab_states(),
            pane_bus_links: if is_all_false_links(&app.pane_bus_links) {
                Vec::new()
            } else {
                app.pane_bus_links.to_vec()
            },
        }
    }

    fn resolve_extract_layouts(app: &super::MathExplorerApp) -> (SplitLayoutMode, ViewportLayout) {
        let split_layout_mode = if app.viewport_layout != ViewportLayout::Single {
            match app.viewport_layout {
                ViewportLayout::Single | ViewportLayout::QuadGrid => SplitLayoutMode::Single,
                ViewportLayout::SplitHorizontal => SplitLayoutMode::Horizontal,
                ViewportLayout::SplitVertical => SplitLayoutMode::Vertical,
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
        (split_layout_mode, viewport_layout)
    }

    /// Restores serialized state into the provided `MathExplorerApp` instance.
    pub fn apply(self, app: &mut super::MathExplorerApp) {
        app.show_info = self.show_info;
        app.show_warnings = self.show_warnings;
        app.show_errors = self.show_errors;
        app.show_diagnostics_panel = self.show_diagnostics_panel;
        app.theme = self.theme;
        app.split_layout_mode = self.split_layout_mode;
        app.split_ratio = self.split_ratio.clamp(0.20, 0.80);
        app.quad_split_ratios = (
            self.quad_split_ratios.0.clamp(0.20, 0.80),
            self.quad_split_ratios.1.clamp(0.20, 0.80),
        );
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
        app.tab_state_cache.clear();
        Self::apply_pane_indices_and_focus(&self, app);
        let mut restored_links = [false; 4];
        for (slot, &linked) in restored_links.iter_mut().zip(self.pane_bus_links.iter()) {
            *slot = linked;
        }
        app.pane_bus_links = restored_links;
        app.workspace_bus.reset();
        app.restore_pane_tab_states(self.pane_tab_states, &self.tab_states);
        app.sync_linked_panes_bus();
    }

    fn apply_pane_indices_and_focus(&self, app: &mut super::MathExplorerApp) {
        if let Some(idx) = self
            .selected_tab_name
            .as_deref()
            .and_then(|n| app.tabs.iter().position(|t| t.name() == n))
        {
            app.selected_tab = idx;
        } else if self.selected_tab < app.tabs.len() {
            app.selected_tab = self.selected_tab;
        }

        if let Some(idx) = self
            .secondary_tab_name
            .as_deref()
            .and_then(|n| app.tabs.iter().position(|t| t.name() == n))
        {
            app.secondary_tab = idx;
        } else if self.secondary_tab < app.tabs.len() {
            app.secondary_tab = self.secondary_tab;
        }

        app.pane_tabs = if self.pane_tabs.is_empty() {
            vec![app.selected_tab, app.secondary_tab, 2, 3]
        } else {
            self.pane_tabs.clone()
        };
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
    }
}
