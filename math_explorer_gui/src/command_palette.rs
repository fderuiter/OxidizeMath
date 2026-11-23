mod actions;
mod scoring;
mod ui;

use actions::score_workspace_action;
pub use actions::WorkspaceAction;
use eframe::egui;
pub(crate) use scoring::score_tool_in_domain;
use scoring::score_tool_metadata;
pub use scoring::{format_domain_title, fuzzy_match_score};

/// Target tab, tool, starter preset, or workspace action selected from the command palette.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteSelection {
    /// Target explorer tab index.
    pub tab_index: usize,
    /// Optional tool name to activate within the domain tab.
    pub tool_name: Option<String>,
    /// Optional validated starter preset card to launch.
    pub preset: Option<crate::starter_gallery::StarterCard>,
    /// Optional global workspace action to execute.
    pub action: Option<WorkspaceAction>,
}

impl PaletteSelection {
    /// Creates a selection pointing to a domain tab.
    #[must_use]
    pub fn tab(tab_index: usize) -> Self {
        Self {
            tab_index,
            tool_name: None,
            preset: None,
            action: None,
        }
    }

    /// Creates a selection pointing to a specific simulation tool within a domain tab.
    #[must_use]
    pub fn tool(tab_index: usize, tool_name: impl Into<String>) -> Self {
        Self {
            tab_index,
            tool_name: Some(tool_name.into()),
            preset: None,
            action: None,
        }
    }

    /// Creates a selection launching a curated [`crate::starter_gallery::StarterCard`].
    #[must_use]
    pub fn preset(tab_index: usize, card: crate::starter_gallery::StarterCard) -> Self {
        let tool_name = card.tool_name.map(str::to_string);
        Self {
            tab_index,
            tool_name,
            preset: Some(card),
            action: None,
        }
    }

    /// Creates a selection executing a [`WorkspaceAction`].
    #[must_use]
    pub const fn action(action: WorkspaceAction) -> Self {
        Self {
            tab_index: 0,
            tool_name: None,
            preset: None,
            action: Some(action),
        }
    }
}

/// Search result item in the command palette representing a tab, tool, starter preset, or action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaletteItem {
    Tab {
        index: usize,
        name: &'static str,
    },
    Tool {
        tab_index: usize,
        name: &'static str,
        domain: &'static str,
        tags: &'static [&'static str],
    },
    Preset {
        tab_index: usize,
        card: crate::starter_gallery::StarterCard,
    },
    Action {
        action: WorkspaceAction,
    },
}

impl PaletteItem {
    /// Converts this item into a [`PaletteSelection`].
    #[must_use]
    pub fn to_selection(&self) -> PaletteSelection {
        match self {
            PaletteItem::Tab { index, .. } => PaletteSelection::tab(*index),
            PaletteItem::Tool {
                tab_index, name, ..
            } => PaletteSelection::tool(*tab_index, *name),
            PaletteItem::Preset { tab_index, card } => {
                PaletteSelection::preset(*tab_index, card.clone())
            }
            PaletteItem::Action { action } => PaletteSelection::action(*action),
        }
    }

    /// Primary display title of this item.
    #[must_use]
    pub fn title(&self) -> &'static str {
        match self {
            PaletteItem::Tab { name, .. } | PaletteItem::Tool { name, .. } => name,
            PaletteItem::Preset { card, .. } => card.title,
            PaletteItem::Action { action } => action.title(),
        }
    }
}

/// Modal command palette providing instant search-driven navigation, presets, and actions.
#[derive(Default)]
pub struct CommandPalette {
    /// Whether the command palette modal is currently open.
    pub is_open: bool,
    /// Current search query entered by the user.
    pub query: String,
    /// Last query captured to detect search term changes and reset selection.
    pub last_query: String,
    /// Currently highlighted item index in the filtered results list.
    pub selected_index: usize,
    /// Flag indicating whether the palette was opened on this frame (for auto-focusing).
    pub just_opened: bool,
    /// Widget ID that held focus prior to opening the command palette.
    pub prior_focus: Option<egui::Id>,
}

pub(crate) const SEARCH_WIDGET_INFO_ID: &str = "command_palette_search_info";

pub(crate) fn resolve_tab_index_for_domain(
    tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    domain_slug: &str,
) -> usize {
    if let Some(idx) = tabs
        .iter()
        .position(|t| crate::url_state::matches_tab_slug(domain_slug, t.url_slug(), t.name()))
    {
        return idx;
    }
    tabs.iter()
        .position(|t| {
            let t_name = t.name().to_ascii_lowercase().replace(' ', "_");
            let t_clean = t_name.replace('&', "").replace("__", "_");
            t_clean.contains(domain_slug)
        })
        .unwrap_or(0)
}

impl CommandPalette {
    /// Opens the command palette modal, clearing the query and resetting selection.
    pub fn open(&mut self) {
        self.prior_focus = None;
        self.reset_state(true);
    }

    /// Closes the command palette modal.
    pub fn close(&mut self) {
        self.reset_state(false);
    }

    fn reset_state(&mut self, open: bool) {
        self.is_open = open;
        self.query.clear();
        self.last_query.clear();
        self.selected_index = 0;
        self.just_opened = open;
    }

    /// Toggles the visibility state of the command palette modal.
    pub fn toggle(&mut self) {
        if self.is_open {
            self.close();
        } else {
            self.open();
        }
    }

    /// Filters registered domain tabs using case-insensitive word-prefix and token matching.
    pub fn filter_tabs(
        &self,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    ) -> Vec<(usize, &'static str)> {
        let mut matched: Vec<(usize, &'static str, usize)> = tabs
            .iter()
            .enumerate()
            .filter_map(|(idx, tab)| {
                let tab_name = tab.name();
                fuzzy_match_score(tab_name, &self.query).map(|score| (idx, tab_name, score))
            })
            .collect();

        matched.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
        matched
            .into_iter()
            .map(|(idx, name, _)| (idx, name))
            .collect()
    }

    /// Filters domain tabs, `ToolMetadata`, `StarterPresetMetadata` cards, and `WorkspaceAction`s.
    pub fn filter_items(&self, tabs: &[Box<dyn crate::tabs::ExplorerTab>]) -> Vec<PaletteItem> {
        let mut matched: Vec<(PaletteItem, usize, usize)> = Vec::new();
        self.collect_tab_matches(tabs, &mut matched);
        self.collect_tool_matches(tabs, &mut matched);
        self.collect_preset_matches(tabs, &mut matched);
        self.collect_action_matches(&mut matched);

        matched.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.title().cmp(b.0.title()))
        });
        matched.into_iter().map(|(item, _, _)| item).collect()
    }

    fn collect_tab_matches(
        &self,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
        matched: &mut Vec<(PaletteItem, usize, usize)>,
    ) {
        for (idx, tab) in tabs.iter().enumerate() {
            let tab_name = tab.name();
            let mut score = fuzzy_match_score(tab_name, &self.query);
            let domain = tab.academic_domain();
            if !domain.is_empty() {
                if let Some(domain_score) = fuzzy_match_score(domain, &self.query) {
                    score = Some(score.map_or(domain_score, |s| s.min(domain_score)));
                }
            }
            if let Some(s) = score {
                matched.push((
                    PaletteItem::Tab {
                        index: idx,
                        name: tab_name,
                    },
                    s,
                    0,
                ));
            }
        }
    }

    fn collect_tool_matches(
        &self,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
        matched: &mut Vec<(PaletteItem, usize, usize)>,
    ) {
        for meta in inventory::iter::<crate::framework::ToolMetadata> {
            if let Some(s) = score_tool_metadata(meta, &self.query) {
                let tab_index = resolve_tab_index_for_domain(tabs, meta.domain);
                matched.push((
                    PaletteItem::Tool {
                        tab_index,
                        name: meta.display_name,
                        domain: meta.domain,
                        tags: meta.tags,
                    },
                    s,
                    1,
                ));
            }
        }
    }

    fn collect_preset_matches(
        &self,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
        matched: &mut Vec<(PaletteItem, usize, usize)>,
    ) {
        for card in crate::starter_gallery::get_all_starter_cards() {
            if let Some(s) = crate::starter_gallery::score_starter_card(&card, &self.query) {
                let tab_index = resolve_tab_index_for_domain(tabs, card.domain);
                matched.push((PaletteItem::Preset { tab_index, card }, s, 2));
            }
        }
    }

    fn collect_action_matches(&self, matched: &mut Vec<(PaletteItem, usize, usize)>) {
        for &action in WorkspaceAction::ALL {
            if let Some(s) = score_workspace_action(action, &self.query) {
                matched.push((PaletteItem::Action { action }, s, 3));
            }
        }
    }

    /// Processes keyboard shortcuts (Arrow Up/Down, Enter, Escape) for navigating and selecting items.
    pub fn handle_keyboard_inputs(
        &mut self,
        ctx: &egui::Context,
        filtered_items: &[PaletteItem],
    ) -> (Option<PaletteSelection>, bool) {
        let mut selected_item = None;
        let mut esc_pressed = false;

        if !filtered_items.is_empty() {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown)) {
                self.selected_index = (self.selected_index + 1) % filtered_items.len();
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp)) {
                self.selected_index = self
                    .selected_index
                    .checked_sub(1)
                    .unwrap_or(filtered_items.len() - 1);
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)) {
                if let Some(item) = filtered_items.get(self.selected_index) {
                    selected_item = Some(item.to_selection());
                }
            }
        }

        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            esc_pressed = true;
        }

        (selected_item, esc_pressed)
    }
}

#[cfg(test)]
#[path = "command_palette_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "command_palette_actions_tests.rs"]
mod actions_tests;
