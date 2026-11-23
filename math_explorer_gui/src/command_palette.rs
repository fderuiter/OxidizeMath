pub mod actions;
mod layout_specs;
mod plot_specs;
mod scoring;
mod theory_index;
mod ui;

pub use actions::PaletteAction;
use actions::{all_builtin_commands, extra_registry_commands, score_builtin_command};
use eframe::egui;
pub(crate) use scoring::score_tool_metadata;
pub use scoring::{format_domain_title, fuzzy_match_score};

/// Target tab, tool, starter preset, or executable action selected from `CommandPalette`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteSelection {
    /// Target explorer tab index.
    pub tab_index: usize,
    /// Optional tool name to activate within the domain tab.
    pub tool_name: Option<String>,
    /// Optional starter preset metadata to launch inside the target tab.
    pub preset: Option<&'static crate::starter_gallery::StarterPresetMetadata>,
    /// Optional executable workspace, layout, theme, export, playback, camera, or plot action.
    pub action: Option<PaletteAction>,
}

impl PaletteSelection {
    /// Creates a selection pointing to a domain tab.
    pub fn tab(tab_index: usize) -> Self {
        Self {
            tab_index,
            tool_name: None,
            preset: None,
            action: None,
        }
    }

    /// Creates a selection pointing to a specific simulation tool within a domain tab.
    pub fn tool(tab_index: usize, tool_name: impl Into<String>) -> Self {
        Self {
            tab_index,
            tool_name: Some(tool_name.into()),
            preset: None,
            action: None,
        }
    }

    /// Creates a selection launching a curated starter preset inside `tab_index`.
    pub fn preset(
        tab_index: usize,
        preset: &'static crate::starter_gallery::StarterPresetMetadata,
    ) -> Self {
        Self {
            tab_index,
            tool_name: preset.tool_name.map(str::to_string),
            preset: Some(preset),
            action: None,
        }
    }

    /// Creates a selection executing a global or contextual command action.
    pub fn command(action: PaletteAction) -> Self {
        Self {
            tab_index: 0,
            tool_name: None,
            preset: None,
            action: Some(action),
        }
    }
}

/// Unified zero-prefix search result item in the command palette.
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
        preset: &'static crate::starter_gallery::StarterPresetMetadata,
    },
    Command {
        action: PaletteAction,
        title: String,
        category: String,
        shortcut: Option<String>,
        description: String,
    },
}

impl PaletteItem {
    /// Converts this item into a `PaletteSelection`.
    pub fn to_selection(&self) -> PaletteSelection {
        match self {
            Self::Tab { index, .. } => PaletteSelection::tab(*index),
            Self::Tool {
                tab_index, name, ..
            } => PaletteSelection::tool(*tab_index, *name),
            Self::Preset { tab_index, preset } => PaletteSelection::preset(*tab_index, preset),
            Self::Command { action, .. } => PaletteSelection::command(action.clone()),
        }
    }

    /// Primary display title of this item.
    pub fn title(&self) -> &str {
        match self {
            Self::Tab { name, .. } => name,
            Self::Tool { name, .. } => name,
            Self::Preset { preset, .. } => preset.title,
            Self::Command { title, .. } => title.as_str(),
        }
    }
}

/// Modal command palette providing unified zero-prefix tab, tool, preset, and action execution.
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

pub(crate) fn resolve_domain_tab_index(
    tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    domain: &str,
) -> usize {
    if let Some(idx) = tabs.iter().position(|t| {
        t.url_slug() == domain || (domain == "biology" && t.url_slug() == "morphogenesis")
    }) {
        return idx;
    }
    tabs.iter()
        .position(|t| {
            let t_name = t.name().to_lowercase().replace(' ', "_");
            let t_clean = t_name.replace('&', "").replace("__", "_");
            t_clean.contains(domain)
                || (domain == "ai"
                    && (t.name() == "Artificial Intelligence" || t.name().starts_with("AI")))
                || (domain == "biology" && t.name() == "Morphogenesis")
        })
        .unwrap_or(0)
}

fn score_preset_metadata(
    preset: &crate::starter_gallery::StarterPresetMetadata,
    query: &str,
) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let domain_title = format_domain_title(preset.domain);
    let mut best = fuzzy_match_score(preset.title, query);
    for (candidate, penalty) in [
        (preset.id, 2usize),
        (preset.description, 12),
        (preset.category, 20),
        (preset.domain, 45),
        (domain_title.as_ref(), 45),
    ] {
        if let Some(s) = fuzzy_match_score(candidate, query) {
            let weighted = s.saturating_add(penalty);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    best.map(|s| if s == 0 { 0 } else { s.saturating_add(5) })
}

type RankedEntry = (PaletteItem, usize, usize, usize);

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

    /// Filters registered domain tabs using word-boundary matching.
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

    /// Filters commands, domain tabs, simulation tools, and starter presets in one zero-prefix list.
    pub fn filter_items(&self, tabs: &[Box<dyn crate::tabs::ExplorerTab>]) -> Vec<PaletteItem> {
        self.filter_items_with_ctx(None, tabs)
    }

    /// Filters unified palette items, including active contextual shortcuts registered on `ctx`.
    pub fn filter_items_with_ctx(
        &self,
        ctx: Option<&egui::Context>,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    ) -> Vec<PaletteItem> {
        let mut matched: Vec<RankedEntry> = Vec::new();
        self.collect_tab_and_tool_entries(tabs, &mut matched);
        self.collect_preset_entries(tabs, &mut matched);
        self.collect_command_entries(ctx, &mut matched);

        matched.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| a.0.title().cmp(b.0.title()))
        });
        matched.into_iter().map(|(item, _, _, _)| item).collect()
    }

    fn collect_tab_and_tool_entries(
        &self,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
        matched: &mut Vec<RankedEntry>,
    ) {
        for (idx, tab) in tabs.iter().enumerate() {
            let tab_name = tab.name();
            let mut score = fuzzy_match_score(tab_name, &self.query);
            let domain = tab.academic_domain();
            if !domain.is_empty() {
                if let Some(ds) = fuzzy_match_score(domain, &self.query) {
                    let weighted = ds.saturating_add(40);
                    score = Some(score.map_or(weighted, |s| s.min(weighted)));
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
                    idx,
                ));
            }
        }
        for meta in inventory::iter::<crate::framework::ToolMetadata> {
            if let Some(s) = score_tool_metadata(meta, &self.query) {
                let tab_index = resolve_domain_tab_index(tabs, meta.domain);
                matched.push((
                    PaletteItem::Tool {
                        tab_index,
                        name: meta.display_name,
                        domain: meta.domain,
                        tags: meta.tags,
                    },
                    s,
                    1,
                    0,
                ));
            }
        }
    }

    fn collect_preset_entries(
        &self,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
        matched: &mut Vec<RankedEntry>,
    ) {
        for preset in inventory::iter::<crate::starter_gallery::StarterPresetMetadata> {
            if let Some(s) = score_preset_metadata(preset, &self.query) {
                let tab_index = resolve_domain_tab_index(tabs, preset.domain);
                matched.push((PaletteItem::Preset { tab_index, preset }, s, 2, 0));
            }
        }
    }

    fn collect_command_entries(&self, ctx: Option<&egui::Context>, matched: &mut Vec<RankedEntry>) {
        for (idx, spec) in all_builtin_commands().enumerate() {
            if let Some(s) = score_builtin_command(spec, &self.query) {
                matched.push((
                    PaletteItem::Command {
                        action: (spec.action)(),
                        title: spec.title.to_string(),
                        category: spec.category.to_string(),
                        shortcut: spec.shortcut.map(str::to_string),
                        description: spec.description.to_string(),
                    },
                    s,
                    3,
                    idx,
                ));
            }
        }
        let Some(ctx) = ctx else {
            return;
        };
        self.collect_extra_ctx_commands(ctx, matched);
    }

    fn collect_extra_ctx_commands(&self, ctx: &egui::Context, matched: &mut Vec<RankedEntry>) {
        let base_idx = 1000usize;
        for (offset, cmd) in extra_registry_commands(ctx).into_iter().enumerate() {
            let mut best = fuzzy_match_score(&cmd.name, &self.query);
            for (extra, penalty) in [(&cmd.context, 20usize), (&cmd.description, 20)] {
                if let Some(s) = fuzzy_match_score(extra, &self.query) {
                    let weighted = s.saturating_add(penalty);
                    best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
                }
            }
            if let Some(s) = best {
                let adj = if s == 0 { 0 } else { s.saturating_add(3) };
                matched.push((
                    PaletteItem::Command {
                        action: PaletteAction::DispatchedCommand {
                            name: cmd.name.clone(),
                            context: cmd.context.clone(),
                        },
                        title: format!("{}: {}", cmd.context, cmd.name),
                        category: cmd.context,
                        shortcut: Some(cmd.trigger.display_label()),
                        description: cmd.description,
                    },
                    adj,
                    3,
                    base_idx + offset,
                ));
            }
        }
    }
}

#[cfg(test)]
#[path = "command_palette_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "command_palette/unified_tests.rs"]
mod unified_tests;
