mod scoring;

use eframe::egui;
use scoring::score_tool_metadata;
pub use scoring::{format_domain_title, fuzzy_match_score};

/// Target tab index and optional tool name selected from the command palette.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteSelection {
    /// Target explorer tab index.
    pub tab_index: usize,
    /// Optional tool name to activate within the domain tab.
    pub tool_name: Option<String>,
}

impl PaletteSelection {
    /// Creates a selection pointing to a domain tab.
    pub fn tab(tab_index: usize) -> Self {
        Self {
            tab_index,
            tool_name: None,
        }
    }

    /// Creates a selection pointing to a specific simulation tool within a domain tab.
    pub fn tool(tab_index: usize, tool_name: impl Into<String>) -> Self {
        Self {
            tab_index,
            tool_name: Some(tool_name.into()),
        }
    }
}

/// Search result item in the command palette representing either a top-level tab or an individual tool.
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
}

impl PaletteItem {
    /// Converts this item into a `PaletteSelection`.
    pub fn to_selection(&self) -> PaletteSelection {
        match self {
            PaletteItem::Tab { index, .. } => PaletteSelection::tab(*index),
            PaletteItem::Tool {
                tab_index, name, ..
            } => PaletteSelection::tool(*tab_index, *name),
        }
    }

    /// Primary display title of this item.
    pub fn title(&self) -> &'static str {
        match self {
            PaletteItem::Tab { name, .. } => name,
            PaletteItem::Tool { name, .. } => name,
        }
    }
}

/// Modal command palette providing instant search-driven tab switching and command execution.
#[derive(Default)]
pub struct CommandPalette {
    /// Whether the command palette modal is currently open.
    pub is_open: bool,
    /// Current search query entered by the user.
    pub query: String,
    /// Last query captured to detect search term changes and reset selection.
    pub last_query: String,
    /// Currently highlighted tab index in the filtered results list.
    pub selected_index: usize,
    /// Flag indicating whether the palette was opened on this frame (for auto-focusing).
    pub just_opened: bool,
    /// Widget ID that held focus prior to opening the command palette.
    pub prior_focus: Option<egui::Id>,
}

const SEARCH_INPUT_LABEL: &str = "Search tabs and tools";
const SEARCH_HINT_TEXT: &str = "Type tab or tool name (e.g. dijkstra, lorenz, cera, fluid)...";
pub(crate) const SEARCH_WIDGET_INFO_ID: &str = "command_palette_search_info";

fn format_search_count_status(count: usize) -> String {
    match count {
        0 => "No matching tools found".to_string(),
        1 => "1 matching result found".to_string(),
        n => format!("{n} matching results found"),
    }
}

fn announce_palette_status(ctx: &egui::Context, message: impl Into<String>) {
    let message = message.into();
    crate::accessibility::announce_status(&message);
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new("aria_live_message"), message);
    });
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

    /// Filters registered domain tabs using case-insensitive fuzzy and substring matching.
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

    /// Filters registered domain tabs and simulation tools using case-insensitive fuzzy and tag matching.
    pub fn filter_items(&self, tabs: &[Box<dyn crate::tabs::ExplorerTab>]) -> Vec<PaletteItem> {
        let mut matched: Vec<(PaletteItem, usize, usize)> = Vec::new();

        // 1. Index domain tabs
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

        // 2. Index simulation tools registered in inventory
        for meta in inventory::iter::<crate::framework::ToolMetadata> {
            if let Some(s) = score_tool_metadata(meta, &self.query) {
                // Find matching tab index for this tool's domain, safely falling back to 0
                let tab_index = tabs
                    .iter()
                    .position(|t| {
                        let t_name = t.name().to_lowercase().replace(' ', "_");
                        let t_clean = t_name.replace('&', "").replace("__", "_");
                        t_clean.contains(meta.domain)
                            || (meta.domain == "ai" && t.name() == "Artificial Intelligence")
                    })
                    .unwrap_or(0);

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

        // Sort by match score ascending (lower score = higher relevance / better match).
        // On tie, sort tabs before tools (item_type_order 0 vs 1), then deterministically by title.
        matched.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.title().cmp(b.0.title()))
        });

        matched.into_iter().map(|(item, _, _)| item).collect()
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

    /// Renders search header and text field with auto-focus and accessible WidgetInfo label.
    fn render_header_and_search(&mut self, ui: &mut egui::Ui, prev_query: &str) {
        ui.horizontal(|ui| {
            ui.heading("🔍 Command Palette");
        });
        ui.separator();

        let text_edit_response = ui.add(
            egui::TextEdit::singleline(&mut self.query)
                .hint_text(SEARCH_HINT_TEXT)
                .desired_width(f32::INFINITY),
        );

        let search_info = egui::WidgetInfo {
            label: Some(SEARCH_INPUT_LABEL.to_string()),
            ..egui::WidgetInfo::text_edit(
                ui.is_enabled(),
                prev_query,
                &self.query,
                SEARCH_HINT_TEXT,
            )
        };
        text_edit_response.widget_info({
            let search_info = search_info.clone();
            move || search_info.clone()
        });
        ui.ctx().data_mut(|d| {
            d.insert_temp(egui::Id::new(SEARCH_WIDGET_INFO_ID), search_info);
        });

        if self.just_opened {
            if self.prior_focus.is_none() {
                self.prior_focus = ui.ctx().memory(|m| m.focused());
            }
            text_edit_response.request_focus();
            self.just_opened = false;
        }
    }

    /// Renders vertical scrollable list of filtered simulation domain tabs and tools.
    fn render_results_list(
        &mut self,
        ui: &mut egui::Ui,
        filtered_items: &[PaletteItem],
    ) -> Option<PaletteSelection> {
        let mut clicked_selection = None;

        egui::ScrollArea::vertical()
            .max_height(280.0)
            .show(ui, |ui| {
                if filtered_items.is_empty() {
                    ui.label(
                        egui::RichText::new("No matching simulation domain tabs or tools found.")
                            .italics(),
                    );
                } else {
                    for (list_idx, item) in filtered_items.iter().enumerate() {
                        let is_selected = list_idx == self.selected_index;
                        let text =
                            match item {
                                PaletteItem::Tab { name, .. } => {
                                    egui::RichText::new(format!("📁 [Tab] {}", name))
                                        .size(14.0)
                                        .strong()
                                }
                                PaletteItem::Tool { name, domain, .. } => egui::RichText::new(
                                    format!("🔧 {} ({})", name, format_domain_title(domain)),
                                )
                                .size(14.0),
                            };
                        let response = ui.selectable_label(is_selected, text);

                        if is_selected {
                            response.scroll_to_me(Some(egui::Align::Center));
                        }

                        if response.clicked() {
                            clicked_selection = Some(item.to_selection());
                        }
                    }
                }
            });

        clicked_selection
    }

    /// Renders footer keyboard hint bar.
    fn render_footer(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("↑↓ Navigate • Enter Select • Esc Close")
                    .small()
                    .weak(),
            );
        });
    }

    /// Renders the command palette modal overlay if open.
    ///
    /// Returns `Some(PaletteSelection)` if a tab or tool was selected (via Enter key or mouse click),
    /// which also dismisses the command palette.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    ) -> Option<PaletteSelection> {
        if !self.is_open {
            if let Some(prev_id) = self.prior_focus.take() {
                ctx.memory_mut(|m| m.request_focus(prev_id));
            }
            return None;
        }

        let prev_query = self.last_query.clone();
        let mut filtered_items = self.filter_items(tabs);
        let mut query_changed = self.sync_selection_with_results(filtered_items.len());

        let (mut selected_item, esc_pressed) = self.handle_keyboard_inputs(ctx, &filtered_items);
        let modal = egui::Modal::new(egui::Id::new("command_palette_modal"))
            .backdrop_color(egui::Color32::from_black_alpha(150));

        let modal_res = modal.show(ctx, |ui| {
            ui.set_width(550.0);
            ui.set_max_height(400.0);
            ui.vertical(|ui| {
                self.render_header_and_search(ui, &prev_query);
                if self.query != self.last_query {
                    filtered_items = self.filter_items(tabs);
                    query_changed |= self.sync_selection_with_results(filtered_items.len());
                }
                ui.add_space(8.0);
                if let Some(clicked) = self.render_results_list(ui, &filtered_items) {
                    selected_item = Some(clicked);
                }
                ui.add_space(4.0);
                self.render_footer(ui);
            });
        });

        if esc_pressed || modal_res.should_close() {
            let prior_focus = self.prior_focus.take();
            self.close();
            if let Some(prev_id) = prior_focus {
                ctx.memory_mut(|m| m.request_focus(prev_id));
            }
            announce_palette_status(ctx, "Command palette closed");
        } else if selected_item.is_some() {
            let prior_focus = self.prior_focus.take();
            self.close();
            if let Some(prev_id) = prior_focus {
                ctx.memory_mut(|m| m.request_focus(prev_id));
            }
        } else if query_changed {
            announce_palette_status(ctx, format_search_count_status(filtered_items.len()));
        }

        selected_item
    }

    fn sync_selection_with_results(&mut self, item_count: usize) -> bool {
        if self.query != self.last_query {
            self.selected_index = 0;
            self.last_query = self.query.clone();
            true
        } else {
            if item_count == 0 {
                self.selected_index = 0;
            } else if self.selected_index >= item_count {
                self.selected_index = item_count - 1;
            }
            false
        }
    }
}

#[cfg(test)]
#[path = "command_palette_tests.rs"]
mod tests;
