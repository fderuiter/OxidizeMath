use eframe::egui;

/// Helper to check if `query` matches `text` using case-insensitive fuzzy or substring filtering.
/// Returns a score if matched (lower score = better match / higher rank).
pub fn fuzzy_match_score(text: &str, query: &str) -> Option<usize> {
    let trimmed_query = query.trim().to_lowercase();
    if trimmed_query.is_empty() {
        return Some(usize::MAX);
    }

    let text_lower = text.to_lowercase();

    // 1. Exact match
    if text_lower == trimmed_query {
        return Some(0);
    }

    // 2. Exact prefix match
    if text_lower.starts_with(&trimmed_query) {
        return Some(1);
    }

    // 3. Exact contiguous substring match
    if let Some(pos) = text_lower.find(&trimmed_query) {
        return Some(10 + pos);
    }

    // 4. All whitespace-separated terms match as contiguous substrings
    let terms: Vec<&str> = trimmed_query.split_whitespace().collect();
    if terms.len() > 1 && terms.iter().all(|term| text_lower.contains(term)) {
        return Some(100);
    }

    // 5. Fuzzy character subsequence match (all non-space query characters appear in order)
    let query_chars: Vec<char> = trimmed_query
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if query_chars.is_empty() {
        return Some(usize::MAX);
    }

    let mut text_chars = text_lower.chars();
    let mut matched_count = 0;
    for q_char in &query_chars {
        if text_chars.any(|t_char| t_char == *q_char) {
            matched_count += 1;
        } else {
            break;
        }
    }

    if matched_count == query_chars.len() {
        Some(200)
    } else {
        None
    }
}

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
}

impl CommandPalette {
    /// Opens the command palette modal, clearing the query and resetting selection.
    pub fn open(&mut self) {
        self.is_open = true;
        self.query.clear();
        self.last_query.clear();
        self.selected_index = 0;
        self.just_opened = true;
    }

    /// Closes the command palette modal.
    pub fn close(&mut self) {
        self.is_open = false;
        self.query.clear();
        self.last_query.clear();
        self.selected_index = 0;
        self.just_opened = false;
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
            let domain = tab.domain();
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
            let mut score = fuzzy_match_score(meta.name, &self.query);
            for &tag in meta.tags {
                if let Some(tag_score) = fuzzy_match_score(tag, &self.query) {
                    score = Some(score.map_or(tag_score, |s| s.min(tag_score)));
                }
            }

            if let Some(s) = score {
                // Find matching tab index for this tool's domain, safely falling back to 0
                let tab_index = tabs
                    .iter()
                    .position(|t| !t.domain().is_empty() && t.domain() == meta.domain)
                    .or_else(|| {
                        tabs.iter().position(|t| {
                            let t_name = t.name().to_lowercase().replace(' ', "_");
                            t_name.contains(meta.domain)
                        })
                    })
                    .unwrap_or(0);

                matched.push((
                    PaletteItem::Tool {
                        tab_index,
                        name: meta.name,
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
                if self.selected_index == 0 {
                    self.selected_index = filtered_items.len() - 1;
                } else {
                    self.selected_index -= 1;
                }
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

    /// Renders search header and text field with auto-focus behavior.
    fn render_header_and_search(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("🔍 Command Palette");
        });
        ui.separator();

        let text_edit_response = ui.add(
            egui::TextEdit::singleline(&mut self.query)
                .hint_text("Type tab or tool name (e.g. dijkstra, lorenz, cera, fluid)...")
                .desired_width(f32::INFINITY),
        );

        if self.just_opened {
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
                        let text = match item {
                            PaletteItem::Tab { name, .. } => {
                                egui::RichText::new(format!("📁 [Tab] {}", name))
                                    .size(14.0)
                                    .strong()
                            }
                            PaletteItem::Tool { name, domain, .. } => {
                                egui::RichText::new(format!("🔧 {}  ({})", name, domain)).size(14.0)
                            }
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
            return None;
        }

        let filtered_items = self.filter_items(tabs);

        if self.query != self.last_query {
            self.selected_index = 0;
            self.last_query = self.query.clone();
        } else if filtered_items.is_empty() {
            self.selected_index = 0;
        } else if self.selected_index >= filtered_items.len() {
            self.selected_index = filtered_items.len() - 1;
        }

        let (kb_selected_item, esc_pressed) = self.handle_keyboard_inputs(ctx, &filtered_items);
        let mut selected_item = kb_selected_item;

        let modal = egui::Modal::new(egui::Id::new("command_palette_modal"))
            .backdrop_color(egui::Color32::from_black_alpha(150));

        let modal_res = modal.show(ctx, |ui| {
            ui.set_width(550.0);
            ui.set_max_height(400.0);

            ui.vertical(|ui| {
                self.render_header_and_search(ui);
                ui.add_space(8.0);
                let mouse_selected = self.render_results_list(ui, &filtered_items);
                if mouse_selected.is_some() {
                    selected_item = mouse_selected;
                }
                ui.add_space(4.0);
                self.render_footer(ui);
            });
        });

        if esc_pressed || modal_res.should_close() {
            self.close();
            ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("aria_live_message"),
                    "Command palette closed".to_string(),
                );
            });
        } else if selected_item.is_some() {
            self.close();
        }

        selected_item
    }
}

#[cfg(test)]
#[path = "command_palette_tests.rs"]
mod tests;
