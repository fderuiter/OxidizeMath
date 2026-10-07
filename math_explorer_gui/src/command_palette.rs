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

    /// Processes keyboard shortcuts (Arrow Up/Down, Enter, Escape) for navigating and selecting tabs.
    pub fn handle_keyboard_inputs(
        &mut self,
        ctx: &egui::Context,
        filtered_tabs: &[(usize, &'static str)],
    ) -> (Option<usize>, bool) {
        let mut selected_tab_index = None;
        let mut esc_pressed = false;

        if !filtered_tabs.is_empty() {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown)) {
                self.selected_index = (self.selected_index + 1) % filtered_tabs.len();
            }

            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp)) {
                if self.selected_index == 0 {
                    self.selected_index = filtered_tabs.len() - 1;
                } else {
                    self.selected_index -= 1;
                }
            }

            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)) {
                if let Some(&(tab_idx, _)) = filtered_tabs.get(self.selected_index) {
                    selected_tab_index = Some(tab_idx);
                }
            }
        }

        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            esc_pressed = true;
        }

        (selected_tab_index, esc_pressed)
    }

    /// Renders search header and text field with auto-focus behavior.
    fn render_header_and_search(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("🔍 Command Palette");
        });
        ui.separator();

        let text_edit_response = ui.add(
            egui::TextEdit::singleline(&mut self.query)
                .hint_text("Type tab name or domain query (e.g. quantum, fluid, ODE)...")
                .desired_width(f32::INFINITY),
        );

        if self.just_opened {
            text_edit_response.request_focus();
            self.just_opened = false;
        }
    }

    /// Renders vertical scrollable list of filtered simulation domain tabs.
    fn render_results_list(
        &mut self,
        ui: &mut egui::Ui,
        filtered_tabs: &[(usize, &'static str)],
    ) -> Option<usize> {
        let mut clicked_tab = None;

        egui::ScrollArea::vertical()
            .max_height(280.0)
            .show(ui, |ui| {
                if filtered_tabs.is_empty() {
                    ui.label(
                        egui::RichText::new("No matching simulation domain tabs found.").italics(),
                    );
                } else {
                    for (list_idx, &(tab_idx, tab_name)) in filtered_tabs.iter().enumerate() {
                        let is_selected = list_idx == self.selected_index;
                        let label_text = egui::RichText::new(tab_name).size(14.0).strong();
                        let response = ui.selectable_label(is_selected, label_text);

                        if is_selected {
                            response.scroll_to_me(Some(egui::Align::Center));
                        }

                        if response.clicked() {
                            clicked_tab = Some(tab_idx);
                        }
                    }
                }
            });

        clicked_tab
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
    /// Returns `Some(tab_index)` if a tab was selected (via Enter key or mouse click),
    /// which also dismisses the command palette.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    ) -> Option<usize> {
        if !self.is_open {
            return None;
        }

        let filtered_tabs = self.filter_tabs(tabs);

        if self.query != self.last_query {
            self.selected_index = 0;
            self.last_query = self.query.clone();
        } else if filtered_tabs.is_empty() {
            self.selected_index = 0;
        } else if self.selected_index >= filtered_tabs.len() {
            self.selected_index = filtered_tabs.len() - 1;
        }

        let (kb_selected_tab, esc_pressed) = self.handle_keyboard_inputs(ctx, &filtered_tabs);
        let mut selected_tab_index = kb_selected_tab;

        let modal = egui::Modal::new(egui::Id::new("command_palette_modal"))
            .backdrop_color(egui::Color32::from_black_alpha(150));

        let modal_res = modal.show(ctx, |ui| {
            ui.set_width(550.0);
            ui.set_max_height(400.0);

            ui.vertical(|ui| {
                self.render_header_and_search(ui);
                ui.add_space(8.0);
                let mouse_selected = self.render_results_list(ui, &filtered_tabs);
                if mouse_selected.is_some() {
                    selected_tab_index = mouse_selected;
                }
                ui.add_space(4.0);
                self.render_footer(ui);
            });
        });

        if esc_pressed || modal_res.should_close() || selected_tab_index.is_some() {
            self.close();
        }

        selected_tab_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTab(&'static str);

    impl crate::tabs::ExplorerTab for MockTab {
        fn name(&self) -> &'static str {
            self.0
        }
        fn show(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {}
    }

    fn mock_tabs() -> Vec<Box<dyn crate::tabs::ExplorerTab>> {
        vec![
            Box::new(MockTab("Quantum / Spin Visualization")),
            Box::new(MockTab("Fluid Dynamics / Lattice Boltzmann")),
            Box::new(MockTab("Analysis / ODE Solver")),
            Box::new(MockTab("AI / Grid World")),
        ]
    }

    #[test]
    fn test_fuzzy_match_score() {
        assert!(fuzzy_match_score("Quantum / Spin Visualization", "quantum").is_some());
        assert!(fuzzy_match_score("Quantum / Spin Visualization", "spin").is_some());
        assert!(fuzzy_match_score("Quantum / Spin Visualization", "qspin").is_some());
        assert!(fuzzy_match_score("Quantum / Spin Visualization", "nonexistent").is_none());
    }

    #[test]
    fn test_palette_open_close_toggle() {
        let mut palette = CommandPalette::default();
        assert!(!palette.is_open);

        palette.open();
        assert!(palette.is_open);
        assert!(palette.just_opened);

        palette.close();
        assert!(!palette.is_open);

        palette.toggle();
        assert!(palette.is_open);
    }

    #[test]
    fn test_palette_filtering_and_selection() {
        let ctx = egui::Context::default();
        let tabs = mock_tabs();
        let mut palette = CommandPalette::default();

        palette.open();
        palette.query = "quantum".to_string();

        let mut selected = None;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            selected = palette.show(ctx, &tabs);
        });

        assert_eq!(selected, None);
        assert!(palette.is_open);

        // Press enter to select top filtered item
        let raw_input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };

        let _ = ctx.run(raw_input, |ctx| {
            selected = palette.show(ctx, &tabs);
        });

        assert_eq!(selected, Some(0)); // Quantum is at index 0
        assert!(!palette.is_open);
    }

    #[test]
    fn test_palette_arrow_navigation() {
        let ctx = egui::Context::default();
        let tabs = mock_tabs();
        let mut palette = CommandPalette::default();

        palette.open();

        // Arrow down
        let raw_input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::ArrowDown,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };

        let _ = ctx.run(raw_input, |ctx| {
            let _ = palette.show(ctx, &tabs);
        });

        assert_eq!(palette.selected_index, 1);

        // Arrow up wraps around or goes back
        let raw_input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::ArrowUp,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };

        let _ = ctx.run(raw_input, |ctx| {
            let _ = palette.show(ctx, &tabs);
        });

        assert_eq!(palette.selected_index, 0);
    }

    #[test]
    fn test_palette_escape_key_closes() {
        let ctx = egui::Context::default();
        let tabs = mock_tabs();
        let mut palette = CommandPalette::default();

        palette.open();

        let raw_input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };

        let mut selected = None;
        let _ = ctx.run(raw_input, |ctx| {
            selected = palette.show(ctx, &tabs);
        });

        assert_eq!(selected, None);
        assert!(!palette.is_open);
    }

    #[test]
    fn test_query_change_resets_selection() {
        let ctx = egui::Context::default();
        let tabs = mock_tabs();
        let mut palette = CommandPalette::default();

        palette.open();
        palette.selected_index = 2;

        palette.query = "fluid".to_string();

        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            let _ = palette.show(ctx, &tabs);
        });

        assert_eq!(palette.selected_index, 0);
    }
}
