#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    format_domain_title, CommandPalette, PaletteItem, PaletteSelection, SEARCH_WIDGET_INFO_ID,
};
use eframe::egui;

const SEARCH_INPUT_LABEL: &str = "Search tabs and tools";
const SEARCH_HINT_TEXT: &str =
    "Type tab, tool, starter preset, or workspace action (e.g. dijkstra, kermack, quad grid)...";

fn format_search_count_status(count: usize) -> String {
    match count {
        0 => "No matching tools found".to_string(),
        1 => "1 matching result found".to_string(),
        n => format!("{n} matching results found"),
    }
}

fn announce_palette_status(ctx: &egui::Context, message: impl Into<String>) {
    crate::accessibility::announce_live_status(ctx, message);
}

fn format_palette_item_text(item: &PaletteItem) -> egui::RichText {
    match item {
        PaletteItem::Tab { name, .. } => egui::RichText::new(format!("📁 [Tab] {name}"))
            .size(14.0)
            .strong(),
        PaletteItem::Tool { name, domain, .. } => {
            egui::RichText::new(format!("🔧 {name} ({})", format_domain_title(domain))).size(14.0)
        }
        PaletteItem::Preset { card, .. } => egui::RichText::new(format!(
            "🎨 [Preset] {} {} ({})",
            card.icon,
            card.title,
            format_domain_title(card.domain)
        ))
        .size(14.0),
        PaletteItem::Action { action } => {
            egui::RichText::new(format!("{} [Action] {}", action.icon(), action.title())).size(14.0)
        }
    }
}

impl CommandPalette {
    /// Renders search header and text field with auto-focus and accessible `WidgetInfo` label.
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

    /// Renders vertical scrollable list of filtered tabs, tools, presets, and workspace actions.
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
                        let response =
                            ui.selectable_label(is_selected, format_palette_item_text(item));

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
    /// Returns `Some(PaletteSelection)` if a tab, tool, preset, or action was selected.
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
            ui.set_width(580.0);
            ui.set_max_height(420.0);
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

    pub(super) fn sync_selection_with_results(&mut self, item_count: usize) -> bool {
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
