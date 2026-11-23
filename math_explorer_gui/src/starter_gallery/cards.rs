//! Shared card widgets and Guided Story sub-tab UI for `StarterGallery`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::filter::{academic_domain_for_slug, module_title_for_slug, StarterGalleryFilterState};
use super::modal_ui::GalleryOverlayMetrics;
use super::{story_to_starter_card, PresetStoryMetadata, StarterCard, StarterGalleryTab};
use crate::app::domain_bar::AcademicDomain;
use eframe::egui;

pub(super) fn render_card_header(
    ui: &mut egui::Ui,
    icon: &str,
    title: &str,
    category: &str,
    domain: &str,
    cat_color: egui::Color32,
    card_width: f32,
) {
    let module_title = module_title_for_slug(domain);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(icon).size(26.0));
        let text_w = (card_width - 42.0).max(56.0);
        ui.vertical(|ui| {
            ui.set_max_width(text_w);
            ui.add(egui::Label::new(egui::RichText::new(title).strong().size(15.0)).wrap());
            let cat = egui::RichText::new(format!("[{category} · {module_title}]"))
                .small()
                .color(cat_color);
            ui.add(egui::Label::new(cat).truncate());
        });
    });
}

pub(super) fn render_starter_card_with_verb(
    ui: &mut egui::Ui,
    card: &StarterCard,
    card_width: f32,
    verb: &str,
) -> bool {
    let mut launched = false;
    egui::Frame::group(ui.style())
        .inner_margin(14.0)
        .corner_radius(8.0)
        .show(ui, |ui| {
            ui.set_width(card_width);
            ui.set_max_width(card_width);
            ui.set_min_height(152.0);
            ui.vertical(|ui| {
                ui.set_max_width(card_width);
                render_card_header(
                    ui,
                    card.icon,
                    card.title,
                    card.category,
                    card.domain,
                    ui.visuals().weak_text_color(),
                    card_width,
                );
                ui.add_space(8.0);
                let desc = egui::RichText::new(card.description)
                    .size(12.0)
                    .color(ui.visuals().text_color());
                ui.add(egui::Label::new(desc).wrap());
                ui.add_space(10.0);
                let launch_btn = egui::Button::new(format!("{verb} {}", card.title)).truncate();
                if ui.add_sized([card_width, 26.0], launch_btn).clicked() {
                    launched = true;
                }
            });
        });
    launched
}

pub(super) fn render_story_domain_chips(
    ui: &mut egui::Ui,
    stories: &[&'static PresetStoryMetadata],
    filter: &mut StarterGalleryFilterState,
    inner_w: f32,
) {
    let all_count = filter.count_matching_stories_in_domain(stories, None);
    ui.horizontal_wrapped(|ui| {
        ui.set_max_width(inner_w);
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
        let all_label = format!("All Domains ({all_count})");
        if ui
            .selectable_label(filter.selected_domain.is_none(), all_label)
            .clicked()
        {
            filter.selected_domain = None;
            filter.preferred_slug = None;
        }
        for &domain in AcademicDomain::ALL {
            let has_domain = stories
                .iter()
                .any(|s| academic_domain_for_slug(s.domain) == domain);
            if !has_domain {
                continue;
            }
            let count = filter.count_matching_stories_in_domain(stories, Some(domain));
            let is_selected = filter.selected_domain == Some(domain);
            let chip_label = format!("{} ({count})", domain.name());
            if ui.selectable_label(is_selected, chip_label).clicked() {
                filter.selected_domain = (!is_selected).then_some(domain);
                filter.preferred_slug = None;
            }
        }
    });
}

pub(super) fn render_stories_scroll_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    filtered_stories: &[&'static PresetStoryMetadata],
    filter: &mut StarterGalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<&'static PresetStoryMetadata> {
    let mut selected = None;
    let scroll_h = ui
        .available_height()
        .clamp(84.0, (m.max_h - 150.0).max(84.0));
    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_max_width(m.inner_w);
            if filtered_stories.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(18.0);
                    ui.label(
                        egui::RichText::new(
                            "No guided stories match the active search or domain filter.",
                        )
                        .italics(),
                    );
                    ui.add_space(8.0);
                    if ui.button("Show All Stories").clicked() {
                        filter.reset_to_all();
                        filter.active_tab = StarterGalleryTab::GuidedStories;
                    }
                });
                return;
            }
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            for row in filtered_stories.chunks(m.cols.max(1)) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
                    for &story in row {
                        let card = story_to_starter_card(story);
                        if render_starter_card_with_verb(ui, &card, m.card_w, "▶ Launch") {
                            selected = Some(story);
                            frame_ctx.close();
                        }
                    }
                });
            }
        });
    selected
}
