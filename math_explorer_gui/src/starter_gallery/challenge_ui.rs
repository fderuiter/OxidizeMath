//! Inverse Challenge sub-tab controls, domain filter chips, and card rendering for `StarterGallery`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::filter::{academic_domain_for_slug, module_title_for_slug, StarterGalleryFilterState};
use super::modal_ui::GalleryOverlayMetrics;
use super::{ChallengeCard, StarterGalleryTab};
use crate::app::domain_bar::AcademicDomain;
use crate::app::SemanticPalette;
use eframe::egui;

pub(super) fn render_gallery_mode_tabs(
    ui: &mut egui::Ui,
    filter: &mut StarterGalleryFilterState,
    starter_count: usize,
    challenge_count: usize,
) {
    ui.horizontal_wrapped(|ui| {
        let starters_label = format!("🚀 Starter Templates ({starter_count})");
        let challenges_label = format!("🎯 Inverse Challenges ({challenge_count})");
        if ui
            .selectable_label(
                filter.active_tab == StarterGalleryTab::Starters,
                starters_label,
            )
            .clicked()
        {
            filter.active_tab = StarterGalleryTab::Starters;
        }
        if ui
            .selectable_label(
                filter.active_tab == StarterGalleryTab::Challenges,
                challenges_label,
            )
            .clicked()
        {
            filter.active_tab = StarterGalleryTab::Challenges;
        }
    });
}

pub(super) fn render_challenge_domain_chips(
    ui: &mut egui::Ui,
    challenges: &[ChallengeCard],
    filter: &mut StarterGalleryFilterState,
    inner_w: f32,
) {
    let all_count = filter.count_matching_challenges_in_domain(challenges, None);
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
            let has_domain = challenges
                .iter()
                .any(|c| academic_domain_for_slug(c.domain) == domain);
            if !has_domain {
                continue;
            }
            let count = filter.count_matching_challenges_in_domain(challenges, Some(domain));
            let is_selected = filter.selected_domain == Some(domain);
            let chip_label = format!("{} ({count})", domain.name());
            if ui.selectable_label(is_selected, chip_label).clicked() {
                filter.selected_domain = (!is_selected).then_some(domain);
                filter.preferred_slug = None;
            }
        }
    });
}

fn render_challenge_card_header(
    ui: &mut egui::Ui,
    card: &ChallengeCard,
    card_width: f32,
    info_color: egui::Color32,
) {
    let module_title = module_title_for_slug(card.domain);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(card.icon).size(26.0));
        let text_w = (card_width - 42.0).max(56.0);
        ui.vertical(|ui| {
            ui.set_max_width(text_w);
            ui.add(egui::Label::new(egui::RichText::new(card.title).strong().size(15.0)).wrap());
            let cat = egui::RichText::new(format!("[{} · {}]", card.category, module_title))
                .small()
                .color(info_color);
            ui.add(egui::Label::new(cat).truncate());
        });
    });
}

fn render_challenge_card(ui: &mut egui::Ui, card: &ChallengeCard, card_width: f32) -> bool {
    let mut launched = false;
    let palette = SemanticPalette::from_visuals(ui.visuals());
    egui::Frame::group(ui.style())
        .inner_margin(14.0)
        .corner_radius(8.0)
        .stroke(egui::Stroke::new(1.5_f32, palette.info))
        .show(ui, |ui| {
            ui.set_width(card_width);
            ui.set_max_width(card_width);
            ui.set_min_height(175.0);
            ui.vertical(|ui| {
                ui.set_max_width(card_width);
                render_challenge_card_header(ui, card, card_width, palette.info);
                ui.add_space(6.0);
                let desc = egui::RichText::new(card.description)
                    .size(12.0)
                    .color(ui.visuals().text_color());
                ui.add(egui::Label::new(desc).wrap());
                ui.add_space(4.0);
                let lock = egui::RichText::new(format!("🔒 {}", card.locked_inputs))
                    .size(11.0)
                    .color(palette.muted_text);
                ui.add(egui::Label::new(lock).wrap());
                let tgt = egui::RichText::new(format!("🎯 {}", card.target_summary))
                    .size(11.0)
                    .color(palette.accent);
                ui.add(egui::Label::new(tgt).wrap());
                ui.add_space(8.0);
                let launch_btn = egui::Button::new(format!("🎯 Start {}", card.title)).truncate();
                if ui.add_sized([card_width, 26.0], launch_btn).clicked() {
                    launched = true;
                }
            });
        });
    launched
}

pub(super) fn render_challenges_scroll_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    filtered_challenges: &[ChallengeCard],
    filter: &mut StarterGalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<ChallengeCard> {
    let mut selected = None;
    let scroll_h = ui
        .available_height()
        .clamp(84.0, (m.max_h - 150.0).max(84.0));
    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_max_width(m.inner_w);
            if filtered_challenges.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(18.0);
                    ui.label(
                        egui::RichText::new(
                            "No inverse challenges match the active search or domain filter.",
                        )
                        .italics(),
                    );
                    ui.add_space(8.0);
                    if ui.button("Show All Challenges").clicked() {
                        filter.reset_to_all();
                        filter.active_tab = StarterGalleryTab::Challenges;
                    }
                });
                return;
            }
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            for row in filtered_challenges.chunks(m.cols.max(1)) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
                    for ch in row {
                        if render_challenge_card(ui, ch, m.card_w) {
                            selected = Some(ch.clone());
                            frame_ctx.close();
                        }
                    }
                });
            }
        });
    selected
}
