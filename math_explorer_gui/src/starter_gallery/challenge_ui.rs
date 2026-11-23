//! Sub-tab mode bar, Challenge domain filter chips, and Inverse + AST Goal Challenge cards.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cards::render_card_header;
use super::filter::{academic_domain_for_slug, StarterGalleryFilterState};
use super::modal_ui::GalleryOverlayMetrics;
use super::{ChallengeCard, GallerySelection, GoalChallengeMetadata, StarterGalleryTab};
use crate::app::domain_bar::AcademicDomain;
use crate::app::SemanticPalette;
use eframe::egui;

pub(super) fn render_gallery_mode_tabs(
    ui: &mut egui::Ui,
    filter: &mut StarterGalleryFilterState,
    counts: (usize, usize, usize),
) {
    let (starter_count, story_count, challenge_count) = counts;
    ui.horizontal_wrapped(|ui| {
        let tabs = [
            (
                StarterGalleryTab::Starters,
                format!("🚀 Starter Templates ({starter_count})"),
            ),
            (
                StarterGalleryTab::GuidedStories,
                format!("📖 Guided Stories ({story_count})"),
            ),
            (
                StarterGalleryTab::Challenges,
                format!("🎯 Challenges ({challenge_count})"),
            ),
        ];
        for (tab, label) in tabs {
            if ui
                .selectable_label(filter.active_tab == tab, label)
                .clicked()
            {
                filter.active_tab = tab;
            }
        }
    });
}

pub(super) fn render_challenge_domain_chips(
    ui: &mut egui::Ui,
    challenges: &[ChallengeCard],
    goals: &[&'static GoalChallengeMetadata],
    filter: &mut StarterGalleryFilterState,
    inner_w: f32,
) {
    let all_count = filter.count_matching_challenges_in_domain(challenges, goals, None);
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
                .any(|c| academic_domain_for_slug(c.domain) == domain)
                || goals
                    .iter()
                    .any(|g| academic_domain_for_slug(g.domain) == domain);
            if !has_domain {
                continue;
            }
            let count = filter.count_matching_challenges_in_domain(challenges, goals, Some(domain));
            let is_selected = filter.selected_domain == Some(domain);
            let chip_label = format!("{} ({count})", domain.name());
            if ui.selectable_label(is_selected, chip_label).clicked() {
                filter.selected_domain = (!is_selected).then_some(domain);
                filter.preferred_slug = None;
            }
        }
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
                render_card_header(
                    ui,
                    card.icon,
                    card.title,
                    card.category,
                    card.domain,
                    palette.info,
                    card_width,
                );
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

fn render_goal_challenge_card(
    ui: &mut egui::Ui,
    goal: &GoalChallengeMetadata,
    card_width: f32,
) -> bool {
    let mut launched = false;
    let palette = SemanticPalette::from_visuals(ui.visuals());
    egui::Frame::group(ui.style())
        .inner_margin(14.0)
        .corner_radius(8.0)
        .stroke(egui::Stroke::new(1.5_f32, palette.accent))
        .show(ui, |ui| {
            ui.set_width(card_width);
            ui.set_max_width(card_width);
            ui.set_min_height(175.0);
            ui.vertical(|ui| {
                ui.set_max_width(card_width);
                render_card_header(
                    ui,
                    goal.icon,
                    goal.title,
                    goal.category,
                    goal.domain,
                    palette.accent,
                    card_width,
                );
                ui.add_space(6.0);
                let desc = egui::RichText::new(goal.description)
                    .size(12.0)
                    .color(ui.visuals().text_color());
                ui.add(egui::Label::new(desc).wrap());
                ui.add_space(4.0);
                let tgt = egui::RichText::new(format!("🎯 {}", goal.goal_summary))
                    .size(11.0)
                    .color(palette.accent);
                ui.add(egui::Label::new(tgt).wrap());
                ui.add_space(8.0);
                let launch_btn = egui::Button::new(format!("🎯 Start {}", goal.title)).truncate();
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
    filtered_goals: &[&'static GoalChallengeMetadata],
    filter: &mut StarterGalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    let mut selected = None;
    let scroll_h = ui
        .available_height()
        .clamp(84.0, (m.max_h - 150.0).max(84.0));
    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_max_width(m.inner_w);
            if filtered_challenges.is_empty() && filtered_goals.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(18.0);
                    ui.label(
                        egui::RichText::new(
                            "No challenges match the active search or domain filter.",
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
                            selected = Some(GallerySelection::Challenge(ch.clone()));
                            frame_ctx.close();
                        }
                    }
                });
            }
            for row in filtered_goals.chunks(m.cols.max(1)) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
                    for &goal in row {
                        if render_goal_challenge_card(ui, goal, m.card_w) {
                            selected = Some(GallerySelection::GoalChallenge(goal));
                            frame_ctx.close();
                        }
                    }
                });
            }
        });
    selected
}
