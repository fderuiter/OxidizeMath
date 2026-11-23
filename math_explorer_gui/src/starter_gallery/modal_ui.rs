//! Responsive WCAG 2.2 AA UI rendering for the global Starter Template Gallery modal.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cards::{
    render_starter_card_with_verb, render_stories_scroll_grid, render_story_domain_chips,
};
use super::challenge_ui::{
    render_challenge_domain_chips, render_challenges_scroll_grid, render_gallery_mode_tabs,
};
use super::filter::{academic_domain_for_slug, StarterGalleryFilterState};
use super::{
    ChallengeCard, GallerySelection, GoalChallengeMetadata, PresetStoryMetadata, StarterCard,
    StarterGalleryTab,
};
use crate::app::domain_bar::AcademicDomain;
use crate::app::SemanticPalette;
use eframe::egui;

/// Temp context key storing the gallery search `WidgetInfo` descriptor for accessibility tests.
pub(crate) const STARTER_GALLERY_SEARCH_INFO_ID: &str = "starter_gallery_search_info";
const GALLERY_SEARCH_LABEL: &str = "Search starter template gallery";
const GALLERY_SEARCH_HINT: &str =
    "Search models, equations, topics, or domains (e.g. Mandelbrot, Kermack, Harmonic)...";

pub(super) struct GalleryItems<'a> {
    pub(super) cards: &'a [StarterCard],
    pub(super) stories: &'a [&'static PresetStoryMetadata],
    pub(super) challenges: &'a [ChallengeCard],
    pub(super) goal_challenges: &'a [&'static GoalChallengeMetadata],
}

pub(super) struct GalleryOverlayMetrics {
    pub(super) max_w: f32,
    pub(super) max_h: f32,
    pub(super) inner_w: f32,
    pub(super) card_w: f32,
    pub(super) cols: usize,
}

pub(super) fn compute_gallery_metrics(ctx: &egui::Context) -> GalleryOverlayMetrics {
    let raw_screen = ctx.content_rect();
    let avail_w = if raw_screen.width().is_finite() && raw_screen.width() > 64.0 {
        raw_screen.width()
    } else {
        1024.0
    };
    let avail_h = if raw_screen.height().is_finite() && raw_screen.height() > 64.0 {
        raw_screen.height()
    } else {
        768.0
    };
    let max_w = (avail_w - 32.0).clamp(180.0, 820.0);
    let max_h = (avail_h - 32.0).clamp(220.0, 620.0);
    let inner_w = (max_w - 36.0).max(144.0);
    let grid_avail_w = (inner_w - 14.0).max(130.0);
    let (cols, card_w) = if grid_avail_w >= 660.0 {
        (3, ((grid_avail_w - 28.0) / 3.0 - 30.0).floor())
    } else if grid_avail_w >= 430.0 {
        (2, ((grid_avail_w - 14.0) / 2.0 - 30.0).floor().min(240.0))
    } else {
        (1, (grid_avail_w - 30.0).clamp(98.0, 240.0))
    };
    GalleryOverlayMetrics {
        max_w,
        max_h,
        inner_w,
        card_w,
        cols,
    }
}

fn render_gallery_header(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    inner_w: f32,
    shown_count: usize,
    total_count: usize,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.horizontal(|ui| {
        ui.set_width(inner_w);
        ui.add(
            egui::Label::new(egui::RichText::new("🚀 Starter Template Gallery").heading())
                .truncate(),
        );
        ui.label(
            egui::RichText::new(format!("({shown_count} of {total_count})"))
                .small()
                .color(palette.muted_text),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("✖ Close Gallery").clicked() {
                frame_ctx.close();
            }
        });
    });
}

fn render_search_bar(ui: &mut egui::Ui, filter: &mut StarterGalleryFilterState, inner_w: f32) {
    let prev_query = filter.search_query.clone();
    let has_query = !filter.search_query.is_empty();
    let text_w = if has_query {
        (inner_w - 68.0).max(80.0)
    } else {
        inner_w
    };
    ui.horizontal(|ui| {
        ui.set_max_width(inner_w);
        let resp = ui.add(
            egui::TextEdit::singleline(&mut filter.search_query)
                .hint_text(GALLERY_SEARCH_HINT)
                .desired_width(text_w),
        );
        let search_info = egui::WidgetInfo {
            label: Some(GALLERY_SEARCH_LABEL.to_string()),
            ..egui::WidgetInfo::text_edit(
                ui.is_enabled(),
                &prev_query,
                &filter.search_query,
                GALLERY_SEARCH_HINT,
            )
        };
        resp.widget_info({
            let info = search_info.clone();
            move || info.clone()
        });
        ui.ctx().data_mut(|d| {
            d.insert_temp(egui::Id::new(STARTER_GALLERY_SEARCH_INFO_ID), search_info);
        });
        if has_query && ui.button("Clear").clicked() {
            filter.search_query.clear();
        }
    });
}

fn render_domain_filter_chips(
    ui: &mut egui::Ui,
    cards: &[StarterCard],
    filter: &mut StarterGalleryFilterState,
    inner_w: f32,
) {
    let all_count = filter.count_matching_in_domain(cards, None);
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
            let total_in_domain = cards
                .iter()
                .any(|c| academic_domain_for_slug(c.domain) == domain);
            if !total_in_domain {
                continue;
            }
            let count = filter.count_matching_in_domain(cards, Some(domain));
            let is_selected = filter.selected_domain == Some(domain);
            let chip_label = format!("{} ({count})", domain.name());
            if ui.selectable_label(is_selected, chip_label).clicked() {
                filter.selected_domain = (!is_selected).then_some(domain);
                filter.preferred_slug = None;
            }
        }
    });
}

fn render_cards_scroll_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    filtered_cards: &[StarterCard],
    filter: &mut StarterGalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<StarterCard> {
    let mut selected_card = None;
    let scroll_h = ui
        .available_height()
        .clamp(84.0, (m.max_h - 130.0).max(84.0));
    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_max_width(m.inner_w);
            if filtered_cards.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(18.0);
                    ui.label(
                        egui::RichText::new(
                            "No starter presets match the active search or domain filter.",
                        )
                        .italics(),
                    );
                    ui.add_space(8.0);
                    if ui.button("Show All Presets").clicked() {
                        filter.reset_to_all();
                    }
                });
                return;
            }
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            for row in filtered_cards.chunks(m.cols.max(1)) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
                    for card in row {
                        if render_starter_card_with_verb(ui, card, m.card_w, "▶ Launch") {
                            selected_card = Some(card.clone());
                            frame_ctx.close();
                        }
                    }
                });
            }
        });
    selected_card
}

pub(super) fn render_searchable_gallery_body(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    items: &GalleryItems<'_>,
    filter: &mut StarterGalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    ui.set_max_width(m.inner_w);
    ui.set_width(m.inner_w);

    let pre_cards = filter.filter_cards(items.cards);
    let pre_stories = filter.filter_stories(items.stories);
    let pre_challenges = filter.filter_challenges(items.challenges);
    let pre_goals = filter.filter_goal_challenges(items.goal_challenges);
    let total_challenges = items.challenges.len() + items.goal_challenges.len();
    let shown_challenges = pre_challenges.len() + pre_goals.len();

    let (shown, total) = match filter.active_tab {
        StarterGalleryTab::Starters => (pre_cards.len(), items.cards.len()),
        StarterGalleryTab::GuidedStories => (pre_stories.len(), items.stories.len()),
        StarterGalleryTab::Challenges => (shown_challenges, total_challenges),
    };
    render_gallery_header(ui, frame_ctx, m.inner_w, shown, total);
    if !items.stories.is_empty() || total_challenges > 0 {
        ui.add_space(4.0);
        let counts = (pre_cards.len(), pre_stories.len(), shown_challenges);
        render_gallery_mode_tabs(ui, filter, counts);
    }
    ui.add_space(4.0);
    render_search_bar(ui, filter, m.inner_w);
    ui.add_space(4.0);
    match filter.active_tab {
        StarterGalleryTab::Starters => {
            render_domain_filter_chips(ui, items.cards, filter, m.inner_w)
        }
        StarterGalleryTab::GuidedStories => {
            render_story_domain_chips(ui, items.stories, filter, m.inner_w);
        }
        StarterGalleryTab::Challenges => render_challenge_domain_chips(
            ui,
            items.challenges,
            items.goal_challenges,
            filter,
            m.inner_w,
        ),
    }
    ui.add_space(4.0);
    ui.separator();
    ui.add_space(6.0);

    render_active_tab_content(ui, frame_ctx, items, filter, m)
}

fn render_active_tab_content(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    items: &GalleryItems<'_>,
    filter: &mut StarterGalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    match filter.active_tab {
        StarterGalleryTab::Starters => {
            let filtered = filter.filter_cards(items.cards);
            if !ui.is_sizing_pass() {
                filter.sync_live_announcement(ui.ctx(), filtered.len());
            }
            render_cards_scroll_grid(ui, frame_ctx, &filtered, filter, m)
                .map(GallerySelection::Starter)
        }
        StarterGalleryTab::GuidedStories => {
            let filtered = filter.filter_stories(items.stories);
            if !ui.is_sizing_pass() {
                filter.sync_live_announcement(ui.ctx(), filtered.len());
            }
            render_stories_scroll_grid(ui, frame_ctx, &filtered, filter, m)
                .map(GallerySelection::Story)
        }
        StarterGalleryTab::Challenges => {
            let filtered_inv = filter.filter_challenges(items.challenges);
            let filtered_goals = filter.filter_goal_challenges(items.goal_challenges);
            if !ui.is_sizing_pass() {
                filter.sync_live_announcement(ui.ctx(), filtered_inv.len() + filtered_goals.len());
            }
            render_challenges_scroll_grid(ui, frame_ctx, &filtered_inv, &filtered_goals, filter, m)
        }
    }
}
