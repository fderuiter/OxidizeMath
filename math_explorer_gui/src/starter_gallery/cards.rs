#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::{
    goal_challenge_to_starter_card, story_to_starter_card, ChallengeCard, GallerySelection,
    GoalChallengeMetadata, PresetStoryMetadata, StarterCard,
};
use crate::app::SemanticPalette;
use eframe::egui;

pub(super) struct GalleryOverlayMetrics {
    pub max_w: f32,
    pub max_h: f32,
    pub inner_w: f32,
    pub scroll_h: f32,
    pub card_w: f32,
    pub cols: usize,
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
    let max_w = (avail_w - 32.0).clamp(180.0, 740.0);
    let max_h = (avail_h - 32.0).clamp(220.0, 620.0);
    let inner_w = (max_w - 36.0).max(144.0);
    let scroll_h = (max_h - 196.0).max(88.0);
    let grid_avail_w = (inner_w - 14.0).max(130.0);
    let (cols, card_w) = if grid_avail_w >= 660.0 {
        (3, ((grid_avail_w - 28.0) / 3.0 - 30.0).floor())
    } else if grid_avail_w >= 430.0 {
        (2, ((grid_avail_w - 14.0) / 2.0 - 30.0).floor().min(215.0))
    } else {
        (1, (grid_avail_w - 30.0).clamp(98.0, 215.0))
    };
    GalleryOverlayMetrics {
        max_w,
        max_h,
        inner_w,
        scroll_h,
        card_w,
        cols,
    }
}

fn render_card_header(
    ui: &mut egui::Ui,
    icon: &str,
    title: &str,
    category: &str,
    cat_color: egui::Color32,
    card_width: f32,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(icon).size(26.0));
        let text_w = (card_width - 42.0).max(56.0);
        ui.vertical(|ui| {
            ui.set_max_width(text_w);
            ui.add(egui::Label::new(egui::RichText::new(title).strong().size(15.0)).wrap());
            let cat = egui::RichText::new(format!("[{category}]"))
                .small()
                .color(cat_color);
            ui.add(egui::Label::new(cat).truncate());
        });
    });
}

pub(super) fn render_starter_card(
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
            ui.set_min_height(150.0);
            ui.vertical(|ui| {
                ui.set_max_width(card_width);
                render_card_header(
                    ui,
                    card.icon,
                    card.title,
                    card.category,
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

pub(super) fn render_challenge_card(
    ui: &mut egui::Ui,
    card: &ChallengeCard,
    card_width: f32,
) -> bool {
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
                    palette.info,
                    card_width,
                );
                ui.add_space(6.0);
                let desc = egui::RichText::new(card.description)
                    .size(12.0)
                    .color(ui.visuals().text_color());
                ui.add(egui::Label::new(desc).wrap());
                ui.add_space(4.0);
                let locked = egui::RichText::new(format!("🔒 {}", card.locked_inputs))
                    .size(11.0)
                    .color(palette.muted_text);
                ui.add(egui::Label::new(locked).wrap());
                let target = egui::RichText::new(format!("🎯 {}", card.target_summary))
                    .size(11.0)
                    .color(palette.accent);
                ui.add(egui::Label::new(target).wrap());
                ui.add_space(8.0);
                let launch_btn = egui::Button::new(format!("🎯 Start {}", card.title)).truncate();
                if ui.add_sized([card_width, 26.0], launch_btn).clicked() {
                    launched = true;
                }
            });
        });
    launched
}

pub(super) fn render_challenges_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    challenges: &[ChallengeCard],
    goal_challenges: &[&'static GoalChallengeMetadata],
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    if challenges.is_empty() && goal_challenges.is_empty() {
        ui.label(
            "No challenges are registered for this domain yet. Try Financial Math, Game Theory, MRI Physics, Clinical Trials, Fluid Dynamics, or Medical Physics.",
        );
        return None;
    }
    let mut selected = None;
    for row in challenges.chunks(m.cols.max(1)) {
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
    for row in goal_challenges.chunks(m.cols.max(1)) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            for &goal in row {
                let card = goal_challenge_to_starter_card(goal);
                if render_starter_card(ui, &card, m.card_w, "🎯 Start") {
                    selected = Some(GallerySelection::GoalChallenge(goal));
                    frame_ctx.close();
                }
            }
        });
    }
    selected
}

pub(super) fn render_stories_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    stories: &[&'static PresetStoryMetadata],
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    if stories.is_empty() {
        ui.label(
            "No guided stories are registered for this domain yet. Try Financial Math, Game Theory, or Fluid Dynamics.",
        );
        return None;
    }
    let mut selected = None;
    for row in stories.chunks(m.cols.max(1)) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            for &story in row {
                let card = story_to_starter_card(story);
                if render_starter_card(ui, &card, m.card_w, "▶ Launch") {
                    selected = Some(GallerySelection::Story(story));
                    frame_ctx.close();
                }
            }
        });
    }
    selected
}

pub(super) struct FilteredGalleryItems {
    pub starters: Vec<StarterCard>,
    pub stories: Vec<&'static PresetStoryMetadata>,
    pub challenges: Vec<ChallengeCard>,
    pub goals: Vec<&'static GoalChallengeMetadata>,
    pub has_starters: bool,
    pub has_stories: bool,
    pub has_challenges: bool,
}

impl FilteredGalleryItems {
    pub(super) fn build(
        cards: &[StarterCard],
        stories: &[&'static PresetStoryMetadata],
        challenges: &[ChallengeCard],
        goals: &[&'static GoalChallengeMetadata],
        filter_state: &crate::starter_gallery::filter_ui::GalleryFilterState,
    ) -> Self {
        use crate::starter_gallery::filter_ui::{
            filter_challenge_cards, filter_goal_challenges, filter_starter_cards, filter_stories,
        };
        Self {
            starters: filter_starter_cards(
                cards,
                &filter_state.query,
                filter_state.selected_domain,
            ),
            stories: filter_stories(stories, &filter_state.query, filter_state.selected_domain),
            challenges: filter_challenge_cards(
                challenges,
                &filter_state.query,
                filter_state.selected_domain,
            ),
            goals: filter_goal_challenges(goals, &filter_state.query, filter_state.selected_domain),
            has_starters: !cards.is_empty(),
            has_stories: !stories.is_empty(),
            has_challenges: !challenges.is_empty() || !goals.is_empty(),
        }
    }

    pub(super) fn counts(&self) -> (usize, usize, usize) {
        (
            self.starters.len(),
            self.stories.len(),
            self.challenges.len() + self.goals.len(),
        )
    }

    pub(super) fn render_tab(
        &self,
        ui: &mut egui::Ui,
        frame_ctx: &mut crate::ModalFrameContext,
        active_tab: super::StarterGalleryTab,
        filter_state: &mut crate::starter_gallery::filter_ui::GalleryFilterState,
        m: &GalleryOverlayMetrics,
    ) -> Option<GallerySelection> {
        use crate::starter_gallery::filter_ui::render_empty_filtered_state;
        match active_tab {
            super::StarterGalleryTab::Starters => {
                if self.starters.is_empty() && self.has_starters {
                    render_empty_filtered_state(ui, filter_state, "starter templates");
                    return None;
                }
                let mut selected = None;
                for row in self.starters.chunks(m.cols.max(1)) {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
                        for card in row {
                            if render_starter_card(ui, card, m.card_w, "▶ Launch") {
                                selected = Some(GallerySelection::Starter(card.clone()));
                                frame_ctx.close();
                            }
                        }
                    });
                }
                selected
            }
            super::StarterGalleryTab::GuidedStories => {
                if self.stories.is_empty() && self.has_stories {
                    render_empty_filtered_state(ui, filter_state, "guided stories");
                    return None;
                }
                render_stories_grid(ui, frame_ctx, &self.stories, m)
            }
            super::StarterGalleryTab::Challenges => {
                if self.challenges.is_empty() && self.goals.is_empty() && self.has_challenges {
                    render_empty_filtered_state(ui, filter_state, "challenges");
                    return None;
                }
                render_challenges_grid(ui, frame_ctx, &self.challenges, &self.goals, m)
            }
        }
    }
}
