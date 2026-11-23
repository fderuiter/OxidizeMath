#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "cards.rs"]
mod cards;

use super::{
    goal_challenge_to_starter_card, story_to_starter_card, ChallengeCard, GallerySelection,
    StarterCard,
};
use cards::{
    compute_gallery_metrics, render_challenges_grid, render_starter_card, render_stories_grid,
    GalleryOverlayMetrics,
};
use eframe::egui;

const DEFER_STARTER_GALLERY_TO_ROOT_KEY: &str = "DEFER_STARTER_GALLERY_TO_ROOT";
const DEFERRED_STARTER_GALLERY_QUEUE_KEY: &str = "DEFERRED_STARTER_GALLERY_QUEUE";

/// Active sub-tab inside the `StarterGallery` modal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StarterGalleryTab {
    /// Guided forward simulation templates / snapshots.
    #[default]
    Starters,
    /// Multi-step narrative tours with synchronized plot callouts.
    GuidedStories,
    /// Goal-driven inverse challenges and playable AST target challenges.
    Challenges,
}

impl StarterGalleryTab {
    /// Alias for [`Self::Starters`] used by snapshot filter callers.
    #[allow(non_upper_case_globals)]
    pub const Snapshots: Self = Self::Starters;
}

#[derive(Clone)]
struct DeferredStarterGalleryRequest {
    pane_idx: usize,
    id_source: String,
    modal_source: String,
    cards: Vec<StarterCard>,
    challenges: Vec<ChallengeCard>,
}

fn resolve_gallery_domain<'a>(
    id_source: &'a str,
    cards: &'a [StarterCard],
    challenges: &'a [ChallengeCard],
) -> &'a str {
    if let Some(first) = cards.first() {
        return first.domain;
    }
    if let Some(first_ch) = challenges.first() {
        return first_ch.domain;
    }
    match id_source.rsplit_once('_') {
        Some((prefix, suffix))
            if !prefix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) =>
        {
            prefix
        }
        _ => id_source,
    }
}

fn load_active_gallery_tab(
    ctx: &egui::Context,
    modal_id: egui::Id,
    id_source: &str,
    counts: (usize, usize, usize),
) -> (egui::Id, egui::Id, StarterGalleryTab) {
    let (starter_count, story_count, challenge_count) = counts;
    let tab_key = modal_id.with("gallery_active_tab");
    let legacy_key = egui::Id::new((id_source, "starter_gallery_active_tab"));
    let default_tab = if starter_count == 0 && challenge_count > 0 {
        StarterGalleryTab::Challenges
    } else if starter_count == 0 && story_count > 0 {
        StarterGalleryTab::GuidedStories
    } else {
        StarterGalleryTab::Starters
    };
    let active_tab = ctx
        .data(|d| {
            d.get_temp::<StarterGalleryTab>(legacy_key)
                .or_else(|| d.get_temp::<StarterGalleryTab>(tab_key))
        })
        .unwrap_or(default_tab);
    (tab_key, legacy_key, active_tab)
}

fn render_gallery_header_and_tabs(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    modal_id: egui::Id,
    id_source: &str,
    counts: (usize, usize, usize),
    inner_w: f32,
) -> StarterGalleryTab {
    let (starter_count, story_count, challenge_count) = counts;
    let (tab_key, legacy_key, mut active_tab) =
        load_active_gallery_tab(ui.ctx(), modal_id, id_source, counts);

    ui.horizontal(|ui| {
        ui.set_width(inner_w);
        let heading = egui::RichText::new("🚀 Starter & Challenge Gallery").heading();
        ui.add(egui::Label::new(heading).truncate());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("✖ Close Gallery").clicked() {
                frame_ctx.close();
            }
        });
    });
    ui.add_space(4.0);
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
            if ui.selectable_label(active_tab == tab, label).clicked() {
                active_tab = tab;
            }
        }
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(tab_key, active_tab);
        d.insert_temp(legacy_key, active_tab);
    });
    let subtitle = match active_tab {
        StarterGalleryTab::Starters => {
            "Choose a guided starter model to launch pre-configured simulation parameters:"
        }
        StarterGalleryTab::GuidedStories => {
            "Walk through a multi-step mathematical paradox tour with synchronized plot callouts:"
        }
        StarterGalleryTab::Challenges => {
            "Solve an inverse boundary challenge or live AST observable target challenge:"
        }
    };
    let sub_text = egui::RichText::new(subtitle).color(ui.visuals().weak_text_color());
    ui.add(egui::Label::new(sub_text).wrap());
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(8.0);
    active_tab
}

fn render_starter_gallery_body(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    modal_id: egui::Id,
    id_source: &str,
    cards: &[StarterCard],
    challenges: &[ChallengeCard],
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    let domain = resolve_gallery_domain(id_source, cards, challenges);
    let stories = super::get_stories_for_domain(domain);
    let goal_challenges = super::get_goal_challenges_for_domain(domain);
    let mut selected = None;
    ui.set_max_width(m.inner_w);
    ui.set_width(m.inner_w);
    let counts = (
        cards.len(),
        stories.len(),
        challenges.len() + goal_challenges.len(),
    );
    let active_tab =
        render_gallery_header_and_tabs(ui, frame_ctx, modal_id, id_source, counts, m.inner_w);

    egui::ScrollArea::vertical()
        .max_height(m.scroll_h)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_max_width(m.inner_w);
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            match active_tab {
                StarterGalleryTab::Starters => {
                    for row in cards.chunks(m.cols.max(1)) {
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
                }
                StarterGalleryTab::GuidedStories => {
                    selected = render_stories_grid(ui, frame_ctx, &stories, m);
                }
                StarterGalleryTab::Challenges => {
                    selected =
                        render_challenges_grid(ui, frame_ctx, challenges, &goal_challenges, m);
                }
            }
        });
    selected
}

/// Renders the central canvas starter, story, & challenge gallery overlay clamped to root window bounds.
pub fn show_starter_gallery_with_challenges(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    challenges: &[ChallengeCard],
    is_open: &mut bool,
) -> Option<GallerySelection> {
    let domain = resolve_gallery_domain(id_source, cards, challenges);
    if cards.is_empty()
        && challenges.is_empty()
        && super::get_stories_for_domain(domain).is_empty()
        && super::get_goal_challenges_for_domain(domain).is_empty()
    {
        return None;
    }
    let m = compute_gallery_metrics(ctx);
    let mut selected = None;
    let modal_id = egui::Id::new(format!("{id_source}_starter_gallery_overlay"));
    let frame = egui::Frame::window(&ctx.style())
        .inner_margin(16)
        .corner_radius(12.0)
        .shadow(ctx.style().visuals.window_shadow);

    crate::ModalLifecycle::new(modal_id)
        .frame(frame)
        .backdrop_color(egui::Color32::from_black_alpha(120))
        .max_width(m.max_w)
        .max_height(m.max_h)
        .show(ctx, is_open, |ui, frame_ctx| {
            selected = render_starter_gallery_body(
                ui, frame_ctx, modal_id, id_source, cards, challenges, &m,
            );
        });

    selected
}

/// Renders the central canvas starter gallery card overlay clamped to root window bounds.
pub fn show_starter_gallery_overlay(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    is_open: &mut bool,
) -> Option<StarterCard> {
    let domain = resolve_gallery_domain(id_source, cards, &[]);
    let challenges = super::get_challenge_cards_for_domain(domain);
    match show_starter_gallery_with_challenges(ctx, id_source, cards, &challenges, is_open) {
        Some(GallerySelection::Starter(card)) => Some(card),
        Some(GallerySelection::Story(story)) => Some(story_to_starter_card(story)),
        Some(GallerySelection::GoalChallenge(goal)) => Some(goal_challenge_to_starter_card(goal)),
        Some(GallerySelection::Challenge(_)) | None => None,
    }
}

/// Enables root-level hoisting of `StarterGallery` modals rendered during pane updates.
pub(crate) fn begin_root_starter_gallery_scope(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(DEFER_STARTER_GALLERY_TO_ROOT_KEY), true);
        d.remove::<Vec<DeferredStarterGalleryRequest>>(egui::Id::new(
            DEFERRED_STARTER_GALLERY_QUEUE_KEY,
        ));
    });
}

/// Enqueues a pane-opened `StarterGallery` for root `egui::Context` rendering if a root scope is active.
pub(crate) fn try_defer_starter_gallery(
    ctx: &egui::Context,
    id_source: &str,
    modal_source: &str,
    cards: Vec<StarterCard>,
    challenges: Vec<ChallengeCard>,
) -> bool {
    ctx.data_mut(|d| {
        let defer = d
            .get_temp::<bool>(egui::Id::new(DEFER_STARTER_GALLERY_TO_ROOT_KEY))
            .unwrap_or(false);
        let Some(pane_idx) = d.get_temp::<usize>(egui::Id::new(
            crate::app::viewport::focus::CURRENT_RENDERING_PANE_KEY,
        )) else {
            return false;
        };
        if !defer {
            return false;
        }
        let key = egui::Id::new(DEFERRED_STARTER_GALLERY_QUEUE_KEY);
        let mut queue = d
            .get_temp::<Vec<DeferredStarterGalleryRequest>>(key)
            .unwrap_or_default();
        queue.push(DeferredStarterGalleryRequest {
            pane_idx,
            id_source: id_source.to_string(),
            modal_source: modal_source.to_string(),
            cards,
            challenges,
        });
        d.insert_temp(key, queue);
        true
    })
}

/// Renders all deferred `StarterGallery` modals at the root `egui::Context` level and applies selections to their owning panes.
pub(crate) fn flush_deferred_starter_galleries(
    app: &mut crate::app::MathExplorerApp,
    ctx: &egui::Context,
) {
    let requests = ctx.data_mut(|d| {
        d.remove::<bool>(egui::Id::new(DEFER_STARTER_GALLERY_TO_ROOT_KEY));
        let key = egui::Id::new(DEFERRED_STARTER_GALLERY_QUEUE_KEY);
        let q = d.get_temp::<Vec<DeferredStarterGalleryRequest>>(key);
        d.remove::<Vec<DeferredStarterGalleryRequest>>(key);
        q.unwrap_or_default()
    });
    for req in requests {
        let mut is_open = true;
        let selected = show_starter_gallery_with_challenges(
            ctx,
            &req.modal_source,
            &req.cards,
            &req.challenges,
            &mut is_open,
        );
        if selected.is_some() {
            is_open = false;
        }
        if let Some(tab) = app.pane_tab_mut(req.pane_idx) {
            tab.apply_starter_gallery_selection(ctx, &req.id_source, is_open, selected);
        }
    }
}
