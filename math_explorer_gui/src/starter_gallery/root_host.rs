#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::filter::StarterGalleryFilterState;
use super::modal_ui::{compute_gallery_metrics, render_searchable_gallery_body, GalleryItems};
use super::{
    goal_challenge_to_starter_card, story_to_starter_card, ChallengeCard, GallerySelection,
    GoalChallengeMetadata, PresetStoryMetadata, StarterCard, StarterGalleryTab,
};
use eframe::egui;
use std::collections::HashSet;

const DEFER_STARTER_GALLERY_TO_ROOT_KEY: &str = "DEFER_STARTER_GALLERY_TO_ROOT";
const DEFERRED_STARTER_GALLERY_QUEUE_KEY: &str = "DEFERRED_STARTER_GALLERY_QUEUE";
const GLOBAL_STARTER_GALLERY_MODAL_ID: &str = "global_starter_gallery_modal";

#[derive(Clone)]
struct DeferredStarterGalleryRequest {
    pane_idx: usize,
    id_source: String,
    modal_source: String,
    cards: Vec<StarterCard>,
    challenges: Vec<ChallengeCard>,
    initial_domain_slug: Option<String>,
}

/// Root-level global Starter Template, Story & Challenge Gallery modal state owned by `MathExplorerApp`.
#[derive(Clone, Debug, Default)]
pub struct GlobalStarterGallery {
    /// Whether the global starter gallery modal is currently open.
    pub is_open: bool,
    /// True only for the initial cold-boot presentation before any session/URL/tab override.
    pub is_cold_boot: bool,
    /// Active text search, sub-tab, and `AcademicDomain` filter state.
    pub filter: StarterGalleryFilterState,
    was_open: bool,
}

impl GlobalStarterGallery {
    /// Creates a cold-boot global starter gallery showing all registered presets across all domains.
    #[must_use]
    pub fn cold_boot() -> Self {
        let has_presets = !super::get_all_starter_cards().is_empty();
        Self {
            is_open: has_presets,
            is_cold_boot: has_presets,
            filter: StarterGalleryFilterState::default(),
            was_open: has_presets,
        }
    }

    /// Opens the global starter gallery across `All Domains` and emits a polite live announcement.
    pub fn open_all(&mut self, ctx: &egui::Context) {
        self.is_open = true;
        self.is_cold_boot = false;
        self.was_open = true;
        self.filter.reset_to_all();
        crate::accessibility::announce_live_status(ctx, "Starter template gallery opened");
    }

    /// Opens the global starter gallery pre-filtered to `domain_slug`'s `AcademicDomain`.
    pub fn open_for_domain(&mut self, ctx: &egui::Context, domain_slug: &str) {
        self.is_open = true;
        self.is_cold_boot = false;
        self.was_open = true;
        self.filter.reset_to_domain(domain_slug);
        crate::accessibility::announce_live_status(ctx, "Starter template gallery opened");
    }

    /// Silently dismisses the gallery if it was only open due to initial cold boot.
    pub fn dismiss_cold_boot(&mut self) {
        if self.is_cold_boot {
            self.is_open = false;
            self.is_cold_boot = false;
            self.was_open = false;
        }
    }

    /// Closes the gallery without emitting a dismissal announcement (e.g. when superseded by `Ctrl+K`).
    pub fn close_silently(&mut self) {
        self.is_open = false;
        self.is_cold_boot = false;
        self.was_open = false;
    }

    /// Renders the global starter, story & challenge gallery via `ModalLifecycle`.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<GallerySelection> {
        let cards = super::get_all_starter_cards();
        let stories = super::get_all_stories();
        let challenges = super::get_all_challenge_cards();
        let goal_challenges = super::get_all_goal_challenges();
        let items = GalleryItems {
            cards: &cards,
            stories: &stories,
            challenges: &challenges,
            goal_challenges: &goal_challenges,
        };
        let was_open = self.was_open;
        let selected = run_gallery_modal_lifecycle(
            ctx,
            egui::Id::new(GLOBAL_STARTER_GALLERY_MODAL_ID),
            &items,
            &mut self.filter,
            &mut self.is_open,
        );
        if selected.is_some() {
            self.is_open = false;
            self.is_cold_boot = false;
        } else if (was_open || self.is_cold_boot) && !self.is_open {
            self.is_cold_boot = false;
            crate::accessibility::announce_live_status(ctx, "Starter template gallery closed");
        }
        self.was_open = self.is_open;
        selected
    }
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

fn spans_multiple_academic_domains(cards: &[StarterCard], challenges: &[ChallengeCard]) -> bool {
    let mut seen = HashSet::new();
    cards
        .iter()
        .map(|c| c.domain)
        .chain(challenges.iter().map(|c| c.domain))
        .any(|dom| seen.insert(super::academic_domain_for_slug(dom)) && seen.len() > 1)
}

fn resolve_stories_and_goals(
    id_source: &str,
    cards: &[StarterCard],
    challenges: &[ChallengeCard],
    multi_domain: bool,
) -> (
    Vec<&'static PresetStoryMetadata>,
    Vec<&'static GoalChallengeMetadata>,
) {
    if multi_domain {
        (super::get_all_stories(), super::get_all_goal_challenges())
    } else {
        let domain = resolve_gallery_domain(id_source, cards, challenges);
        (
            super::get_stories_for_domain(domain),
            super::get_goal_challenges_for_domain(domain),
        )
    }
}

fn run_gallery_modal_lifecycle(
    ctx: &egui::Context,
    modal_id: egui::Id,
    items: &GalleryItems<'_>,
    filter: &mut StarterGalleryFilterState,
    is_open: &mut bool,
) -> Option<GallerySelection> {
    if items.cards.is_empty()
        && items.stories.is_empty()
        && items.challenges.is_empty()
        && items.goal_challenges.is_empty()
    {
        return None;
    }
    let m = compute_gallery_metrics(ctx);
    let mut selected = None;
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
            selected = render_searchable_gallery_body(ui, frame_ctx, items, filter, &m);
        });

    selected
}

/// Renders a starter, story & challenge gallery overlay with optional initial domain pre-filtering.
pub fn show_starter_gallery_with_challenges_for_domain(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    challenges: &[ChallengeCard],
    initial_domain_slug: Option<&str>,
    is_open: &mut bool,
) -> Option<GallerySelection> {
    let multi_domain = spans_multiple_academic_domains(cards, challenges);
    let (stories, goal_challenges) =
        resolve_stories_and_goals(id_source, cards, challenges, multi_domain);
    if cards.is_empty() && stories.is_empty() && challenges.is_empty() && goal_challenges.is_empty()
    {
        return None;
    }

    let modal_id = egui::Id::new(format!("{id_source}_starter_gallery_overlay"));
    let filter_key = modal_id.with("filter_state");
    let open_key = modal_id.with("was_open");
    let legacy_tab_key = egui::Id::new((id_source, "starter_gallery_active_tab"));
    let was_open = ctx.data(|d| d.get_temp::<bool>(open_key)).unwrap_or(false);

    if !*is_open {
        if was_open {
            ctx.data_mut(|d| {
                d.remove::<StarterGalleryFilterState>(filter_key);
                d.remove::<StarterGalleryTab>(legacy_tab_key);
                d.insert_temp(open_key, false);
            });
        }
        return None;
    }

    let mut filter = load_or_init_gallery_filter(
        ctx,
        filter_key,
        legacy_tab_key,
        initial_domain_slug,
        multi_domain,
        (
            cards.len(),
            stories.len(),
            challenges.len() + goal_challenges.len(),
        ),
    );
    let prev_tab = filter.active_tab;
    let items = GalleryItems {
        cards,
        stories: &stories,
        challenges,
        goal_challenges: &goal_challenges,
    };
    let selected = run_gallery_modal_lifecycle(ctx, modal_id, &items, &mut filter, is_open);
    if selected.is_some() {
        *is_open = false;
    } else if !*is_open {
        crate::accessibility::announce_live_status(ctx, "Starter template gallery closed");
    }

    persist_gallery_filter_state(
        ctx,
        (filter_key, open_key, legacy_tab_key),
        *is_open,
        prev_tab,
        filter,
    );
    selected
}

fn load_or_init_gallery_filter(
    ctx: &egui::Context,
    filter_key: egui::Id,
    legacy_tab_key: egui::Id,
    initial_domain_slug: Option<&str>,
    multi_domain: bool,
    counts: (usize, usize, usize),
) -> StarterGalleryFilterState {
    let (starter_count, story_count, challenge_count) = counts;
    let mut filter = ctx
        .data(|d| d.get_temp::<StarterGalleryFilterState>(filter_key))
        .unwrap_or_else(|| {
            let mut f = StarterGalleryFilterState::default();
            if let Some(slug) = initial_domain_slug {
                if multi_domain {
                    f.reset_to_domain(slug);
                }
            }
            if starter_count == 0 && challenge_count > 0 {
                f.active_tab = StarterGalleryTab::Challenges;
            } else if starter_count == 0 && story_count > 0 {
                f.active_tab = StarterGalleryTab::GuidedStories;
            }
            f
        });
    if let Some(legacy_tab) = ctx.data(|d| d.get_temp::<StarterGalleryTab>(legacy_tab_key)) {
        filter.active_tab = legacy_tab;
    }
    filter
}

fn persist_gallery_filter_state(
    ctx: &egui::Context,
    keys: (egui::Id, egui::Id, egui::Id),
    is_open: bool,
    prev_tab: StarterGalleryTab,
    filter: StarterGalleryFilterState,
) {
    let (filter_key, open_key, legacy_tab_key) = keys;
    ctx.data_mut(|d| {
        if is_open {
            if filter.active_tab != prev_tab {
                d.insert_temp(legacy_tab_key, filter.active_tab);
            }
            d.insert_temp(filter_key, filter);
            d.insert_temp(open_key, true);
        } else {
            d.remove::<StarterGalleryFilterState>(filter_key);
            d.remove::<StarterGalleryTab>(legacy_tab_key);
            d.insert_temp(open_key, false);
        }
    });
}

/// Renders the central canvas starter, story & challenge gallery overlay clamped to root window bounds.
pub fn show_starter_gallery_with_challenges(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    challenges: &[ChallengeCard],
    is_open: &mut bool,
) -> Option<GallerySelection> {
    show_starter_gallery_with_challenges_for_domain(
        ctx, id_source, cards, challenges, None, is_open,
    )
}

/// Renders a starter gallery overlay with optional initial domain pre-filtering.
pub fn show_starter_gallery_overlay_for_domain(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    initial_domain_slug: Option<&str>,
    is_open: &mut bool,
) -> Option<StarterCard> {
    let domain =
        initial_domain_slug.unwrap_or_else(|| resolve_gallery_domain(id_source, cards, &[]));
    let challenges = super::get_challenge_cards_for_domain(domain);
    match show_starter_gallery_with_challenges_for_domain(
        ctx,
        id_source,
        cards,
        &challenges,
        initial_domain_slug,
        is_open,
    ) {
        Some(GallerySelection::Starter(card)) => Some(card),
        Some(GallerySelection::Story(story)) => Some(story_to_starter_card(story)),
        Some(GallerySelection::GoalChallenge(goal)) => Some(goal_challenge_to_starter_card(goal)),
        Some(GallerySelection::Challenge(_)) | None => None,
    }
}

/// Renders the central canvas starter gallery card overlay clamped to root window bounds.
pub fn show_starter_gallery_overlay(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    is_open: &mut bool,
) -> Option<StarterCard> {
    show_starter_gallery_overlay_for_domain(ctx, id_source, cards, None, is_open)
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
    initial_domain_slug: Option<&str>,
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
            initial_domain_slug: initial_domain_slug.map(str::to_string),
        });
        d.insert_temp(key, queue);
        true
    })
}

/// Renders all deferred `StarterGallery` modals at the root `egui::Context` level and routes selections.
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
    if !requests.is_empty() {
        app.starter_gallery.close_silently();
    }
    for req in requests {
        dispatch_deferred_gallery_request(app, ctx, req);
    }
}

fn dispatch_deferred_gallery_request(
    app: &mut crate::app::MathExplorerApp,
    ctx: &egui::Context,
    req: DeferredStarterGalleryRequest,
) {
    let mut is_open = true;
    let selected = show_starter_gallery_with_challenges_for_domain(
        ctx,
        &req.modal_source,
        &req.cards,
        &req.challenges,
        req.initial_domain_slug.as_deref(),
        &mut is_open,
    );
    match selected {
        Some(GallerySelection::Starter(card)) => {
            if let Some(tab) = app.pane_tab_mut(req.pane_idx) {
                tab.apply_starter_gallery_selection(ctx, &req.id_source, false, None);
            }
            let _ = app.launch_starter_card_in_pane(ctx, req.pane_idx, &card);
        }
        Some(GallerySelection::Story(story)) => {
            if let Some(tab) = app.pane_tab_mut(req.pane_idx) {
                tab.apply_starter_gallery_selection(ctx, &req.id_source, false, None);
            }
            let _ = app.launch_story_in_pane(ctx, req.pane_idx, story);
        }
        Some(GallerySelection::Challenge(ch)) => {
            if let Some(tab) = app.pane_tab_mut(req.pane_idx) {
                tab.apply_starter_gallery_selection(ctx, &req.id_source, false, None);
            }
            let _ = app.launch_challenge_in_pane(ctx, req.pane_idx, &ch);
        }
        Some(GallerySelection::GoalChallenge(goal)) => {
            if let Some(tab) = app.pane_tab_mut(req.pane_idx) {
                tab.apply_starter_gallery_selection(ctx, &req.id_source, false, None);
            }
            let _ = app.launch_goal_challenge_in_pane(ctx, req.pane_idx, goal);
        }
        None => {
            if let Some(tab) = app.pane_tab_mut(req.pane_idx) {
                tab.apply_starter_gallery_selection(ctx, &req.id_source, is_open, None);
            }
        }
    }
}
