#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ChallengeCard, GallerySelection, StarterCard};
use eframe::egui;

const DEFER_STARTER_GALLERY_TO_ROOT_KEY: &str = "DEFER_STARTER_GALLERY_TO_ROOT";
const DEFERRED_STARTER_GALLERY_QUEUE_KEY: &str = "DEFERRED_STARTER_GALLERY_QUEUE";

/// Active sub-tab inside the `StarterGallery` modal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StarterGalleryTab {
    /// Guided forward simulation templates.
    #[default]
    Starters,
    /// Goal-driven inverse challenges with locked boundary conditions.
    Challenges,
}

#[derive(Clone)]
struct DeferredStarterGalleryRequest {
    pane_idx: usize,
    id_source: String,
    modal_source: String,
    cards: Vec<StarterCard>,
    challenges: Vec<ChallengeCard>,
}

struct GalleryOverlayMetrics {
    max_w: f32,
    max_h: f32,
    inner_w: f32,
    scroll_h: f32,
    card_w: f32,
    cols: usize,
}

fn compute_gallery_metrics(ctx: &egui::Context) -> GalleryOverlayMetrics {
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

fn render_gallery_header_and_tabs(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    modal_id: egui::Id,
    default_tab: StarterGalleryTab,
    starter_count: usize,
    challenge_count: usize,
    inner_w: f32,
) -> StarterGalleryTab {
    let tab_key = modal_id.with("gallery_active_tab");
    let mut active_tab = ui
        .ctx()
        .data(|d| d.get_temp::<StarterGalleryTab>(tab_key))
        .unwrap_or(default_tab);

    ui.horizontal(|ui| {
        ui.set_width(inner_w);
        ui.add(
            egui::Label::new(egui::RichText::new("🚀 Starter & Challenge Gallery").heading())
                .truncate(),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("✖ Close Gallery").clicked() {
                frame_ctx.close();
            }
        });
    });

    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        let starters_label = format!("🚀 Starter Templates ({starter_count})");
        let challenges_label = format!("🎯 Inverse Challenges ({challenge_count})");
        if ui
            .selectable_label(active_tab == StarterGalleryTab::Starters, starters_label)
            .clicked()
        {
            active_tab = StarterGalleryTab::Starters;
        }
        if ui
            .selectable_label(
                active_tab == StarterGalleryTab::Challenges,
                challenges_label,
            )
            .clicked()
        {
            active_tab = StarterGalleryTab::Challenges;
        }
    });
    ui.ctx().data_mut(|d| d.insert_temp(tab_key, active_tab));

    let subtitle = match active_tab {
        StarterGalleryTab::Starters => {
            "Search or filter across domains to launch pre-configured simulation templates:"
        }
        StarterGalleryTab::Challenges => {
            "Solve an inverse challenge by tuning budgeted controls against locked physical or clinical bounds:"
        }
    };
    ui.add(
        egui::Label::new(egui::RichText::new(subtitle).color(ui.visuals().weak_text_color()))
            .wrap(),
    );
    ui.add_space(4.0);
    active_tab
}

fn render_starters_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    cards: &[StarterCard],
    filter_state: &mut super::filter_ui::GalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    let filtered = super::filter_ui::filter_starter_cards(
        cards,
        &filter_state.query,
        filter_state.selected_domain,
    );
    if filtered.is_empty() {
        super::filter_ui::render_empty_filtered_state(ui, filter_state, "starter templates");
        return None;
    }
    let mut selected = None;
    for row in filtered.chunks(m.cols.max(1)) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            for card in row {
                if super::challenges::render_starter_card(ui, card, m.card_w) {
                    selected = Some(GallerySelection::Starter(card.clone()));
                    frame_ctx.close();
                }
            }
        });
    }
    selected
}

fn render_challenges_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    challenges: &[ChallengeCard],
    filter_state: &mut super::filter_ui::GalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    if challenges.is_empty() {
        ui.label(
            "No inverse challenges are registered for this domain yet. Try MRI Physics, Clinical Trials, Fluid Dynamics, or Medical Physics.",
        );
        return None;
    }
    let filtered = super::filter_ui::filter_challenge_cards(
        challenges,
        &filter_state.query,
        filter_state.selected_domain,
    );
    if filtered.is_empty() {
        super::filter_ui::render_empty_filtered_state(ui, filter_state, "inverse challenges");
        return None;
    }
    let mut selected = None;
    for row in filtered.chunks(m.cols.max(1)) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            for ch in row {
                if super::challenges::render_challenge_card(ui, ch, m.card_w) {
                    selected = Some(GallerySelection::Challenge(ch.clone()));
                    frame_ctx.close();
                }
            }
        });
    }
    selected
}

fn render_gallery_cards_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    active_tab: StarterGalleryTab,
    cards: &[StarterCard],
    challenges: &[ChallengeCard],
    filter_state: &mut super::filter_ui::GalleryFilterState,
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
    match active_tab {
        StarterGalleryTab::Starters => render_starters_grid(ui, frame_ctx, cards, filter_state, m),
        StarterGalleryTab::Challenges => {
            render_challenges_grid(ui, frame_ctx, challenges, filter_state, m)
        }
    }
}

fn render_starter_gallery_body(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    modal_id: egui::Id,
    cards: &[StarterCard],
    challenges: &[ChallengeCard],
    m: &GalleryOverlayMetrics,
) -> Option<GallerySelection> {
    let mut selected = None;
    let top_y = ui.cursor().top();
    ui.set_max_width(m.inner_w);
    ui.set_width(m.inner_w);

    let default_tab = if cards.is_empty() && !challenges.is_empty() {
        StarterGalleryTab::Challenges
    } else {
        StarterGalleryTab::Starters
    };
    let mut filter_state = super::filter_ui::load_gallery_filter_state(ui.ctx(), modal_id);
    let starter_count = super::filter_ui::filter_starter_cards(
        cards,
        &filter_state.query,
        filter_state.selected_domain,
    )
    .len();
    let challenge_count = super::filter_ui::filter_challenge_cards(
        challenges,
        &filter_state.query,
        filter_state.selected_domain,
    )
    .len();
    let active_tab = render_gallery_header_and_tabs(
        ui,
        frame_ctx,
        modal_id,
        default_tab,
        starter_count,
        challenge_count,
        m.inner_w,
    );
    super::filter_ui::render_gallery_filter_bar(ui, frame_ctx, &mut filter_state, m.inner_w);
    ui.add_space(4.0);
    ui.separator();
    ui.add_space(6.0);

    let header_used = (ui.cursor().top() - top_y).max(0.0);
    let scroll_h = (m.max_h - 32.0 - header_used)
        .max(88.0)
        .min(m.scroll_h.max(220.0));

    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_max_width(m.inner_w);
            selected = render_gallery_cards_grid(
                ui,
                frame_ctx,
                active_tab,
                cards,
                challenges,
                &mut filter_state,
                m,
            );
        });
    super::filter_ui::store_gallery_filter_state(ui.ctx(), modal_id, filter_state);
    selected
}

/// Renders the central canvas starter & inverse challenge gallery overlay clamped to root window bounds.
pub fn show_starter_gallery_with_challenges(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    challenges: &[ChallengeCard],
    is_open: &mut bool,
) -> Option<GallerySelection> {
    let modal_id = egui::Id::new(format!("{id_source}_starter_gallery_overlay"));
    if !*is_open {
        super::filter_ui::clear_gallery_filter_state(ctx, modal_id);
    }
    if cards.is_empty() && challenges.is_empty() {
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
            selected = render_starter_gallery_body(ui, frame_ctx, modal_id, cards, challenges, &m);
        });

    if !*is_open || selected.is_some() {
        super::filter_ui::clear_gallery_filter_state(ctx, modal_id);
    }
    selected
}

/// Renders the central canvas starter gallery card overlay clamped to root window bounds.
pub fn show_starter_gallery_overlay(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    is_open: &mut bool,
) -> Option<StarterCard> {
    let domain = cards.first().map_or(id_source, |c| c.domain);
    let challenges = if id_source == "global_workspace" {
        super::get_all_challenge_cards()
    } else {
        super::get_challenge_cards_for_domain(domain)
    };
    match show_starter_gallery_with_challenges(ctx, id_source, cards, &challenges, is_open) {
        Some(GallerySelection::Starter(card)) => Some(card),
        Some(GallerySelection::Challenge(_)) | None => None,
    }
}

/// Returns whether a root-level `StarterGallery` hoisting scope is currently active on `ctx`.
pub(crate) fn is_root_starter_gallery_scope_active(ctx: &egui::Context) -> bool {
    ctx.data(|d| {
        d.get_temp::<bool>(egui::Id::new(DEFER_STARTER_GALLERY_TO_ROOT_KEY))
            .unwrap_or(false)
    })
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
        app.apply_deferred_starter_gallery_selection(
            ctx,
            req.pane_idx,
            &req.id_source,
            is_open,
            selected,
        );
    }
}
