#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::StarterCard;
use eframe::egui;

const DEFER_STARTER_GALLERY_TO_ROOT_KEY: &str = "DEFER_STARTER_GALLERY_TO_ROOT";
const DEFERRED_STARTER_GALLERY_QUEUE_KEY: &str = "DEFERRED_STARTER_GALLERY_QUEUE";

#[derive(Clone)]
struct DeferredStarterGalleryRequest {
    pane_idx: usize,
    id_source: String,
    modal_source: String,
    cards: Vec<StarterCard>,
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
    let max_h = (avail_h - 32.0).clamp(160.0, 520.0);
    let inner_w = (max_w - 36.0).max(144.0);
    let scroll_h = (max_h - 124.0).max(64.0);
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

fn render_starter_card(ui: &mut egui::Ui, card: &StarterCard, card_width: f32) -> bool {
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
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(card.icon).size(26.0));
                    let text_w = (card_width - 42.0).max(56.0);
                    ui.vertical(|ui| {
                        ui.set_max_width(text_w);
                        ui.add(
                            egui::Label::new(egui::RichText::new(card.title).strong().size(15.0))
                                .wrap(),
                        );
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(format!("[{}]", card.category))
                                    .small()
                                    .color(ui.visuals().weak_text_color()),
                            )
                            .truncate(),
                        );
                    });
                });
                ui.add_space(8.0);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(card.description)
                            .size(12.0)
                            .color(ui.visuals().text_color()),
                    )
                    .wrap(),
                );
                ui.add_space(10.0);
                let launch_btn = egui::Button::new(format!("▶ Launch {}", card.title)).truncate();
                if ui.add_sized([card_width, 26.0], launch_btn).clicked() {
                    launched = true;
                }
            });
        });
    launched
}

fn resolve_gallery_domain<'a>(id_source: &'a str, cards: &'a [StarterCard]) -> &'a str {
    if let Some(first) = cards.first() {
        return first.domain;
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

fn render_gallery_filter_tabs(
    ui: &mut egui::Ui,
    tab_state_id: egui::Id,
    snapshot_count: usize,
    story_count: usize,
    challenge_count: usize,
) -> super::StarterGalleryTab {
    let mut active_tab = ui
        .ctx()
        .data(|d| d.get_temp::<super::StarterGalleryTab>(tab_state_id))
        .unwrap_or_default();

    ui.horizontal_wrapped(|ui| {
        for (tab, label) in [
            (
                super::StarterGalleryTab::Snapshots,
                format!("📸 Snapshots ({snapshot_count})"),
            ),
            (
                super::StarterGalleryTab::GuidedStories,
                format!("📖 Guided Stories ({story_count})"),
            ),
            (
                super::StarterGalleryTab::Challenges,
                format!("🎯 Challenges ({challenge_count})"),
            ),
        ] {
            if ui.selectable_label(active_tab == tab, label).clicked() {
                active_tab = tab;
            }
        }
    });
    ui.ctx()
        .data_mut(|d| d.insert_temp(tab_state_id, active_tab));
    active_tab
}

fn render_gallery_card_grid(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    visible_cards: &[StarterCard],
    m: &GalleryOverlayMetrics,
) -> Option<StarterCard> {
    let mut selected_card = None;
    egui::ScrollArea::vertical()
        .max_height((m.scroll_h - 28.0).max(64.0))
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_max_width(m.inner_w);
            if visible_cards.is_empty() {
                ui.label(
                    egui::RichText::new(
                        "No guided stories or challenges are registered for this domain yet.",
                    )
                    .weak(),
                );
                return;
            }
            ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
            for row in visible_cards.chunks(m.cols.max(1)) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(14.0, 14.0);
                    for card in row {
                        if render_starter_card(ui, card, m.card_w) {
                            selected_card = Some(card.clone());
                            frame_ctx.close();
                        }
                    }
                });
            }
        });
    selected_card
}

fn render_starter_gallery_body(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    id_source: &str,
    cards: &[StarterCard],
    m: &GalleryOverlayMetrics,
) -> Option<StarterCard> {
    ui.set_max_width(m.inner_w);
    ui.set_width(m.inner_w);
    ui.horizontal(|ui| {
        ui.set_width(m.inner_w);
        ui.add(
            egui::Label::new(egui::RichText::new("🚀 Starter Template Gallery").heading())
                .truncate(),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("✖ Close Gallery").clicked() {
                frame_ctx.close();
            }
        });
    });

    ui.add(
        egui::Label::new(
            egui::RichText::new(
                "Choose a snapshot, guided paradox story, or playable AST target challenge:",
            )
            .color(ui.visuals().weak_text_color()),
        )
        .wrap(),
    );
    ui.add_space(4.0);

    let domain = resolve_gallery_domain(id_source, cards);
    let story_cards = super::get_story_cards_for_domain(domain);
    let challenge_cards = super::get_challenge_cards_for_domain(domain);
    let tab_id = egui::Id::new((id_source, "starter_gallery_active_tab"));
    let active_tab = render_gallery_filter_tabs(
        ui,
        tab_id,
        cards.len(),
        story_cards.len(),
        challenge_cards.len(),
    );
    let visible_cards: &[StarterCard] = match active_tab {
        super::StarterGalleryTab::Snapshots => cards,
        super::StarterGalleryTab::GuidedStories => &story_cards,
        super::StarterGalleryTab::Challenges => &challenge_cards,
    };

    ui.add_space(4.0);
    ui.separator();
    ui.add_space(6.0);
    render_gallery_card_grid(ui, frame_ctx, visible_cards, m)
}

/// Renders the central canvas starter gallery card overlay clamped to root window bounds.
pub fn show_starter_gallery_overlay(
    ctx: &egui::Context,
    id_source: &str,
    cards: &[StarterCard],
    is_open: &mut bool,
) -> Option<StarterCard> {
    if cards.is_empty() {
        return None;
    }
    let m = compute_gallery_metrics(ctx);
    let mut selected_card = None;
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
            selected_card = render_starter_gallery_body(ui, frame_ctx, id_source, cards, &m);
        });

    selected_card
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
        let selected =
            show_starter_gallery_overlay(ctx, &req.modal_source, &req.cards, &mut is_open);
        if selected.is_some() {
            is_open = false;
        }
        if let Some(tab) = app.pane_tab_mut(req.pane_idx) {
            tab.apply_starter_gallery_outcome(ctx, &req.id_source, is_open, selected);
        }
    }
}
