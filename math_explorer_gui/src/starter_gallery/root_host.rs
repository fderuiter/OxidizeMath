#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ChallengeCard, GallerySelection, StarterCard};
use crate::app::SemanticPalette;
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
    let max_h = (avail_h - 32.0).clamp(160.0, 540.0);
    let inner_w = (max_w - 36.0).max(144.0);
    let scroll_h = (max_h - 156.0).max(64.0);
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

fn render_challenge_card_header(
    ui: &mut egui::Ui,
    card: &ChallengeCard,
    card_width: f32,
    info_color: egui::Color32,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(card.icon).size(26.0));
        let text_w = (card_width - 42.0).max(56.0);
        ui.vertical(|ui| {
            ui.set_max_width(text_w);
            ui.add(egui::Label::new(egui::RichText::new(card.title).strong().size(15.0)).wrap());
            let cat = egui::RichText::new(format!("[{}]", card.category))
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

fn render_gallery_header_and_tabs(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    modal_id: egui::Id,
    starter_count: usize,
    challenge_count: usize,
    inner_w: f32,
) -> StarterGalleryTab {
    let tab_key = modal_id.with("gallery_active_tab");
    let default_tab = if starter_count == 0 && challenge_count > 0 {
        StarterGalleryTab::Challenges
    } else {
        StarterGalleryTab::Starters
    };
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
            "Choose a guided starter model to launch pre-configured simulation parameters:"
        }
        StarterGalleryTab::Challenges => {
            "Solve an inverse challenge by tuning budgeted controls against locked physical or clinical bounds:"
        }
    };
    ui.add(
        egui::Label::new(egui::RichText::new(subtitle).color(ui.visuals().weak_text_color()))
            .wrap(),
    );
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(8.0);
    active_tab
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
    ui.set_max_width(m.inner_w);
    ui.set_width(m.inner_w);
    let active_tab = render_gallery_header_and_tabs(
        ui,
        frame_ctx,
        modal_id,
        cards.len(),
        challenges.len(),
        m.inner_w,
    );

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
                                if render_starter_card(ui, card, m.card_w) {
                                    selected = Some(GallerySelection::Starter(card.clone()));
                                    frame_ctx.close();
                                }
                            }
                        });
                    }
                }
                StarterGalleryTab::Challenges => {
                    if challenges.is_empty() {
                        ui.label(
                            "No inverse challenges are registered for this domain yet. Try MRI Physics, Clinical Trials, Fluid Dynamics, or Medical Physics.",
                        );
                    } else {
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
                    }
                }
            }
        });
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
    if cards.is_empty() && challenges.is_empty() {
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
            selected = render_starter_gallery_body(ui, frame_ctx, modal_id, cards, challenges, &m);
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
    let domain = cards.first().map_or(id_source, |c| c.domain);
    let challenges = super::get_challenge_cards_for_domain(domain);
    match show_starter_gallery_with_challenges(ctx, id_source, cards, &challenges, is_open) {
        Some(GallerySelection::Starter(card)) => Some(card),
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
