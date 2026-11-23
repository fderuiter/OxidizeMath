#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Search input, domain filter pills, and word-boundary card ranking for `StarterGallery`.

use super::StarterCard;
use crate::app::AcademicDomain;
use crate::command_palette::{format_domain_title, fuzzy_match_score};
use crate::framework::ToolMetadata;
use eframe::egui;

#[derive(Clone, Debug, Default)]
pub(crate) struct GalleryFilterState {
    pub query: String,
    pub selected_domain: Option<AcademicDomain>,
}

fn filter_state_id(modal_id: egui::Id) -> egui::Id {
    modal_id.with("starter_gallery_filter_state")
}

pub(crate) fn load_gallery_filter_state(
    ctx: &egui::Context,
    modal_id: egui::Id,
) -> GalleryFilterState {
    ctx.data(|d| d.get_temp::<GalleryFilterState>(filter_state_id(modal_id)))
        .unwrap_or_default()
}

pub(crate) fn store_gallery_filter_state(
    ctx: &egui::Context,
    modal_id: egui::Id,
    state: GalleryFilterState,
) {
    ctx.data_mut(|d| d.insert_temp(filter_state_id(modal_id), state));
}

pub(crate) fn clear_gallery_filter_state(ctx: &egui::Context, modal_id: egui::Id) {
    ctx.data_mut(|d| {
        d.remove::<GalleryFilterState>(filter_state_id(modal_id));
    });
}

/// Maps a tool/preset domain identifier to its top-level [`AcademicDomain`].
pub fn card_academic_domain(domain: &str) -> AcademicDomain {
    match domain {
        "quantum" | "mri" | "fluid_dynamics" | "chaos" | "solid_state" | "medical" => {
            AcademicDomain::Physics
        }
        "analysis" | "number_theory" | "geometry_topology" | "graph_theory" | "financial_math" => {
            AcademicDomain::PureMath
        }
        "biology" | "morphogenesis" | "neuroscience" | "epidemiology" => AcademicDomain::Biology,
        "climate" => AcademicDomain::Climate,
        "ai" => AcademicDomain::Ai,
        _ => AcademicDomain::AppliedMath,
    }
}

fn score_starter_card(card: &StarterCard, query: &str) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let mut best = fuzzy_match_score(card.title, query);
    let domain_title = format_domain_title(card.domain);
    let academic_name = card_academic_domain(card.domain).name();
    for (candidate, penalty) in [
        (card.id, 2usize),
        (card.description, 10),
        (card.category, 16),
        (card.domain, 28),
        (domain_title.as_ref(), 28),
        (academic_name, 32),
    ] {
        if let Some(s) = fuzzy_match_score(candidate, query) {
            let weighted = s.saturating_add(penalty);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    if let Some(tool_name) = card.tool_name {
        if let Some(s) = fuzzy_match_score(tool_name, query) {
            let weighted = s.saturating_add(8);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
        for meta in inventory::iter::<ToolMetadata> {
            if meta.id == tool_name || meta.display_name == tool_name {
                if let Some(s) = fuzzy_match_score(meta.display_name, query) {
                    let weighted = s.saturating_add(8);
                    best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
                }
                for &tag in meta.tags {
                    if let Some(s) = fuzzy_match_score(tag, query) {
                        let weighted = s.saturating_add(14);
                        best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
                    }
                }
            }
        }
    }
    best
}

/// Filters and ranks starter cards by optional [`AcademicDomain`] pill and word-boundary query.
pub fn filter_starter_cards(
    cards: &[StarterCard],
    query: &str,
    selected_domain: Option<AcademicDomain>,
) -> Vec<StarterCard> {
    let trimmed = query.trim();
    let domain_iter = cards
        .iter()
        .filter(|card| selected_domain.is_none_or(|dom| card_academic_domain(card.domain) == dom));
    if trimmed.is_empty() {
        return domain_iter.cloned().collect();
    }
    let mut scored: Vec<(usize, usize, StarterCard)> = domain_iter
        .enumerate()
        .filter_map(|(idx, card)| {
            score_starter_card(card, trimmed).map(|score| (score, idx, card.clone()))
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.2.title.cmp(b.2.title))
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().map(|(_, _, card)| card).collect()
}

pub(super) fn render_gallery_filter_bar(
    ui: &mut egui::Ui,
    frame_ctx: &mut crate::ModalFrameContext,
    state: &mut GalleryFilterState,
    inner_w: f32,
) {
    let esc_pressed = ui.input(|i| i.key_pressed(egui::Key::Escape) && i.modifiers.is_none());
    let prev_query_empty = state.query.is_empty();

    ui.horizontal(|ui| {
        ui.set_max_width(inner_w);
        let clear_reserve = if state.query.is_empty() { 0.0 } else { 34.0 };
        let edit_w = (inner_w - clear_reserve).max(80.0);
        let edit = egui::TextEdit::singleline(&mut state.query)
            .desired_width(edit_w)
            .hint_text("🔍 Filter starter templates, equations, or tools...");
        let resp = ui.add(edit);
        if !state.query.is_empty() && ui.small_button("✖").clicked() {
            state.query.clear();
            resp.request_focus();
        }
        if esc_pressed && (resp.has_focus() || resp.lost_focus()) {
            if prev_query_empty {
                frame_ctx.close();
            } else {
                state.query.clear();
                resp.request_focus();
            }
        }
    });

    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        ui.set_max_width(inner_w);
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
        if ui
            .selectable_label(state.selected_domain.is_none(), "All Domains")
            .clicked()
        {
            state.selected_domain = None;
        }
        for &domain in AcademicDomain::ALL {
            let active = state.selected_domain == Some(domain);
            if ui.selectable_label(active, domain.name()).clicked() {
                state.selected_domain = if active { None } else { Some(domain) };
            }
        }
    });
}
