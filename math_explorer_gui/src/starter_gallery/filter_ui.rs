#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Search input, domain filter pills, and word-boundary card ranking for `StarterGallery`.

use super::{ChallengeCard, StarterCard, StarterPresetMetadata};
use crate::app::AcademicDomain;
use crate::command_palette::{format_domain_title, fuzzy_match_score};
use crate::framework::ToolMetadata;
use eframe::egui;

fn domains_match(a: &str, b: &str) -> bool {
    a == b || (a == "biology" && b == "morphogenesis") || (a == "morphogenesis" && b == "biology")
}

fn preset_to_card(p: &StarterPresetMetadata) -> StarterCard {
    StarterCard {
        id: p.id,
        title: p.title,
        description: p.description,
        category: p.category,
        icon: p.icon,
        domain: p.domain,
        tool_name: p.tool_name,
        preset_json: Some((p.preset_generator)()),
    }
}

fn fallback_card_for_tool(meta: &'static ToolMetadata) -> StarterCard {
    StarterCard {
        id: meta.id,
        title: meta.display_name,
        description: "Interactive simulation tool for mathematical exploration and analysis.",
        category: meta.domain,
        icon: "🔬",
        domain: meta.domain,
        tool_name: Some(meta.id),
        preset_json: None,
    }
}

fn append_missing_tool_fallbacks(
    cards: &mut Vec<StarterCard>,
    domain_filter: Option<&str>,
    tools: &[&'static ToolMetadata],
) {
    let matching_tool_count = tools
        .iter()
        .filter(|m| domain_filter.is_none_or(|d| domains_match(m.domain, d)))
        .count();
    for &meta in tools {
        if domain_filter.is_some_and(|d| !domains_match(meta.domain, d)) {
            continue;
        }
        let has_card = cards.iter().any(|c| {
            c.tool_name == Some(meta.id)
                || c.tool_name == Some(meta.display_name)
                || (c.tool_name.is_none()
                    && domains_match(c.domain, meta.domain)
                    && matching_tool_count == 1)
        });
        if !has_card {
            cards.push(fallback_card_for_tool(meta));
        }
    }
}

/// Returns curated starter cards for `domain`, plus per-tool fallback cards for any tool in `available_tools` lacking a preset.
pub fn get_starter_cards_for_domain(
    domain: &str,
    available_tools: &[&'static ToolMetadata],
) -> Vec<StarterCard> {
    let mut cards: Vec<StarterCard> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| domains_match(p.domain, domain))
        .map(preset_to_card)
        .collect();

    append_missing_tool_fallbacks(&mut cards, Some(domain), available_tools);
    cards.sort_by_key(|c| (c.title, c.id));
    cards
}

/// Returns all registered starter cards across every domain, generating per-tool fallback cards for any registered tool lacking a preset.
pub fn get_all_starter_cards() -> Vec<StarterCard> {
    let all_tools: Vec<&'static ToolMetadata> =
        inventory::iter::<ToolMetadata>.into_iter().collect();
    get_all_starter_cards_with_tools(&all_tools)
}

/// Returns all registered starter cards across domains plus per-tool fallback cards for `extra_tools`.
pub fn get_all_starter_cards_with_tools(extra_tools: &[&'static ToolMetadata]) -> Vec<StarterCard> {
    let mut cards: Vec<StarterCard> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .map(preset_to_card)
        .collect();
    let mut tools: Vec<&'static ToolMetadata> =
        inventory::iter::<ToolMetadata>.into_iter().collect();
    for &extra in extra_tools {
        if !tools
            .iter()
            .any(|t| t.id == extra.id && t.domain == extra.domain)
        {
            tools.push(extra);
        }
    }
    append_missing_tool_fallbacks(&mut cards, None, &tools);
    cards.sort_by_key(|c| (c.title, c.id));
    cards
}

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
        "quantum" | "mri" | "fluid_dynamics" | "chaos" | "solid_state" | "medical"
        | "astrophysics" | "relativity" => AcademicDomain::Physics,
        "analysis"
        | "elliptic_curves"
        | "number_theory"
        | "geometry_topology"
        | "graph_theory"
        | "financial_math"
        | "algorithmic_information" => AcademicDomain::PureMath,
        "biology" | "morphogenesis" | "neuroscience" | "epidemiology" => AcademicDomain::Biology,
        "climate" => AcademicDomain::Climate,
        "ai" => AcademicDomain::Ai,
        _ => AcademicDomain::AppliedMath,
    }
}

fn score_tool_association(best: &mut Option<usize>, tool_name: &str, query: &str) {
    if let Some(s) = fuzzy_match_score(tool_name, query) {
        let weighted = s.saturating_add(8);
        *best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
    }
    for meta in inventory::iter::<ToolMetadata> {
        if meta.id == tool_name || meta.display_name == tool_name {
            if let Some(s) = fuzzy_match_score(meta.display_name, query) {
                let weighted = s.saturating_add(8);
                *best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
            }
            for &tag in meta.tags {
                if let Some(s) = fuzzy_match_score(tag, query) {
                    let weighted = s.saturating_add(14);
                    *best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
                }
            }
        }
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
        score_tool_association(&mut best, tool_name, query);
    }
    best
}

fn score_challenge_card(card: &ChallengeCard, query: &str) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let mut best = fuzzy_match_score(card.title, query);
    let domain_title = format_domain_title(card.domain);
    let academic_name = card_academic_domain(card.domain).name();
    for (candidate, penalty) in [
        (card.id, 2usize),
        (card.description, 10),
        (card.target_summary, 12),
        (card.locked_inputs, 14),
        (card.tunable_budget, 14),
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
    score_tool_association(&mut best, card.tool_name, query);
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

/// Filters and ranks inverse challenge cards by optional [`AcademicDomain`] pill and word-boundary query.
pub fn filter_challenge_cards(
    challenges: &[ChallengeCard],
    query: &str,
    selected_domain: Option<AcademicDomain>,
) -> Vec<ChallengeCard> {
    let trimmed = query.trim();
    let domain_iter = challenges
        .iter()
        .filter(|card| selected_domain.is_none_or(|dom| card_academic_domain(card.domain) == dom));
    if trimmed.is_empty() {
        return domain_iter.cloned().collect();
    }
    let mut scored: Vec<(usize, usize, ChallengeCard)> = domain_iter
        .enumerate()
        .filter_map(|(idx, card)| {
            score_challenge_card(card, trimmed).map(|score| (score, idx, card.clone()))
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.2.title.cmp(b.2.title))
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().map(|(_, _, card)| card).collect()
}

/// Filters and ranks guided stories by optional [`AcademicDomain`] pill and word-boundary query.
pub fn filter_stories(
    stories: &[&'static super::PresetStoryMetadata],
    query: &str,
    selected_domain: Option<AcademicDomain>,
) -> Vec<&'static super::PresetStoryMetadata> {
    let trimmed = query.trim();
    let domain_iter = stories
        .iter()
        .copied()
        .filter(|s| selected_domain.is_none_or(|dom| card_academic_domain(s.domain) == dom));
    if trimmed.is_empty() {
        return domain_iter.collect();
    }
    let mut scored: Vec<(usize, usize, &'static super::PresetStoryMetadata)> = domain_iter
        .enumerate()
        .filter_map(|(idx, s)| {
            let card = super::story_to_starter_card(s);
            score_starter_card(&card, trimmed).map(|score| (score, idx, s))
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.2.title.cmp(b.2.title))
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().map(|(_, _, s)| s).collect()
}

/// Filters and ranks playable AST goal challenges by optional [`AcademicDomain`] pill and word-boundary query.
pub fn filter_goal_challenges(
    goals: &[&'static super::GoalChallengeMetadata],
    query: &str,
    selected_domain: Option<AcademicDomain>,
) -> Vec<&'static super::GoalChallengeMetadata> {
    let trimmed = query.trim();
    let domain_iter = goals
        .iter()
        .copied()
        .filter(|g| selected_domain.is_none_or(|dom| card_academic_domain(g.domain) == dom));
    if trimmed.is_empty() {
        return domain_iter.collect();
    }
    let mut scored: Vec<(usize, usize, &'static super::GoalChallengeMetadata)> = domain_iter
        .enumerate()
        .filter_map(|(idx, g)| {
            let card = super::goal_challenge_to_starter_card(g);
            let mut best = score_starter_card(&card, trimmed);
            if let Some(s) = fuzzy_match_score(g.goal_summary, trimmed) {
                let weighted = s.saturating_add(12);
                best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
            }
            best.map(|score| (score, idx, g))
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.2.title.cmp(b.2.title))
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().map(|(_, _, g)| g).collect()
}

pub(crate) fn render_empty_filtered_state(
    ui: &mut egui::Ui,
    filter_state: &mut GalleryFilterState,
    item_noun: &str,
) {
    ui.vertical_centered(|ui| {
        ui.add_space(16.0);
        ui.label(format!("No {item_noun} match the active filter."));
        ui.add_space(6.0);
        if ui.button("Reset Gallery Filters").clicked() {
            filter_state.query.clear();
            filter_state.selected_domain = None;
        }
        ui.add_space(12.0);
    });
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
            .hint_text("🔍 Filter starter templates, stories, challenges, or tools...");
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
