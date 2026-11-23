#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Word-boundary card ranking and per-tool fallback card builders for `StarterGallery`.

use super::{ChallengeCard, StarterCard, StarterPresetMetadata};
use crate::app::AcademicDomain;
use crate::command_palette::{format_domain_title, fuzzy_match_score};
use crate::framework::ToolMetadata;

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
#[must_use]
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
#[must_use]
pub fn get_all_starter_cards() -> Vec<StarterCard> {
    let all_tools: Vec<&'static ToolMetadata> =
        inventory::iter::<ToolMetadata>.into_iter().collect();
    get_all_starter_cards_with_tools(&all_tools)
}

/// Returns all registered starter cards across domains plus per-tool fallback cards for `extra_tools`.
#[must_use]
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

/// Maps a tool/preset domain identifier to its top-level [`AcademicDomain`].
#[must_use]
pub fn card_academic_domain(domain: &str) -> AcademicDomain {
    super::filter::academic_domain_for_slug(domain)
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
#[must_use]
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
#[must_use]
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
#[must_use]
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
#[must_use]
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
