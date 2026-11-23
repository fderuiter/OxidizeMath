//! Live text search, `AcademicDomain` mapping, and `TwinSlotLiveRegion` filter announcements
//! for the global Starter Template Gallery, Guided Stories, Challenges, and Command Palette.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    goal_challenge_to_starter_card, story_to_starter_card, ChallengeCard, GoalChallengeMetadata,
    PresetStoryMetadata, StarterCard, StarterGalleryTab,
};
use crate::app::domain_bar::AcademicDomain;
use crate::command_palette::fuzzy_match_score;
use crate::framework::ToolMetadata;
use eframe::egui;

/// Maps a tool/preset `domain` or tab URL slug to its canonical [`AcademicDomain`].
#[must_use]
pub fn academic_domain_for_slug(domain_slug: &str) -> AcademicDomain {
    match domain_slug.trim() {
        "algorithmic_information"
        | "analysis"
        | "elliptic_curves"
        | "financial_math"
        | "geometry_topology"
        | "graph_theory"
        | "number_theory" => AcademicDomain::PureMath,
        "astrophysics" | "chaos" | "fluid_dynamics" | "medical" | "mri" | "quantum"
        | "relativity" | "solid_state" => AcademicDomain::Physics,
        "battery_degradation"
        | "clinical_trials"
        | "favoritism"
        | "game_theory"
        | "paradox_lab"
        | "traceability" => AcademicDomain::AppliedMath,
        "biology" | "morphogenesis" | "neuroscience" | "epidemiology" => AcademicDomain::Biology,
        "climate" => AcademicDomain::Climate,
        "ai" => AcademicDomain::Ai,
        other => AcademicDomain::from_str(other).unwrap_or(AcademicDomain::AppliedMath),
    }
}

/// Maps a tool/preset `domain` slug to the human-readable `ExplorerTab` display name.
#[must_use]
pub fn module_title_for_slug(domain_slug: &str) -> &'static str {
    match domain_slug.trim() {
        "ai" => "Artificial Intelligence",
        "algorithmic_information" => "Algorithmic Information",
        "analysis" => "Analysis & Calculus",
        "astrophysics" => "Astrophysics",
        "battery_degradation" => "Battery Degradation",
        "biology" | "morphogenesis" => "Morphogenesis",
        "chaos" => "Chaos Theory",
        "climate" => "Climate",
        "clinical_trials" => "Clinical Trials",
        "elliptic_curves" => "Elliptic Curves",
        "epidemiology" => "Epidemiology",
        "favoritism" => "Favoritism",
        "financial_math" => "Financial Math",
        "fluid_dynamics" => "Fluid Dynamics",
        "game_theory" => "Game Theory",
        "geometry_topology" => "Geometry & Topology",
        "graph_theory" => "Graph Theory",
        "medical" => "Medical Physics",
        "mri" => "MRI Physics",
        "neuroscience" => "Neuroscience",
        "number_theory" => "Number Theory",
        "paradox_lab" => "Paradox Lab",
        "quantum" => "Quantum Mechanics",
        "relativity" => "General Relativity",
        "solid_state" => "Solid State Physics",
        "traceability" => "Traceability Portal",
        _ => "Simulation",
    }
}

pub(crate) fn same_module_slug(a: &str, b: &str) -> bool {
    a == b || (a == "biology" && b == "morphogenesis") || (a == "morphogenesis" && b == "biology")
}

fn tool_tags_for_domain_tool(domain: &str, tool_id: &str) -> &'static [&'static str] {
    for meta in inventory::iter::<ToolMetadata> {
        if same_module_slug(meta.domain, domain)
            && (meta.id == tool_id || meta.display_name == tool_id)
        {
            return meta.tags;
        }
    }
    &[]
}

/// Looks up curated `ToolMetadata::tags` associated with `card`, if registered.
#[must_use]
pub fn tool_tags_for_card(card: &StarterCard) -> &'static [&'static str] {
    card.tool_name.map_or(&[], |tool_id| {
        tool_tags_for_domain_tool(card.domain, tool_id)
    })
}

/// Scores a `StarterCard` against `query` across title, tool id, curated tags, category, and domain.
#[must_use]
pub fn score_starter_card(card: &StarterCard, query: &str) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let mut best =
        fuzzy_match_score(card.title, query).map(|s| if s == 0 { 0 } else { s.saturating_add(2) });
    if let Some(tool_id) = card.tool_name {
        if let Some(s) = fuzzy_match_score(tool_id, query) {
            let weighted = s.saturating_add(6);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    for &tag in tool_tags_for_card(card) {
        if let Some(s) = fuzzy_match_score(tag, query) {
            let weighted = s.saturating_add(8);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    if let Some(s) = fuzzy_match_score(card.category, query) {
        let weighted = s.saturating_add(20);
        best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
    }
    score_domain_candidates(card.domain, query, best)
}

/// Scores a `ChallengeCard` against `query` across title, id, tool, tags, category, and domain.
#[must_use]
pub fn score_challenge_card(card: &ChallengeCard, query: &str) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let mut best =
        fuzzy_match_score(card.title, query).map(|s| if s == 0 { 0 } else { s.saturating_add(2) });
    for candidate in [card.id, card.tool_name] {
        if let Some(s) = fuzzy_match_score(candidate, query) {
            let weighted = s.saturating_add(6);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    for &tag in tool_tags_for_domain_tool(card.domain, card.tool_name) {
        if let Some(s) = fuzzy_match_score(tag, query) {
            let weighted = s.saturating_add(8);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    for field in [card.category, card.target_summary] {
        if let Some(s) = fuzzy_match_score(field, query) {
            let weighted = s.saturating_add(20);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    score_domain_candidates(card.domain, query, best)
}

pub(super) fn score_story_metadata(story: &PresetStoryMetadata, query: &str) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let card = story_to_starter_card(story);
    let mut best = score_starter_card(&card, query);
    if let Some(s) = fuzzy_match_score(story.id, query) {
        let weighted = s.saturating_add(6);
        best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
    }
    best
}

pub(super) fn score_goal_challenge_metadata(
    goal: &GoalChallengeMetadata,
    query: &str,
) -> Option<usize> {
    if query.trim().is_empty() {
        return Some(usize::MAX);
    }
    let card = goal_challenge_to_starter_card(goal);
    let mut best = score_starter_card(&card, query);
    if let Some(s) = fuzzy_match_score(goal.id, query) {
        let weighted = s.saturating_add(6);
        best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
    }
    if let Some(s) = fuzzy_match_score(goal.goal_summary, query) {
        let weighted = s.saturating_add(20);
        best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
    }
    best
}

fn score_domain_candidates(domain: &str, query: &str, mut best: Option<usize>) -> Option<usize> {
    let acad = academic_domain_for_slug(domain);
    let mod_title = module_title_for_slug(domain);
    for dom in [domain, mod_title, acad.name()] {
        if let Some(s) = fuzzy_match_score(dom, query) {
            let weighted = s.saturating_add(35);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    best
}

/// Interactive search, sub-tab, and `AcademicDomain` filter state for the Starter Gallery modal.
#[derive(Clone, Debug, Default)]
pub struct StarterGalleryFilterState {
    /// Active gallery sub-tab (`Starters`, `GuidedStories`, or `Challenges`).
    pub active_tab: StarterGalleryTab,
    /// Live search query entered by the user.
    pub search_query: String,
    /// Active academic domain chip (`None` = `All Domains`).
    pub selected_domain: Option<AcademicDomain>,
    /// Optional preferred module slug (keeps the active tab's presets at the top when pre-filtered).
    pub preferred_slug: Option<String>,
    last_announced: Option<(StarterGalleryTab, String, Option<AcademicDomain>, usize)>,
}

impl StarterGalleryFilterState {
    /// Resets the filter state to `All Domains` on `Starters` with an empty search query.
    pub fn reset_to_all(&mut self) {
        self.active_tab = StarterGalleryTab::Starters;
        self.search_query.clear();
        self.selected_domain = None;
        self.preferred_slug = None;
        self.last_announced = None;
    }

    /// Pre-filters the gallery to the `AcademicDomain` owning `domain_slug`, prioritizing `domain_slug` presets.
    pub fn reset_to_domain(&mut self, domain_slug: &str) {
        self.active_tab = StarterGalleryTab::Starters;
        self.search_query.clear();
        self.selected_domain = Some(academic_domain_for_slug(domain_slug));
        self.preferred_slug = Some(domain_slug.to_string());
        self.last_announced = None;
    }

    fn pref_rank(&self, item_domain: &str) -> u8 {
        u8::from(
            !self
                .preferred_slug
                .as_deref()
                .is_some_and(|slug| same_module_slug(slug, item_domain)),
        )
    }

    /// Counts starter cards matching `self.search_query` and the given `domain` (`None` = `All Domains`).
    #[must_use]
    pub fn count_matching_in_domain(
        &self,
        cards: &[StarterCard],
        domain: Option<AcademicDomain>,
    ) -> usize {
        cards
            .iter()
            .filter(|c| {
                domain.is_none_or(|d| academic_domain_for_slug(c.domain) == d)
                    && score_starter_card(c, &self.search_query).is_some()
            })
            .count()
    }

    /// Counts guided stories matching `self.search_query` and `domain` (`None` = `All Domains`).
    #[must_use]
    pub fn count_matching_stories_in_domain(
        &self,
        stories: &[&'static PresetStoryMetadata],
        domain: Option<AcademicDomain>,
    ) -> usize {
        stories
            .iter()
            .filter(|s| {
                domain.is_none_or(|d| academic_domain_for_slug(s.domain) == d)
                    && score_story_metadata(s, &self.search_query).is_some()
            })
            .count()
    }

    /// Counts inverse + AST goal challenges matching `self.search_query` and `domain`.
    #[must_use]
    pub fn count_matching_challenges_in_domain(
        &self,
        challenges: &[ChallengeCard],
        goals: &[&'static GoalChallengeMetadata],
        domain: Option<AcademicDomain>,
    ) -> usize {
        let inv = challenges
            .iter()
            .filter(|c| {
                domain.is_none_or(|d| academic_domain_for_slug(c.domain) == d)
                    && score_challenge_card(c, &self.search_query).is_some()
            })
            .count();
        let ast = goals
            .iter()
            .filter(|g| {
                domain.is_none_or(|d| academic_domain_for_slug(g.domain) == d)
                    && score_goal_challenge_metadata(g, &self.search_query).is_some()
            })
            .count();
        inv + ast
    }

    /// Filters and ranks `cards` according to `selected_domain`, `search_query`, and `preferred_slug`.
    #[must_use]
    pub fn filter_cards(&self, cards: &[StarterCard]) -> Vec<StarterCard> {
        let mut matched: Vec<(StarterCard, usize, u8)> = cards
            .iter()
            .filter(|c| {
                self.selected_domain
                    .is_none_or(|dom| academic_domain_for_slug(c.domain) == dom)
            })
            .filter_map(|c| {
                let score = score_starter_card(c, &self.search_query)?;
                Some((c.clone(), score, self.pref_rank(c.domain)))
            })
            .collect();
        matched.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.title.cmp(b.0.title))
                .then_with(|| a.0.id.cmp(b.0.id))
        });
        matched.into_iter().map(|(card, _, _)| card).collect()
    }

    /// Filters and ranks guided `stories` according to `selected_domain`, `search_query`, and `preferred_slug`.
    #[must_use]
    pub fn filter_stories(
        &self,
        stories: &[&'static PresetStoryMetadata],
    ) -> Vec<&'static PresetStoryMetadata> {
        let mut matched: Vec<(&'static PresetStoryMetadata, usize, u8)> = stories
            .iter()
            .copied()
            .filter(|s| {
                self.selected_domain
                    .is_none_or(|dom| academic_domain_for_slug(s.domain) == dom)
            })
            .filter_map(|s| {
                let score = score_story_metadata(s, &self.search_query)?;
                Some((s, score, self.pref_rank(s.domain)))
            })
            .collect();
        matched.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.title.cmp(b.0.title))
                .then_with(|| a.0.id.cmp(b.0.id))
        });
        matched.into_iter().map(|(story, _, _)| story).collect()
    }

    /// Filters and ranks inverse `challenges` according to `selected_domain`, `search_query`, and `preferred_slug`.
    #[must_use]
    pub fn filter_challenges(&self, challenges: &[ChallengeCard]) -> Vec<ChallengeCard> {
        let mut matched: Vec<(ChallengeCard, usize, u8)> = challenges
            .iter()
            .filter(|c| {
                self.selected_domain
                    .is_none_or(|dom| academic_domain_for_slug(c.domain) == dom)
            })
            .filter_map(|c| {
                let score = score_challenge_card(c, &self.search_query)?;
                Some((c.clone(), score, self.pref_rank(c.domain)))
            })
            .collect();
        matched.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.title.cmp(b.0.title))
                .then_with(|| a.0.id.cmp(b.0.id))
        });
        matched.into_iter().map(|(card, _, _)| card).collect()
    }

    /// Filters and ranks playable AST `goals` according to `selected_domain`, `search_query`, and `preferred_slug`.
    #[must_use]
    pub fn filter_goal_challenges(
        &self,
        goals: &[&'static GoalChallengeMetadata],
    ) -> Vec<&'static GoalChallengeMetadata> {
        let mut matched: Vec<(&'static GoalChallengeMetadata, usize, u8)> = goals
            .iter()
            .copied()
            .filter(|g| {
                self.selected_domain
                    .is_none_or(|dom| academic_domain_for_slug(g.domain) == dom)
            })
            .filter_map(|g| {
                let score = score_goal_challenge_metadata(g, &self.search_query)?;
                Some((g, score, self.pref_rank(g.domain)))
            })
            .collect();
        matched.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.title.cmp(b.0.title))
                .then_with(|| a.0.id.cmp(b.0.id))
        });
        matched.into_iter().map(|(goal, _, _)| goal).collect()
    }

    /// Emits a polite `TwinSlotLiveRegion` result-count update when the user changes tab, query, or domain filter.
    pub fn sync_live_announcement(&mut self, ctx: &egui::Context, match_count: usize) {
        let normalized_q = self.search_query.trim().to_lowercase();
        let current = (
            self.active_tab,
            normalized_q,
            self.selected_domain,
            match_count,
        );
        let Some(ref prev) = self.last_announced else {
            self.last_announced = Some(current);
            return;
        };
        if prev == &current {
            return;
        }
        self.last_announced = Some(current);
        let msg = format_gallery_filter_status(self.active_tab, match_count, self.selected_domain);
        crate::accessibility::announce_live_status(ctx, msg);
    }
}

fn format_gallery_filter_status(
    tab: StarterGalleryTab,
    count: usize,
    domain: Option<AcademicDomain>,
) -> String {
    let (singular, plural) = match tab {
        StarterGalleryTab::Starters => ("starter preset", "starter presets"),
        StarterGalleryTab::GuidedStories => ("guided story", "guided stories"),
        StarterGalleryTab::Challenges => ("challenge", "challenges"),
    };
    match (count, domain) {
        (0, Some(dom)) => format!("No matching {plural} in {}", dom.name()),
        (0, None) => format!("No matching {plural} found"),
        (1, Some(dom)) => format!("1 {singular} shown in {}", dom.name()),
        (1, None) => format!("1 {singular} shown"),
        (n, Some(dom)) => format!("{n} {plural} shown in {}", dom.name()),
        (n, None) => format!("{n} {plural} shown"),
    }
}
