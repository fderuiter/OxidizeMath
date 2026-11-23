//! Live text search, `AcademicDomain` mapping, and `TwinSlotLiveRegion` filter announcements
//! for the global Starter Template Gallery and Command Palette.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::StarterCard;
use crate::app::domain_bar::AcademicDomain;
use crate::command_palette::fuzzy_match_score;
use crate::framework::ToolMetadata;
use eframe::egui;

/// Maps a tool/preset `domain` or tab URL slug to its canonical [`AcademicDomain`].
#[must_use]
pub fn academic_domain_for_slug(domain_slug: &str) -> AcademicDomain {
    match domain_slug.trim() {
        "analysis" | "financial_math" | "geometry_topology" | "graph_theory" | "number_theory" => {
            AcademicDomain::PureMath
        }
        "chaos" | "fluid_dynamics" | "medical" | "mri" | "quantum" | "solid_state" => {
            AcademicDomain::Physics
        }
        "battery_degradation"
        | "clinical_trials"
        | "favoritism"
        | "game_theory"
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
        "analysis" => "Analysis & Calculus",
        "battery_degradation" => "Battery Degradation",
        "biology" | "morphogenesis" => "Morphogenesis",
        "chaos" => "Chaos Theory",
        "climate" => "Climate",
        "clinical_trials" => "Clinical Trials",
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
        "quantum" => "Quantum Mechanics",
        "solid_state" => "Solid State Physics",
        "traceability" => "Traceability Portal",
        _ => "Simulation",
    }
}

pub(crate) fn same_module_slug(a: &str, b: &str) -> bool {
    a == b || (a == "biology" && b == "morphogenesis") || (a == "morphogenesis" && b == "biology")
}

/// Looks up curated `ToolMetadata::tags` associated with `card`, if registered.
#[must_use]
pub fn tool_tags_for_card(card: &StarterCard) -> &'static [&'static str] {
    let Some(tool_id) = card.tool_name else {
        return &[];
    };
    for meta in inventory::iter::<ToolMetadata> {
        if same_module_slug(meta.domain, card.domain)
            && (meta.id == tool_id || meta.display_name == tool_id)
        {
            return meta.tags;
        }
    }
    &[]
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
    let acad = academic_domain_for_slug(card.domain);
    let mod_title = module_title_for_slug(card.domain);
    for dom in [card.domain, mod_title, acad.name()] {
        if let Some(s) = fuzzy_match_score(dom, query) {
            let weighted = s.saturating_add(35);
            best = Some(best.map_or(weighted, |cur| cur.min(weighted)));
        }
    }
    best
}

/// Interactive search and `AcademicDomain` filter state for the Starter Template Gallery modal.
#[derive(Clone, Debug, Default)]
pub struct StarterGalleryFilterState {
    /// Live search query entered by the user.
    pub search_query: String,
    /// Active academic domain chip (`None` = `All Domains`).
    pub selected_domain: Option<AcademicDomain>,
    /// Optional preferred module slug (keeps the active tab's presets at the top when pre-filtered).
    pub preferred_slug: Option<String>,
    last_announced: Option<(String, Option<AcademicDomain>, usize)>,
}

impl StarterGalleryFilterState {
    /// Resets the filter state to `All Domains` with an empty search query.
    pub fn reset_to_all(&mut self) {
        self.search_query.clear();
        self.selected_domain = None;
        self.preferred_slug = None;
        self.last_announced = None;
    }

    /// Pre-filters the gallery to the `AcademicDomain` owning `domain_slug`, prioritizing `domain_slug` presets.
    pub fn reset_to_domain(&mut self, domain_slug: &str) {
        self.search_query.clear();
        self.selected_domain = Some(academic_domain_for_slug(domain_slug));
        self.preferred_slug = Some(domain_slug.to_string());
        self.last_announced = None;
    }

    /// Counts cards matching `self.search_query` and the given `domain` (`None` = `All Domains`).
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
                let pref = if self
                    .preferred_slug
                    .as_deref()
                    .is_some_and(|slug| same_module_slug(slug, c.domain))
                {
                    0
                } else {
                    1
                };
                Some((c.clone(), score, pref))
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

    /// Emits a polite `TwinSlotLiveRegion` result-count update when the user changes query or domain filter.
    pub fn sync_live_announcement(&mut self, ctx: &egui::Context, match_count: usize) {
        let normalized_q = self.search_query.trim().to_lowercase();
        let current = (normalized_q, self.selected_domain, match_count);
        let Some(ref prev) = self.last_announced else {
            self.last_announced = Some(current);
            return;
        };
        if prev == &current {
            return;
        }
        self.last_announced = Some(current);
        let msg = format_gallery_filter_status(match_count, self.selected_domain);
        crate::accessibility::announce_live_status(ctx, msg);
    }
}

fn format_gallery_filter_status(count: usize, domain: Option<AcademicDomain>) -> String {
    match (count, domain) {
        (0, Some(dom)) => format!("No matching starter presets in {}", dom.name()),
        (0, None) => "No matching starter presets found".to_string(),
        (1, Some(dom)) => format!("1 starter preset shown in {}", dom.name()),
        (1, None) => "1 starter preset shown".to_string(),
        (n, Some(dom)) => format!("{n} starter presets shown in {}", dom.name()),
        (n, None) => format!("{n} starter presets shown"),
    }
}
