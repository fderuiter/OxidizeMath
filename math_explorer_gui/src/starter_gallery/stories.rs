//! Declarative metadata and federated `inventory` registries for multi-step guided stories
//! and playable AST target challenges across `SimulationFramework` tools.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{StarterCard, StarterPresetMetadata};
use crate::framework::CalloutSpec;

/// Single step in a guided mathematical paradox tour.
#[derive(Clone, Copy, Debug)]
pub struct StoryStep {
    /// Short headline for this tour step.
    pub title: &'static str,
    /// Narrative explanation shown in the HUD banner.
    pub narrative: &'static str,
    /// Generator returning a valid `PresetWrapper` JSON string.
    pub preset_generator: fn() -> &'static str,
    /// Plot callout pins synchronized when entering this step.
    pub callouts: &'static [CalloutSpec],
    /// Observable names highlighted in the HUD readout strip.
    pub focus_observables: &'static [&'static str],
}

/// Federated guided story registration metadata submitted across flagship tools.
#[derive(Clone, Copy, Debug)]
pub struct PresetStoryMetadata {
    /// Unique URL-safe story identifier (`&preset=<id>`).
    pub id: &'static str,
    /// User-facing story card title.
    pub title: &'static str,
    /// Concise summary shown in `StarterGallery`.
    pub description: &'static str,
    /// Thematic category badge.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain matching `InteractiveTool::domain`.
    pub domain: &'static str,
    /// Canonical `ToolMetadata::id` for the target tool.
    pub tool_name: &'static str,
    /// Ordered sequence of narrative parameter steps.
    pub steps: &'static [StoryStep],
}

impl PartialEq for PresetStoryMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.domain == other.domain && self.tool_name == other.tool_name
    }
}

impl Eq for PresetStoryMetadata {}

inventory::collect!(PresetStoryMetadata);

/// Federated playable AST challenge metadata submitted across flagship tools.
#[derive(Clone, Copy, Debug)]
pub struct GoalChallengeMetadata {
    /// Unique URL-safe challenge identifier (`&challenge=<id>`).
    pub id: &'static str,
    /// User-facing challenge card title.
    pub title: &'static str,
    /// Concise summary shown in `StarterGallery`.
    pub description: &'static str,
    /// Thematic category badge.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain matching `InteractiveTool::domain`.
    pub domain: &'static str,
    /// Canonical `ToolMetadata::id` for the target tool.
    pub tool_name: &'static str,
    /// Generator returning the unsolved initial `PresetWrapper` JSON.
    pub initial_preset_generator: fn() -> &'static str,
    /// Plain-language target prompt displayed in the Challenge HUD.
    pub goal_summary: &'static str,
    /// Conjunction of relational expressions compiled with `math_commons::ast::parse`.
    pub predicate_expr: &'static str,
    /// Optional hint revealed on request.
    pub hint: &'static str,
    /// Screen-reader and HUD celebration message announced on completion.
    pub win_message: &'static str,
    /// Initial plot callout pins synchronized when starting the challenge.
    pub callouts: &'static [CalloutSpec],
    /// Observable names highlighted in the HUD readout strip.
    pub focus_observables: &'static [&'static str],
}

impl PartialEq for GoalChallengeMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.domain == other.domain && self.tool_name == other.tool_name
    }
}

impl Eq for GoalChallengeMetadata {}

inventory::collect!(GoalChallengeMetadata);

/// Returns `true` when `entry_domain` matches `requested_domain` (accounting for `biology`/`morphogenesis`).
#[must_use]
pub fn domain_matches_gallery(entry_domain: &str, requested_domain: &str) -> bool {
    entry_domain == requested_domain
        || (requested_domain == "biology" && entry_domain == "morphogenesis")
        || (requested_domain == "morphogenesis" && entry_domain == "biology")
}

/// Returns all registered guided stories for `domain`, sorted deterministically.
#[must_use]
pub fn get_stories_for_domain(domain: &str) -> Vec<&'static PresetStoryMetadata> {
    let mut out: Vec<&'static PresetStoryMetadata> = inventory::iter::<PresetStoryMetadata>
        .into_iter()
        .filter(|s| domain_matches_gallery(s.domain, domain) && !s.steps.is_empty())
        .collect();
    out.sort_by_key(|s| (s.title, s.id));
    out
}

/// Returns all registered playable AST goal challenges for `domain`, sorted deterministically.
#[must_use]
pub fn get_goal_challenges_for_domain(domain: &str) -> Vec<&'static GoalChallengeMetadata> {
    let mut out: Vec<&'static GoalChallengeMetadata> = inventory::iter::<GoalChallengeMetadata>
        .into_iter()
        .filter(|c| domain_matches_gallery(c.domain, domain))
        .collect();
    out.sort_by_key(|c| (c.title, c.id));
    out
}

/// Converts a [`PresetStoryMetadata`] entry into a [`StarterCard`].
#[must_use]
pub fn story_to_starter_card(s: &PresetStoryMetadata) -> StarterCard {
    StarterCard {
        id: s.id,
        title: s.title,
        description: s.description,
        category: s.category,
        icon: s.icon,
        domain: s.domain,
        tool_name: Some(s.tool_name),
        preset_json: s.steps.first().map(|step| (step.preset_generator)()),
    }
}

/// Converts a [`GoalChallengeMetadata`] entry into a [`StarterCard`].
#[must_use]
pub fn goal_challenge_to_starter_card(c: &GoalChallengeMetadata) -> StarterCard {
    StarterCard {
        id: c.id,
        title: c.title,
        description: c.description,
        category: c.category,
        icon: c.icon,
        domain: c.domain,
        tool_name: Some(c.tool_name),
        preset_json: Some((c.initial_preset_generator)()),
    }
}

/// Converts registered guided stories for `domain` into `StarterCard` items.
#[must_use]
pub fn get_story_cards_for_domain(domain: &str) -> Vec<StarterCard> {
    get_stories_for_domain(domain)
        .into_iter()
        .map(story_to_starter_card)
        .collect()
}

/// Converts registered AST goal challenges for `domain` into `StarterCard` items.
#[must_use]
pub fn get_goal_challenge_cards_for_domain(domain: &str) -> Vec<StarterCard> {
    get_goal_challenges_for_domain(domain)
        .into_iter()
        .map(goal_challenge_to_starter_card)
        .collect()
}

/// Looks up a registered `PresetStoryMetadata` by ID (optionally scoped to `domain`).
#[must_use]
pub fn find_story_by_id(id: &str, domain: Option<&str>) -> Option<&'static PresetStoryMetadata> {
    let trimmed = id.trim();
    inventory::iter::<PresetStoryMetadata>
        .into_iter()
        .find(|s| {
            s.id.eq_ignore_ascii_case(trimmed)
                && !s.steps.is_empty()
                && domain.is_none_or(|d| domain_matches_gallery(s.domain, d))
        })
}

/// Looks up a registered `GoalChallengeMetadata` by ID (optionally scoped to `domain`).
#[must_use]
pub fn find_goal_challenge_by_id(
    id: &str,
    domain: Option<&str>,
) -> Option<&'static GoalChallengeMetadata> {
    let trimmed = id.trim();
    inventory::iter::<GoalChallengeMetadata>
        .into_iter()
        .find(|c| {
            c.id.eq_ignore_ascii_case(trimmed)
                && domain.is_none_or(|d| domain_matches_gallery(c.domain, d))
        })
}

/// Alias for [`find_goal_challenge_by_id`] looking up an AST target challenge by ID.
#[must_use]
pub fn find_challenge_by_id(
    id: &str,
    domain: Option<&str>,
) -> Option<&'static GoalChallengeMetadata> {
    find_goal_challenge_by_id(id, domain)
}

/// Looks up a registered static `StarterPresetMetadata` snapshot by ID within `domain`.
#[must_use]
pub fn find_starter_preset_by_id(
    id: &str,
    domain: Option<&str>,
) -> Option<&'static StarterPresetMetadata> {
    let trimmed = id.trim();
    inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .find(|p| {
            p.id.eq_ignore_ascii_case(trimmed)
                && domain.is_none_or(|d| domain_matches_gallery(p.domain, d))
        })
}
