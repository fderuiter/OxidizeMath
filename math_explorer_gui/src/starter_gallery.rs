//! Starter Template, Guided Story & Challenge Gallery component providing a searchable cross-domain
//! catalog for cold-launch discovery, per-tab pre-filtered browsing, and instant preset/story/challenge execution.

#[path = "starter_gallery/cards.rs"]
mod cards;
#[path = "starter_gallery/challenge_ui.rs"]
mod challenge_ui;
#[path = "starter_gallery/challenges.rs"]
pub mod challenges;
#[path = "starter_gallery/filter.rs"]
mod filter;
#[path = "starter_gallery/filter_ui.rs"]
pub mod filter_ui;
#[path = "starter_gallery/modal_ui.rs"]
mod modal_ui;
#[path = "starter_gallery/presets_builtin.rs"]
mod presets_builtin;
#[path = "starter_gallery/root_host.rs"]
mod root_host;
#[path = "starter_gallery/stories.rs"]
pub mod stories;
#[path = "starter_gallery/story_packs.rs"]
mod story_packs;

pub use challenges::{
    challenge_matches_domain, find_challenge_metadata, get_all_challenge_cards,
    get_challenge_cards_for_domain, ChallengeCard, ChallengeMetadata, GallerySelection,
    StarterGalleryTab,
};
pub use filter::{
    academic_domain_for_slug, module_title_for_slug, score_challenge_card, score_starter_card,
    tool_tags_for_card, StarterGalleryFilterState,
};
pub use filter_ui::{
    card_academic_domain, filter_challenge_cards, filter_goal_challenges, filter_starter_cards,
    filter_stories, get_all_starter_cards, get_all_starter_cards_with_tools,
    get_starter_cards_for_domain,
};
pub use presets_builtin::*;
pub(crate) use root_host::{
    begin_root_starter_gallery_scope, flush_deferred_starter_galleries,
    is_root_starter_gallery_scope_active, try_defer_starter_gallery,
};
pub use root_host::{
    show_starter_gallery_overlay, show_starter_gallery_overlay_for_domain,
    show_starter_gallery_with_challenges, show_starter_gallery_with_challenges_for_domain,
    GlobalStarterGallery,
};
pub use stories::*;

#[cfg(test)]
use crate::framework::ToolMetadata;
#[cfg(test)]
use eframe::egui;

/// Pre-configured starter simulation template card.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarterCard {
    /// Unique identifier for the starter card.
    pub id: &'static str,
    /// User-facing display title.
    pub title: &'static str,
    /// Concise mathematical description or governing equation.
    pub description: &'static str,
    /// Category or domain tag.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain string matching [`crate::framework::InteractiveTool::domain`].
    pub domain: &'static str,
    /// Tool metadata name in the domain to select/activate.
    pub tool_name: Option<&'static str>,
    /// Serialized JSON `PresetWrapper` configuration to load.
    pub preset_json: Option<&'static str>,
}

/// Federated starter preset registration metadata submitted across tools.
#[derive(Clone, Copy, Debug)]
pub struct StarterPresetMetadata {
    /// Unique identifier for the preset template.
    pub id: &'static str,
    /// User-facing display title.
    pub title: &'static str,
    /// Concise mathematical description or governing equation.
    pub description: &'static str,
    /// Category or domain tag.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain string matching the simulation tool domain.
    pub domain: &'static str,
    /// Target tool metadata name (if associated with a registered ToolMetadata).
    pub tool_name: Option<&'static str>,
    /// Function pointer returning the serialized JSON preset configuration envelope.
    pub preset_generator: fn() -> &'static str,
}

impl PartialEq for StarterPresetMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.title == other.title && self.domain == other.domain
    }
}

impl Eq for StarterPresetMetadata {}

impl StarterPresetMetadata {
    /// Generates the JSON preset payload.
    #[inline]
    pub fn generate_preset(&self) -> &'static str {
        (self.preset_generator)()
    }

    /// Converts static registration metadata into a launchable [`StarterCard`].
    #[must_use]
    pub fn to_card(&self) -> StarterCard {
        StarterCard {
            id: self.id,
            title: self.title,
            description: self.description,
            category: self.category,
            icon: self.icon,
            domain: self.domain,
            tool_name: self.tool_name,
            preset_json: Some((self.preset_generator)()),
        }
    }
}

inventory::collect!(StarterPresetMetadata);

#[cfg(test)]
#[path = "starter_gallery/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "starter_gallery/contract_tests.rs"]
mod contract_tests;

#[cfg(test)]
#[path = "starter_gallery/search_tests.rs"]
mod search_tests;
