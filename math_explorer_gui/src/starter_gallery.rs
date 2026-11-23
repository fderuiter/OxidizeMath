//! Starter Template & Inverse Challenge Gallery component providing a searchable cross-domain catalog
//! for cold-launch discovery, per-tab pre-filtered browsing, and instant preset/challenge execution.

use crate::framework::ToolMetadata;
#[cfg(test)]
use eframe::egui;

#[path = "starter_gallery/challenge_ui.rs"]
mod challenge_ui;
#[path = "starter_gallery/challenges.rs"]
pub mod challenges;
#[path = "starter_gallery/filter.rs"]
mod filter;
#[path = "starter_gallery/modal_ui.rs"]
mod modal_ui;
#[path = "starter_gallery/presets_builtin.rs"]
mod presets_builtin;
#[path = "starter_gallery/root_host.rs"]
mod root_host;

pub use challenges::{
    challenge_matches_domain, find_challenge_metadata, get_all_challenge_cards,
    get_challenge_cards_for_domain, ChallengeCard, ChallengeMetadata, GallerySelection,
    StarterGalleryTab,
};
pub use filter::{
    academic_domain_for_slug, module_title_for_slug, score_challenge_card, score_starter_card,
    tool_tags_for_card, StarterGalleryFilterState,
};
pub use presets_builtin::*;
pub(crate) use root_host::{
    begin_root_starter_gallery_scope, flush_deferred_starter_galleries, try_defer_starter_gallery,
};
pub use root_host::{
    show_starter_gallery_overlay, show_starter_gallery_overlay_for_domain,
    show_starter_gallery_with_challenges, show_starter_gallery_with_challenges_for_domain,
    GlobalStarterGallery,
};

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

/// Returns all registered `StarterPresetMetadata` cards across all compiled domains, deterministically sorted.
#[must_use]
pub fn get_all_starter_cards() -> Vec<StarterCard> {
    let mut cards: Vec<StarterCard> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .map(StarterPresetMetadata::to_card)
        .collect();
    cards.sort_by_key(|c| (c.title, c.id));
    cards
}

/// Returns all global starter cards plus any dynamic fallback cards for `domain` if none were registered.
#[must_use]
pub fn get_global_or_domain_starter_cards(
    domain: &str,
    available_tools: &[&'static ToolMetadata],
) -> Vec<StarterCard> {
    let mut all = get_all_starter_cards();
    let has_domain = all
        .iter()
        .any(|c| filter::same_module_slug(c.domain, domain));
    if !has_domain {
        all.extend(get_starter_cards_for_domain(domain, available_tools));
        all.sort_by_key(|c| (c.title, c.id));
    }
    all
}

/// Returns curated or dynamically generated starter cards for a single simulation domain.
pub fn get_starter_cards_for_domain(
    domain: &str,
    available_tools: &[&'static ToolMetadata],
) -> Vec<StarterCard> {
    let mut cards: Vec<StarterCard> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| filter::same_module_slug(p.domain, domain))
        .map(StarterPresetMetadata::to_card)
        .collect();

    if cards.is_empty() {
        for meta in available_tools {
            if meta.domain == domain {
                cards.push(StarterCard {
                    id: meta.id,
                    title: meta.display_name,
                    description:
                        "Interactive simulation tool for mathematical exploration and analysis.",
                    category: meta.domain,
                    icon: "🔬",
                    domain: meta.domain,
                    tool_name: Some(meta.id),
                    preset_json: None,
                });
            }
        }
    }

    cards.sort_by_key(|c| (c.title, c.id));
    cards
}

#[cfg(test)]
#[path = "starter_gallery/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "starter_gallery/contract_tests.rs"]
mod contract_tests;
