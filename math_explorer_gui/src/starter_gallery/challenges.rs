//! Federated inverse challenge metadata registry and gallery card types.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::SemanticPalette;
use eframe::egui;

/// Renderable inverse challenge card displayed in the `StarterGallery` Challenges tab.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChallengeCard {
    /// Canonical URL-safe identifier (`challenge=<id>`).
    pub id: &'static str,
    /// User-facing challenge title.
    pub title: &'static str,
    /// Concise problem description and governing objective.
    pub description: &'static str,
    /// Scientific category tag.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain matching [`crate::framework::InteractiveTool::domain`].
    pub domain: &'static str,
    /// Target `ToolMetadata::id` or `InteractiveTool::name` within `domain`.
    pub tool_name: &'static str,
    /// Summary of locked boundary conditions / constants.
    pub locked_inputs: &'static str,
    /// Summary of tunable parameter budgets.
    pub tunable_budget: &'static str,
    /// Quantitative pass criteria summary.
    pub target_summary: &'static str,
}

/// Federated inverse challenge registration metadata submitted via `inventory::submit!`.
#[derive(Clone, Copy, Debug)]
pub struct ChallengeMetadata {
    /// Canonical URL-safe identifier (`challenge=<id>`).
    pub id: &'static str,
    /// User-facing challenge title.
    pub title: &'static str,
    /// Concise problem description and governing objective.
    pub description: &'static str,
    /// Scientific category tag.
    pub category: &'static str,
    /// Visual icon or emoji.
    pub icon: &'static str,
    /// Target domain matching the simulation tool domain.
    pub domain: &'static str,
    /// Target `ToolMetadata::id` or `InteractiveTool::name`.
    pub tool_name: &'static str,
    /// Human-readable summary of locked boundary conditions.
    pub locked_inputs: &'static str,
    /// Human-readable summary of tunable control budgets.
    pub tunable_budget: &'static str,
    /// Human-readable summary of quantitative pass threshold.
    pub target_summary: &'static str,
    /// Generates canonical unsolved baseline tool-config JSON for this challenge.
    pub baseline_state_generator: fn() -> String,
    /// Clamps any tampered locked fields in a candidate tool-config JSON string back to canonical values.
    pub clamp_locked_state: fn(&str) -> Result<String, String>,
}

impl PartialEq for ChallengeMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.domain == other.domain && self.tool_name == other.tool_name
    }
}

impl Eq for ChallengeMetadata {}

impl ChallengeMetadata {
    /// Converts static registration metadata into a gallery [`ChallengeCard`].
    #[must_use]
    pub fn to_card(&self) -> ChallengeCard {
        ChallengeCard {
            id: self.id,
            title: self.title,
            description: self.description,
            category: self.category,
            icon: self.icon,
            domain: self.domain,
            tool_name: self.tool_name,
            locked_inputs: self.locked_inputs,
            tunable_budget: self.tunable_budget,
            target_summary: self.target_summary,
        }
    }

    /// Generates the canonical baseline tool-config JSON string.
    #[must_use]
    pub fn generate_baseline_state(&self) -> String {
        (self.baseline_state_generator)()
    }

    /// Clamps locked parameters on a raw tool-config JSON string.
    pub fn clamp_state_json(&self, raw_config_json: &str) -> Result<String, String> {
        (self.clamp_locked_state)(raw_config_json)
    }
}

inventory::collect!(ChallengeMetadata);

/// Outcome selected by the user inside the `StarterGallery` modal.
#[derive(Clone, Debug, PartialEq)]
pub enum GallerySelection {
    /// Forward starter preset card.
    Starter(super::StarterCard),
    /// Inverse challenge card.
    Challenge(ChallengeCard),
}

fn domains_match(requested: &str, candidate: &str) -> bool {
    requested == candidate
        || (requested == "biology" && candidate == "morphogenesis")
        || (requested == "morphogenesis" && candidate == "biology")
}

/// Returns sorted [`ChallengeCard`] entries registered for `domain`.
#[must_use]
pub fn get_challenge_cards_for_domain(domain: &str) -> Vec<ChallengeCard> {
    let mut cards: Vec<ChallengeCard> = inventory::iter::<ChallengeMetadata>
        .into_iter()
        .filter(|c| domains_match(domain, c.domain))
        .map(ChallengeMetadata::to_card)
        .collect();
    cards.sort_by_key(|c| (c.title, c.id));
    cards
}

/// Returns all registered [`ChallengeCard`] entries across every domain in deterministic order.
#[must_use]
pub fn get_all_challenge_cards() -> Vec<ChallengeCard> {
    let mut cards: Vec<ChallengeCard> = inventory::iter::<ChallengeMetadata>
        .into_iter()
        .map(ChallengeMetadata::to_card)
        .collect();
    cards.sort_by_key(|c| (c.title, c.id));
    cards
}

/// Looks up a registered [`ChallengeMetadata`] by canonical `id` (ASCII case-insensitive).
#[must_use]
pub fn find_challenge_metadata(id: &str) -> Option<&'static ChallengeMetadata> {
    let trimmed = id.trim();
    inventory::iter::<ChallengeMetadata>
        .into_iter()
        .find(|meta| meta.id.eq_ignore_ascii_case(trimmed))
}

/// Checks whether `challenge` belongs to `domain`.
#[must_use]
pub fn challenge_matches_domain(challenge: &ChallengeMetadata, domain: &str) -> bool {
    domains_match(domain, challenge.domain)
}

pub(super) fn render_starter_card(
    ui: &mut egui::Ui,
    card: &super::StarterCard,
    card_width: f32,
) -> bool {
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

pub(super) fn render_challenge_card(
    ui: &mut egui::Ui,
    card: &ChallengeCard,
    card_width: f32,
) -> bool {
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

// [cite:favorite_child]
