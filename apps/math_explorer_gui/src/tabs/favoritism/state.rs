#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::applied::favoritism::FavoritismInputs;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Serializable GUI parameter payload for a single child's [`FavoritismInputs`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChildInputsState {
    /// Integration horizon `t` in days (`1.0..=3650.0`).
    pub days_integrated: f64,
    /// Geographic distance `x_0` in kilometers (`0.0..=1000.0`).
    pub distance_km: f64,
    /// Practical gift score (`0.0..=10.0`).
    pub g_practical: f64,
    /// Emotional gift score (`0.0..=10.0`).
    pub g_emotional: f64,
    /// Financial wealth metric (`0.0..=10.0`).
    pub wealth: f64,
    /// Perceived talent metric (`0.0..=10.0`).
    pub talent: f64,
    /// Perceived intelligence metric (`0.0..=10.0`).
    pub intelligence: f64,
    /// Emotional sensitivity metric (`0.0..=10.0`).
    pub emotional_sensitivity: f64,
    /// Whether this family member assisted during a crisis.
    pub helped_during_crisis: bool,
    /// Whether this family member interacts on social media.
    pub active_on_social_media: bool,
    /// Birth order weight multiplier (`0.5..=1.5`).
    pub birth_order_weight: f64,
    /// Days elapsed since last phone call (`0.0..=365.0`).
    pub time_since_last_contact: f64,
    /// Baseline monthly phone call frequency (`0.0..=30.0`).
    pub f_initial: f64,
}

impl ChildInputsState {
    /// Extracts all 13 interactive parameters from a domain [`FavoritismInputs`] struct.
    pub fn from_inputs(inputs: &FavoritismInputs) -> Self {
        Self {
            days_integrated: inputs.time.t,
            distance_km: inputs.time.x_0,
            g_practical: inputs.gifts.g_practical,
            g_emotional: inputs.gifts.g_emotional,
            wealth: inputs.personality.wealth,
            talent: inputs.personality.talent,
            intelligence: inputs.personality.intelligence,
            emotional_sensitivity: inputs.personality.emotional_sensitivity,
            helped_during_crisis: inputs.social.helped_during_crisis,
            active_on_social_media: inputs.social.active_on_social_media,
            birth_order_weight: inputs.social.birth_order_weight,
            time_since_last_contact: inputs.contact.time_since_last_contact,
            f_initial: inputs.contact.f_initial,
        }
    }

    /// Converts validated GUI parameters into a domain [`FavoritismInputs`] struct.
    pub fn to_inputs(&self) -> FavoritismInputs {
        let mut inputs = FavoritismInputs::default();
        inputs.time.t = self.days_integrated;
        inputs.time.x_0 = self.distance_km;
        inputs.gifts.g_practical = self.g_practical;
        inputs.gifts.g_emotional = self.g_emotional;
        inputs.personality.wealth = self.wealth;
        inputs.personality.talent = self.talent;
        inputs.personality.intelligence = self.intelligence;
        inputs.personality.emotional_sensitivity = self.emotional_sensitivity;
        inputs.social.helped_during_crisis = self.helped_during_crisis;
        inputs.social.active_on_social_media = self.active_on_social_media;
        inputs.social.birth_order_weight = self.birth_order_weight;
        inputs.contact.time_since_last_contact = self.time_since_last_contact;
        inputs.contact.f_initial = self.f_initial;
        inputs
    }

    /// Validates that every numeric parameter is finite and within safe domain bounds.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("days_integrated", self.days_integrated, 1.0, 3650.0),
            ("distance_km", self.distance_km, 0.0, 1000.0),
            ("g_practical", self.g_practical, 0.0, 10.0),
            ("g_emotional", self.g_emotional, 0.0, 10.0),
            ("wealth", self.wealth, 0.0, 10.0),
            ("talent", self.talent, 0.0, 10.0),
            ("intelligence", self.intelligence, 0.0, 10.0),
            (
                "emotional_sensitivity",
                self.emotional_sensitivity,
                0.0,
                10.0,
            ),
            ("birth_order_weight", self.birth_order_weight, 0.5, 1.5),
            (
                "time_since_last_contact",
                self.time_since_last_contact,
                0.0,
                365.0,
            ),
            ("f_initial", self.f_initial, 0.0, 30.0),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        Ok(())
    }

    /// Hashes every persisted field into `hasher` for state-revision tracking.
    pub fn hash_into<H: Hasher>(&self, hasher: &mut H) {
        self.days_integrated.to_bits().hash(hasher);
        self.distance_km.to_bits().hash(hasher);
        self.g_practical.to_bits().hash(hasher);
        self.g_emotional.to_bits().hash(hasher);
        self.wealth.to_bits().hash(hasher);
        self.talent.to_bits().hash(hasher);
        self.intelligence.to_bits().hash(hasher);
        self.emotional_sensitivity.to_bits().hash(hasher);
        self.helped_during_crisis.hash(hasher);
        self.active_on_social_media.hash(hasher);
        self.birth_order_weight.to_bits().hash(hasher);
        self.time_since_last_contact.to_bits().hash(hasher);
        self.f_initial.to_bits().hash(hasher);
    }
}

/// Serializable state for a single family member entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChildState {
    /// Display name of the family member.
    pub name: String,
    /// All 13 strategy parameters for this family member.
    pub inputs: ChildInputsState,
}

/// Complete validated state schema for [`super::FavoritismTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FavoritismToolState {
    /// Family members and their configured strategy parameters.
    pub children: Vec<ChildState>,
    /// Currently selected family member index (clamped to `children.len()` on load).
    pub selected_child_index: Option<usize>,
    /// Draft text in the add-family-member input field.
    pub new_child_name: String,
}

/// Legacy flat `FavoritismState` payload saved by pre-framework `.mws` workspaces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyFavoritismState {
    /// Previously persisted selected child index.
    pub selected_child_index: Option<usize>,
    /// Previously persisted draft child name.
    pub new_child_name: String,
}

/// Clamps `selected` so out-of-range indices resolve to the last valid child (or `None` when empty).
pub fn clamp_selected_child_index(selected: Option<usize>, children_len: usize) -> Option<usize> {
    if children_len == 0 {
        None
    } else {
        selected.map(|idx| idx.min(children_len - 1))
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "FavoritismTool",
        display_name: "Favoritism",
        domain: "favoritism",
        tags: &[
            "favoritism",
            "inheritance",
            "scoring",
            "strategy",
            "applied math",
        ],
        build: || Box::new(super::FavoritismTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "favoritism_sibling_rivalry",
        title: "Crisis & Proximity Showdown",
        description: "Compares a nearby sibling who helped during a crisis against a wealthy distant heir under Unified Favoritism Theory.",
        category: "Behavioral Modeling",
        icon: "👑",
        domain: "favoritism",
        tool_name: Some("FavoritismTool"),
        preset_generator: || crate::starter_gallery::PRESET_FAVORITISM_SIBLING_RIVALRY,
    }
}
