#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::presets::{
    deserialize_preset_envelope_bounded, serialize_compact_preset_envelope,
    serialize_preset_envelope,
};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

const MAX_FAVORITISM_CHILDREN: usize = 32;
const MAX_CHILD_NAME_BYTES: usize = 128;

/// Serializable and bounded representation of a single family member's strategy inputs.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FavoritismChildConfig {
    /// Display name of the family member.
    pub name: String,
    /// Integrated time horizon `t` (days).
    pub days_integrated: f64,
    /// Geographical distance `x_0` (km).
    pub distance_km: f64,
    /// Practical gift score.
    pub g_practical: f64,
    /// Emotional gift score.
    pub g_emotional: f64,
    /// Wealth score.
    pub wealth: f64,
    /// Talent score.
    pub talent: f64,
    /// Intelligence score.
    pub intelligence: f64,
    /// Emotional sensitivity score.
    pub emotional_sensitivity: f64,
    /// Whether the member helped during a crisis.
    pub helped_during_crisis: bool,
    /// Whether the member is active on social media.
    pub active_on_social_media: bool,
    /// Birth order weight factor.
    pub birth_order_weight: f64,
    /// Days elapsed since last contact.
    pub time_since_last_contact: f64,
    /// Baseline monthly call frequency.
    pub f_initial: f64,
}

impl FavoritismChildConfig {
    fn from_child(child: &Child) -> Self {
        let inp = &child.inputs;
        Self {
            name: child.name.clone(),
            days_integrated: inp.time.t,
            distance_km: inp.time.x_0,
            g_practical: inp.gifts.g_practical,
            g_emotional: inp.gifts.g_emotional,
            wealth: inp.personality.wealth,
            talent: inp.personality.talent,
            intelligence: inp.personality.intelligence,
            emotional_sensitivity: inp.personality.emotional_sensitivity,
            helped_during_crisis: inp.social.helped_during_crisis,
            active_on_social_media: inp.social.active_on_social_media,
            birth_order_weight: inp.social.birth_order_weight,
            time_since_last_contact: inp.contact.time_since_last_contact,
            f_initial: inp.contact.f_initial,
        }
    }

    fn validate_and_build(&self) -> Result<Child, String> {
        if self.name.len() > MAX_CHILD_NAME_BYTES {
            return Err("Family member name exceeds 128-byte limit".to_string());
        }
        let floats = [
            ("days_integrated", self.days_integrated, 1.0, 3_650.0),
            ("distance_km", self.distance_km, 0.0, 1_000.0),
            ("g_practical", self.g_practical, 0.0, 10.0),
            ("g_emotional", self.g_emotional, 0.0, 10.0),
            ("wealth", self.wealth, 0.0, 10.0),
            ("talent", self.talent, 0.0, 10.0),
            ("intelligence", self.intelligence, 0.0, 10.0),
            ("emotional_sensitivity", self.emotional_sensitivity, 0.0, 10.0),
            ("birth_order_weight", self.birth_order_weight, 0.5, 1.5),
            ("time_since_last_contact", self.time_since_last_contact, 0.0, 365.0),
            ("f_initial", self.f_initial, 0.0, 30.0),
        ];
        for (field, val, min, max) in floats {
            if !val.is_finite() || !(min..=max).contains(&val) {
                return Err(format!(
                    "Favoritism parameter '{field}' ({val}) is non-finite or outside [{min}, {max}]"
                ));
            }
        }
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
        Ok(Child::new(self.name.clone(), inputs))
    }
}

/// Validated preset configuration for `FavoritismTab`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FavoritismPresetConfig {
    /// Configured family members and their input parameters.
    pub children: Vec<FavoritismChildConfig>,
    /// Currently selected family member index.
    pub selected_child_index: Option<usize>,
    /// Pending draft name for adding a new family member.
    pub new_child_name: String,
}

impl FavoritismTab {
    /// Domain identifier used in `PresetWrapper` envelopes for `FavoritismTab`.
    #[must_use]
    pub fn domain(&self) -> &'static str {
        "favoritism"
    }

    /// Exports active `FavoritismTab` parameters into [`FavoritismPresetConfig`].
    #[must_use]
    pub fn export_config(&self) -> FavoritismPresetConfig {
        FavoritismPresetConfig {
            children: self.children.iter().map(FavoritismChildConfig::from_child).collect(),
            selected_child_index: self.selected_child_index,
            new_child_name: self.new_child_name.clone(),
        }
    }

    /// Computes a deterministic hash of all shareable `FavoritismTab` parameters.
    #[must_use]
    pub fn url_param_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.selected_child_index.hash(&mut h);
        self.new_child_name.hash(&mut h);
        self.children.len().hash(&mut h);
        for child in &self.children {
            child.name.hash(&mut h);
            child.score.to_bits().hash(&mut h);
            child.inputs.time.t.to_bits().hash(&mut h);
            child.inputs.time.x_0.to_bits().hash(&mut h);
            child.inputs.gifts.g_practical.to_bits().hash(&mut h);
            child.inputs.gifts.g_emotional.to_bits().hash(&mut h);
            child.inputs.personality.wealth.to_bits().hash(&mut h);
            child.inputs.personality.talent.to_bits().hash(&mut h);
            child.inputs.personality.intelligence.to_bits().hash(&mut h);
            child.inputs.personality.emotional_sensitivity.to_bits().hash(&mut h);
            child.inputs.social.helped_during_crisis.hash(&mut h);
            child.inputs.social.active_on_social_media.hash(&mut h);
            child.inputs.social.birth_order_weight.to_bits().hash(&mut h);
            child.inputs.contact.time_since_last_contact.to_bits().hash(&mut h);
            child.inputs.contact.f_initial.to_bits().hash(&mut h);
        }
        h.finish().max(1)
    }

    /// Atomically validates and applies a [`FavoritismPresetConfig`].
    pub fn apply_config(&mut self, config: FavoritismPresetConfig) -> Result<(), String> {
        if config.children.len() > MAX_FAVORITISM_CHILDREN {
            return Err(format!(
                "Favoritism preset exceeds maximum of {MAX_FAVORITISM_CHILDREN} members"
            ));
        }
        if config.new_child_name.len() > MAX_CHILD_NAME_BYTES {
            return Err("Favoritism new_child_name exceeds 128-byte limit".to_string());
        }
        if let Some(sel) = config.selected_child_index {
            if sel >= config.children.len() {
                return Err(format!(
                    "Favoritism selected_child_index ({sel}) out of bounds for {} members",
                    config.children.len()
                ));
            }
        }
        let built: Vec<Child> = config
            .children
            .iter()
            .map(FavoritismChildConfig::validate_and_build)
            .collect::<Result<Vec<_>, _>>()?;
        self.children = built;
        self.selected_child_index = config.selected_child_index;
        self.new_child_name = config.new_child_name;
        Ok(())
    }

    /// Serializes `FavoritismTab` parameters as a pretty `PresetWrapper` JSON string.
    pub fn to_json_string(&self) -> Result<String, String> {
        serialize_preset_envelope("Favoritism Strategy", self.domain(), self.export_config())
    }

    /// Serializes `FavoritismTab` parameters as a compact deterministic `PresetWrapper` JSON string.
    pub fn to_compact_url_json(&self) -> Result<String, String> {
        serialize_compact_preset_envelope(
            "Favoritism Strategy",
            self.domain(),
            self.export_config(),
        )
    }

    /// Parses and applies a `PresetWrapper` JSON string to `FavoritismTab`.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_json_str(&mut self, json_str: &str) -> Result<(), String> {
        let bounds = crate::framework::ToolStateBounds::new(
            crate::url_state::MAX_DECODED_URL_STATE_BYTES,
            crate::framework::MAX_JSON_NESTING_DEPTH,
            crate::framework::MAX_JSON_ARRAY_LEN,
        );
        let config: FavoritismPresetConfig =
            deserialize_preset_envelope_bounded(json_str, self.domain(), self.name(), bounds)?;
        self.apply_config(config)
    }

    /// Builds a shareable web simulation URL for the current `FavoritismTab` state.
    pub fn encode_share_url(&self) -> Result<String, String> {
        let json = self.to_compact_url_json()?;
        let frag = crate::url_state::encode_share_fragment(self.url_slug(), None, Some(&json))?;
        Ok(crate::url_state::build_share_url_from_fragment(&frag))
    }
}
