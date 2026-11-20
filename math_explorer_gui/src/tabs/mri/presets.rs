#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::presets::{
    deserialize_preset_envelope_bounded, serialize_compact_preset_envelope,
    serialize_preset_envelope,
};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Validated preset configuration for the Bloch equation MRI Physics tab.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MriPresetConfig {
    /// Longitudinal relaxation constant T1 (s).
    pub t1: f64,
    /// Transverse relaxation constant T2 (s).
    pub t2: f64,
    /// Transverse RF field Bx (T).
    pub bx: f64,
    /// Transverse RF field By (T).
    pub by: f64,
    /// Main static field Bz (T).
    pub bz: f64,
    /// Playback speed multiplier.
    pub time_scale: f64,
}

impl MriPresetConfig {
    /// Validates finiteness and physical parameter ranges prior to mutating `MriTab`.
    pub fn validate(&self) -> Result<(), String> {
        let all_finite = self.t1.is_finite()
            && self.t2.is_finite()
            && self.bx.is_finite()
            && self.by.is_finite()
            && self.bz.is_finite()
            && self.time_scale.is_finite();
        if !all_finite {
            return Err("MRI preset contains non-finite numeric parameters".to_string());
        }
        if !(0.01..=100.0).contains(&self.t1) {
            return Err(format!("MRI T1 ({}) is outside [0.01, 100.0] s", self.t1));
        }
        if !(0.001..=100.0).contains(&self.t2) {
            return Err(format!("MRI T2 ({}) is outside [0.001, 100.0] s", self.t2));
        }
        if self.bx.abs() > 1_000.0 || self.by.abs() > 1_000.0 || self.bz.abs() > 1_000.0 {
            return Err("MRI magnetic field component exceeds +/-1000 T bound".to_string());
        }
        if !(0.1..=1_000.0).contains(&self.time_scale) {
            return Err(format!(
                "MRI time_scale ({}) is outside [0.1, 1000.0]",
                self.time_scale
            ));
        }
        Ok(())
    }
}

impl MriTab {
    /// Domain identifier used in `PresetWrapper` envelopes for MRI Physics.
    #[must_use]
    pub fn domain(&self) -> &'static str {
        "mri"
    }

    /// Exports current Bloch simulation parameters as an [`MriPresetConfig`].
    #[must_use]
    pub fn export_config(&self) -> MriPresetConfig {
        MriPresetConfig {
            t1: self.simulator.t1,
            t2: self.simulator.t2,
            bx: self.simulator.b_field.x,
            by: self.simulator.b_field.y,
            bz: self.simulator.b_field.z,
            time_scale: self.time_scale,
        }
    }

    /// Computes a deterministic hash of shareable MRI simulation parameters.
    #[must_use]
    pub fn url_param_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.simulator.t1.to_bits().hash(&mut h);
        self.simulator.t2.to_bits().hash(&mut h);
        self.simulator.b_field.x.to_bits().hash(&mut h);
        self.simulator.b_field.y.to_bits().hash(&mut h);
        self.simulator.b_field.z.to_bits().hash(&mut h);
        self.time_scale.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    /// Atomically validates and applies an [`MriPresetConfig`].
    pub fn apply_config(&mut self, config: MriPresetConfig) -> Result<(), String> {
        config.validate()?;
        let b_field = Vector3::new(config.bx, config.by, config.bz);
        self.simulator.set_relaxation(config.t1, config.t2);
        self.simulator.set_b_field(b_field);
        self.time_scale = config.time_scale;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.t1 = config.t1;
            p.t2 = config.t2;
            p.b_field = b_field;
            p.time_scale = config.time_scale;
        }
        self.reset();
        Ok(())
    }

    /// Serializes MRI parameters as a pretty `PresetWrapper` JSON string.
    pub fn to_json_string(&self) -> Result<String, String> {
        serialize_preset_envelope("MRI Bloch Simulation", self.domain(), self.export_config())
    }

    /// Serializes MRI parameters as a compact deterministic `PresetWrapper` JSON string for URLs.
    pub fn to_compact_url_json(&self) -> Result<String, String> {
        serialize_compact_preset_envelope(
            "MRI Bloch Simulation",
            self.domain(),
            self.export_config(),
        )
    }

    /// Parses and applies a `PresetWrapper` JSON string to `MriTab`.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_json_str(&mut self, json_str: &str) -> Result<(), String> {
        let bounds = crate::framework::ToolStateBounds::new(
            crate::url_state::MAX_DECODED_URL_STATE_BYTES,
            crate::framework::MAX_JSON_NESTING_DEPTH,
            crate::framework::MAX_JSON_ARRAY_LEN,
        );
        let config: MriPresetConfig =
            deserialize_preset_envelope_bounded(json_str, self.domain(), self.name(), bounds)?;
        self.apply_config(config)
    }

    /// Builds a shareable web simulation URL for the current MRI Physics parameters.
    pub fn encode_share_url(&self) -> Result<String, String> {
        let json = self.to_compact_url_json()?;
        let frag = crate::url_state::encode_share_fragment(self.url_slug(), None, Some(&json))?;
        Ok(crate::url_state::build_share_url_from_fragment(&frag))
    }
}
