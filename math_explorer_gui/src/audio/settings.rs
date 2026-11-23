//! Persisted non-mute `AcousticSettings` bounded for safe synthesis, JSON presets, and share URLs.

// theory_verification!

use serde::{Deserialize, Serialize};

/// Default safe user master gain (`0.5` = 50%).
pub const DEFAULT_AUDIO_GAIN: f32 = 0.5;

/// Default user pitch multiplier (`1.0` = nominal pitch).
pub const DEFAULT_PITCH_SCALE: f32 = 1.0;

const fn default_gain() -> f32 {
    DEFAULT_AUDIO_GAIN
}

const fn default_pitch_scale() -> f32 {
    DEFAULT_PITCH_SCALE
}

/// Persisted non-mute acoustic parameters for JSON presets and `#tab=...&tool=...&state=...` URLs.
///
/// Deliberately omits mute state so application startup, preset import, and URL hydration
/// always boot muted until an explicit user gesture unmutes playback.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticSettings {
    /// Master output gain scalar (`0.0..=1.0`).
    #[serde(default = "default_gain")]
    pub gain: f32,
    /// Pitch transposition multiplier (`0.25..=4.0`).
    #[serde(default = "default_pitch_scale")]
    pub pitch_scale: f32,
}

impl Default for AcousticSettings {
    fn default() -> Self {
        Self {
            gain: DEFAULT_AUDIO_GAIN,
            pitch_scale: DEFAULT_PITCH_SCALE,
        }
    }
}

impl AcousticSettings {
    /// Returns `true` when settings match factory defaults (used for `skip_serializing_if`).
    #[must_use]
    pub fn is_default(&self) -> bool {
        (self.gain - DEFAULT_AUDIO_GAIN).abs() < 1e-6
            && (self.pitch_scale - DEFAULT_PITCH_SCALE).abs() < 1e-6
    }

    /// Validates finiteness and safe acoustic bounds for preset and URL deserialization.
    pub fn validate(&self) -> Result<(), String> {
        if !self.gain.is_finite() || !(0.0..=1.0).contains(&self.gain) {
            return Err(format!(
                "Acoustic gain {} must be finite within [0.0, 1.0]",
                self.gain
            ));
        }
        if !self.pitch_scale.is_finite() || !(0.25..=4.0).contains(&self.pitch_scale) {
            return Err(format!(
                "Acoustic pitch_scale {} must be finite within [0.25, 4.0]",
                self.pitch_scale
            ));
        }
        Ok(())
    }

    /// Clamps potentially untrusted values into safe finite ranges.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let gain = if self.gain.is_finite() {
            self.gain.clamp(0.0, 1.0)
        } else {
            DEFAULT_AUDIO_GAIN
        };
        let pitch_scale = if self.pitch_scale.is_finite() {
            self.pitch_scale.clamp(0.25, 4.0)
        } else {
            DEFAULT_PITCH_SCALE
        };
        Self { gain, pitch_scale }
    }
}
