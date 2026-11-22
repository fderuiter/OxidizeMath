//! Fixed-size, zero-allocation `SonificationFrame` acoustic descriptors and persisted
//! non-mute `AcousticSettings` bounded for safe synthesis and preset/URL sharing.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use serde::{Deserialize, Serialize};

/// Number of `PeriodicWave` Fourier slots (`0` = DC offset `0.0`, `1..8` = harmonics 1..=7).
pub const MAX_HARMONIC_SLOTS: usize = 8;

/// Number of simultaneous polyrhythmic step-sequencer voices.
pub const MAX_POLYRHYTHM_VOICES: usize = 4;

/// Minimum audible carrier frequency in Hz.
pub const MIN_CARRIER_HZ: f32 = 20.0;
/// Maximum audible carrier frequency in Hz.
pub const MAX_CARRIER_HZ: f32 = 12_000.0;
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

/// Fixed-size, stack-allocated acoustic frame emitted per simulation step or UI tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonificationFrame {
    /// Whether this frame actively drives audible output when unmuted.
    pub active: bool,
    /// Primary oscillator carrier frequency in Hz (`20.0..=12_000.0`).
    pub carrier_hz: f32,
    /// Normalized voice amplitude envelope (`0.0..=1.0`).
    pub amplitude: f32,
    /// Equal-power stereo pan (`-1.0` = hard left, `0.0` = center, `+1.0` = hard right).
    pub pan: f32,
    /// Two-operator FM modulator frequency in Hz (`0.0..=8_000.0`).
    pub fm_modulator_hz: f32,
    /// Two-operator FM peak frequency deviation in Hz (`0.0..=4_000.0`).
    pub fm_index_hz: f32,
    /// Biquad filter cutoff / center frequency in Hz (`40.0..=16_000.0`).
    pub filter_cutoff_hz: f32,
    /// Biquad filter resonance `Q` (`0.25..=18.0`).
    pub filter_q: f32,
    /// Selects band-pass (`true`) vs. low-pass (`false`) biquad response.
    pub filter_bandpass: bool,
    /// Cosine (`real`) Fourier coefficients for `PeriodicWave` synthesis (`[0]` = DC `0.0`).
    pub harmonics_real: [f32; MAX_HARMONIC_SLOTS],
    /// Sine (`imag`) Fourier coefficients for `PeriodicWave` synthesis (`[0]` = `0.0`).
    pub harmonics_imag: [f32; MAX_HARMONIC_SLOTS],
    /// Whether `harmonics_real` / `harmonics_imag` drive a custom `PeriodicWave` spectrum.
    pub use_periodic_wave: bool,
    /// Stochastic patch-clamp spike click burst intensity (`0.0..=1.0`).
    pub click_impulse: f32,
    /// Exponential ring-down decay rate `1/T2` in `s^-1` (`0.0..=200.0`) for NMR FID bells.
    pub bell_decay_rate: f32,
    /// Frequencies in Hz for up to 4 polyrhythmic step-sequencer voices (`0.0` = silent).
    pub poly_voices_hz: [f32; MAX_POLYRHYTHM_VOICES],
    /// Amplitudes (`0.0..=1.0`) for up to 4 polyrhythmic step-sequencer voices.
    pub poly_voices_amp: [f32; MAX_POLYRHYTHM_VOICES],
}

impl Default for SonificationFrame {
    fn default() -> Self {
        Self::silent()
    }
}

fn sanitize_bounded(val: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

impl SonificationFrame {
    /// Returns a completely silent, zero-energy `SonificationFrame`.
    #[must_use]
    pub const fn silent() -> Self {
        Self {
            active: false,
            carrier_hz: 220.0,
            amplitude: 0.0,
            pan: 0.0,
            fm_modulator_hz: 0.0,
            fm_index_hz: 0.0,
            filter_cutoff_hz: 2_400.0,
            filter_q: 0.707,
            filter_bandpass: false,
            harmonics_real: [0.0; MAX_HARMONIC_SLOTS],
            harmonics_imag: [0.0; MAX_HARMONIC_SLOTS],
            use_periodic_wave: false,
            click_impulse: 0.0,
            bell_decay_rate: 0.0,
            poly_voices_hz: [0.0; MAX_POLYRHYTHM_VOICES],
            poly_voices_amp: [0.0; MAX_POLYRHYTHM_VOICES],
        }
    }

    /// Sanitizes every field to finite physical bounds without heap allocation.
    /// Non-finite primary carrier or amplitude forces the frame silent.
    #[must_use]
    pub fn sanitized(mut self) -> Self {
        if !self.carrier_hz.is_finite() || !self.amplitude.is_finite() || !self.pan.is_finite() {
            return Self::silent();
        }
        self.carrier_hz = sanitize_bounded(self.carrier_hz, MIN_CARRIER_HZ, MAX_CARRIER_HZ, 220.0);
        self.amplitude = if self.active {
            sanitize_bounded(self.amplitude, 0.0, 1.0, 0.0)
        } else {
            0.0
        };
        self.pan = sanitize_bounded(self.pan, -1.0, 1.0, 0.0);
        self.fm_modulator_hz = sanitize_bounded(self.fm_modulator_hz, 0.0, 8_000.0, 0.0);
        self.fm_index_hz = sanitize_bounded(self.fm_index_hz, 0.0, 4_000.0, 0.0);
        self.filter_cutoff_hz = sanitize_bounded(self.filter_cutoff_hz, 40.0, 16_000.0, 2_400.0);
        self.filter_q = sanitize_bounded(self.filter_q, 0.25, 18.0, 0.707);
        self.click_impulse = if self.active {
            sanitize_bounded(self.click_impulse, 0.0, 1.0, 0.0)
        } else {
            0.0
        };
        self.bell_decay_rate = sanitize_bounded(self.bell_decay_rate, 0.0, 200.0, 0.0);

        self.harmonics_real[0] = 0.0;
        self.harmonics_imag[0] = 0.0;
        for slot in 1..MAX_HARMONIC_SLOTS {
            self.harmonics_real[slot] = sanitize_bounded(self.harmonics_real[slot], -1.0, 1.0, 0.0);
            self.harmonics_imag[slot] = sanitize_bounded(self.harmonics_imag[slot], -1.0, 1.0, 0.0);
        }
        for v in 0..MAX_POLYRHYTHM_VOICES {
            self.poly_voices_hz[v] =
                sanitize_bounded(self.poly_voices_hz[v], 0.0, MAX_CARRIER_HZ, 0.0);
            self.poly_voices_amp[v] = if self.active && self.poly_voices_hz[v] >= MIN_CARRIER_HZ {
                sanitize_bounded(self.poly_voices_amp[v], 0.0, 1.0, 0.0)
            } else {
                0.0
            };
        }
        self
    }
}
