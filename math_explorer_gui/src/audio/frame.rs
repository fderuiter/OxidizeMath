//! Stack-allocated acoustic frame types carried across simulation snapshots.

use serde::{Deserialize, Serialize};

/// Maximum simultaneous polyphonic voices per [`SonificationFrame`].
pub const MAX_VOICES: usize = 8;

/// Number of Fourier partial harmonics (`n = 1..=8`) per custom `PeriodicWave`.
pub const NUM_HARMONICS: usize = 8;

/// Minimum audible carrier or filter cutoff frequency in `Hz`.
pub const MIN_FREQ_HZ: f32 = 20.0;

/// Maximum carrier or filter cutoff frequency in `Hz`.
pub const MAX_FREQ_HZ: f32 = 8000.0;

/// Hard master limiter ceiling in decibels relative to full scale (`dBFS`).
pub const LIMITER_CEILING_DBFS: f32 = -3.0;

/// Linear amplitude ceiling corresponding to `-3.0 dBFS` (`10^(-3/20)`).
pub const LIMITER_CEILING_LINEAR: f32 = 0.707_945_76;

/// Silence floor displayed by the visual level meter in `dBFS`.
pub const FLOOR_DBFS: f32 = -60.0;

/// Exponential `AudioParam::set_target_at_time` smoothing time constant (`15 ms`).
pub const PARAM_SMOOTH_TAU_SEC: f64 = 0.015;

/// Fast ramp-to-silence time constant (`3 ms`, reaching `< -80 dBFS` within `<= 30 ms`).
pub const MUTE_RAMP_TAU_SEC: f64 = 0.003;

/// Built-in custom waveform timbres synthesized via continuous Fourier coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaveformPreset {
    /// Pure fundamental sine tone (`b_1 = 1`).
    #[default]
    Sine,
    /// Warm string-like series with smooth even and odd overtone roll-off.
    FourierWarm,
    /// Asymmetric patch-clamp action-potential impulse rich in upper harmonics.
    PatchClamp,
    /// Standing-wave modal blend emphasizing low-order quantum eigenharmonics.
    QuantumModal,
    /// Odd-harmonic saturated square wave suited to period-doubling cascades.
    SubharmonicSquare,
    /// Clear harmonic brass timbre for just-intonation prime chords.
    JustHarmonic,
}

impl WaveformPreset {
    /// All selectable presets for UI combo boxes and precomputed harmonic caches.
    pub const ALL: [Self; 6] = [
        Self::Sine,
        Self::FourierWarm,
        Self::PatchClamp,
        Self::QuantumModal,
        Self::SubharmonicSquare,
        Self::JustHarmonic,
    ];

    /// Human-readable label displayed in the sonification timbre selector.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sine => "Pure Sine",
            Self::FourierWarm => "Fourier Strings",
            Self::PatchClamp => "Patch-Clamp Pulse",
            Self::QuantumModal => "Quantum Eigenmodes",
            Self::SubharmonicSquare => "Subharmonic Reed",
            Self::JustHarmonic => "Just Harmonic Organ",
        }
    }

    /// Zero-based index into [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Sine => 0,
            Self::FourierWarm => 1,
            Self::PatchClamp => 2,
            Self::QuantumModal => 3,
            Self::SubharmonicSquare => 4,
            Self::JustHarmonic => 5,
        }
    }
}

/// Stack-resident real (`cos_a`) and imaginary (`sin_b`) harmonic partial weights (`n = 1..=8`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HarmonicWeights {
    /// Cosine coefficients `a_1 .. a_8` (`PeriodicWaveOptions::real[1..=8]`).
    pub cos_a: [f32; NUM_HARMONICS],
    /// Sine coefficients `b_1 .. b_8` (`PeriodicWaveOptions::imag[1..=8]`).
    pub sin_b: [f32; NUM_HARMONICS],
}

impl Default for HarmonicWeights {
    fn default() -> Self {
        Self::pure_sine()
    }
}

impl HarmonicWeights {
    /// Fundamental-only sine table (`b_1 = 1.0`).
    #[must_use]
    pub const fn pure_sine() -> Self {
        let mut sin_b = [0.0; NUM_HARMONICS];
        sin_b[0] = 1.0;
        Self {
            cos_a: [0.0; NUM_HARMONICS],
            sin_b,
        }
    }

    /// Sanitizes non-finite entries and normalizes L2 partial energy to unit RMS scale.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        let mut sum_sq = 0.0_f32;
        for i in 0..NUM_HARMONICS {
            let a = sanitize_f32(self.cos_a[i], 0.0, -4.0, 4.0);
            let b = sanitize_f32(self.sin_b[i], 0.0, -4.0, 4.0);
            self.cos_a[i] = a;
            self.sin_b[i] = b;
            sum_sq += a * a + b * b;
        }
        if sum_sq <= 1e-8 {
            return Self::pure_sine();
        }
        let inv = 1.0 / sum_sq.sqrt();
        for i in 0..NUM_HARMONICS {
            self.cos_a[i] *= inv;
            self.sin_b[i] *= inv;
        }
        self
    }
}

/// Parameters for a single polyphonic synthesis voice (`Copy`, stack-only).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoiceSpec {
    /// Carrier fundamental frequency in `Hz` (clamped to `20.0..=8000.0`).
    pub freq_hz: f32,
    /// Linear voice gain (`0.0..=1.0`).
    pub gain: f32,
    /// Stereo pan position (`-1.0` full left to `+1.0` full right).
    pub pan: f32,
    /// Frequency-modulation oscillator rate in `Hz` (`0.0..=4000.0`).
    pub fm_mod_hz: f32,
    /// Frequency-modulation depth index (`0.0..=8.0`).
    pub fm_index: f32,
    /// Low-pass biquad filter cutoff frequency in `Hz` (`20.0..=8000.0`).
    pub filter_cutoff_hz: f32,
    /// Patch-clamp spike click impulse intensity (`0.0..=1.0`).
    pub transient_impulse: f32,
}

impl Default for VoiceSpec {
    fn default() -> Self {
        Self::silent()
    }
}

impl VoiceSpec {
    /// Returns an inactive voice initialized to safe audible defaults.
    #[must_use]
    pub const fn silent() -> Self {
        Self {
            freq_hz: 220.0,
            gain: 0.0,
            pan: 0.0,
            fm_mod_hz: 0.0,
            fm_index: 0.0,
            filter_cutoff_hz: 4000.0,
            transient_impulse: 0.0,
        }
    }

    /// Clamps every parameter to finite, speaker-safe bounds.
    #[must_use]
    pub fn sanitize(self) -> Self {
        Self {
            freq_hz: clamp_freq_hz(self.freq_hz),
            gain: sanitize_f32(self.gain, 0.0, 0.0, 1.0),
            pan: sanitize_f32(self.pan, 0.0, -1.0, 1.0),
            fm_mod_hz: sanitize_f32(self.fm_mod_hz, 0.0, 0.0, 4000.0),
            fm_index: sanitize_f32(self.fm_index, 0.0, 0.0, 8.0),
            filter_cutoff_hz: clamp_freq_hz(self.filter_cutoff_hz),
            transient_impulse: sanitize_f32(self.transient_impulse, 0.0, 0.0, 1.0),
        }
    }
}

/// Stack-allocated polyphonic control frame emitted by mathematical simulations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SonificationFrame {
    /// Fixed bank of up to [`MAX_VOICES`] simultaneous voices.
    pub voices: [VoiceSpec; MAX_VOICES],
    /// Count of active voices in `0..=MAX_VOICES`.
    pub active_voices: u8,
    /// Base Fourier waveform timbre applied when `custom_harmonics` is `None`.
    pub waveform: WaveformPreset,
    /// Optional dynamic harmonic table (used e.g. by quantum eigenmode projections).
    pub custom_harmonics: Option<HarmonicWeights>,
    /// Filtered noise amplitude (`0.0..=1.0`, used in chaotic logistic regimes).
    pub noise_gain: f32,
    /// Low-pass cutoff frequency for the chaotic noise generator (`20.0..=8000.0 Hz`).
    pub noise_cutoff_hz: f32,
    /// Frame-level master gain (`0.0..=1.0`).
    pub master_gain: f32,
}

impl Default for SonificationFrame {
    fn default() -> Self {
        Self::silent()
    }
}

impl SonificationFrame {
    /// Returns a zero-energy silent frame (`Copy`, zero heap allocations).
    #[must_use]
    pub const fn silent() -> Self {
        Self {
            voices: [VoiceSpec::silent(); MAX_VOICES],
            active_voices: 0,
            waveform: WaveformPreset::Sine,
            custom_harmonics: None,
            noise_gain: 0.0,
            noise_cutoff_hz: 1200.0,
            master_gain: 0.0,
        }
    }

    /// Clamps all voice frequencies (`20..=8000 Hz`), gains, pans, and harmonics.
    #[must_use]
    pub fn sanitize(mut self) -> Self {
        let count = (self.active_voices as usize).min(MAX_VOICES);
        self.active_voices = count as u8;
        for (i, voice) in self.voices.iter_mut().enumerate() {
            if i < count {
                *voice = voice.sanitize();
            } else {
                *voice = VoiceSpec::silent();
            }
        }
        self.custom_harmonics = self.custom_harmonics.map(HarmonicWeights::normalized);
        self.noise_gain = sanitize_f32(self.noise_gain, 0.0, 0.0, 1.0);
        self.noise_cutoff_hz = clamp_freq_hz(self.noise_cutoff_hz);
        self.master_gain = sanitize_f32(self.master_gain, 0.0, 0.0, 1.0);
        self
    }

    /// Returns `true` when at least one voice or noise source carries audible gain.
    #[must_use]
    pub fn is_audible(&self) -> bool {
        if self.master_gain <= 1e-4 {
            return false;
        }
        let count = (self.active_voices as usize).min(MAX_VOICES);
        let voice_energy = self.voices[..count]
            .iter()
            .any(|v| v.gain > 1e-4 || v.transient_impulse > 1e-3);
        voice_energy || self.noise_gain > 1e-4
    }

    /// Estimates real-time meter telemetry directly from sanitized voice parameters.
    #[must_use]
    pub fn estimate_meter(&self, user_volume: f32, muted: bool) -> MeterReading {
        let clean = self.sanitize();
        let vol = sanitize_f32(user_volume, 0.75, 0.0, 1.0);
        if muted || vol <= 1e-4 || !clean.is_audible() {
            return MeterReading::silent();
        }
        let count = clean.active_voices as usize;
        let mut dominant_hz = 0.0_f32;
        let mut top_gain = 0.0_f32;
        let mut sum_amp = clean.noise_gain * 0.5;
        for voice in &clean.voices[..count] {
            let effective = voice.gain + 0.35 * voice.transient_impulse;
            sum_amp += effective;
            if effective > top_gain {
                top_gain = effective;
                dominant_hz = voice.freq_hz;
            }
        }
        if dominant_hz <= 0.0 && clean.noise_gain > 1e-4 {
            dominant_hz = clean.noise_cutoff_hz;
        }
        let raw_peak = sum_amp * clean.master_gain * vol;
        let is_limiter_engaged = raw_peak > LIMITER_CEILING_LINEAR;
        let limited_peak = soft_limit_sample(raw_peak);
        MeterReading {
            peak_dbfs: linear_to_dbfs(limited_peak),
            dominant_hz,
            active_voices: clean.active_voices,
            is_limiter_engaged,
        }
    }
}

/// Real-time acoustic telemetry rendered by the accessible visual dBFS and `Hz` meter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeterReading {
    /// Peak output level in `dBFS`, bounded within `[FLOOR_DBFS, LIMITER_CEILING_DBFS]`.
    pub peak_dbfs: f32,
    /// Dominant audible frequency in `Hz` (`0.0` when silent).
    pub dominant_hz: f32,
    /// Number of currently sounding voices (`0..=8`).
    pub active_voices: u8,
    /// Whether the `-3.0 dBFS` master limiter is actively compressing peaks.
    pub is_limiter_engaged: bool,
}

impl Default for MeterReading {
    fn default() -> Self {
        Self::silent()
    }
}

impl MeterReading {
    /// Silent meter reading at `-60.0 dBFS` and `0.0 Hz`.
    #[must_use]
    pub const fn silent() -> Self {
        Self {
            peak_dbfs: FLOOR_DBFS,
            dominant_hz: 0.0,
            active_voices: 0,
            is_limiter_engaged: false,
        }
    }
}

/// Clamps any frequency value to `[20.0, 8000.0] Hz`, replacing non-finite values with `220.0 Hz`.
#[inline]
#[must_use]
pub fn clamp_freq_hz(hz: f32) -> f32 {
    sanitize_f32(hz, 220.0, MIN_FREQ_HZ, MAX_FREQ_HZ)
}

/// Sanitizes a floating-point parameter, replacing `NaN`/`Inf` with `fallback` and clamping.
#[inline]
#[must_use]
pub fn sanitize_f32(val: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback.clamp(min, max)
    }
}

/// Applies a smooth knee limiter strictly capped at `-3.0 dBFS` (`LIMITER_CEILING_LINEAR`).
#[inline]
#[must_use]
pub fn soft_limit_sample(sample: f32) -> f32 {
    if !sample.is_finite() {
        return 0.0;
    }
    let normalized = sample / LIMITER_CEILING_LINEAR;
    (normalized.tanh() * LIMITER_CEILING_LINEAR)
        .clamp(-LIMITER_CEILING_LINEAR, LIMITER_CEILING_LINEAR)
}

/// Converts linear amplitude to decibels full-scale (`[FLOOR_DBFS, LIMITER_CEILING_DBFS]`).
#[inline]
#[must_use]
pub fn linear_to_dbfs(amp: f32) -> f32 {
    let mag = if amp.is_finite() { amp.abs() } else { 0.0 };
    if mag <= 1e-3 {
        FLOOR_DBFS
    } else {
        (20.0 * mag.log10()).clamp(FLOOR_DBFS, LIMITER_CEILING_DBFS)
    }
}
