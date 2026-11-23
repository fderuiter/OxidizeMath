//! Deterministic zero-allocation `OfflinePcmSynthesizer` for native verification and `.wav` export.

use std::f64::consts::TAU;

use super::backend::{MuteReason, SonificationBackend};
use super::fourier::PresetHarmonicBank;
use super::frame::{
    clamp_freq_hz, linear_to_dbfs, sanitize_f32, soft_limit_sample, HarmonicWeights, MeterReading,
    SonificationFrame, FLOOR_DBFS, LIMITER_CEILING_LINEAR, MAX_VOICES, MUTE_RAMP_TAU_SEC,
    NUM_HARMONICS, PARAM_SMOOTH_TAU_SEC,
};

#[derive(Clone, Copy, Debug)]
struct VoiceDspState {
    carrier_phase: f64,
    mod_phase: f64,
    smooth_freq_hz: f32,
    smooth_gain: f32,
    smooth_pan: f32,
    smooth_cutoff_hz: f32,
    lp_state: f32,
    click_env: f32,
    prev_impulse: f32,
}

impl Default for VoiceDspState {
    fn default() -> Self {
        Self {
            carrier_phase: 0.0,
            mod_phase: 0.0,
            smooth_freq_hz: 220.0,
            smooth_gain: 0.0,
            smooth_pan: 0.0,
            smooth_cutoff_hz: 4000.0,
            lp_state: 0.0,
            click_env: 0.0,
            prev_impulse: 0.0,
        }
    }
}

/// Pre-allocated deterministic stereo PCM synthesizer implementing [`SonificationBackend`].
///
/// All heap storage (`stereo_buffer`) is allocated once in [`Self::new`] prior to
/// `VerifiedAllocator::lock_allocations()`, guaranteeing zero dynamic allocations during
/// `apply_frame` and `render_frames`.
pub struct OfflinePcmSynthesizer {
    sample_rate: u32,
    stereo_buffer: Vec<f32>,
    rendered_frames: usize,
    preset_bank: PresetHarmonicBank,
    current_frame: SonificationFrame,
    voices: [VoiceDspState; MAX_VOICES],
    smooth_master: f32,
    noise_lp_state: f32,
    rng_state: u64,
    master_volume: f32,
    muted: bool,
    last_mute_reason: MuteReason,
    last_meter: MeterReading,
}

impl OfflinePcmSynthesizer {
    /// Pre-allocates stereo PCM storage for up to `max_frames` stereo sample pairs.
    #[must_use]
    pub fn new(sample_rate: u32, max_frames: usize) -> Self {
        let sr = sample_rate.clamp(8_000, 192_000);
        let cap = max_frames.max(64);
        Self {
            sample_rate: sr,
            stereo_buffer: vec![0.0_f32; cap.saturating_mul(2)],
            rendered_frames: 0,
            preset_bank: PresetHarmonicBank::new(),
            current_frame: SonificationFrame::silent(),
            voices: [VoiceDspState::default(); MAX_VOICES],
            smooth_master: 0.0,
            noise_lp_state: 0.0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
            master_volume: 0.80,
            muted: true,
            last_mute_reason: MuteReason::InitialDefault,
            last_meter: MeterReading::silent(),
        }
    }

    /// Sample rate in `Hz`.
    #[inline]
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Maximum stereo frames pre-allocated in this synthesizer.
    #[inline]
    #[must_use]
    pub fn capacity_frames(&self) -> usize {
        self.stereo_buffer.len() / 2
    }

    /// Most recent lifecycle reason that muted or initialized the synthesizer.
    #[inline]
    #[must_use]
    pub const fn last_mute_reason(&self) -> MuteReason {
        self.last_mute_reason
    }

    /// Returns the most recently rendered interleaved stereo `[L0, R0, L1, R1, ...]` slice.
    #[inline]
    #[must_use]
    pub fn stereo_samples(&self) -> &[f32] {
        &self.stereo_buffer[..self.rendered_frames.saturating_mul(2)]
    }

    /// Renders `frame_count` stereo samples in place without heap allocation.
    pub fn render_frames(&mut self, frame_count: usize) -> &[f32] {
        let frames = frame_count.min(self.capacity_frames());
        self.rendered_frames = frames;
        let sr = self.sample_rate as f64;
        let param_alpha = one_pole_alpha(sr, PARAM_SMOOTH_TAU_SEC);
        let mute_alpha = one_pole_alpha(sr, MUTE_RAMP_TAU_SEC);
        let harmonics = self
            .current_frame
            .custom_harmonics
            .unwrap_or_else(|| self.preset_bank.get(self.current_frame.waveform));
        let target_master = if self.muted {
            0.0
        } else {
            self.current_frame.master_gain * self.master_volume
        };

        let mut peak = 0.0_f32;
        let mut limiter_hit = false;
        for idx in 0..frames {
            let master_alpha = if self.muted { mute_alpha } else { param_alpha };
            self.smooth_master += master_alpha * (target_master - self.smooth_master);
            if self.muted && self.smooth_master < 1e-4 {
                self.smooth_master = 0.0;
            }
            let (raw_l, raw_r) = self.step_stereo_frame(sr, param_alpha, &harmonics);
            if raw_l.abs() > LIMITER_CEILING_LINEAR || raw_r.abs() > LIMITER_CEILING_LINEAR {
                limiter_hit = true;
            }
            let out_l = soft_limit_sample(raw_l);
            let out_r = soft_limit_sample(raw_r);
            peak = peak.max(out_l.abs()).max(out_r.abs());
            let base = idx * 2;
            self.stereo_buffer[base] = out_l;
            self.stereo_buffer[base + 1] = out_r;
        }
        self.update_meter_from_peak(peak, limiter_hit);
        &self.stereo_buffer[..frames * 2]
    }

    fn step_stereo_frame(
        &mut self,
        sr: f64,
        param_alpha: f32,
        harmonics: &HarmonicWeights,
    ) -> (f32, f32) {
        let active = self.current_frame.active_voices as usize;
        let mut mix_l = 0.0_f32;
        let mut mix_r = 0.0_f32;
        for i in 0..MAX_VOICES {
            let spec = if i < active {
                self.current_frame.voices[i]
            } else {
                super::frame::VoiceSpec::silent()
            };
            let (vl, vr) =
                step_single_voice(&mut self.voices[i], &spec, sr, param_alpha, harmonics);
            mix_l += vl;
            mix_r += vr;
        }
        if self.current_frame.noise_gain > 1e-5 {
            let white = next_bipolar_noise(&mut self.rng_state);
            let cutoff = clamp_freq_hz(self.current_frame.noise_cutoff_hz) as f64;
            let lp_alpha = cutoff_alpha(sr, cutoff);
            self.noise_lp_state += lp_alpha * (white - self.noise_lp_state);
            let n_sig = self.noise_lp_state * self.current_frame.noise_gain * 0.35;
            mix_l += n_sig;
            mix_r += n_sig;
        }
        (mix_l * self.smooth_master, mix_r * self.smooth_master)
    }

    fn update_meter_from_peak(&mut self, peak: f32, limiter_hit: bool) {
        if (self.muted && self.smooth_master <= 1e-4) || peak <= 1e-4 {
            self.last_meter = MeterReading {
                peak_dbfs: FLOOR_DBFS,
                dominant_hz: 0.0,
                active_voices: 0,
                is_limiter_engaged: false,
            };
            return;
        }
        let est = self.current_frame.estimate_meter(self.master_volume, false);
        self.last_meter = MeterReading {
            peak_dbfs: linear_to_dbfs(peak),
            dominant_hz: est.dominant_hz,
            active_voices: self.current_frame.active_voices,
            is_limiter_engaged: limiter_hit || est.is_limiter_engaged,
        };
    }
}

impl SonificationBackend for OfflinePcmSynthesizer {
    fn is_muted(&self) -> bool {
        self.muted
    }

    fn set_muted_from_user_gesture(&mut self, muted: bool) -> Result<(), String> {
        self.muted = muted;
        if muted {
            self.last_mute_reason = MuteReason::UserMuted;
            self.last_meter = MeterReading::silent();
        }
        Ok(())
    }

    fn mute_and_suspend(&mut self, reason: MuteReason) {
        self.muted = true;
        self.last_mute_reason = reason;
        self.current_frame = SonificationFrame::silent();
        self.last_meter = MeterReading::silent();
    }

    fn apply_frame(&mut self, frame: &SonificationFrame) {
        let clean = frame.sanitize();
        for (state, spec) in self.voices.iter_mut().zip(clean.voices.iter()) {
            if spec.transient_impulse > 0.25 && state.prev_impulse <= 0.25 {
                state.click_env = spec.transient_impulse;
            }
            state.prev_impulse = spec.transient_impulse;
        }
        self.current_frame = clean;
        self.last_meter = clean.estimate_meter(self.master_volume, self.muted);
    }

    fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = sanitize_f32(volume, 0.80, 0.0, 1.0);
        self.last_meter = self
            .current_frame
            .estimate_meter(self.master_volume, self.muted);
    }

    fn master_volume(&self) -> f32 {
        self.master_volume
    }

    fn meter_reading(&self) -> MeterReading {
        self.last_meter
    }
}

fn step_single_voice(
    state: &mut VoiceDspState,
    spec: &super::frame::VoiceSpec,
    sr: f64,
    param_alpha: f32,
    harmonics: &HarmonicWeights,
) -> (f32, f32) {
    state.smooth_freq_hz += param_alpha * (spec.freq_hz - state.smooth_freq_hz);
    state.smooth_gain += param_alpha * (spec.gain - state.smooth_gain);
    state.smooth_pan += param_alpha * (spec.pan - state.smooth_pan);
    state.smooth_cutoff_hz += param_alpha * (spec.filter_cutoff_hz - state.smooth_cutoff_hz);

    if state.smooth_gain <= 1e-5 && state.click_env <= 1e-4 {
        return (0.0, 0.0);
    }
    let mod_inc = TAU * (spec.fm_mod_hz as f64) / sr;
    state.mod_phase = (state.mod_phase + mod_inc).rem_euclid(TAU);
    let fm_offset = (spec.fm_index as f64) * state.mod_phase.sin();
    let carrier_inc = TAU * (state.smooth_freq_hz as f64) / sr;
    state.carrier_phase = (state.carrier_phase + carrier_inc).rem_euclid(TAU);

    let inst_phase = state.carrier_phase + fm_offset;
    let raw = eval_harmonic_series(inst_phase, state.smooth_freq_hz, sr as f32, harmonics);
    let click = state.click_env * (inst_phase * 3.0).sin() as f32;
    state.click_env *= (-1.0_f64 / (0.0025 * sr)).exp() as f32;

    let lp_alpha = cutoff_alpha(sr, state.smooth_cutoff_hz as f64);
    state.lp_state += lp_alpha * ((raw * state.smooth_gain + click * 0.45) - state.lp_state);
    let angle = (state.smooth_pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (state.lp_state * angle.cos(), state.lp_state * angle.sin())
}

fn eval_harmonic_series(
    phase: f64,
    base_freq_hz: f32,
    sample_rate_hz: f32,
    harmonics: &HarmonicWeights,
) -> f32 {
    let nyquist = 0.48 * sample_rate_hz;
    let mut sum = 0.0_f32;
    for n in 0..NUM_HARMONICS {
        let k = (n + 1) as f32;
        if base_freq_hz * k >= nyquist {
            break;
        }
        let a = harmonics.cos_a[n];
        let b = harmonics.sin_b[n];
        if a.abs() <= 1e-5 && b.abs() <= 1e-5 {
            continue;
        }
        let arg = phase * (k as f64);
        sum += a * (arg.cos() as f32) + b * (arg.sin() as f32);
    }
    sum
}

#[inline]
fn one_pole_alpha(sample_rate: f64, tau_sec: f64) -> f32 {
    (1.0 - (-1.0 / (sample_rate * tau_sec.max(1e-4))).exp()).clamp(0.0, 1.0) as f32
}

#[inline]
fn cutoff_alpha(sample_rate: f64, cutoff_hz: f64) -> f32 {
    let rc_inv = TAU * cutoff_hz.clamp(20.0, 0.45 * sample_rate);
    (rc_inv / (rc_inv + sample_rate)).clamp(0.001, 1.0) as f32
}

#[inline]
fn next_bipolar_noise(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    let unit = ((x >> 40) as f32) * (1.0 / 16_777_216.0);
    unit * 2.0 - 1.0
}
