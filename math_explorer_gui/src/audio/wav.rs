//! Pure-Rust deterministic 16-bit stereo PCM `.wav` synthesizer with zero inner-loop allocations.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::frame::{
    AcousticSettings, SonificationFrame, MAX_HARMONIC_SLOTS, MAX_POLYRHYTHM_VOICES,
};
use super::limiter::{equal_power_pan_gains, soft_clip_stereo};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Standard CD-quality PCM sample rate in Hz (`44,100 Hz`).
pub const DEFAULT_WAV_SAMPLE_RATE: u32 = 44_100;

/// Default exported `.wav` clip duration in seconds (`2.0 s`).
pub const DEFAULT_WAV_DURATION_SECS: f32 = 2.0;

/// Default number of simulation frames sampled per exported `.wav` clip (`120` frames @ 60 Hz = 2 s).
pub const DEFAULT_WAV_FRAME_COUNT: usize = 120;

/// Canonical 44-byte RIFF/WAVE PCM header byte length.
pub const WAV_HEADER_BYTES: usize = 44;

/// Zero-allocation DSP state machine for rendering `SonificationFrame` streams into 16-bit PCM.
#[derive(Clone)]
pub struct PcmSynthState {
    carrier_phase: f32,
    modulator_phase: f32,
    harmonic_phases: [f32; MAX_HARMONIC_SLOTS],
    poly_phases: [f32; MAX_POLYRHYTHM_VOICES],
    bell_phases: [f32; 3],
    bell_env: f32,
    click_env: f32,
    svf_ic1: [f32; 2],
    svf_ic2: [f32; 2],
    rng: OxidizeRng,
}

impl PcmSynthState {
    /// Creates a deterministic DSP synthesizer seeded with `seed` (`OxidizeRng`).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            carrier_phase: 0.0,
            modulator_phase: 0.0,
            harmonic_phases: [0.0; MAX_HARMONIC_SLOTS],
            poly_phases: [0.0; MAX_POLYRHYTHM_VOICES],
            bell_phases: [0.0; 3],
            bell_env: 1.0,
            click_env: 0.0,
            svf_ic1: [0.0; 2],
            svf_ic2: [0.0; 2],
            rng: OxidizeRng::new(seed),
        }
    }

    #[inline]
    fn advance_phase(phase: &mut f32, freq_hz: f32, inv_sr: f32) -> f32 {
        let current = *phase;
        let next = (current + freq_hz * inv_sr).fract();
        *phase = if next < 0.0 { next + 1.0 } else { next };
        current * core::f32::consts::TAU
    }

    #[inline]
    fn render_periodic_harmonics(
        &mut self,
        frame: &SonificationFrame,
        base_hz: f32,
        inv_sr: f32,
    ) -> f32 {
        let mut sum = 0.0_f32;
        for k in 1..MAX_HARMONIC_SLOTS {
            let re = frame.harmonics_real[k];
            let im = frame.harmonics_imag[k];
            if re.abs() < 1e-5 && im.abs() < 1e-5 {
                continue;
            }
            let angle =
                Self::advance_phase(&mut self.harmonic_phases[k], base_hz * k as f32, inv_sr);
            let (s, c) = angle.sin_cos();
            sum += re * c + im * s;
        }
        sum * 0.45
    }

    #[inline]
    fn render_poly_voices(
        &mut self,
        frame: &SonificationFrame,
        pitch_scale: f32,
        inv_sr: f32,
    ) -> f32 {
        let mut sum = 0.0_f32;
        for v in 0..MAX_POLYRHYTHM_VOICES {
            let amp = frame.poly_voices_amp[v];
            let hz = frame.poly_voices_hz[v] * pitch_scale;
            if amp <= 1e-5 || hz < 20.0 {
                continue;
            }
            let angle = Self::advance_phase(&mut self.poly_phases[v], hz, inv_sr);
            let s1 = angle.sin();
            let s2 = (angle * 2.0).sin() * 0.25;
            sum += amp * (s1 + s2) * 0.35;
        }
        sum
    }

    #[inline]
    fn render_bell_partials(&mut self, base_hz: f32, decay_rate: f32, inv_sr: f32) -> f32 {
        const RATIOS: [f32; 3] = [1.0, 2.756, 5.404];
        const WEIGHTS: [f32; 3] = [0.65, 0.23, 0.12];
        self.bell_env *= (-decay_rate * inv_sr).exp();
        let mut sum = 0.0_f32;
        for i in 0..3 {
            let angle = Self::advance_phase(&mut self.bell_phases[i], base_hz * RATIOS[i], inv_sr);
            sum += WEIGHTS[i] * angle.sin();
        }
        sum * self.bell_env.max(0.15)
    }

    #[inline]
    fn apply_svf_filter(
        &mut self,
        channel: usize,
        input: f32,
        cutoff_hz: f32,
        q: f32,
        bandpass: bool,
        sample_rate: f32,
    ) -> f32 {
        let max_cutoff = (sample_rate * 0.45).max(80.0);
        let fc = cutoff_hz.clamp(40.0, max_cutoff);
        let g = (core::f32::consts::PI * fc / sample_rate).tan();
        let k = 1.0 / q.clamp(0.25, 18.0);
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        let v3 = input - self.svf_ic2[channel];
        let v1 = a1 * self.svf_ic1[channel] + a2 * v3;
        let v2 = self.svf_ic2[channel] + a2 * self.svf_ic1[channel] + a3 * v3;
        self.svf_ic1[channel] = 2.0 * v1 - self.svf_ic1[channel];
        self.svf_ic2[channel] = 2.0 * v2 - self.svf_ic2[channel];
        if bandpass {
            v1
        } else {
            v2
        }
    }

    /// Renders a block of stereo 16-bit LE PCM bytes into `out_pcm_le` (`4 * sample_frames` bytes)
    /// with strictly zero heap allocations.
    pub fn render_chunk_into(
        &mut self,
        frame: SonificationFrame,
        settings: AcousticSettings,
        sample_rate: u32,
        out_pcm_le: &mut [u8],
    ) {
        let f = frame.sanitized();
        let cfg = settings.sanitized();
        let sr = (sample_rate.max(8_000)) as f32;
        let inv_sr = 1.0 / sr;
        if f.click_impulse > 0.02 {
            self.click_env = self.click_env.max(f.click_impulse);
        }
        let base_hz = (f.carrier_hz * cfg.pitch_scale).clamp(20.0, sr * 0.45);
        let mod_hz = (f.fm_modulator_hz * cfg.pitch_scale).clamp(0.0, sr * 0.45);
        let (pan_l, pan_r) = equal_power_pan_gains(f.pan);

        for stereo_bytes in out_pcm_le.as_chunks_mut::<4>().0 {
            let mut dry = 0.0_f32;
            if f.active {
                let mod_angle = Self::advance_phase(&mut self.modulator_phase, mod_hz, inv_sr);
                let inst_hz = (base_hz + f.fm_index_hz * mod_angle.sin()).clamp(20.0, sr * 0.45);
                if f.use_periodic_wave {
                    dry += f.amplitude * self.render_periodic_harmonics(&f, inst_hz, inv_sr);
                } else if f.bell_decay_rate > 0.0 {
                    dry +=
                        f.amplitude * self.render_bell_partials(inst_hz, f.bell_decay_rate, inv_sr);
                } else {
                    let car_angle = Self::advance_phase(&mut self.carrier_phase, inst_hz, inv_sr);
                    dry += f.amplitude * car_angle.sin() * 0.55;
                }
                dry += self.render_poly_voices(&f, cfg.pitch_scale, inv_sr);
            }

            if self.click_env > 1e-4 {
                let white = ((self.rng.next_u32() as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
                dry += white * self.click_env * 0.65;
                self.click_env *= (-950.0 * inv_sr).exp();
            }

            let flt_l = self.apply_svf_filter(
                0,
                dry * pan_l,
                f.filter_cutoff_hz,
                f.filter_q,
                f.filter_bandpass,
                sr,
            );
            let flt_r = self.apply_svf_filter(
                1,
                dry * pan_r,
                f.filter_cutoff_hz,
                f.filter_q,
                f.filter_bandpass,
                sr,
            );
            let (lim_l, lim_r) = soft_clip_stereo(flt_l * cfg.gain, flt_r * cfg.gain);
            let pcm_l = (lim_l * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16;
            let pcm_r = (lim_r * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16;
            let lb = pcm_l.to_le_bytes();
            let rb = pcm_r.to_le_bytes();
            *stereo_bytes = [lb[0], lb[1], rb[0], rb[1]];
        }
    }
}

/// Writes the canonical 44-byte RIFF/WAVE 16-bit stereo PCM header into `header`.
pub fn write_wav_header(sample_rate: u32, num_stereo_frames: u32) -> [u8; WAV_HEADER_BYTES] {
    let channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let block_align: u16 = channels * (bits_per_sample / 8);
    let byte_rate: u32 = sample_rate.saturating_mul(u32::from(block_align));
    let data_bytes: u32 = num_stereo_frames.saturating_mul(u32::from(block_align));
    let riff_chunk_size: u32 = 36_u32.saturating_add(data_bytes);

    let mut h = [0_u8; WAV_HEADER_BYTES];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&riff_chunk_size.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16_u32.to_le_bytes());
    h[20..22].copy_from_slice(&1_u16.to_le_bytes());
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&bits_per_sample.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    h
}

/// Synthesizes a complete 16-bit stereo PCM `.wav` file from a sequence of `SonificationFrame`s.
#[must_use]
pub fn synthesize_wav_16bit(
    frames: &[SonificationFrame],
    settings: AcousticSettings,
    sample_rate: u32,
    duration_secs: f32,
    seed: u64,
) -> Vec<u8> {
    let sr = sample_rate.clamp(8_000, 96_000);
    let dur = if duration_secs.is_finite() {
        duration_secs.clamp(0.1, 10.0)
    } else {
        DEFAULT_WAV_DURATION_SECS
    };
    let total_samples = ((sr as f32 * dur).round() as usize).max(1);
    let data_len = total_samples * 4;
    let mut wav = vec![0_u8; WAV_HEADER_BYTES + data_len];
    let header = write_wav_header(sr, total_samples as u32);
    wav[..WAV_HEADER_BYTES].copy_from_slice(&header);

    let mut synth = PcmSynthState::new(seed);
    let mut effective_cfg = settings.sanitized();
    if effective_cfg.gain < 0.05 {
        effective_cfg.gain = 0.5;
    }

    if frames.is_empty() {
        synth.render_chunk_into(
            SonificationFrame::silent(),
            effective_cfg,
            sr,
            &mut wav[WAV_HEADER_BYTES..],
        );
        return wav;
    }

    let n_frames = frames.len();
    for (idx, frame) in frames.iter().copied().enumerate() {
        let start_sample = (idx * total_samples) / n_frames;
        let end_sample = ((idx + 1) * total_samples) / n_frames;
        if end_sample > start_sample {
            let byte_start = WAV_HEADER_BYTES + start_sample * 4;
            let byte_end = WAV_HEADER_BYTES + end_sample * 4;
            synth.render_chunk_into(frame, effective_cfg, sr, &mut wav[byte_start..byte_end]);
        }
    }
    wav
}
