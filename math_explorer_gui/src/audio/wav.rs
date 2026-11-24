//! Deterministic RIFF `WAVE` (`PCM16` stereo) encoding and export helpers.

// theory_verification!

use super::backend::SonificationBackend;
use super::frame::{sanitize_f32, SonificationFrame, LIMITER_CEILING_LINEAR};
use super::settings::AcousticSettings;
use super::synth::OfflinePcmSynthesizer;

/// Default CD-quality sample rate (`44.1 kHz`) used when exporting `.wav` clips.
pub const WAV_EXPORT_SAMPLE_RATE: u32 = 44_100;

/// Studio broadcast sample rate (`48.0 kHz`) selectable for offline `.wav` rendering.
pub const WAV_STUDIO_SAMPLE_RATE: u32 = 48_000;

/// Minimum deterministic offline export duration (`1.0 s`).
pub const MIN_OFFLINE_DURATION_SECS: f32 = 1.0;

/// Maximum deterministic offline export duration (`15.0 s`) to keep synchronous browser exports responsive.
pub const MAX_OFFLINE_DURATION_SECS: f32 = 15.0;

/// Default deterministic offline export duration (`5.0 s`).
pub const DEFAULT_OFFLINE_DURATION_SECS: f32 = 5.0;

/// Hard upper bound on stereo sample frames (`15.0 s * 48,000 Hz = 720,000` frames).
pub const MAX_OFFLINE_STEREO_FRAMES: usize = 720_000;

/// Control-frame update rate (`60 Hz`) used during deterministic offline `.wav` & `.mid` rendering.
pub const OFFLINE_CONTROL_RATE_HZ: usize = 60;

/// Validated configuration for deterministic frame-rate-independent offline `.wav` synthesis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OfflineRenderConfig {
    /// Output sample rate (`44_100` or `48_000 Hz`).
    pub sample_rate: u32,
    /// Target clip duration in seconds (`1.0..=15.0 s`).
    pub duration_secs: f32,
    /// Master gain multiplier (`0.0..=1.0`).
    pub master_volume: f32,
}

impl Default for OfflineRenderConfig {
    fn default() -> Self {
        Self {
            sample_rate: WAV_EXPORT_SAMPLE_RATE,
            duration_secs: DEFAULT_OFFLINE_DURATION_SECS,
            master_volume: 0.80,
        }
    }
}

impl OfflineRenderConfig {
    /// Normalizes sample rate to `44.1 kHz` or `48 kHz` and clamps duration to `[1.0, 15.0] s`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let sample_rate = if self.sample_rate >= 46_000 {
            WAV_STUDIO_SAMPLE_RATE
        } else {
            WAV_EXPORT_SAMPLE_RATE
        };
        let duration_secs = sanitize_f32(
            self.duration_secs,
            DEFAULT_OFFLINE_DURATION_SECS,
            MIN_OFFLINE_DURATION_SECS,
            MAX_OFFLINE_DURATION_SECS,
        );
        let master_volume = sanitize_f32(self.master_volume, 0.80, 0.0, 1.0);
        Self {
            sample_rate,
            duration_secs,
            master_volume,
        }
    }

    /// Computes exact stereo sample frame count bounded by [`MAX_OFFLINE_STEREO_FRAMES`].
    #[must_use]
    pub fn total_stereo_frames(self) -> usize {
        let clean = self.sanitized();
        let raw = (f64::from(clean.duration_secs) * f64::from(clean.sample_rate)).round() as usize;
        raw.clamp(clean.sample_rate as usize, MAX_OFFLINE_STEREO_FRAMES)
    }

    /// Computes stereo frames per `60 Hz` control step (`735` at `44.1 kHz`, `800` at `48 kHz`).
    #[must_use]
    pub fn step_stereo_frames(self) -> usize {
        let clean = self.sanitized();
        ((clean.sample_rate as usize) / OFFLINE_CONTROL_RATE_HZ).max(64)
    }

    /// Scales `master_volume` relative to [`super::settings::DEFAULT_AUDIO_GAIN`] (`0.5`).
    #[must_use]
    pub fn with_tool_gain(mut self, tool_gain: f32) -> Self {
        let ratio = if tool_gain.is_finite() {
            (tool_gain.clamp(0.0, 1.0) / super::settings::DEFAULT_AUDIO_GAIN).clamp(0.0, 2.0)
        } else {
            1.0
        };
        self.master_volume = (self.master_volume * ratio).clamp(0.0, 1.0);
        self
    }
}

/// Renders deterministic offline PCM stereo samples into pre-allocated `out_interleaved`
/// using pre-allocated `synth` with **zero dynamic heap allocations** inside the loop.
pub fn render_offline_pcm_into<F>(
    synth: &mut OfflinePcmSynthesizer,
    step_frames: usize,
    out_interleaved: &mut [f32],
    mut frame_at_step: F,
) -> usize
where
    F: FnMut(usize, f64) -> SonificationFrame,
{
    let total_frames = (out_interleaved.len() / 2).min(MAX_OFFLINE_STEREO_FRAMES);
    let chunk_cap = step_frames.max(64).min(synth.capacity_frames().max(1));
    let inv_sr = 1.0 / f64::from(synth.sample_rate().max(1));
    let mut written_frames = 0_usize;
    let mut step_idx = 0_usize;

    while written_frames < total_frames {
        let chunk_frames = (total_frames - written_frames).min(chunk_cap);
        let t_sec = (written_frames as f64) * inv_sr;
        let frame = frame_at_step(step_idx, t_sec);
        synth.apply_frame(&frame);
        let rendered = synth.render_frames(chunk_frames);
        let start = written_frames * 2;
        let end = start + rendered.len();
        out_interleaved[start..end].copy_from_slice(rendered);
        written_frames += rendered.len() / 2;
        step_idx += 1;
    }
    written_frames
}

/// Pre-allocates synthesis buffers, steps `frame_at_step` deterministically over `config`,
/// and encodes a 16-bit stereo RIFF `WAVE` file.
#[must_use]
pub fn synthesize_offline_wav<F>(config: OfflineRenderConfig, frame_at_step: F) -> Vec<u8>
where
    F: FnMut(usize, f64) -> SonificationFrame,
{
    let clean = config.sanitized();
    let total_frames = clean.total_stereo_frames();
    let step_frames = clean.step_stereo_frames();
    let mut synth = OfflinePcmSynthesizer::new(clean.sample_rate, step_frames);
    synth.set_master_volume(clean.master_volume);
    let _ = synth.set_muted_from_user_gesture(false);

    let mut samples = vec![0.0_f32; total_frames * 2];
    let _ = render_offline_pcm_into(&mut synth, step_frames, &mut samples, frame_at_step);
    encode_wav_stereo_pcm16(clean.sample_rate, &samples)
}

/// Standard CD-quality PCM sample rate in `Hz` (`44,100 Hz`).
pub const DEFAULT_WAV_SAMPLE_RATE: u32 = WAV_EXPORT_SAMPLE_RATE;

/// Default exported `.wav` clip duration in seconds (`2.0 s`).
pub const DEFAULT_WAV_DURATION_SECS: f32 = 2.0;

/// Default number of simulation frames sampled per exported `.wav` clip (`120` frames @ 60 Hz = 2 s).
pub const DEFAULT_WAV_FRAME_COUNT: usize = 120;

/// Canonical 44-byte RIFF/WAVE PCM header byte length.
pub const WAV_HEADER_BYTES: usize = 44;

/// Writes the canonical 44-byte RIFF/WAVE 16-bit stereo PCM header.
#[must_use]
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

/// Encodes interleaved stereo `[-1.0, 1.0]` samples into a 44-byte RIFF `WAVE` PCM16 byte buffer.
#[must_use]
pub fn encode_wav_stereo_pcm16(sample_rate: u32, interleaved_stereo: &[f32]) -> Vec<u8> {
    let frame_count = interleaved_stereo.len() / 2;
    let sample_count = frame_count * 2;
    let header = write_wav_header(sample_rate, frame_count as u32);

    let mut out = Vec::with_capacity(WAV_HEADER_BYTES + sample_count * 2);
    out.extend_from_slice(&header);
    for &sample in &interleaved_stereo[..sample_count] {
        let clamped = if sample.is_finite() {
            sample.clamp(-LIMITER_CEILING_LINEAR, LIMITER_CEILING_LINEAR)
        } else {
            0.0
        };
        let quantized = (clamped * 32_767.0).round() as i16;
        out.extend_from_slice(&quantized.to_le_bytes());
    }
    out
}

/// Renders a sequence of [`SonificationFrame`] slices into a complete stereo PCM16 `.wav` file.
#[must_use]
pub fn synthesize_frames_to_wav(
    frames: &[SonificationFrame],
    sample_rate: u32,
    samples_per_step: usize,
    master_volume: f32,
) -> Vec<u8> {
    let step_len = samples_per_step.max(64);
    let total_steps = frames.len().max(1);
    let mut synth = OfflinePcmSynthesizer::new(sample_rate, step_len);
    synth.set_master_volume(master_volume);
    let _ = synth.set_muted_from_user_gesture(false);

    let mut all_samples = Vec::with_capacity(total_steps * step_len * 2);
    if frames.is_empty() {
        synth.apply_frame(&SonificationFrame::silent());
        all_samples.extend_from_slice(synth.render_frames(step_len));
    } else {
        for frame in frames {
            synth.apply_frame(frame);
            all_samples.extend_from_slice(synth.render_frames(step_len));
        }
    }
    encode_wav_stereo_pcm16(synth.sample_rate(), &all_samples)
}

/// Synthesizes a complete 16-bit stereo PCM `.wav` file from `frames` and [`AcousticSettings`]
/// via [`OfflinePcmSynthesizer`].
#[must_use]
pub fn synthesize_wav_16bit(
    frames: &[SonificationFrame],
    settings: AcousticSettings,
    sample_rate: u32,
    duration_secs: f32,
    _seed: u64,
) -> Vec<u8> {
    let sr = sample_rate.clamp(8_000, 96_000);
    let dur = if duration_secs.is_finite() {
        duration_secs.clamp(0.1, 10.0)
    } else {
        DEFAULT_WAV_DURATION_SECS
    };
    let total_samples = ((sr as f32 * dur).round() as usize).max(64);
    let mut effective_cfg = settings.sanitized();
    if effective_cfg.gain < 0.05 {
        effective_cfg.gain = 0.5;
    }

    let n_frames = frames.len().max(1);
    let max_step = total_samples.div_ceil(n_frames).max(64);
    let mut synth = OfflinePcmSynthesizer::new(sr, max_step);
    synth.set_master_volume(effective_cfg.gain);
    let _ = synth.set_muted_from_user_gesture(false);

    let mut all_samples = Vec::with_capacity(total_samples * 2);
    if frames.is_empty() {
        synth.apply_frame(&SonificationFrame::silent());
        let mut remaining = total_samples;
        while remaining > 0 {
            let chunk = remaining.min(max_step);
            all_samples.extend_from_slice(synth.render_frames(chunk));
            remaining -= chunk;
        }
        return encode_wav_stereo_pcm16(synth.sample_rate(), &all_samples);
    }

    for (idx, &frame) in frames.iter().enumerate() {
        let start_sample = (idx * total_samples) / n_frames;
        let end_sample = ((idx + 1) * total_samples) / n_frames;
        if end_sample > start_sample {
            synth.apply_frame(&frame.with_pitch_scale(effective_cfg.pitch_scale));
            all_samples.extend_from_slice(synth.render_frames(end_sample - start_sample));
        }
    }
    encode_wav_stereo_pcm16(synth.sample_rate(), &all_samples)
}
