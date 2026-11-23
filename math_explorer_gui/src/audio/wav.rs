//! Deterministic RIFF `WAVE` (`PCM16` stereo) encoding and export helpers.

// theory_verification!

use super::backend::SonificationBackend;
use super::frame::{SonificationFrame, LIMITER_CEILING_LINEAR};
use super::settings::AcousticSettings;
use super::synth::OfflinePcmSynthesizer;

/// Default sample rate (`44.1 kHz`) used when exporting `.wav` clips.
pub const WAV_EXPORT_SAMPLE_RATE: u32 = 44_100;

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
