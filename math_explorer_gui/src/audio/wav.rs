//! Deterministic RIFF `WAVE` (`PCM16` stereo) encoding and export helpers.

// theory_verification!

use super::backend::SonificationBackend;
use super::frame::{SonificationFrame, LIMITER_CEILING_LINEAR};
use super::synth::OfflinePcmSynthesizer;

/// Default sample rate (`44.1 kHz`) used when exporting `.wav` clips.
pub const WAV_EXPORT_SAMPLE_RATE: u32 = 44_100;

/// Encodes interleaved stereo `[-1.0, 1.0]` samples into a 44-byte RIFF `WAVE` PCM16 byte buffer.
#[must_use]
pub fn encode_wav_stereo_pcm16(sample_rate: u32, interleaved_stereo: &[f32]) -> Vec<u8> {
    let channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let bytes_per_sample: u32 = 2;
    let frame_count = interleaved_stereo.len() / 2;
    let sample_count = frame_count * 2;
    let data_bytes = (sample_count as u32).saturating_mul(bytes_per_sample);
    let riff_size = 36_u32.saturating_add(data_bytes);
    let byte_rate = sample_rate
        .saturating_mul(u32::from(channels))
        .saturating_mul(bytes_per_sample);
    let block_align = channels.saturating_mul(bits_per_sample / 8);

    let mut out = Vec::with_capacity(44 + sample_count * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

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
