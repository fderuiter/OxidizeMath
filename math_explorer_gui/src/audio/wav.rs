//! Deterministic RIFF `WAVE` (`PCM16` and `IEEE Float32`, mono/stereo, `44.1`/`48 kHz`) and
//! synthesizer wavetable (`clm ` / `uhWT`) chunk encoding helpers.

// theory_verification!

use serde::{Deserialize, Serialize};

use super::backend::SonificationBackend;
use super::frame::{SonificationFrame, LIMITER_CEILING_LINEAR};
use super::settings::AcousticSettings;
use super::synth::OfflinePcmSynthesizer;
use super::wav_riff::build_wavetable_extra_chunks;
pub use super::wav_riff::{inspect_wav_riff, RiffChunkView, RiffWavSummary};

/// Default sample rate (`44.1 kHz`) used when exporting standard sonification `.wav` clips.
pub const WAV_EXPORT_SAMPLE_RATE: u32 = 44_100;

/// Standard CD-quality PCM sample rate in `Hz` (`44,100 Hz`).
pub const DEFAULT_WAV_SAMPLE_RATE: u32 = WAV_EXPORT_SAMPLE_RATE;

/// Default exported `.wav` clip duration in seconds (`2.0 s`).
pub const DEFAULT_WAV_DURATION_SECS: f32 = 2.0;

/// Default number of simulation frames sampled per exported `.wav` clip (`120` frames @ 60 Hz = 2 s).
pub const DEFAULT_WAV_FRAME_COUNT: usize = 120;

/// Canonical 44-byte RIFF/WAVE PCM header byte length.
pub const WAV_HEADER_BYTES: usize = 44;

/// Supported studio `.wav` export sample rates (`44.1 kHz` and `48 kHz`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WavSampleRate {
    /// Standard `44,100 Hz` sample rate.
    Hz44100,
    /// Studio `48,000 Hz` sample rate.
    #[default]
    Hz48000,
}

impl WavSampleRate {
    /// All selectable studio sample rates in UI display order.
    pub const ALL: [Self; 2] = [Self::Hz44100, Self::Hz48000];

    /// Numeric sample rate in Hertz.
    #[must_use]
    pub const fn hz(self) -> u32 {
        match self {
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
        }
    }

    /// Concise human-readable label for UI combo boxes.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Hz44100 => "44.1 kHz",
            Self::Hz48000 => "48 kHz",
        }
    }
}

/// Supported RIFF `WAVE` sample encodings (`PCM16` `fmt = 1` and `IEEE Float32` `fmt = 3`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WavSampleFormat {
    /// 16-bit signed little-endian linear PCM (`WAVE_FORMAT_PCM = 1`).
    Pcm16,
    /// 32-bit little-endian IEEE-754 floating point (`WAVE_FORMAT_IEEE_FLOAT = 3`).
    #[default]
    Float32,
}

impl WavSampleFormat {
    /// All selectable `.wav` sample formats in UI display order.
    pub const ALL: [Self; 2] = [Self::Pcm16, Self::Float32];

    /// RIFF `fmt ` chunk format tag (`1` for PCM16, `3` for IEEE Float32).
    #[must_use]
    pub const fn format_tag(self) -> u16 {
        match self {
            Self::Pcm16 => 1,
            Self::Float32 => 3,
        }
    }

    /// Bits per individual channel sample.
    #[must_use]
    pub const fn bits_per_sample(self) -> u16 {
        match self {
            Self::Pcm16 => 16,
            Self::Float32 => 32,
        }
    }

    /// Bytes per individual channel sample.
    #[must_use]
    pub const fn bytes_per_sample(self) -> u32 {
        match self {
            Self::Pcm16 => 2,
            Self::Float32 => 4,
        }
    }

    /// Concise human-readable label for UI combo boxes.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pcm16 => "PCM 16-bit",
            Self::Float32 => "IEEE Float32",
        }
    }
}

/// Supported channel layouts (`Mono = 1` and `Stereo = 2`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WavChannelLayout {
    /// Single-channel mono layout (standard for single-oscillator wavetable banks).
    Mono,
    /// Two-channel interleaved `[L, R, ...]` layout (standard for stereo impulse responses).
    #[default]
    Stereo,
}

impl WavChannelLayout {
    /// All supported channel layouts.
    pub const ALL: [Self; 2] = [Self::Mono, Self::Stereo];

    /// Channel count (`1` or `2`).
    #[must_use]
    pub const fn channels(self) -> u16 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

/// Optional `clm ` and `uhWT` metadata chunks appended for Vital, Serum, and Surge XT wavetables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavetableChunkSpec {
    /// Samples per single-cycle wavetable frame (`2048` for Serum/Vital/Surge XT).
    pub frame_samples: u16,
    /// Number of consecutive frames stored in the `data` chunk (`1..=256`).
    pub frame_count: u16,
}

/// Complete encoding specification for RIFF `WAVE` output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavEncodeSpec {
    /// Sample rate in Hz (`44_100` or `48_000`, or custom positive rate).
    pub sample_rate_hz: u32,
    /// Mono (`1`) or interleaved Stereo (`2`).
    pub channels: WavChannelLayout,
    /// `PCM16` (`fmt = 1`) or `IEEE Float32` (`fmt = 3`).
    pub format: WavSampleFormat,
    /// Peak magnitude clamp applied to finite samples before quantization.
    pub peak_ceiling: f32,
    /// Optional wavetable `clm ` + `uhWT` chunk metadata appended after `data`.
    pub wavetable_chunks: Option<WavetableChunkSpec>,
}

struct RiffWaveHeader {
    riff_size: u32,
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    data_bytes: u32,
}

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

/// Encodes interleaved or mono samples according to a [`WavEncodeSpec`].
#[must_use]
pub fn encode_wav_samples(spec: &WavEncodeSpec, samples: &[f32]) -> Vec<u8> {
    let ch = spec.channels.channels();
    let ch_usize = usize::from(ch);
    let frame_count = samples.len() / ch_usize;
    let sample_count = frame_count * ch_usize;
    let bytes_per_sample = spec.format.bytes_per_sample();
    let bits_per_sample = spec.format.bits_per_sample();
    let sample_rate = spec.sample_rate_hz.max(1);
    let data_bytes = (sample_count as u32).saturating_mul(bytes_per_sample);
    let extra_chunks = build_wavetable_extra_chunks(spec.wavetable_chunks);
    let riff_size = 36_u32
        .saturating_add(data_bytes)
        .saturating_add(extra_chunks.len() as u32);
    let byte_rate = sample_rate
        .saturating_mul(u32::from(ch))
        .saturating_mul(bytes_per_sample);
    let block_align = ch.saturating_mul(bits_per_sample / 8);
    let ceiling = if spec.peak_ceiling.is_finite() && spec.peak_ceiling > 0.0 {
        spec.peak_ceiling.min(1.0)
    } else {
        LIMITER_CEILING_LINEAR
    };

    let mut out = Vec::with_capacity(
        WAV_HEADER_BYTES + sample_count * (bytes_per_sample as usize) + extra_chunks.len(),
    );
    write_riff_fmt_and_data_header(
        &mut out,
        &RiffWaveHeader {
            riff_size,
            format_tag: spec.format.format_tag(),
            channels: ch,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            data_bytes,
        },
    );
    write_sample_payload(&mut out, &samples[..sample_count], spec.format, ceiling);
    out.extend_from_slice(&extra_chunks);
    out
}

fn write_riff_fmt_and_data_header(out: &mut Vec<u8>, hdr: &RiffWaveHeader) {
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&hdr.riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&hdr.format_tag.to_le_bytes());
    out.extend_from_slice(&hdr.channels.to_le_bytes());
    out.extend_from_slice(&hdr.sample_rate.to_le_bytes());
    out.extend_from_slice(&hdr.byte_rate.to_le_bytes());
    out.extend_from_slice(&hdr.block_align.to_le_bytes());
    out.extend_from_slice(&hdr.bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&hdr.data_bytes.to_le_bytes());
}

fn write_sample_payload(out: &mut Vec<u8>, samples: &[f32], format: WavSampleFormat, ceiling: f32) {
    match format {
        WavSampleFormat::Pcm16 => {
            for &sample in samples {
                let clamped = if sample.is_finite() {
                    sample.clamp(-ceiling, ceiling)
                } else {
                    0.0
                };
                let quantized = (clamped * 32_767.0).round() as i16;
                out.extend_from_slice(&quantized.to_le_bytes());
            }
        }
        WavSampleFormat::Float32 => {
            for &sample in samples {
                let clamped = if sample.is_finite() {
                    sample.clamp(-ceiling, ceiling)
                } else {
                    0.0
                };
                out.extend_from_slice(&clamped.to_le_bytes());
            }
        }
    }
}

/// Encodes separate stereo left/right impulse-response channels into a studio `.wav` buffer.
#[must_use]
pub fn encode_stereo_ir_wav(
    sample_rate: WavSampleRate,
    format: WavSampleFormat,
    left: &[f32],
    right: &[f32],
) -> Vec<u8> {
    let frames = left.len().min(right.len());
    let mut interleaved = Vec::with_capacity(frames * 2);
    for idx in 0..frames {
        interleaved.push(left[idx]);
        interleaved.push(right[idx]);
    }
    let spec = WavEncodeSpec {
        sample_rate_hz: sample_rate.hz(),
        channels: WavChannelLayout::Stereo,
        format,
        peak_ceiling: LIMITER_CEILING_LINEAR,
        wavetable_chunks: None,
    };
    encode_wav_samples(&spec, &interleaved)
}

/// Encodes a mono `2048 * frame_count` wavetable bank with `clm ` and `uhWT` chunks.
#[must_use]
pub fn encode_wavetable_wav(
    sample_rate: WavSampleRate,
    format: WavSampleFormat,
    frame_samples: u16,
    frame_count: u16,
    mono_samples: &[f32],
) -> Vec<u8> {
    let expected = usize::from(frame_samples) * usize::from(frame_count);
    let slice = &mono_samples[..mono_samples.len().min(expected)];
    let spec = WavEncodeSpec {
        sample_rate_hz: sample_rate.hz(),
        channels: WavChannelLayout::Mono,
        format,
        peak_ceiling: 1.0,
        wavetable_chunks: Some(WavetableChunkSpec {
            frame_samples,
            frame_count,
        }),
    };
    encode_wav_samples(&spec, slice)
}

/// Encodes interleaved stereo `[-1.0, 1.0]` samples into a 44-byte RIFF `WAVE` PCM16 byte buffer.
#[must_use]
pub fn encode_wav_stereo_pcm16(sample_rate: u32, interleaved_stereo: &[f32]) -> Vec<u8> {
    let frame_count = interleaved_stereo.len() / 2;
    let sample_count = frame_count * 2;
    let header = write_wav_header(sample_rate.max(1), frame_count as u32);

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
