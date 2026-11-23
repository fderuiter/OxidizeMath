//! Deterministic RIFF `WAVE` (`PCM16` and `IEEE Float32`, mono/stereo, `44.1`/`48 kHz`) and
//! synthesizer wavetable (`clm ` / `uhWT`) chunk encoding and inspection helpers.

// theory_verification!

use serde::{Deserialize, Serialize};

use super::backend::SonificationBackend;
use super::frame::{SonificationFrame, LIMITER_CEILING_LINEAR};
use super::synth::OfflinePcmSynthesizer;

/// Default sample rate (`44.1 kHz`) used when exporting standard sonification `.wav` clips.
pub const WAV_EXPORT_SAMPLE_RATE: u32 = 44_100;

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

/// Parsed metadata chunk descriptor returned by [`inspect_wav_riff`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiffChunkView {
    /// Four-character RIFF chunk tag.
    pub id: [u8; 4],
    /// Unpadded payload byte length declared in the chunk header.
    pub size: u32,
    /// Raw payload bytes (excluding trailing odd-length pad byte).
    pub payload: Vec<u8>,
}

/// Parsed RIFF `WAVE` summary used to verify exported headers and metadata chunks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiffWavSummary {
    /// Declared `RIFF` payload size (`file_len - 8`).
    pub riff_size: u32,
    /// Format tag (`1` = PCM, `3` = IEEE Float).
    pub format_tag: u16,
    /// Number of channels (`1` or `2`).
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Byte rate (`sample_rate * channels * bytes_per_sample`).
    pub byte_rate: u32,
    /// Block alignment in bytes (`channels * bytes_per_sample`).
    pub block_align: u16,
    /// Bits per sample (`16` or `32`).
    pub bits_per_sample: u16,
    /// Byte length of the `data` chunk payload.
    pub data_bytes: u32,
    /// Byte offset where the `data` payload starts in the `.wav` buffer.
    pub data_offset: usize,
    /// Additional chunks following `fmt ` / `data` (such as `clm ` and `uhWT`).
    pub extra_chunks: Vec<RiffChunkView>,
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

    let mut out =
        Vec::with_capacity(44 + sample_count * (bytes_per_sample as usize) + extra_chunks.len());
    write_riff_fmt_and_data_header(
        &mut out,
        riff_size,
        spec.format.format_tag(),
        ch,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        data_bytes,
    );
    write_sample_payload(&mut out, &samples[..sample_count], spec.format, ceiling);
    out.extend_from_slice(&extra_chunks);
    out
}

#[allow(clippy::too_many_arguments)]
fn write_riff_fmt_and_data_header(
    out: &mut Vec<u8>,
    riff_size: u32,
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    data_bytes: u32,
) {
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
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

fn build_wavetable_extra_chunks(spec: Option<WavetableChunkSpec>) -> Vec<u8> {
    let Some(wt) = spec else {
        return Vec::new();
    };
    let frame_samples = wt.frame_samples.max(64);
    let frame_count = wt.frame_count.clamp(1, 256);
    // Serum / Vital / Surge XT `clm ` chunk: `<!>2048 01000000 OxidizeMath` (28 bytes, even).
    let clm_text = format!("<!>{frame_samples:04} 01000000 OxidizeMath");
    let clm_bytes = clm_text.as_bytes();
    let mut chunks = Vec::with_capacity(8 + clm_bytes.len() + 1 + 20);
    append_riff_chunk(&mut chunks, *b"clm ", clm_bytes);

    // u-he / Surge XT `uhWT` chunk: [version = 1_u32, frame_count: u32, frame_samples: u32].
    let mut uhwt_payload = [0_u8; 12];
    uhwt_payload[0..4].copy_from_slice(&1_u32.to_le_bytes());
    uhwt_payload[4..8].copy_from_slice(&u32::from(frame_count).to_le_bytes());
    uhwt_payload[8..12].copy_from_slice(&u32::from(frame_samples).to_le_bytes());
    append_riff_chunk(&mut chunks, *b"uhWT", &uhwt_payload);
    chunks
}

fn append_riff_chunk(buf: &mut Vec<u8>, id: [u8; 4], payload: &[u8]) {
    buf.extend_from_slice(&id);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        buf.push(0);
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
    let spec = WavEncodeSpec {
        sample_rate_hz: sample_rate,
        channels: WavChannelLayout::Stereo,
        format: WavSampleFormat::Pcm16,
        peak_ceiling: LIMITER_CEILING_LINEAR,
        wavetable_chunks: None,
    };
    encode_wav_samples(&spec, interleaved_stereo)
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

/// Parses and validates a RIFF `WAVE` byte buffer, extracting `fmt `, `data`, and extra chunks.
pub fn inspect_wav_riff(bytes: &[u8]) -> Result<RiffWavSummary, String> {
    if bytes.len() < 44 {
        return Err(format!("WAV buffer too short: {} bytes", bytes.len()));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Missing RIFF/WAVE magic header".to_string());
    }
    let riff_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if usize::try_from(riff_size).unwrap_or(0) + 8 != bytes.len() {
        return Err(format!(
            "RIFF size mismatch: header {} + 8 != {}",
            riff_size,
            bytes.len()
        ));
    }
    parse_wave_subchunks(bytes, riff_size)
}

type FmtFields = (u16, u16, u32, u32, u16, u16);

fn parse_fmt_fields(payload: &[u8]) -> Result<FmtFields, String> {
    if payload.len() < 16 {
        return Err("fmt chunk shorter than 16 bytes".to_string());
    }
    Ok((
        u16::from_le_bytes([payload[0], payload[1]]),
        u16::from_le_bytes([payload[2], payload[3]]),
        u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]),
        u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]),
        u16::from_le_bytes([payload[12], payload[13]]),
        u16::from_le_bytes([payload[14], payload[15]]),
    ))
}

fn parse_wave_subchunks(bytes: &[u8], riff_size: u32) -> Result<RiffWavSummary, String> {
    let mut cursor = 12_usize;
    let mut fmt_fields: Option<FmtFields> = None;
    let mut data_info: Option<(u32, usize)> = None;
    let mut extra_chunks = Vec::new();

    while cursor + 8 <= bytes.len() {
        let id: [u8; 4] = bytes[cursor..cursor + 4].try_into().unwrap_or([0; 4]);
        let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap_or([0; 4]));
        let payload_start = cursor + 8;
        let payload_end = payload_start.saturating_add(size as usize);
        if payload_end > bytes.len() {
            return Err(format!("Chunk {:?} overruns WAV buffer", id));
        }
        let payload = &bytes[payload_start..payload_end];
        if &id == b"fmt " {
            fmt_fields = Some(parse_fmt_fields(payload)?);
        } else if &id == b"data" {
            data_info = Some((size, payload_start));
        } else {
            extra_chunks.push(RiffChunkView {
                id,
                size,
                payload: payload.to_vec(),
            });
        }
        cursor = payload_start + (size as usize) + ((size as usize) % 2);
    }

    let (format_tag, channels, sample_rate_hz, byte_rate, block_align, bits_per_sample) =
        fmt_fields.ok_or_else(|| "Missing fmt chunk".to_string())?;
    let (data_bytes, data_offset) = data_info.ok_or_else(|| "Missing data chunk".to_string())?;
    Ok(RiffWavSummary {
        riff_size,
        format_tag,
        channels,
        sample_rate_hz,
        byte_rate,
        block_align,
        bits_per_sample,
        data_bytes,
        data_offset,
        extra_chunks,
    })
}
