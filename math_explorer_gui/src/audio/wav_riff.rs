//! RIFF `WAVE` chunk inspection and synthesizer (`clm `, `uhWT`) metadata helpers.

use super::wav::WavetableChunkSpec;

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

pub(super) fn build_wavetable_extra_chunks(spec: Option<WavetableChunkSpec>) -> Vec<u8> {
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
