#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::url_state::{decode_base64url, encode_base64url};

/// Maximum `base64url` string length accepted for an RLE raster mask (`<= URL_JSON_BOUNDS`).
pub const MAX_RLE_ENCODED_LEN: usize = 3_584;
/// Maximum raster width or height accepted by the RLE mask codec.
pub const MAX_RLE_GRID_DIM: usize = 512;
/// Maximum total decoded cells accepted by the RLE mask codec (`512 * 512`).
pub const MAX_RLE_CELLS: usize = MAX_RLE_GRID_DIM * MAX_RLE_GRID_DIM;

const RLE_MAGIC: [u8; 2] = *b"RM";
const RLE_VERSION: u8 = 1;
const HEADER_LEN: usize = 7;

fn push_uleb128(mut value: usize, out: &mut Vec<u8>) {
    for _ in 0..5 {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
            out.push(byte);
        } else {
            out.push(byte);
            break;
        }
    }
}

fn read_uleb128(bytes: &[u8], cursor: &mut usize) -> Result<usize, &'static str> {
    let mut result: usize = 0;
    let mut shift: u32 = 0;
    for byte_idx in 0..4 {
        if *cursor >= bytes.len() {
            return Err("Truncated RLE run-length varint");
        }
        let byte = bytes[*cursor];
        *cursor += 1;
        let low = (byte & 0x7F) as usize;
        if byte_idx == 3 && (byte & 0x80) != 0 {
            return Err("RLE run-length varint exceeds 28-bit bound");
        }
        result |= low << shift;
        if (byte & 0x80) == 0 {
            if result == 0 {
                return Err("Zero-length RLE run is invalid");
            }
            return Ok(result);
        }
        shift += 7;
    }
    Err("Malformed RLE varint")
}

fn validate_dimensions(width: usize, height: usize) -> Result<usize, String> {
    if width == 0 || width > MAX_RLE_GRID_DIM || height == 0 || height > MAX_RLE_GRID_DIM {
        return Err(format!(
            "RLE grid dimensions {width}x{height} outside 1..={MAX_RLE_GRID_DIM}"
        ));
    }
    let total = width
        .checked_mul(height)
        .filter(|&n| n <= MAX_RLE_CELLS)
        .ok_or_else(|| "RLE grid cell count overflow".to_string())?;
    Ok(total)
}

/// Encodes a flattened `width * height` grid of discrete symbols `0..=max_symbol` as `base64url` RLE.
pub fn encode_u8_grid(
    width: usize,
    height: usize,
    cells: &[u8],
    max_symbol: u8,
) -> Result<String, String> {
    let total = validate_dimensions(width, height)?;
    if cells.len() != total {
        return Err(format!(
            "RLE encode cell buffer length {} does not match {width}x{height} ({total})",
            cells.len()
        ));
    }
    let mut raw = Vec::with_capacity(32);
    raw.extend_from_slice(&RLE_MAGIC);
    raw.push(RLE_VERSION);
    raw.extend_from_slice(&(width as u16).to_le_bytes());
    raw.extend_from_slice(&(height as u16).to_le_bytes());

    let mut idx = 0;
    while idx < total {
        let sym = cells[idx];
        if sym > max_symbol {
            return Err(format!(
                "RLE symbol {sym} exceeds allowed maximum {max_symbol}"
            ));
        }
        let mut run_len = 1usize;
        while idx + run_len < total && cells[idx + run_len] == sym {
            run_len += 1;
        }
        raw.push(sym);
        push_uleb128(run_len, &mut raw);
        idx += run_len;
    }

    let encoded = encode_base64url(&raw);
    if encoded.len() > MAX_RLE_ENCODED_LEN {
        return Err(format!(
            "Encoded RLE mask length {} exceeds {MAX_RLE_ENCODED_LEN}-byte bound",
            encoded.len()
        ));
    }
    Ok(encoded)
}

pub(super) fn verify_rle_stream_without_grid_alloc(
    bytes: &[u8],
    expected_width: usize,
    expected_height: usize,
    expected_total: usize,
    max_symbol: u8,
) -> Result<(), &'static str> {
    if bytes.len() < HEADER_LEN {
        return Err("RLE payload shorter than 7-byte header");
    }
    if bytes[0..2] != RLE_MAGIC {
        return Err("Invalid RLE magic header");
    }
    if bytes[2] != RLE_VERSION {
        return Err("Unsupported RLE version");
    }
    let hdr_w = u16::from_le_bytes([bytes[3], bytes[4]]) as usize;
    let hdr_h = u16::from_le_bytes([bytes[5], bytes[6]]) as usize;
    if hdr_w != expected_width || hdr_h != expected_height {
        return Err("RLE header dimensions do not match expected grid dimensions");
    }

    let mut cursor = HEADER_LEN;
    let mut accumulated: usize = 0;
    let max_runs = bytes.len().saturating_sub(HEADER_LEN);
    for _ in 0..=max_runs {
        if cursor >= bytes.len() {
            break;
        }
        let sym = bytes[cursor];
        cursor += 1;
        if sym > max_symbol {
            return Err("RLE symbol exceeds allowed maximum");
        }
        let run_len = read_uleb128(bytes, &mut cursor)?;
        accumulated = accumulated
            .checked_add(run_len)
            .filter(|&sum| sum <= expected_total)
            .ok_or("RLE run sum exceeds expected grid cell count")?;
    }
    if cursor != bytes.len() {
        return Err("Unconsumed trailing bytes in RLE stream");
    }
    if accumulated != expected_total {
        return Err("RLE run sum does not equal expected grid cell count");
    }
    Ok(())
}

/// Decodes a `base64url` RLE mask after strictly validating header dimensions and run sums
/// *before* allocating the decoded grid vector.
pub fn decode_u8_grid(
    encoded: &str,
    expected_width: usize,
    expected_height: usize,
    max_symbol: u8,
) -> Result<Vec<u8>, String> {
    let expected_total = validate_dimensions(expected_width, expected_height)?;
    if encoded.is_empty() || encoded.len() > MAX_RLE_ENCODED_LEN {
        return Err(format!(
            "RLE string length {} outside 1..={MAX_RLE_ENCODED_LEN}",
            encoded.len()
        ));
    }
    let bytes = decode_base64url(encoded, MAX_RLE_ENCODED_LEN)
        .map_err(|e| format!("Invalid RLE base64url: {e}"))?;
    verify_rle_stream_without_grid_alloc(
        &bytes,
        expected_width,
        expected_height,
        expected_total,
        max_symbol,
    )
    .map_err(str::to_string)?;

    let mut out = Vec::with_capacity(expected_total);
    let mut cursor = HEADER_LEN;
    let max_runs = bytes.len().saturating_sub(HEADER_LEN);
    for _ in 0..=max_runs {
        if cursor >= bytes.len() {
            break;
        }
        let sym = bytes[cursor];
        cursor += 1;
        let run_len = read_uleb128(&bytes, &mut cursor).map_err(str::to_string)?;
        out.resize(out.len() + run_len, sym);
    }
    Ok(out)
}

/// Encodes a flattened boolean mask (`false` -> `0`, `true` -> `1`) as `base64url` RLE.
pub fn encode_bool_grid(width: usize, height: usize, cells: &[bool]) -> Result<String, String> {
    let total = validate_dimensions(width, height)?;
    if cells.len() != total {
        return Err(format!(
            "Boolean mask length {} does not match {width}x{height}",
            cells.len()
        ));
    }
    let bytes: Vec<u8> = cells.iter().map(|&b| u8::from(b)).collect();
    encode_u8_grid(width, height, &bytes, 1)
}

/// Decodes a `base64url` RLE boolean mask, validating dimensions and bounds prior to grid allocation.
pub fn decode_bool_grid(
    encoded: &str,
    expected_width: usize,
    expected_height: usize,
) -> Result<Vec<bool>, String> {
    let decoded = decode_u8_grid(encoded, expected_width, expected_height, 1)?;
    Ok(decoded.into_iter().map(|b| b != 0).collect())
}
