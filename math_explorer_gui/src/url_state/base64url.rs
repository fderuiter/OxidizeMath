//! Pure-Rust RFC 4648 §5 URL-safe unpadded Base64 (`base64url`) encoder and strict decoder.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Encodes arbitrary bytes as unpadded URL-safe Base64 (RFC 4648 §5).
#[must_use]
pub fn encode(input: &[u8]) -> String {
    if input.is_empty() {
        return String::new();
    }
    let out_len = (input.len() * 4).div_ceil(3);
    let mut out = String::with_capacity(out_len);
    let (chunks, rem) = input.as_chunks::<3>();
    for &[b0, b1, b2] in chunks {
        out.push(ALPHABET[usize::from(b0 >> 2)] as char);
        out.push(ALPHABET[usize::from(((b0 & 0x03) << 4) | (b1 >> 4))] as char);
        out.push(ALPHABET[usize::from(((b1 & 0x0f) << 2) | (b2 >> 6))] as char);
        out.push(ALPHABET[usize::from(b2 & 0x3f)] as char);
    }
    match rem {
        [b0] => {
            out.push(ALPHABET[usize::from(b0 >> 2)] as char);
            out.push(ALPHABET[usize::from((b0 & 0x03) << 4)] as char);
        }
        [b0, b1] => {
            out.push(ALPHABET[usize::from(b0 >> 2)] as char);
            out.push(ALPHABET[usize::from(((b0 & 0x03) << 4) | (b1 >> 4))] as char);
            out.push(ALPHABET[usize::from((b1 & 0x0f) << 2)] as char);
        }
        _ => {}
    }
    out
}

#[inline]
fn decode_sextet(byte: u8) -> Result<u8, String> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'-' => Ok(62),
        b'_' => Ok(63),
        _ => Err(format!(
            "Invalid Base64URL character 0x{byte:02x} in shared state"
        )),
    }
}

fn strip_optional_padding(raw: &str) -> Result<&[u8], String> {
    let bytes = raw.as_bytes();
    let trimmed = bytes.strip_suffix(b"==").map_or_else(
        || bytes.strip_suffix(b"=").unwrap_or(bytes),
        |without_two| without_two,
    );
    let pad_count = bytes.len() - trimmed.len();
    if pad_count > 0 && !bytes.len().is_multiple_of(4) {
        return Err("Invalid Base64URL padding alignment".to_string());
    }
    if trimmed.contains(&b'=') {
        return Err("Unexpected '=' character inside Base64URL payload".to_string());
    }
    if trimmed.len() % 4 == 1 {
        return Err("Invalid Base64URL payload length (mod 4 == 1)".to_string());
    }
    Ok(trimmed)
}

/// Strictly decodes an RFC 4648 §5 `base64url` string while enforcing `max_decoded_bytes`
/// prior to heap allocation and rejecting non-canonical trailing bits.
pub fn decode(input: &str, max_decoded_bytes: usize) -> Result<Vec<u8>, String> {
    let bytes = strip_optional_padding(input)?;
    let full_quads = bytes.len() / 4;
    let rem = bytes.len() % 4;
    let extra_bytes = match rem {
        0 => 0,
        2 => 1,
        3 => 2,
        _ => unreachable!(),
    };
    let decoded_len = full_quads
        .checked_mul(3)
        .and_then(|n| n.checked_add(extra_bytes))
        .ok_or_else(|| "Base64URL length overflow".to_string())?;
    if decoded_len > max_decoded_bytes {
        return Err(format!(
            "Decoded URL state size ({decoded_len} bytes) exceeds maximum allowed limit ({max_decoded_bytes} bytes)"
        ));
    }

    let mut out = Vec::with_capacity(decoded_len);
    let (chunks, tail) = bytes.as_chunks::<4>();
    for &[c0, c1, c2, c3] in chunks {
        let v0 = decode_sextet(c0)?;
        let v1 = decode_sextet(c1)?;
        let v2 = decode_sextet(c2)?;
        let v3 = decode_sextet(c3)?;
        out.push((v0 << 2) | (v1 >> 4));
        out.push((v1 << 4) | (v2 >> 2));
        out.push((v2 << 6) | v3);
    }
    decode_remainder(tail, &mut out)?;
    Ok(out)
}

fn decode_remainder(rem: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    match rem {
        [] => Ok(()),
        [c0, c1] => {
            let v0 = decode_sextet(*c0)?;
            let v1 = decode_sextet(*c1)?;
            if (v1 & 0x0f) != 0 {
                return Err("Non-canonical trailing bits in Base64URL payload".to_string());
            }
            out.push((v0 << 2) | (v1 >> 4));
            Ok(())
        }
        [c0, c1, c2] => {
            let v0 = decode_sextet(*c0)?;
            let v1 = decode_sextet(*c1)?;
            let v2 = decode_sextet(*c2)?;
            if (v2 & 0x03) != 0 {
                return Err("Non-canonical trailing bits in Base64URL payload".to_string());
            }
            out.push((v0 << 2) | (v1 >> 4));
            out.push((v1 << 4) | (v2 >> 2));
            Ok(())
        }
        _ => Err("Invalid Base64URL remainder".to_string()),
    }
}
