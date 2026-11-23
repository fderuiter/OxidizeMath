#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::TracePoints;
use crate::url_state::{decode_base64url, encode_base64url};
use serde::de::{Error as DeError, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

const COMPACT_PREFIX: &str = "tp1:";
const MIN_COMPACT_POINTS: usize = 8;
const MAX_COMPACT_POINTS: usize = 131_072;
const MAX_DECODED_TRACE_BYTES: usize = 3 * 1024 * 1024;

#[inline]
fn zigzag_encode(val: u64) -> u64 {
    (val << 1) ^ ((val as i64 >> 63) as u64)
}

#[inline]
fn zigzag_decode(zz: u64) -> u64 {
    (zz >> 1) ^ (zz & 1).wrapping_neg()
}

#[inline]
fn push_uleb128(out: &mut Vec<u8>, mut val: u64) {
    while val >= 0x80 {
        out.push((val as u8) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

#[inline]
fn read_uleb128(bytes: &[u8], cursor: &mut usize) -> Result<u64, &'static str> {
    let mut result = 0_u64;
    let mut shift = 0_u32;
    loop {
        let Some(&byte) = bytes.get(*cursor) else {
            return Err("truncated TracePoints varint");
        };
        *cursor += 1;
        if shift == 63 && (byte & 0xfe) != 0 {
            return Err("TracePoints varint overflow");
        }
        result |= u64::from(byte & 0x7f) << shift;
        if (byte & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

pub(super) fn encode_compact_points(points: &[[f64; 2]]) -> String {
    let mut raw = Vec::with_capacity(points.len().saturating_mul(4).saturating_add(8));
    push_uleb128(&mut raw, points.len() as u64);
    let mut prev_x = 0_u64;
    let mut prev_dx = 0_u64;
    let mut prev_y = 0_u64;
    let mut prev_dy = 0_u64;

    for &[x, y] in points {
        let bx = x.to_bits();
        let dx = bx.wrapping_sub(prev_x);
        let ddx = dx.wrapping_sub(prev_dx);
        push_uleb128(&mut raw, zigzag_encode(ddx));
        prev_x = bx;
        prev_dx = dx;

        let by = y.to_bits();
        let dy = by.wrapping_sub(prev_y);
        let ddy = dy.wrapping_sub(prev_dy);
        push_uleb128(&mut raw, zigzag_encode(ddy));
        prev_y = by;
        prev_dy = dy;
    }

    let b64 = encode_base64url(&raw);
    let mut out = String::with_capacity(COMPACT_PREFIX.len() + b64.len());
    out.push_str(COMPACT_PREFIX);
    out.push_str(&b64);
    out
}

pub(super) fn decode_compact_points(encoded: &str) -> Result<Vec<[f64; 2]>, String> {
    let payload = encoded
        .strip_prefix(COMPACT_PREFIX)
        .ok_or_else(|| "missing tp1: prefix in TracePoints string".to_string())?;
    let bytes = decode_base64url(payload, MAX_DECODED_TRACE_BYTES)?;
    let mut cursor = 0_usize;
    let count_u64 = read_uleb128(&bytes, &mut cursor).map_err(str::to_string)?;
    let count =
        usize::try_from(count_u64).map_err(|_| "TracePoints length overflow".to_string())?;
    if count > MAX_COMPACT_POINTS {
        return Err(format!(
            "TracePoints count ({count}) exceeds maximum ({MAX_COMPACT_POINTS})"
        ));
    }
    let remaining = bytes.len().saturating_sub(cursor);
    if remaining < count.saturating_mul(2) {
        return Err("truncated TracePoints coordinate stream".to_string());
    }

    let mut out = Vec::with_capacity(count);
    let mut prev_x = 0_u64;
    let mut prev_dx = 0_u64;
    let mut prev_y = 0_u64;
    let mut prev_dy = 0_u64;

    for _ in 0..count {
        let ddx = zigzag_decode(read_uleb128(&bytes, &mut cursor).map_err(str::to_string)?);
        let dx = prev_dx.wrapping_add(ddx);
        let bx = prev_x.wrapping_add(dx);
        prev_dx = dx;
        prev_x = bx;

        let ddy = zigzag_decode(read_uleb128(&bytes, &mut cursor).map_err(str::to_string)?);
        let dy = prev_dy.wrapping_add(ddy);
        let by = prev_y.wrapping_add(dy);
        prev_dy = dy;
        prev_y = by;

        out.push([f64::from_bits(bx), f64::from_bits(by)]);
    }
    if cursor != bytes.len() {
        return Err("trailing bytes in TracePoints payload".to_string());
    }
    Ok(out)
}

impl Serialize for TracePoints {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let slice = self.0.as_ref();
        if (MIN_COMPACT_POINTS..=MAX_COMPACT_POINTS).contains(&slice.len()) {
            let compact = encode_compact_points(slice);
            serializer.serialize_str(&compact)
        } else {
            slice.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for TracePoints {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TracePointsVisitor;

        impl<'de> Visitor<'de> for TracePointsVisitor {
            type Value = TracePoints;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an array of [x, y] pairs or a compact tp1: coordinate string")
            }

            fn visit_str<E: DeError>(self, v: &str) -> Result<Self::Value, E> {
                decode_compact_points(v)
                    .map(TracePoints::from)
                    .map_err(E::custom)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let cap = seq.size_hint().unwrap_or(0).min(MAX_COMPACT_POINTS);
                let mut pts = Vec::with_capacity(cap);
                while let Some(pair) = seq.next_element::<[f64; 2]>()? {
                    pts.push(pair);
                }
                Ok(TracePoints::from(pts))
            }
        }

        deserializer.deserialize_any(TracePointsVisitor)
    }
}
