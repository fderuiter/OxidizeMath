//! Zero-allocation single-pass streaming JSON structural and finiteness validator for URL states.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::codec::URL_JSON_BOUNDS;
use crate::framework::ToolStateBounds;
use serde::de::{DeserializeSeed, Error as DeError, MapAccess, SeqAccess, Visitor};

const NON_FINITE_NUMBER_ERR: &str = "URL state contains non-finite or out-of-range numeric value";

struct IgnoredKeySeed;

impl<'de> DeserializeSeed<'de> for IgnoredKeySeed {
    type Value = ();

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        struct KeyStrVisitor;

        impl<'de> Visitor<'de> for KeyStrVisitor {
            type Value = ();

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object key string")
            }

            fn visit_borrowed_str<E: DeError>(self, _v: &'de str) -> Result<(), E> {
                Ok(())
            }

            fn visit_str<E: DeError>(self, _v: &str) -> Result<(), E> {
                Ok(())
            }
        }

        deserializer.deserialize_str(KeyStrVisitor)
    }
}

#[derive(Clone, Copy)]
struct StreamingJsonValidator {
    depth: usize,
    max_depth: usize,
    max_array_len: usize,
}

struct ArrayElementSeed {
    next_index: usize,
    inner: StreamingJsonValidator,
}

impl<'de> DeserializeSeed<'de> for ArrayElementSeed {
    type Value = ();

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        if self.next_index > self.inner.max_array_len {
            return Err(D::Error::custom(format!(
                "JSON array length ({}) exceeds maximum allowed limit ({})",
                self.next_index, self.inner.max_array_len
            )));
        }
        self.inner.deserialize(deserializer)
    }
}

impl<'de> DeserializeSeed<'de> for StreamingJsonValidator {
    type Value = ();

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl StreamingJsonValidator {
    fn descend<E: DeError>(self) -> Result<Self, E> {
        let next_depth = self.depth.saturating_add(1);
        if next_depth > self.max_depth {
            return Err(E::custom(format!(
                "JSON nesting depth ({next_depth}) exceeds maximum allowed limit ({})",
                self.max_depth
            )));
        }
        Ok(Self {
            depth: next_depth,
            ..self
        })
    }
}

impl<'de> Visitor<'de> for StreamingJsonValidator {
    type Value = ();

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a bounded JSON value with finite numbers")
    }

    fn visit_bool<E: DeError>(self, _v: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E: DeError>(self, _v: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E: DeError>(self, _v: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_i128<E: DeError>(self, v: i128) -> Result<(), E> {
        if i64::try_from(v).is_ok() || u64::try_from(v).is_ok() {
            Ok(())
        } else {
            Err(E::custom(NON_FINITE_NUMBER_ERR))
        }
    }

    fn visit_u128<E: DeError>(self, v: u128) -> Result<(), E> {
        u64::try_from(v)
            .map(|_| ())
            .map_err(|_| E::custom(NON_FINITE_NUMBER_ERR))
    }

    fn visit_f32<E: DeError>(self, v: f32) -> Result<(), E> {
        if v.is_finite() {
            Ok(())
        } else {
            Err(E::custom(NON_FINITE_NUMBER_ERR))
        }
    }

    fn visit_f64<E: DeError>(self, v: f64) -> Result<(), E> {
        if v.is_finite() {
            Ok(())
        } else {
            Err(E::custom(NON_FINITE_NUMBER_ERR))
        }
    }

    fn visit_borrowed_str<E: DeError>(self, _v: &'de str) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E: DeError>(self, _v: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E: DeError>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let child = self.descend()?;
        let mut len = 0_usize;
        while seq
            .next_element_seed(ArrayElementSeed {
                next_index: len.saturating_add(1),
                inner: child,
            })?
            .is_some()
        {
            len = len.saturating_add(1);
        }
        Ok(())
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
        let child = self.descend()?;
        while map.next_key_seed(IgnoredKeySeed)?.is_some() {
            map.next_value_seed(child)?;
        }
        Ok(())
    }
}

/// Validates that `json_str` fits within the 4,096-byte URL state cap, obeys structural nesting
/// and array bounds, and parses as JSON containing only finite numbers in a single streaming pass.
pub fn validate_url_preset_json(json_str: &str) -> Result<(), String> {
    validate_json_streaming_with_bounds(json_str, URL_JSON_BOUNDS)
}

pub(crate) fn validate_json_streaming_with_bounds(
    json_str: &str,
    bounds: ToolStateBounds,
) -> Result<(), String> {
    let cleaned = egui_plot::export::strip_utf8_bom(json_str);
    if cleaned.len() > bounds.max_bytes {
        return Err(format!(
            "JSON state size ({} bytes) exceeds maximum allowed limit ({} bytes)",
            cleaned.len(),
            bounds.max_bytes
        ));
    }
    let mut de = serde_json::Deserializer::from_str(cleaned);
    let validator = StreamingJsonValidator {
        depth: 0,
        max_depth: bounds.max_depth,
        max_array_len: bounds.max_array_len,
    };
    validator
        .deserialize(&mut de)
        .map_err(|e| format!("Invalid URL state JSON: {e}"))?;
    de.end().map_err(|e| format!("Invalid URL state JSON: {e}"))
}
