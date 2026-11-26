#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use nalgebra::Vector3;
use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserializer, Serializer};
use std::fmt;

struct F64Def(f64);

impl serde::Serialize for F64Def {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.0.is_nan() {
            serializer.serialize_str("NaN")
        } else if self.0 == f64::INFINITY {
            serializer.serialize_str("Infinity")
        } else if self.0 == f64::NEG_INFINITY {
            serializer.serialize_str("-Infinity")
        } else {
            serializer.serialize_f64(self.0)
        }
    }
}

struct F64Visitor;

impl<'de> Visitor<'de> for F64Visitor {
    type Value = f64;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a float, null, or special float string (NaN, Infinity, -Infinity)")
    }

    fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(v)
    }

    fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(v as f64)
    }

    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(v as f64)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(f64::NAN)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(f64::NAN)
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        match s {
            "NaN" | "nan" => Ok(f64::NAN),
            "Infinity" | "+Infinity" | "inf" | "+inf" => Ok(f64::INFINITY),
            "-Infinity" | "-inf" => Ok(f64::NEG_INFINITY),
            other => other.parse::<f64>().map_err(de::Error::custom),
        }
    }
}

pub fn serialize<S>(vec: &Vector3<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(3))?;
    seq.serialize_element(&F64Def(vec.x))?;
    seq.serialize_element(&F64Def(vec.y))?;
    seq.serialize_element(&F64Def(vec.z))?;
    seq.end()
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<Vector3<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    struct Vec3Visitor;

    impl<'de> Visitor<'de> for Vec3Visitor {
        type Value = Vector3<f64>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a 3-element array of coordinates")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let x = seq
                .next_element_seed(F64Seed)?
                .ok_or_else(|| de::Error::invalid_length(0, &self))?;
            let y = seq
                .next_element_seed(F64Seed)?
                .ok_or_else(|| de::Error::invalid_length(1, &self))?;
            let z = seq
                .next_element_seed(F64Seed)?
                .ok_or_else(|| de::Error::invalid_length(2, &self))?;
            Ok(Vector3::new(x, y, z))
        }
    }

    struct F64Seed;

    impl<'de> de::DeserializeSeed<'de> for F64Seed {
        type Value = f64;

        fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserializer.deserialize_any(F64Visitor)
        }
    }

    deserializer.deserialize_seq(Vec3Visitor)
}
