// [cite:math_scalars]
//! Zero-allocation `#![no_std]` `serde` implementations with domain-invariant validation.

use core::fmt;
use core::marker::PhantomData;

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::ball::Ball;
use crate::dual_vec::DualVec;
use crate::interval::Interval;
use crate::traits::RealScalar;

struct EpsSlice<'a, T>(&'a [T]);

impl<T: Serialize> Serialize for EpsSlice<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter())
    }
}

impl<T: Serialize, const N: usize> Serialize for DualVec<T, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut st = serializer.serialize_struct("DualVec", 2)?;
        st.serialize_field("val", &self.val)?;
        st.serialize_field("eps", &EpsSlice(&self.eps))?;
        st.end()
    }
}

struct FixedArray<T, const N: usize>([T; N]);

struct FixedArrayVisitor<T, const N: usize>(PhantomData<fn() -> [T; N]>);

impl<'de, T, const N: usize> Visitor<'de> for FixedArrayVisitor<T, N>
where
    T: Copy + Default + Deserialize<'de>,
{
    type Value = FixedArray<T, N>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a tangent array of exact length {N}")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut arr = [T::default(); N];
        for (idx, slot) in arr.iter_mut().enumerate() {
            *slot = seq
                .next_element()?
                .ok_or_else(|| de::Error::invalid_length(idx, &self))?;
        }
        if seq.next_element::<de::IgnoredAny>()?.is_some() {
            return Err(de::Error::invalid_length(N + 1, &self));
        }
        Ok(FixedArray(arr))
    }
}

impl<'de, T, const N: usize> Deserialize<'de> for FixedArray<T, N>
where
    T: Copy + Default + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(FixedArrayVisitor(PhantomData))
    }
}

#[derive(Deserialize)]
struct DualVecWire<T, const N: usize>
where
    T: Copy + Default,
{
    val: T,
    eps: FixedArray<T, N>,
}

impl<'de, T, const N: usize> Deserialize<'de> for DualVec<T, N>
where
    T: Copy + Default + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = DualVecWire::<T, N>::deserialize(deserializer)?;
        Ok(Self {
            val: wire.val,
            eps: wire.eps.0,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct IntervalWire<T> {
    lo: T,
    hi: T,
}

impl<T: Serialize> Serialize for Interval<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut st = serializer.serialize_struct("Interval", 2)?;
        st.serialize_field("lo", &self.lo)?;
        st.serialize_field("hi", &self.hi)?;
        st.end()
    }
}

impl<'de, T: RealScalar + Deserialize<'de>> Deserialize<'de> for Interval<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = IntervalWire::<T>::deserialize(deserializer)?;
        let both_nan = w.lo.is_nan() && w.hi.is_nan();
        let ordered = !w.lo.is_nan() && !w.hi.is_nan() && w.lo <= w.hi;
        if both_nan || ordered {
            Ok(Self { lo: w.lo, hi: w.hi })
        } else {
            Err(de::Error::custom(
                "invalid Interval bounds: expected lo <= hi or both NaN",
            ))
        }
    }
}

#[derive(Serialize, Deserialize)]
struct BallWire<T> {
    mid: T,
    rad: T,
}

impl<T: Serialize> Serialize for Ball<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut st = serializer.serialize_struct("Ball", 2)?;
        st.serialize_field("mid", &self.mid)?;
        st.serialize_field("rad", &self.rad)?;
        st.end()
    }
}

impl<'de, T: RealScalar + Deserialize<'de>> Deserialize<'de> for Ball<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let w = BallWire::<T>::deserialize(deserializer)?;
        let both_nan = w.mid.is_nan() && w.rad.is_nan();
        let valid_non_nan = !w.mid.is_nan()
            && !w.rad.is_nan()
            && w.rad >= T::ZERO
            && (w.mid.is_finite() || (w.mid == T::ZERO && w.rad == T::INFINITY));
        if both_nan || valid_non_nan {
            Ok(Self {
                mid: w.mid,
                rad: w.rad,
            })
        } else {
            Err(de::Error::custom(
                "invalid Ball: expected finite mid with rad >= 0, (0, +inf), or both NaN",
            ))
        }
    }
}
