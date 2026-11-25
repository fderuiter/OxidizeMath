//! Optional `#![no_std]` `serde` implementations for forward-AD scalars and validated enclosures.
// [cite:math_commons]

use crate::ball::Ball;
use crate::directed::DirectedRounding;
use crate::dual::Dual;
use crate::dual_vec::DualVec;
use crate::enclosure_cert::{CertifiedBallRoot, CertifiedIntervalRoot, ReferenceEnclosureSummary};
use crate::hyper_dual::HyperDual;
use crate::interval::Interval;
use core::fmt;
use core::marker::PhantomData;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct DualRepr<T> {
    val: T,
    eps: T,
}

impl<T: Serialize> Serialize for Dual<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        DualRepr {
            val: &self.val,
            eps: &self.eps,
        }
        .serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Dual<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = DualRepr::<T>::deserialize(deserializer)?;
        Ok(Self {
            val: repr.val,
            eps: repr.eps,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct HyperDualRepr<T> {
    val: T,
    eps1: T,
    eps2: T,
    eps12: T,
}

impl<T: Serialize> Serialize for HyperDual<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        HyperDualRepr {
            val: &self.val,
            eps1: &self.eps1,
            eps2: &self.eps2,
            eps12: &self.eps12,
        }
        .serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for HyperDual<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = HyperDualRepr::<T>::deserialize(deserializer)?;
        Ok(Self {
            val: repr.val,
            eps1: repr.eps1,
            eps2: repr.eps2,
            eps12: repr.eps12,
        })
    }
}

struct FixedArray<T, const N: usize>([T; N]);

impl<T: Serialize, const N: usize> Serialize for FixedArray<&T, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().copied())
    }
}

struct FixedArrayVisitor<T, const N: usize>(PhantomData<T>);

impl<'de, T: Deserialize<'de> + Copy + Default, const N: usize> Visitor<'de>
    for FixedArrayVisitor<T, N>
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

impl<'de, T: Deserialize<'de> + Copy + Default, const N: usize> Deserialize<'de>
    for FixedArray<T, N>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(FixedArrayVisitor::<T, N>(PhantomData))
    }
}

impl<T: Serialize, const N: usize> Serialize for DualVec<T, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut st = serializer.serialize_struct("DualVec", 2)?;
        st.serialize_field("val", &self.val)?;
        let refs: [&T; N] = core::array::from_fn(|i| &self.eps[i]);
        st.serialize_field("eps", &FixedArray(refs))?;
        st.end()
    }
}

#[derive(Deserialize)]
struct DualVecOwned<T, const N: usize>
where
    T: Copy + Default,
{
    val: T,
    eps: FixedArray<T, N>,
}

impl<'de, T: Deserialize<'de> + Copy + Default, const N: usize> Deserialize<'de> for DualVec<T, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = DualVecOwned::<T, N>::deserialize(deserializer)?;
        Ok(Self {
            val: repr.val,
            eps: repr.eps.0,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct IntervalRepr<T> {
    lo: T,
    hi: T,
}

impl<T: Serialize> Serialize for Interval<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        IntervalRepr {
            lo: &self.lo,
            hi: &self.hi,
        }
        .serialize(serializer)
    }
}

impl<'de, T: DirectedRounding + Deserialize<'de>> Deserialize<'de> for Interval<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = IntervalRepr::<T>::deserialize(deserializer)?;
        if repr.lo.is_nan() || repr.hi.is_nan() || repr.lo > repr.hi {
            return Err(de::Error::custom(
                "Interval endpoints must be non-NaN with lo <= hi",
            ));
        }
        Ok(Self {
            lo: repr.lo,
            hi: repr.hi,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct BallRepr<T> {
    mid: T,
    rad: T,
}

impl<T: Serialize> Serialize for Ball<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        BallRepr {
            mid: &self.mid,
            rad: &self.rad,
        }
        .serialize(serializer)
    }
}

impl<'de, T: DirectedRounding + Deserialize<'de>> Deserialize<'de> for Ball<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = BallRepr::<T>::deserialize(deserializer)?;
        if repr.mid.is_nan() || repr.rad.is_nan() || repr.rad < T::ZERO {
            return Err(de::Error::custom(
                "Ball requires non-NaN midpoint and non-negative radius",
            ));
        }
        Ok(Self {
            mid: repr.mid,
            rad: repr.rad,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct CertRootRepr<E> {
    enclosure: E,
    certified_unique: bool,
    iterations: usize,
}

impl<T: Serialize> Serialize for CertifiedIntervalRoot<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CertRootRepr {
            enclosure: &self.enclosure,
            certified_unique: self.certified_unique,
            iterations: self.iterations,
        }
        .serialize(serializer)
    }
}

impl<'de, T: DirectedRounding + Deserialize<'de>> Deserialize<'de> for CertifiedIntervalRoot<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = CertRootRepr::<Interval<T>>::deserialize(deserializer)?;
        Ok(Self {
            enclosure: repr.enclosure,
            certified_unique: repr.certified_unique,
            iterations: repr.iterations,
        })
    }
}

impl<T: Serialize> Serialize for CertifiedBallRoot<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CertRootRepr {
            enclosure: &self.enclosure,
            certified_unique: self.certified_unique,
            iterations: self.iterations,
        }
        .serialize(serializer)
    }
}

impl<'de, T: DirectedRounding + Deserialize<'de>> Deserialize<'de> for CertifiedBallRoot<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = CertRootRepr::<Ball<T>>::deserialize(deserializer)?;
        Ok(Self {
            enclosure: repr.enclosure,
            certified_unique: repr.certified_unique,
            iterations: repr.iterations,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct SummaryRepr {
    max_interval_ulp_width: u64,
    max_ball_ulp_width: u64,
    all_contained: bool,
}

impl Serialize for ReferenceEnclosureSummary {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SummaryRepr {
            max_interval_ulp_width: self.max_interval_ulp_width,
            max_ball_ulp_width: self.max_ball_ulp_width,
            all_contained: self.all_contained,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ReferenceEnclosureSummary {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = SummaryRepr::deserialize(deserializer)?;
        Ok(Self {
            max_interval_ulp_width: repr.max_interval_ulp_width,
            max_ball_ulp_width: repr.max_ball_ulp_width,
            all_contained: repr.all_contained,
        })
    }
}
