// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
use super::StreamingQr;
use crate::scalar::Scalar;
use crate::types::check_all_finite;
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

struct ConstVecRef<'a, T, const N: usize>(&'a [T; N]);

impl<T: Serialize, const N: usize> Serialize for ConstVecRef<'_, T, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(N)?;
        for elem in self.0 {
            tup.serialize_element(elem)?;
        }
        tup.end()
    }
}

struct ConstMatRef<'a, T, const N: usize>(&'a [[T; N]; N]);

impl<T: Serialize, const N: usize> Serialize for ConstMatRef<'_, T, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(N)?;
        for row in self.0 {
            tup.serialize_element(&ConstVecRef(row))?;
        }
        tup.end()
    }
}

struct ConstVec<T, const N: usize>([T; N]);

impl<'de, T: Scalar + Deserialize<'de>, const N: usize> Deserialize<'de> for ConstVec<T, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct VecVisitor<U, const M: usize>(core::marker::PhantomData<U>);

        impl<'de, U: Scalar + Deserialize<'de>, const M: usize> serde::de::Visitor<'de>
            for VecVisitor<U, M>
        {
            type Value = [U; M];

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(formatter, "an array of {M} finite scalar values")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = [U::ZERO; M];
                for (idx, slot) in out.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(idx, &self))?;
                }
                Ok(out)
            }
        }

        deserializer
            .deserialize_tuple(N, VecVisitor::<T, N>(core::marker::PhantomData))
            .map(Self)
    }
}

struct ConstMat<T, const N: usize>([[T; N]; N]);

impl<'de, T: Scalar + Deserialize<'de>, const N: usize> Deserialize<'de> for ConstMat<T, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MatVisitor<U, const M: usize>(core::marker::PhantomData<U>);

        impl<'de, U: Scalar + Deserialize<'de>, const M: usize> serde::de::Visitor<'de>
            for MatVisitor<U, M>
        {
            type Value = [[U; M]; M];

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(formatter, "a {M}x{M} matrix of finite scalar values")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = [[U::ZERO; M]; M];
                for (idx, row) in out.iter_mut().enumerate() {
                    let parsed: ConstVec<U, M> = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(idx, &self))?;
                    *row = parsed.0;
                }
                Ok(out)
            }
        }

        deserializer
            .deserialize_tuple(N, MatVisitor::<T, N>(core::marker::PhantomData))
            .map(Self)
    }
}

#[derive(Serialize)]
struct StreamingQrSer<'a, T: Serialize, const MAX_P: usize> {
    r: ConstMatRef<'a, T, MAX_P>,
    qty: ConstVecRef<'a, T, MAX_P>,
    ncols: usize,
    nrows: usize,
}

#[derive(Deserialize)]
struct StreamingQrDe<T: Scalar, const MAX_P: usize> {
    #[serde(bound(deserialize = "T: Scalar + Deserialize<'de>"))]
    r: ConstMat<T, MAX_P>,
    #[serde(bound(deserialize = "T: Scalar + Deserialize<'de>"))]
    qty: ConstVec<T, MAX_P>,
    ncols: usize,
    nrows: usize,
}

impl<T: Scalar + Serialize, const MAX_P: usize> Serialize for StreamingQr<T, MAX_P> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        StreamingQrSer {
            r: ConstMatRef(&self.r),
            qty: ConstVecRef(&self.qty),
            ncols: self.ncols,
            nrows: self.nrows,
        }
        .serialize(serializer)
    }
}

impl<'de, T: Scalar + Deserialize<'de>, const MAX_P: usize> Deserialize<'de>
    for StreamingQr<T, MAX_P>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = StreamingQrDe::<T, MAX_P>::deserialize(deserializer)?;
        if repr.ncols == 0 || repr.ncols > MAX_P {
            return Err(serde::de::Error::custom(
                "StreamingQr ncols must lie in 1..=MAX_P",
            ));
        }
        for row in &repr.r.0 {
            if check_all_finite(row).is_err() {
                return Err(serde::de::Error::custom(
                    "StreamingQr matrix r contains non-finite scalar",
                ));
            }
        }
        if check_all_finite(&repr.qty.0).is_err() {
            return Err(serde::de::Error::custom(
                "StreamingQr rhs vector qty contains non-finite scalar",
            ));
        }
        Ok(Self {
            r: repr.r.0,
            qty: repr.qty.0,
            ncols: repr.ncols,
            nrows: repr.nrows,
        })
    }
}
