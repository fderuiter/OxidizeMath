#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
// [cite:modular_polynomials_review]
//! Optional `#![no_std]` `serde` implementations preserving algebraic invariants across
//! [`UnitQuaternion`], [`Fp`], [`StackPoly`], [`FpExt`], [`Qp`], and `alloc`-gated [`BigFloat`].

#[cfg(feature = "alloc")]
use crate::bigint::{BigFloat, BinaryFloat};
use crate::fp::Fp;
use crate::fp_ext::{FpExt, IrreducibleModulus};
use crate::padic::Qp;
use crate::poly::{PolyField, StackPoly};
use crate::quaternion::{Quaternion, UnitQuaternion};
use crate::traits::RealScalar;
use core::fmt;
use core::marker::PhantomData;
use serde::ser::{SerializeSeq, SerializeTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<T: Serialize> Serialize for UnitQuaternion<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de, T: RealScalar + Deserialize<'de>> Deserialize<'de> for UnitQuaternion<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let q = Quaternion::<T>::deserialize(deserializer)?;
        let norm_sqr = q.norm_sqr();
        if !norm_sqr.is_finite() || (norm_sqr - T::ONE).abs() > T::from_f64(1e-3) {
            return Err(serde::de::Error::custom(
                "UnitQuaternion requires a finite quaternion with unit norm",
            ));
        }
        Self::try_from_quaternion(q).map_err(serde::de::Error::custom)
    }
}

impl<const P: u64> Serialize for Fp<P> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.value().serialize(serializer)
    }
}

impl<'de, const P: u64> Deserialize<'de> for Fp<P> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let () = Self::VALIDATE_PRIME;
        let raw = u64::deserialize(deserializer)?;
        if raw >= P {
            return Err(serde::de::Error::custom(
                "Fp canonical residue must be strictly less than modulus P",
            ));
        }
        Ok(Self::from_u64(raw))
    }
}

impl<T: Serialize, const CAP: usize> Serialize for StackPoly<T, CAP> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.len))?;
        for c in &self.coeffs[..self.len] {
            seq.serialize_element(c)?;
        }
        seq.end()
    }
}

impl<'de, T: PolyField + Deserialize<'de>, const CAP: usize> Deserialize<'de>
    for StackPoly<T, CAP>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PolyVisitor<U, const C: usize>(PhantomData<U>);

        impl<'de, U: PolyField + Deserialize<'de>, const C: usize> serde::de::Visitor<'de>
            for PolyVisitor<U, C>
        {
            type Value = StackPoly<U, C>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "at most {C} polynomial coefficients")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut buf = [U::ZERO; C];
                let mut count = 0_usize;
                while let Some(elem) = seq.next_element::<U>()? {
                    if count >= C {
                        return Err(serde::de::Error::custom(
                            "StackPoly coefficient sequence exceeds capacity CAP",
                        ));
                    }
                    buf[count] = elem;
                    count += 1;
                }
                StackPoly::from_slice(&buf[..count]).map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_seq(PolyVisitor::<T, CAP>(PhantomData))
    }
}

impl<const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Serialize for FpExt<P, DEG, M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(DEG)?;
        for c in &self.coeffs {
            tup.serialize_element(c)?;
        }
        tup.end()
    }
}

impl<'de, const P: u64, const DEG: usize, M: IrreducibleModulus<P, DEG>> Deserialize<'de>
    for FpExt<P, DEG, M>
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let () = Self::VALIDATE_EXTENSION;
        struct ExtVisitor<const Q: u64, const D_LEN: usize, Mod>(PhantomData<Mod>);

        impl<'de, const Q: u64, const D_LEN: usize, Mod: IrreducibleModulus<Q, D_LEN>>
            serde::de::Visitor<'de> for ExtVisitor<Q, D_LEN, Mod>
        {
            type Value = FpExt<Q, D_LEN, Mod>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a tuple of {D_LEN} Fp<{Q}> extension coefficients")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut coeffs = [Fp::<Q>::ZERO; D_LEN];
                for (idx, slot) in coeffs.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(idx, &self))?;
                }
                Ok(FpExt::from_coeffs(coeffs))
            }
        }

        deserializer.deserialize_tuple(DEG, ExtVisitor::<P, DEG, M>(PhantomData))
    }
}

struct ConstDigitsRef<'a, const N: usize>(&'a [u64; N]);

impl<const N: usize> Serialize for ConstDigitsRef<'_, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(N)?;
        for d in self.0 {
            tup.serialize_element(d)?;
        }
        tup.end()
    }
}

struct ConstDigits<const N: usize>([u64; N]);

impl<'de, const N: usize> Deserialize<'de> for ConstDigits<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DigitsVisitor<const M: usize>;

        impl<'de, const M: usize> serde::de::Visitor<'de> for DigitsVisitor<M> {
            type Value = [u64; M];

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a tuple of {M} p-adic digits")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = [0_u64; M];
                for (idx, slot) in out.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(idx, &self))?;
                }
                Ok(out)
            }
        }

        deserializer
            .deserialize_tuple(N, DigitsVisitor::<N>)
            .map(Self)
    }
}

#[derive(Serialize)]
struct QpSer<'a, const N: usize> {
    digits: ConstDigitsRef<'a, N>,
    valuation: i32,
    is_zero: bool,
}

#[derive(Deserialize)]
struct QpDe<const N: usize> {
    digits: ConstDigits<N>,
    valuation: i32,
    is_zero: bool,
}

impl<const P: u64, const N: usize> Serialize for Qp<P, N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        QpSer {
            digits: ConstDigitsRef(&self.digits),
            valuation: self.valuation,
            is_zero: self.is_zero,
        }
        .serialize(serializer)
    }
}

impl<'de, const P: u64, const N: usize> Deserialize<'de> for Qp<P, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let () = Self::VALIDATE_PADIC;
        let repr = QpDe::<N>::deserialize(deserializer)?;
        if repr.is_zero {
            if repr.valuation != 0 || repr.digits.0.iter().any(|&d| d != 0) {
                return Err(serde::de::Error::custom(
                    "canonical zero Qp must have valuation 0 and all-zero digits",
                ));
            }
            return Ok(Self::zero());
        }
        if repr.digits.0[0] == 0 || repr.digits.0.iter().any(|&d| d >= P) {
            return Err(serde::de::Error::custom(
                "canonical non-zero Qp requires digits[0] != 0 and all digits < P",
            ));
        }
        Ok(Self {
            digits: repr.digits.0,
            valuation: repr.valuation,
            is_zero: false,
        })
    }
}

#[cfg(feature = "alloc")]
#[derive(Serialize)]
struct BigFloatSer<'a> {
    prec: usize,
    value: &'a BinaryFloat,
}

#[cfg(feature = "alloc")]
#[derive(Deserialize)]
struct BigFloatDe {
    prec: usize,
    value: BinaryFloat,
}

#[cfg(feature = "alloc")]
impl Serialize for BigFloat {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        BigFloatSer {
            prec: self.precision(),
            value: &self.0,
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "alloc")]
impl<'de> Deserialize<'de> for BigFloat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = BigFloatDe::deserialize(deserializer)?;
        if repr.prec == 0 {
            return Err(serde::de::Error::custom(
                "BigFloat precision must be at least 1 bit",
            ));
        }
        Ok(Self(repr.value.with_precision(repr.prec).value()))
    }
}
