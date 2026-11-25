// [cite:pure_math]
use super::{Float, Integer, Rational};
use dashu::float::FBig;
use dashu::float::round::mode::Zero;
use dashu::integer::IBig;
use dashu::rational::RBig;
use math_scalars::{Field, Ring, Scalar};
use std::ops::{AddAssign, DivAssign, MulAssign, Neg, SubAssign};

impl Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self {
        Integer(-self.0)
    }
}

impl Scalar for Integer {}

impl Ring for Integer {
    fn zero() -> Self {
        Integer(IBig::from(0u8))
    }

    fn one() -> Self {
        Integer(IBig::from(1u8))
    }

    fn is_zero(&self) -> bool {
        self.0 == IBig::from(0u8)
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self {
        Rational(-self.0)
    }
}

impl AddAssign for Rational {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Rational {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for Rational {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for Rational {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Scalar for Rational {}

impl Ring for Rational {
    fn zero() -> Self {
        Rational(RBig::from(0u8))
    }

    fn one() -> Self {
        Rational(RBig::from(1u8))
    }

    fn is_zero(&self) -> bool {
        <RBig as Ring>::is_zero(&self.0)
    }
}

impl Field for Rational {
    fn checked_inverse(&self) -> Option<Self> {
        <RBig as Field>::checked_inverse(&self.0).map(Rational)
    }

    fn multiplicative_inverse(&self) -> Self {
        Rational(<RBig as Field>::multiplicative_inverse(&self.0))
    }
}

impl Neg for Float {
    type Output = Self;

    fn neg(self) -> Self {
        Float(-self.0)
    }
}

impl MulAssign for Float {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for Float {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Scalar for Float {}

impl Ring for Float {
    fn zero() -> Self {
        Float(FBig::from(0u8))
    }

    fn one() -> Self {
        Float(FBig::from(1u8))
    }

    fn is_zero(&self) -> bool {
        self.0 == FBig::<Zero>::ZERO
    }
}

impl Field for Float {
    fn checked_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            let prec = self.0.precision().max(53);
            let one = FBig::from(1u8).with_precision(prec).value();
            Some(Float(one / self.0.clone()))
        }
    }

    fn multiplicative_inverse(&self) -> Self {
        let prec = self.0.precision().max(53);
        let one = FBig::from(1u8).with_precision(prec).value();
        Float(one / self.0.clone())
    }
}
