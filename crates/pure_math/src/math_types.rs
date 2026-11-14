use dashu::base::Abs;
use dashu::float::FBig;
use dashu::integer::IBig;
use dashu::rational::RBig;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, RemAssign, Sub, SubAssign};
#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
#[allow(missing_docs)]
pub struct Integer(pub IBig);
impl From<Integer> for IBig {
    fn from(i: Integer) -> Self {
        i.0
    }
}
impl Default for Integer {
    fn default() -> Self {
        Self::new()
    }
}
impl Integer {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Integer(IBig::from(0u8))
    }
    #[allow(missing_docs)]
    pub fn from<T>(v: T) -> Self
    where
        IBig: From<T>,
    {
        Integer(IBig::from(v))
    }
    #[allow(missing_docs)]
    pub fn to_f64(&self) -> f64 {
        use dashu::float::round::mode::HalfAway;
        let f: FBig<HalfAway, 2> = FBig::from(self.0.clone());
        f.to_f64().value()
    }
    #[allow(missing_docs)]
    pub fn to_i32(&self) -> Option<i32> {
        i32::try_from(&self.0).ok()
    }
    #[allow(missing_docs)]
    pub fn to_usize(&self) -> Option<usize> {
        usize::try_from(&self.0).ok()
    }
    #[allow(missing_docs)]
    pub fn to_u64(&self) -> Option<u64> {
        u64::try_from(&self.0).ok()
    }
    #[allow(missing_docs)]
    pub fn assign(&mut self, v: i128) {
        self.0 = IBig::from(v);
    }
    #[allow(clippy::result_unit_err)]
    #[allow(missing_docs)]
    pub fn parse(s: &str) -> Result<Self, ()> {
        IBig::from_str_radix(s, 10).map(Integer).map_err(|_| ())
    }
    #[allow(clippy::result_unit_err)]
    #[allow(missing_docs)]
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, ()> {
        IBig::from_str_radix(s, radix).map(Integer).map_err(|_| ())
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn is_even(&self) -> bool {
        &self.0 % IBig::from(2u8) == IBig::from(0u8)
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn is_divisible(&self, other: &Integer) -> bool {
        (&self.0 % &other.0) == IBig::from(0u8)
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn sqrt(&self) -> Self {
        Integer(self.0.nth_root(2))
    }
    #[allow(clippy::result_unit_err)]
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn pow_mod(&self, exp: &Integer, m: &Integer) -> Result<Self, ()> {
        if m.0 == IBig::from(0u8) {
            return Err(());
        }
        let mut base = &self.0 % &m.0;
        let mut exp_val = exp.0.clone();
        if exp_val < IBig::from(0u8) {
            if let Ok(inv) = self.invert(m) {
                base = inv.0;
                exp_val = -exp_val;
            } else {
                return Err(());
            }
        }
        let mut result = IBig::from(1u8);
        while exp_val > IBig::from(0u8) {
            if &exp_val % IBig::from(2u8) == IBig::from(1u8) {
                result = (result * &base) % &m.0;
            }
            exp_val /= IBig::from(2u8);
            base = (&base * &base) % &m.0;
        }
        Ok(Integer(result))
    }
    #[allow(clippy::result_unit_err)]
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn invert(&self, m: &Integer) -> Result<Self, ()> {
        let (gcd, x, _y) = self.extended_gcd(m);
        if gcd.0 == IBig::from(1u8) {
            let mut r = x.0 % &m.0;
            if r < IBig::from(0u8) {
                r += m.0.clone().abs();
            }
            Ok(Integer(r))
        } else {
            Err(())
        }
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn abs(self) -> Self {
        Integer(self.0.abs())
    }
    #[verified_engine::verified]
    fn extended_gcd(&self, other: &Integer) -> (Integer, Integer, Integer) {
        let mut old_r = self.0.clone();
        let mut r = other.0.clone();
        let mut old_s = IBig::from(1u8);
        let mut s = IBig::from(0u8);
        let mut old_t = IBig::from(0u8);
        let mut t = IBig::from(1u8);
        while r != IBig::from(0u8) {
            let quotient = &old_r / &r;
            let temp = r.clone();
            r = &old_r - &quotient * &r;
            old_r = temp;
            let temp = s.clone();
            s = &old_s - &quotient * &s;
            old_s = temp;
            let temp = t.clone();
            t = &old_t - &quotient * &t;
            old_t = temp;
        }
        (Integer(old_r), Integer(old_s), Integer(old_t))
    }
}

impl PartialEq<i32> for Integer {
    fn eq(&self, other: &i32) -> bool {
        self.0 == IBig::from(*other)
    }
}
impl PartialOrd<i32> for Integer {
    fn partial_cmp(&self, other: &i32) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&IBig::from(*other))
    }
}

math_commons::impl_binop!(Integer, Add, add);
math_commons::impl_binop!(Integer, Sub, sub);
math_commons::impl_binop!(Integer, Mul, mul);
math_commons::impl_binop!(Integer, Div, div);
math_commons::impl_binop!(Integer, Rem, rem);

impl AddAssign for Integer {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl SubAssign for Integer {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl MulAssign for Integer {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}
impl DivAssign for Integer {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}
impl RemAssign for Integer {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}
impl MulAssign<&Integer> for Integer {
    fn mul_assign(&mut self, rhs: &Self) {
        self.0 *= &rhs.0;
    }
}
impl AddAssign<&Integer> for Integer {
    fn add_assign(&mut self, rhs: &Self) {
        self.0 += &rhs.0;
    }
}
impl SubAssign<&Integer> for Integer {
    fn sub_assign(&mut self, rhs: &Self) {
        self.0 -= &rhs.0;
    }
}

type F = dashu::float::FBig;

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct Float(pub F);
impl Float {
    #[allow(missing_docs)]
    pub fn with_val(_prec: u32, v: f64) -> Self {
        if let Ok(f) = F::try_from(v) {
            Float(f)
        } else {
            Float(F::from(0u8))
        }
    }
    #[allow(missing_docs)]
    pub fn abs(self) -> Self {
        Float(self.0.abs())
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn round(self) -> f64 {
        use dashu::float::round::mode::HalfAway;
        let f: FBig<HalfAway, 2> = self.0.clone().with_rounding();
        f.to_f64().value().round()
    }
}
impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Add for Float {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Float(self.0 + rhs.0)
    }
}
impl Sub for Float {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Float(self.0 - rhs.0)
    }
}
impl Mul for Float {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Float(self.0 * rhs.0)
    }
}
impl Div for Float {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Float(self.0 / rhs.0)
    }
}
impl<'a> Add<&'a Float> for Float {
    type Output = Float;
    fn add(self, rhs: &'a Float) -> Float {
        Float(self.0 + rhs.0.clone())
    }
}
impl<'a> Sub<&'a Float> for Float {
    type Output = Float;
    fn sub(self, rhs: &'a Float) -> Float {
        Float(self.0 - rhs.0.clone())
    }
}
impl<'a> Mul<&'a Float> for Float {
    type Output = Float;
    fn mul(self, rhs: &'a Float) -> Float {
        Float(self.0 * rhs.0.clone())
    }
}
impl<'b> Add<&'b Float> for &Float {
    type Output = Float;
    fn add(self, rhs: &'b Float) -> Float {
        Float(self.0.clone() + rhs.0.clone())
    }
}
impl<'b> Sub<&'b Float> for &Float {
    type Output = Float;
    fn sub(self, rhs: &'b Float) -> Float {
        Float(self.0.clone() - rhs.0.clone())
    }
}
impl<'b> Mul<&'b Float> for &Float {
    type Output = Float;
    fn mul(self, rhs: &'b Float) -> Float {
        Float(self.0.clone() * rhs.0.clone())
    }
}
impl<'b> Div<&'b Float> for &Float {
    type Output = Float;
    fn div(self, rhs: &'b Float) -> Float {
        Float(self.0.clone() / rhs.0.clone())
    }
}
impl AddAssign for Float {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl SubAssign for Float {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl PartialEq<f64> for Float {
    fn eq(&self, other: &f64) -> bool {
        if let Ok(f) = F::try_from(*other) {
            self.0 == f
        } else {
            false
        }
    }
}
impl PartialOrd<f64> for Float {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        if let Ok(f) = F::try_from(*other) {
            self.0.partial_cmp(&f)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct Rational(pub RBig);
impl Default for Rational {
    fn default() -> Self {
        Self::new()
    }
}

impl Rational {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Rational(RBig::from(0u8))
    }
    #[allow(missing_docs)]
    pub fn from(v: (i128, i128)) -> Self {
        let n = dashu::integer::IBig::from(v.0);
        let d = dashu::integer::IBig::from(v.1);
        let (sign_d, mag_d) = d.into_parts();
        let mut real_n = n;
        if sign_d == dashu::base::Sign::Negative {
            real_n = -real_n;
        }
        Rational(RBig::from_parts(real_n, mag_d))
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn numer(&self) -> f64 {
        use dashu::float::round::mode::HalfAway;
        let n_f: FBig<HalfAway, 2> = FBig::from(self.0.numerator().clone());
        n_f.to_f64().value()
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn denom(&self) -> f64 {
        use dashu::float::round::mode::HalfAway;
        let d_f: FBig<HalfAway, 2> = FBig::from(self.0.denominator().clone());
        d_f.to_f64().value()
    }
}
impl Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Rational(self.0 + rhs.0)
    }
}
impl Sub for Rational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Rational(self.0 - rhs.0)
    }
}
impl Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Rational(self.0 * rhs.0)
    }
}
impl Div for Rational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Rational(self.0 / rhs.0)
    }
}
impl<'b> Mul<&'b Rational> for &Rational {
    type Output = Rational;
    fn mul(self, rhs: &'b Rational) -> Rational {
        Rational(self.0.clone() * rhs.0.clone())
    }
}
impl<'b> Add<&'b Rational> for &Rational {
    type Output = Rational;
    fn add(self, rhs: &'b Rational) -> Rational {
        Rational(self.0.clone() + rhs.0.clone())
    }
}
impl Mul<Rational> for &Rational {
    type Output = Rational;
    fn mul(self, rhs: Rational) -> Rational {
        Rational(self.0.clone() * rhs.0)
    }
}
impl Add<Rational> for &Rational {
    type Output = Rational;
    fn add(self, rhs: Rational) -> Rational {
        Rational(self.0.clone() + rhs.0)
    }
}

#[allow(missing_docs)]
pub mod ops {
    #[allow(missing_docs)]
    pub trait Pow<Rhs> {
        #[allow(missing_docs)]
        type Output;
        #[allow(missing_docs)]
        fn pow(self, rhs: Rhs) -> Self::Output;
    }
    impl Pow<u32> for super::Integer {
        type Output = super::Integer;
        fn pow(self, rhs: u32) -> Self::Output {
            super::Integer(self.0.pow(rhs as usize))
        }
    }
    #[allow(missing_docs)]
    pub trait RemRounding<Rhs> {
        #[allow(missing_docs)]
        type Output;
        #[allow(missing_docs)]
        fn rem_euc(self, rhs: Rhs) -> Self::Output;
    }
    impl RemRounding<super::Integer> for super::Integer {
        type Output = super::Integer;
        fn rem_euc(self, rhs: super::Integer) -> Self::Output {
            let mut r = &self.0 % &rhs.0;
            if r < dashu::integer::IBig::from(0u8) {
                use dashu::base::Abs;
                r += rhs.0.abs();
            }
            super::Integer(r)
        }
    }
    impl RemRounding<&super::Integer> for super::Integer {
        type Output = super::Integer;
        fn rem_euc(self, rhs: &super::Integer) -> Self::Output {
            let mut r = &self.0 % &rhs.0;
            if r < dashu::integer::IBig::from(0u8) {
                use dashu::base::Abs;
                r += rhs.0.clone().abs();
            }
            super::Integer(r)
        }
    }
}
