//! Portable IEEE-754 directed-rounding arithmetic (`add`, `sub`, `mul`, `div`, `sqrt`)
//! using error-free transformations (`TwoSum` and Dekker-Veltkamp `TwoProd`) with
//! conservative `next_down` / `next_up` fallbacks near subnormal or overflow regimes.
// [cite:math_commons]

use crate::traits::RealScalar;

/// Directed rounding trait implemented for IEEE-754 `f32` and `f64`.
pub trait DirectedRounding: RealScalar {
    /// Greatest representable float `<= \pi`.
    const PI_LO: Self;
    /// Least representable float `>= \pi`.
    const PI_HI: Self;
    /// Greatest representable float `<= 2\pi`.
    const TAU_LO: Self;
    /// Least representable float `>= 2\pi`.
    const TAU_HI: Self;
    /// Tight outward-rounded radius enclosing `|\pi - Self::PI|`.
    const PI_RAD: Self;
    /// Tight outward-rounded radius enclosing `|2\pi - Self::TAU|`.
    const TAU_RAD: Self;

    /// Greatest representable float `<= self + rhs`.
    fn add_down(self, rhs: Self) -> Self;
    /// Least representable float `>= self + rhs`.
    fn add_up(self, rhs: Self) -> Self;
    /// Greatest representable float `<= self - rhs`.
    #[inline]
    fn sub_down(self, rhs: Self) -> Self {
        self.add_down(-rhs)
    }
    /// Least representable float `>= self - rhs`.
    #[inline]
    fn sub_up(self, rhs: Self) -> Self {
        self.add_up(-rhs)
    }
    /// Greatest representable float `<= self * rhs`.
    fn mul_down(self, rhs: Self) -> Self;
    /// Least representable float `>= self * rhs`.
    fn mul_up(self, rhs: Self) -> Self;
    /// Greatest representable float `<= self / rhs`.
    fn div_down(self, rhs: Self) -> Self;
    /// Least representable float `>= self / rhs`.
    fn div_up(self, rhs: Self) -> Self;
    /// Greatest representable float `<= sqrt(self)`.
    fn sqrt_down(self) -> Self;
    /// Least representable float `>= sqrt(self)`.
    fn sqrt_up(self) -> Self;
}

#[inline]
fn two_sum_f64(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let a_prime = s - b;
    let b_prime = s - a_prime;
    let err = (a - a_prime) + (b - b_prime);
    (s, err)
}

#[inline]
fn split_f64(x: f64) -> (f64, f64) {
    const SPLITTER: f64 = 134_217_729.0; // 2^27 + 1
    let c = SPLITTER * x;
    let hi = c - (c - x);
    let lo = x - hi;
    (hi, lo)
}

#[inline]
fn in_exact_prod_range_f64(a: f64, b: f64) -> bool {
    let aa = libm::fabs(a);
    let ab = libm::fabs(b);
    const LO: f64 = 1.0e-145;
    const HI: f64 = 1.0e145;
    (aa == 0.0 || (LO..=HI).contains(&aa)) && (ab == 0.0 || (LO..=HI).contains(&ab))
}

#[inline]
fn two_prod_f64(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (a_hi, a_lo) = split_f64(a);
    let (b_hi, b_lo) = split_f64(b);
    let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, err)
}

impl DirectedRounding for f64 {
    const PI_LO: Self = core::f64::consts::PI;
    const PI_HI: Self = f64::from_bits(core::f64::consts::PI.to_bits() + 1);
    const TAU_LO: Self = core::f64::consts::TAU;
    const TAU_HI: Self = f64::from_bits(core::f64::consts::TAU.to_bits() + 1);
    const PI_RAD: Self = f64::EPSILON;
    const TAU_RAD: Self = 2.0 * f64::EPSILON;

    #[inline]
    fn add_down(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return f64::NEG_INFINITY;
        }
        let (s, err) = two_sum_f64(self, rhs);
        if s.is_finite() && err.is_finite() {
            if err < 0.0 { s.next_down() } else { s }
        } else if s == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            s.pad_ulp_down(1)
        }
    }

    #[inline]
    fn add_up(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return f64::INFINITY;
        }
        let (s, err) = two_sum_f64(self, rhs);
        if s.is_finite() && err.is_finite() {
            if err > 0.0 { s.next_up() } else { s }
        } else if s == f64::INFINITY {
            f64::INFINITY
        } else {
            s.pad_ulp_up(1)
        }
    }

    #[inline]
    fn mul_down(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return f64::NEG_INFINITY;
        }
        if in_exact_prod_range_f64(self, rhs) {
            let (p, err) = two_prod_f64(self, rhs);
            if err < 0.0 { p.next_down() } else { p }
        } else {
            let p = self * rhs;
            if p == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                p.pad_ulp_down(1)
            }
        }
    }

    #[inline]
    fn mul_up(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return f64::INFINITY;
        }
        if in_exact_prod_range_f64(self, rhs) {
            let (p, err) = two_prod_f64(self, rhs);
            if err > 0.0 { p.next_up() } else { p }
        } else {
            let p = self * rhs;
            if p == f64::INFINITY {
                f64::INFINITY
            } else {
                p.pad_ulp_up(1)
            }
        }
    }

    #[inline]
    fn div_down(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() || rhs == 0.0 {
            return f64::NEG_INFINITY;
        }
        let q = self / rhs;
        if q == 0.0 && self == 0.0 {
            return 0.0;
        }
        if in_exact_prod_range_f64(q, rhs) && self.is_finite() {
            let (p, err) = two_prod_f64(q, rhs);
            let rem = (self - p) - err;
            let too_high = (rhs > 0.0 && rem < 0.0) || (rhs < 0.0 && rem > 0.0);
            if too_high { q.next_down() } else { q }
        } else if q == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            q.pad_ulp_down(1)
        }
    }

    #[inline]
    fn div_up(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() || rhs == 0.0 {
            return f64::INFINITY;
        }
        let q = self / rhs;
        if q == 0.0 && self == 0.0 {
            return 0.0;
        }
        if in_exact_prod_range_f64(q, rhs) && self.is_finite() {
            let (p, err) = two_prod_f64(q, rhs);
            let rem = (self - p) - err;
            let too_low = (rhs > 0.0 && rem > 0.0) || (rhs < 0.0 && rem < 0.0);
            if too_low { q.next_up() } else { q }
        } else if q == f64::INFINITY {
            f64::INFINITY
        } else {
            q.pad_ulp_up(1)
        }
    }

    #[inline]
    fn sqrt_down(self) -> Self {
        if self < 0.0 || self.is_nan() {
            return f64::NEG_INFINITY;
        }
        if self == 0.0 || self == f64::INFINITY {
            return self;
        }
        let s = libm::sqrt(self);
        if in_exact_prod_range_f64(s, s) {
            let (p, err) = two_prod_f64(s, s);
            let rem = (self - p) - err;
            if rem < 0.0 { s.next_down() } else { s }
        } else {
            s.pad_ulp_down(1)
        }
    }

    #[inline]
    fn sqrt_up(self) -> Self {
        if self < 0.0 || self.is_nan() {
            return f64::INFINITY;
        }
        if self == 0.0 || self == f64::INFINITY {
            return self;
        }
        let s = libm::sqrt(self);
        if in_exact_prod_range_f64(s, s) {
            let (p, err) = two_prod_f64(s, s);
            let rem = (self - p) - err;
            if rem > 0.0 { s.next_up() } else { s }
        } else {
            s.pad_ulp_up(1)
        }
    }
}

#[inline]
fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let a_prime = s - b;
    let b_prime = s - a_prime;
    let err = (a - a_prime) + (b - b_prime);
    (s, err)
}

impl DirectedRounding for f32 {
    const PI_LO: Self = f32::from_bits(core::f32::consts::PI.to_bits() - 1);
    const PI_HI: Self = core::f32::consts::PI;
    const TAU_LO: Self = f32::from_bits(core::f32::consts::TAU.to_bits() - 1);
    const TAU_HI: Self = core::f32::consts::TAU;
    const PI_RAD: Self = f32::EPSILON;
    const TAU_RAD: Self = 2.0 * f32::EPSILON;

    #[inline]
    fn add_down(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return f32::NEG_INFINITY;
        }
        let (s, err) = two_sum_f32(self, rhs);
        if s.is_finite() && err.is_finite() {
            if err < 0.0 { s.next_down() } else { s }
        } else if s == f32::NEG_INFINITY {
            f32::NEG_INFINITY
        } else {
            s.pad_ulp_down(1)
        }
    }

    #[inline]
    fn add_up(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return f32::INFINITY;
        }
        let (s, err) = two_sum_f32(self, rhs);
        if s.is_finite() && err.is_finite() {
            if err > 0.0 { s.next_up() } else { s }
        } else if s == f32::INFINITY {
            f32::INFINITY
        } else {
            s.pad_ulp_up(1)
        }
    }

    #[inline]
    fn mul_down(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return f32::NEG_INFINITY;
        }
        let exact = f64::from(self) * f64::from(rhs);
        if !exact.is_finite() {
            let p = self * rhs;
            return if p == f32::NEG_INFINITY {
                p
            } else {
                p.pad_ulp_down(1)
            };
        }
        let r = exact as f32;
        if f64::from(r) > exact {
            r.next_down()
        } else {
            r
        }
    }

    #[inline]
    fn mul_up(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            return f32::INFINITY;
        }
        let exact = f64::from(self) * f64::from(rhs);
        if !exact.is_finite() {
            let p = self * rhs;
            return if p == f32::INFINITY {
                p
            } else {
                p.pad_ulp_up(1)
            };
        }
        let r = exact as f32;
        if f64::from(r) < exact { r.next_up() } else { r }
    }

    #[inline]
    fn div_down(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() || rhs == 0.0 {
            return f32::NEG_INFINITY;
        }
        let exact = f64::from(self) / f64::from(rhs);
        if !exact.is_finite() {
            let q = self / rhs;
            return if q == f32::NEG_INFINITY {
                q
            } else {
                q.pad_ulp_down(1)
            };
        }
        let r = exact as f32;
        if f64::from(r) > exact {
            r.next_down()
        } else {
            r
        }
    }

    #[inline]
    fn div_up(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() || rhs == 0.0 {
            return f32::INFINITY;
        }
        let exact = f64::from(self) / f64::from(rhs);
        if !exact.is_finite() {
            let q = self / rhs;
            return if q == f32::INFINITY {
                q
            } else {
                q.pad_ulp_up(1)
            };
        }
        let r = exact as f32;
        if f64::from(r) < exact { r.next_up() } else { r }
    }

    #[inline]
    fn sqrt_down(self) -> Self {
        if self < 0.0 || self.is_nan() {
            return f32::NEG_INFINITY;
        }
        let exact = libm::sqrt(f64::from(self));
        if !exact.is_finite() {
            return self;
        }
        let r = exact as f32;
        if f64::from(r) > exact {
            r.next_down()
        } else {
            r
        }
    }

    #[inline]
    fn sqrt_up(self) -> Self {
        if self < 0.0 || self.is_nan() {
            return f32::INFINITY;
        }
        let exact = libm::sqrt(f64::from(self));
        if !exact.is_finite() {
            return self;
        }
        let r = exact as f32;
        if f64::from(r) < exact { r.next_up() } else { r }
    }
}
