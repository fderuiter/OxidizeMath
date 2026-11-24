// [cite:math_scalars]
//! Elementary `libm` transcendental enclosures and [`Scalar`] implementation for [`Interval`].

use crate::interval::Interval;
use crate::traits::{RealScalar, Scalar, powi_generic};

/// Conservative per-side ULP padding (+/-2 ULPs, <= 4 ULPs span) for deterministic `libm` evaluations.
const LIBM_PAD_ULPS: u32 = 2;

#[inline]
fn contains_integer<T: RealScalar>(iv: Interval<T>) -> bool {
    !iv.is_nan() && iv.lo.ceil() <= iv.hi.floor()
}

#[inline]
fn pad_monotone<T: RealScalar>(lo_raw: T, hi_raw: T) -> Interval<T> {
    if lo_raw.is_nan() || hi_raw.is_nan() {
        Interval::nan()
    } else {
        Interval::new(
            lo_raw.step_down(LIBM_PAD_ULPS),
            hi_raw.step_up(LIBM_PAD_ULPS),
        )
    }
}

#[inline]
fn powi_positive<T: RealScalar>(iv: Interval<T>, exp: u32) -> Interval<T> {
    let mut n = exp;
    let mut acc = Interval::point(T::ONE);
    let mut cur = iv;
    while n > 0 {
        if n & 1 == 1 {
            acc *= cur;
        }
        n >>= 1;
        if n > 0 {
            cur = cur.sqr_interval();
        }
    }
    acc
}

impl<T: RealScalar> Scalar for Interval<T> {
    const ZERO: Self = Self::point(T::ZERO);
    const ONE: Self = Self::point(T::ONE);
    const TWO: Self = Self::point(T::TWO);
    const HALF: Self = Self::point(T::HALF);
    const PI: Self = Self {
        lo: T::PI,
        hi: T::PI,
    };

    #[inline]
    fn from_f64(v: f64) -> Self {
        let mid = T::from_f64(v);
        if mid.to_f64() == v {
            Self::point(mid)
        } else if mid.to_f64() < v {
            Self::new(mid, mid.next_up())
        } else {
            Self::new(mid.next_down(), mid)
        }
    }

    #[inline]
    fn real_part(self) -> f64 {
        self.midpoint().to_f64()
    }

    #[inline]
    fn is_finite_val(self) -> bool {
        self.is_finite()
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs_interval()
    }

    #[inline]
    fn sqr(self) -> Self {
        self.sqr_interval()
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt_interval()
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        if n == 0 {
            return Self::ONE;
        }
        if n < 0 {
            return Self::ONE / self.powi(-n);
        }
        let exp = n.unsigned_abs();
        if exp.is_multiple_of(2) {
            let mag = self.abs_interval();
            let p = powi_positive(mag, exp);
            Self::new(p.lo.max_val(T::ZERO), p.hi)
        } else if self.lo >= T::ZERO {
            powi_positive(self, exp)
        } else if self.hi <= T::ZERO {
            -powi_positive(-self, exp)
        } else {
            let hi = powi_positive(Self::point(self.hi), exp).hi;
            let lo = -powi_positive(Self::point(-self.lo), exp).hi;
            Self::new(lo, hi)
        }
    }

    #[inline]
    fn powf(self, exp: Self) -> Self {
        if self.is_nan() || exp.is_nan() || self.lo <= T::ZERO {
            return Self::nan();
        }
        (exp * self.ln()).exp()
    }

    #[inline]
    fn exp(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        let lo = if self.lo == T::NEG_INFINITY {
            T::ZERO
        } else {
            self.lo.exp().step_down(LIBM_PAD_ULPS).max_val(T::ZERO)
        };
        let hi = if self.hi == T::INFINITY {
            T::INFINITY
        } else {
            self.hi.exp().step_up(LIBM_PAD_ULPS)
        };
        Self::new(lo, hi)
    }

    #[inline]
    fn ln(self) -> Self {
        if self.is_nan() || self.lo < T::ZERO || self.hi <= T::ZERO {
            return Self::nan();
        }
        let lo = if self.lo == T::ZERO {
            T::NEG_INFINITY
        } else {
            self.lo.ln().step_down(LIBM_PAD_ULPS)
        };
        let hi = if self.hi == T::INFINITY {
            T::INFINITY
        } else {
            self.hi.ln().step_up(LIBM_PAD_ULPS)
        };
        Self::new(lo, hi)
    }

    #[inline]
    fn log2(self) -> Self {
        if self.is_nan() || self.lo < T::ZERO || self.hi <= T::ZERO {
            return Self::nan();
        }
        let lo = if self.lo == T::ZERO {
            T::NEG_INFINITY
        } else {
            self.lo.log2().step_down(LIBM_PAD_ULPS)
        };
        let hi = self.hi.log2().step_up(LIBM_PAD_ULPS);
        Self::new(lo, hi)
    }

    #[inline]
    fn log10(self) -> Self {
        if self.is_nan() || self.lo < T::ZERO || self.hi <= T::ZERO {
            return Self::nan();
        }
        let lo = if self.lo == T::ZERO {
            T::NEG_INFINITY
        } else {
            self.lo.log10().step_down(LIBM_PAD_ULPS)
        };
        let hi = self.hi.log10().step_up(LIBM_PAD_ULPS);
        Self::new(lo, hi)
    }

    #[inline]
    fn sin(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        let unit = Self::new(-T::ONE, T::ONE);
        if !self.is_finite() || self.width() >= T::TAU.next_down() {
            return unit;
        }
        let sa = pad_monotone(self.lo.sin(), self.lo.sin());
        let sb = pad_monotone(self.hi.sin(), self.hi.sin());
        let mut lo = sa.lo.min_val(sb.lo).max_val(-T::ONE);
        let mut hi = sa.hi.max_val(sb.hi).min_val(T::ONE);
        let tau = Self::tau_enclosure();
        let hp = Self::frac_pi_2_enclosure();
        if contains_integer((self - hp) / tau) {
            hi = T::ONE;
        }
        if contains_integer((self + hp) / tau) {
            lo = -T::ONE;
        }
        Self::new(lo, hi)
    }

    #[inline]
    fn cos(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        let unit = Self::new(-T::ONE, T::ONE);
        if !self.is_finite() || self.width() >= T::TAU.next_down() {
            return unit;
        }
        let ca = pad_monotone(self.lo.cos(), self.lo.cos());
        let cb = pad_monotone(self.hi.cos(), self.hi.cos());
        let mut lo = ca.lo.min_val(cb.lo).max_val(-T::ONE);
        let mut hi = ca.hi.max_val(cb.hi).min_val(T::ONE);
        let tau = Self::tau_enclosure();
        let pi = Self::pi_enclosure();
        if contains_integer(self / tau) {
            hi = T::ONE;
        }
        if contains_integer((self - pi) / tau) {
            lo = -T::ONE;
        }
        Self::new(lo, hi)
    }

    #[inline]
    fn tan(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        if !self.is_finite() || self.width() >= T::PI.next_down() {
            return Self::entire();
        }
        let hp = Self::frac_pi_2_enclosure();
        let pi = Self::pi_enclosure();
        if contains_integer((self - hp) / pi) {
            return Self::entire();
        }
        pad_monotone(self.lo.tan(), self.hi.tan())
    }

    #[inline]
    fn asin(self) -> Self {
        if self.is_nan() || self.lo < -T::ONE || self.hi > T::ONE {
            return Self::nan();
        }
        let hp_up = T::FRAC_PI_2.next_up();
        let raw = pad_monotone(self.lo.asin(), self.hi.asin());
        Self::new(raw.lo.max_val(-hp_up), raw.hi.min_val(hp_up))
    }

    #[inline]
    fn acos(self) -> Self {
        if self.is_nan() || self.lo < -T::ONE || self.hi > T::ONE {
            return Self::nan();
        }
        let raw = pad_monotone(self.hi.acos(), self.lo.acos());
        Self::new(raw.lo.max_val(T::ZERO), raw.hi.min_val(T::PI.next_up()))
    }

    #[inline]
    fn atan(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        let hp_up = T::FRAC_PI_2.next_up();
        let raw = pad_monotone(self.lo.atan(), self.hi.atan());
        Self::new(raw.lo.max_val(-hp_up), raw.hi.min_val(hp_up))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::nan();
        }
        if other.lo > T::ZERO {
            return (self / other).atan();
        }
        let pi = Self::pi_enclosure();
        if other.hi < T::ZERO && self.lo >= T::ZERO {
            return (self / other).atan() + pi;
        }
        if other.hi < T::ZERO && self.hi < T::ZERO {
            return (self / other).atan() - pi;
        }
        Self::new(-T::PI.next_up(), T::PI.next_up())
    }

    #[inline]
    fn sinh(self) -> Self {
        if self.is_nan() {
            Self::nan()
        } else {
            pad_monotone(self.lo.sinh(), self.hi.sinh())
        }
    }

    #[inline]
    fn cosh(self) -> Self {
        let mag = self.abs_interval();
        if mag.is_nan() {
            return Self::nan();
        }
        let raw = pad_monotone(mag.lo.cosh(), mag.hi.cosh());
        Self::new(raw.lo.max_val(T::ONE), raw.hi)
    }

    #[inline]
    fn tanh(self) -> Self {
        if self.is_nan() {
            return Self::nan();
        }
        let raw = pad_monotone(self.lo.tanh(), self.hi.tanh());
        Self::new(raw.lo.max_val(-T::ONE), raw.hi.min_val(T::ONE))
    }

    #[inline]
    fn asinh(self) -> Self {
        if self.is_nan() {
            Self::nan()
        } else {
            pad_monotone(self.lo.asinh(), self.hi.asinh())
        }
    }

    #[inline]
    fn acosh(self) -> Self {
        if self.is_nan() || self.lo < T::ONE {
            return Self::nan();
        }
        let raw = pad_monotone(self.lo.acosh(), self.hi.acosh());
        Self::new(raw.lo.max_val(T::ZERO), raw.hi)
    }

    #[inline]
    fn atanh(self) -> Self {
        if self.is_nan() || self.lo <= -T::ONE || self.hi >= T::ONE {
            Self::nan()
        } else {
            pad_monotone(self.lo.atanh(), self.hi.atanh())
        }
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        let _ = powi_generic::<f64>;
        (self.sqr_interval() + other.sqr_interval()).sqrt_interval()
    }
}
