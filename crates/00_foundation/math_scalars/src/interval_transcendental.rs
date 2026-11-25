//! Certified outward-rounded `libm` elementary functions for `Interval<T>`.
//!
//! Applies monotone endpoint evaluation with certified `libm` ULP error padding
//! (`LIBM_ULP_PAD = 1` for core FDLIBM minimax transcendentals, `LIBM_COMPOUND_ULP_PAD = 2`
//! for compound transcendentals `tan`/`atan2`/`sinh`/`cosh`/`tanh`)
//! and rigorous stationary-point checks for periodic trigonometric functions.
// [cite:math_commons]

use crate::directed::DirectedRounding;
use crate::interval::Interval;
use crate::traits::Scalar;

/// Certified `libm` accuracy bound (in ULPs) for monotone FDLIBM unary transcendentals (`exp`, `ln`, `sin`, `cos`, `atan`, `asin`, `acos`, `hypot`).
const LIBM_ULP_PAD: u32 = 1;
/// Certified `libm` accuracy bound (in ULPs) for compound transcendentals (`tan`, `sinh`, `cosh`, `tanh`, `atan2`).
const LIBM_COMPOUND_ULP_PAD: u32 = 2;

#[inline]
fn contains_integer<T: DirectedRounding>(iv: Interval<T>) -> bool {
    if !iv.lo.is_finite() || !iv.hi.is_finite() {
        return true;
    }
    iv.lo.ceil() <= iv.hi.floor()
}

#[inline]
fn pi_enclosure<T: DirectedRounding>() -> Interval<T> {
    Interval::PI
}

#[inline]
fn tau_enclosure<T: DirectedRounding>() -> Interval<T> {
    Interval::TAU
}

#[inline]
fn half_pi_enclosure<T: DirectedRounding>() -> Interval<T> {
    let hp = T::PI * T::HALF;
    Interval::new(hp.next_down(), hp.next_up())
}

#[inline]
fn powi_positive<T: DirectedRounding>(iv: Interval<T>, n: u32) -> Interval<T> {
    if n == 0 {
        return Interval::point(T::ONE);
    }
    if n == 1 {
        return iv;
    }
    if (n & 1) == 0 {
        let mut base = iv.sqr();
        let mut exp = n >> 1;
        let mut acc = Interval::point(T::ONE);
        while exp > 0 {
            if (exp & 1) != 0 {
                acc *= base;
            }
            exp >>= 1;
            if exp > 0 {
                base = base.sqr();
            }
        }
        acc
    } else {
        // Odd positive power x^(2k+1) is strictly monotonically increasing on R!
        let lo = pow_monotone_down(iv.lo, n);
        let hi = pow_monotone_up(iv.hi, n);
        Interval::new(lo, hi)
    }
}

#[inline]
fn pow_monotone_down<T: DirectedRounding>(x: T, mut n: u32) -> T {
    if x == T::ZERO {
        return T::ZERO;
    }
    let neg = x < T::ZERO;
    let abs_x = x.abs();
    let mut base = abs_x;
    let mut acc = T::ONE;
    while n > 0 {
        if (n & 1) != 0 {
            acc = if neg {
                acc.mul_up(base)
            } else {
                acc.mul_down(base)
            };
        }
        n >>= 1;
        if n > 0 {
            base = if neg {
                base.mul_up(base)
            } else {
                base.mul_down(base)
            };
        }
    }
    if neg { -acc } else { acc }
}

#[inline]
fn pow_monotone_up<T: DirectedRounding>(x: T, mut n: u32) -> T {
    if x == T::ZERO {
        return T::ZERO;
    }
    let neg = x < T::ZERO;
    let abs_x = x.abs();
    let mut base = abs_x;
    let mut acc = T::ONE;
    while n > 0 {
        if (n & 1) != 0 {
            acc = if neg {
                acc.mul_down(base)
            } else {
                acc.mul_up(base)
            };
        }
        n >>= 1;
        if n > 0 {
            base = if neg {
                base.mul_down(base)
            } else {
                base.mul_up(base)
            };
        }
    }
    if neg { -acc } else { acc }
}

impl<T: DirectedRounding> Scalar for Interval<T> {
    const ZERO: Self = Self {
        lo: T::ZERO,
        hi: T::ZERO,
    };
    const ONE: Self = Self {
        lo: T::ONE,
        hi: T::ONE,
    };
    const TWO: Self = Self {
        lo: T::TWO,
        hi: T::TWO,
    };
    const HALF: Self = Self {
        lo: T::HALF,
        hi: T::HALF,
    };
    const PI: Self = Self {
        lo: T::PI_LO,
        hi: T::PI_HI,
    };
    const TAU: Self = Self {
        lo: T::TAU_LO,
        hi: T::TAU_HI,
    };

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::from_f64_enclosure(v)
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    #[inline]
    fn abs(self) -> Self {
        Self::new(self.mig(), self.mag())
    }

    #[inline]
    fn signum(self) -> Self {
        if self.lo > T::ZERO {
            Self::ONE
        } else if self.hi < T::ZERO {
            -Self::ONE
        } else if self.lo == T::ZERO && self.hi == T::ZERO {
            Self::ZERO
        } else if self.lo == T::ZERO {
            Self::new(T::ZERO, T::ONE)
        } else if self.hi == T::ZERO {
            Self::new(-T::ONE, T::ZERO)
        } else {
            Self::new(-T::ONE, T::ONE)
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        if self.hi < T::ZERO {
            return Self::entire();
        }
        let lo_in = self.lo.max(T::ZERO);
        Self::new(lo_in.sqrt_down().max(T::ZERO), self.hi.sqrt_up())
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::ONE;
        }
        let pos = powi_positive(self, n.unsigned_abs());
        if n > 0 { pos } else { Self::ONE / pos }
    }

    #[inline]
    fn powf(self, exp: Self) -> Self {
        if self.lo <= T::ZERO {
            return Self::entire();
        }
        (exp * self.ln()).exp()
    }

    #[inline]
    fn exp(self) -> Self {
        let lo = if self.lo == T::NEG_INFINITY {
            T::ZERO
        } else {
            self.lo.exp().pad_ulp_down(LIBM_ULP_PAD).max(T::ZERO)
        };
        let hi = if self.hi == T::INFINITY {
            T::INFINITY
        } else {
            self.hi.exp().pad_ulp_up(LIBM_ULP_PAD)
        };
        Self::new(lo, hi)
    }

    #[inline]
    fn ln(self) -> Self {
        if self.hi <= T::ZERO {
            return Self::entire();
        }
        let lo = if self.lo <= T::ZERO {
            T::NEG_INFINITY
        } else {
            self.lo.ln().pad_ulp_down(LIBM_ULP_PAD)
        };
        let hi = if self.hi == T::INFINITY {
            T::INFINITY
        } else {
            self.hi.ln().pad_ulp_up(LIBM_ULP_PAD)
        };
        Self::new(lo, hi)
    }

    #[inline]
    fn sin(self) -> Self {
        let tau = tau_enclosure::<T>();
        if !self.is_finite() || self.width() >= tau.lo {
            return Self::new(-T::ONE, T::ONE);
        }
        let s_lo_d = self.lo.sin().pad_ulp_down(LIBM_ULP_PAD);
        let s_lo_u = self.lo.sin().pad_ulp_up(LIBM_ULP_PAD);
        let s_hi_d = self.hi.sin().pad_ulp_down(LIBM_ULP_PAD);
        let s_hi_u = self.hi.sin().pad_ulp_up(LIBM_ULP_PAD);

        let mut lo = s_lo_d.min(s_hi_d).max(-T::ONE);
        let mut hi = s_lo_u.max(s_hi_u).min(T::ONE);

        let hp = half_pi_enclosure::<T>();
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
        let tau = tau_enclosure::<T>();
        if !self.is_finite() || self.width() >= tau.lo {
            return Self::new(-T::ONE, T::ONE);
        }
        let c_lo_d = self.lo.cos().pad_ulp_down(LIBM_ULP_PAD);
        let c_lo_u = self.lo.cos().pad_ulp_up(LIBM_ULP_PAD);
        let c_hi_d = self.hi.cos().pad_ulp_down(LIBM_ULP_PAD);
        let c_hi_u = self.hi.cos().pad_ulp_up(LIBM_ULP_PAD);

        let mut lo = c_lo_d.min(c_hi_d).max(-T::ONE);
        let mut hi = c_lo_u.max(c_hi_u).min(T::ONE);

        let pi = pi_enclosure::<T>();
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
        let pi = pi_enclosure::<T>();
        let hp = half_pi_enclosure::<T>();
        if !self.is_finite() || self.width() >= pi.lo || contains_integer((self - hp) / pi) {
            return Self::entire();
        }
        let lo = self.lo.tan().pad_ulp_down(LIBM_COMPOUND_ULP_PAD);
        let hi = self.hi.tan().pad_ulp_up(LIBM_COMPOUND_ULP_PAD);
        Self::new(lo, hi)
    }

    #[inline]
    fn asin(self) -> Self {
        if self.lo < -T::ONE || self.hi > T::ONE {
            return Self::entire();
        }
        let hp_hi = (T::PI * T::HALF).next_up();
        let lo = self.lo.asin().pad_ulp_down(LIBM_ULP_PAD).max(-hp_hi);
        let hi = self.hi.asin().pad_ulp_up(LIBM_ULP_PAD).min(hp_hi);
        Self::new(lo, hi)
    }

    #[inline]
    fn acos(self) -> Self {
        if self.lo < -T::ONE || self.hi > T::ONE {
            return Self::entire();
        }
        // acos is strictly monotonically decreasing on [-1, 1]
        let lo = self.hi.acos().pad_ulp_down(LIBM_ULP_PAD).max(T::ZERO);
        let hi = self.lo.acos().pad_ulp_up(LIBM_ULP_PAD).min(T::PI.next_up());
        Self::new(lo, hi)
    }

    #[inline]
    fn atan(self) -> Self {
        let hp_hi = (T::PI * T::HALF).next_up();
        let lo = if self.lo == T::NEG_INFINITY {
            -hp_hi
        } else {
            self.lo.atan().pad_ulp_down(LIBM_ULP_PAD).max(-hp_hi)
        };
        let hi = if self.hi == T::INFINITY {
            hp_hi
        } else {
            self.hi.atan().pad_ulp_up(LIBM_ULP_PAD).min(hp_hi)
        };
        Self::new(lo, hi)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        // Branch cut along negative real axis (other.lo <= 0 and self contains 0)
        if other.lo <= T::ZERO && self.contains_zero() {
            let p = T::PI.next_up();
            return Self::new(-p, p);
        }
        let c1_d = self.lo.atan2(other.lo).pad_ulp_down(LIBM_COMPOUND_ULP_PAD);
        let c2_d = self.lo.atan2(other.hi).pad_ulp_down(LIBM_COMPOUND_ULP_PAD);
        let c3_d = self.hi.atan2(other.lo).pad_ulp_down(LIBM_COMPOUND_ULP_PAD);
        let c4_d = self.hi.atan2(other.hi).pad_ulp_down(LIBM_COMPOUND_ULP_PAD);

        let c1_u = self.lo.atan2(other.lo).pad_ulp_up(LIBM_COMPOUND_ULP_PAD);
        let c2_u = self.lo.atan2(other.hi).pad_ulp_up(LIBM_COMPOUND_ULP_PAD);
        let c3_u = self.hi.atan2(other.lo).pad_ulp_up(LIBM_COMPOUND_ULP_PAD);
        let c4_u = self.hi.atan2(other.hi).pad_ulp_up(LIBM_COMPOUND_ULP_PAD);

        let lo = c1_d.min(c2_d).min(c3_d.min(c4_d));
        let hi = c1_u.max(c2_u).max(c3_u.max(c4_u));
        Self::new(lo, hi)
    }

    #[inline]
    fn sinh(self) -> Self {
        let lo = if self.lo == T::NEG_INFINITY {
            T::NEG_INFINITY
        } else {
            self.lo.sinh().pad_ulp_down(LIBM_COMPOUND_ULP_PAD)
        };
        let hi = if self.hi == T::INFINITY {
            T::INFINITY
        } else {
            self.hi.sinh().pad_ulp_up(LIBM_COMPOUND_ULP_PAD)
        };
        Self::new(lo, hi)
    }

    #[inline]
    fn cosh(self) -> Self {
        let abs_iv = self.abs();
        let lo = if abs_iv.lo == T::ZERO {
            T::ONE
        } else {
            abs_iv
                .lo
                .cosh()
                .pad_ulp_down(LIBM_COMPOUND_ULP_PAD)
                .max(T::ONE)
        };
        let hi = if abs_iv.hi == T::INFINITY {
            T::INFINITY
        } else {
            abs_iv.hi.cosh().pad_ulp_up(LIBM_COMPOUND_ULP_PAD)
        };
        Self::new(lo, hi)
    }

    #[inline]
    fn tanh(self) -> Self {
        let lo = if self.lo == T::NEG_INFINITY {
            -T::ONE
        } else {
            self.lo
                .tanh()
                .pad_ulp_down(LIBM_COMPOUND_ULP_PAD)
                .max(-T::ONE)
        };
        let hi = if self.hi == T::INFINITY {
            T::ONE
        } else {
            self.hi.tanh().pad_ulp_up(LIBM_COMPOUND_ULP_PAD).min(T::ONE)
        };
        Self::new(lo, hi)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        let ax = self.abs();
        let ay = other.abs();
        let lo = ax.lo.hypot(ay.lo).pad_ulp_down(LIBM_ULP_PAD).max(T::ZERO);
        let hi = ax.hi.hypot(ay.hi).pad_ulp_up(LIBM_ULP_PAD);
        Self::new(lo, hi)
    }
}
