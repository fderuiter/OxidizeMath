//! Forward-mode `Dual` scalar numbers over real and complex bases with Clarke subgradients.
// [cite:engine_symbolic]

use core::ops::{Add, Div, Mul, Neg, Sub};
use num_complex::Complex;
use num_traits::Float;

/// Base scalar trait supporting real (`f64`) and complex (`Complex<f64>`) forward-mode [`Dual`] numbers.
pub trait DualScalar:
    Copy
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Additive identity ($0$).
    const ZERO: Self;
    /// Multiplicative identity ($1$).
    const ONE: Self;
    /// Canonical quiet NaN value.
    const NAN: Self;

    /// Embeds a real `f64` constant into `Self`.
    fn from_f64(val: f64) -> Self;
    /// Extracts the real part of `Self`.
    fn real_part(self) -> f64;
    /// Returns `true` if any component of `Self` is NaN.
    fn is_nan_val(self) -> bool;
    /// Returns `true` if all components of `Self` are finite.
    fn is_finite_val(self) -> bool;
    /// Returns `true` if `Self` is exact zero.
    fn is_zero_val(self) -> bool;

    /// Evaluates `(sin(x), d/dx sin(x))` = `(sin(x), cos(x))`.
    fn sin_with_deriv(self) -> (Self, Self);
    /// Evaluates `(cos(x), d/dx cos(x))` = `(cos(x), -sin(x))`.
    fn cos_with_deriv(self) -> (Self, Self);
    /// Evaluates `(tan(x), d/dx tan(x))` = `(tan(x), sec^2(x))`.
    fn tan_with_deriv(self) -> (Self, Self);
    /// Evaluates `(exp(x), d/dx exp(x))` = `(exp(x), exp(x))`.
    fn exp_with_deriv(self) -> (Self, Self);
    /// Evaluates `(ln(x), d/dx ln(x))` = `(ln(x), 1/x)`.
    fn ln_with_deriv(self) -> (Self, Self);
    /// Evaluates `(log10(x), d/dx log10(x))`.
    fn log10_with_deriv(self) -> (Self, Self);
    /// Evaluates `(sqrt(x), d/dx sqrt(x))`.
    fn sqrt_with_deriv(self) -> (Self, Self);
    /// Evaluates `(sinh(x), cosh(x))`.
    fn sinh_with_deriv(self) -> (Self, Self);
    /// Evaluates `(cosh(x), sinh(x))`.
    fn cosh_with_deriv(self) -> (Self, Self);
    /// Evaluates `(tanh(x), sech^2(x))`.
    fn tanh_with_deriv(self) -> (Self, Self);
    /// Evaluates `(asin(x), 1 / sqrt(1 - x^2))`.
    fn asin_with_deriv(self) -> (Self, Self);
    /// Evaluates `(acos(x), -1 / sqrt(1 - x^2))`.
    fn acos_with_deriv(self) -> (Self, Self);
    /// Evaluates `(atan(x), 1 / (1 + x^2))`.
    fn atan_with_deriv(self) -> (Self, Self);

    /// Evaluates `a^b` with tangent `(da, db)` on [`Dual<Self>`].
    fn pow_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self>;
    /// Evaluates $|x|$ with centered Clarke subgradient ($0$ at $x = 0$).
    fn abs_dual(x: Dual<Self>) -> Dual<Self>;
    /// Evaluates Heaviside `step(x)` ($1$ for $\operatorname{Re}(x) \ge 0$, $0$ otherwise) with Clarke subgradient $0$.
    fn step_dual(x: Dual<Self>) -> Dual<Self>;
    /// Evaluates boxcar `rect(x)` ($1$ for $|\operatorname{Re}(x)| \le 0.5$, $0$ otherwise) with Clarke subgradient $0$.
    fn rect_dual(x: Dual<Self>) -> Dual<Self>;
    /// Evaluates `min(a, b)` with midpoint Clarke subgradient at ties.
    fn min_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self>;
    /// Evaluates `max(a, b)` with midpoint Clarke subgradient at ties.
    fn max_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self>;
    /// Evaluates `clamp(x, lo, hi)` with Clarke subgradient at active bounds.
    fn clamp_dual(x: Dual<Self>, lo: Dual<Self>, hi: Dual<Self>) -> Dual<Self>;
    /// Evaluates `atan2(y, x)` with exact partial derivatives.
    fn atan2_dual(y: Dual<Self>, x: Dual<Self>) -> Dual<Self>;
}

/// First-order forward-mode dual number $x + \varepsilon \dot{x}$ ($\varepsilon^2 = 0$) modeled on `num-dual`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Dual<T = f64> {
    /// Primal value $x$.
    pub re: T,
    /// First-order directional derivative (tangent) $\dot{x}$.
    pub eps: T,
}

/// Real `f64` dual number alias (`Dual<f64>`).
pub type Dual64 = Dual<f64>;

/// Complex `Complex<f64>` dual number alias (`Dual<Complex<f64>>`).
pub type DualComplex64 = Dual<Complex<f64>>;

impl<T: DualScalar> Dual<T> {
    /// Additive identity ($0 + 0\varepsilon$).
    pub const ZERO: Self = Self {
        re: T::ZERO,
        eps: T::ZERO,
    };

    /// Multiplicative identity ($1 + 0\varepsilon$).
    pub const ONE: Self = Self {
        re: T::ONE,
        eps: T::ZERO,
    };

    /// Canonical quiet NaN dual value.
    pub const NAN: Self = Self {
        re: T::NAN,
        eps: T::NAN,
    };

    /// Constructs a [`Dual`] number from primal `re` and tangent `eps`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(re: T, eps: T) -> Self {
        Self { re, eps }
    }

    /// Constructs a constant [`Dual`] number with zero tangent ($\dot{x} = 0$).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn constant(re: T) -> Self {
        Self { re, eps: T::ZERO }
    }

    /// Constructs an active differentiation variable with unit tangent ($\dot{x} = 1$).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn variable(re: T) -> Self {
        Self { re, eps: T::ONE }
    }

    /// Applies univariate chain rule $(f(x), f'(x) \cdot \dot{x})$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn chain(self, val_and_deriv: (T, T)) -> Self {
        Self {
            re: val_and_deriv.0,
            eps: val_and_deriv.1 * self.eps,
        }
    }

    /// Multiplicative reciprocal $1 / x$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn recip(self) -> Self {
        let inv = T::ONE / self.re;
        Self {
            re: inv,
            eps: -(self.eps * inv * inv),
        }
    }

    /// Integer power $x^n$ computed via binary exponentiation.
    #[must_use]
    #[verified_engine::verified]
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(T::ONE);
        }
        let mut base = if n < 0 { self.recip() } else { self };
        let mut exp = n.unsigned_abs();
        let mut acc = Self::constant(T::ONE);
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            exp >>= 1;
            if exp > 0 {
                base *= base;
            }
        }
        acc
    }

    /// General power $x^p$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn powf(self, p: Self) -> Self {
        T::pow_dual(self, p)
    }

    /// Sine $\sin(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn sin(self) -> Self {
        self.chain(self.re.sin_with_deriv())
    }

    /// Cosine $\cos(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn cos(self) -> Self {
        self.chain(self.re.cos_with_deriv())
    }

    /// Simultaneous $(\sin(x), \cos(x))$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    /// Tangent $\tan(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn tan(self) -> Self {
        self.chain(self.re.tan_with_deriv())
    }

    /// Natural exponential $e^x$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn exp(self) -> Self {
        self.chain(self.re.exp_with_deriv())
    }

    /// Natural logarithm $\ln(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn ln(self) -> Self {
        self.chain(self.re.ln_with_deriv())
    }

    /// Base-10 logarithm $\log_{10}(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn log10(self) -> Self {
        self.chain(self.re.log10_with_deriv())
    }

    /// Square root $\sqrt{x}$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn sqrt(self) -> Self {
        self.chain(self.re.sqrt_with_deriv())
    }

    /// Hyperbolic sine $\sinh(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn sinh(self) -> Self {
        self.chain(self.re.sinh_with_deriv())
    }

    /// Hyperbolic cosine $\cosh(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn cosh(self) -> Self {
        self.chain(self.re.cosh_with_deriv())
    }

    /// Hyperbolic tangent $\tanh(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn tanh(self) -> Self {
        self.chain(self.re.tanh_with_deriv())
    }

    /// Inverse sine $\arcsin(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn asin(self) -> Self {
        self.chain(self.re.asin_with_deriv())
    }

    /// Inverse cosine $\arccos(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn acos(self) -> Self {
        self.chain(self.re.acos_with_deriv())
    }

    /// Inverse tangent $\arctan(x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn atan(self) -> Self {
        self.chain(self.re.atan_with_deriv())
    }

    /// Four-quadrant inverse tangent $\operatorname{atan2}(y, x)$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn atan2(self, other: Self) -> Self {
        T::atan2_dual(self, other)
    }

    /// Absolute value $|x|$ with centered Clarke subgradient ($0$ at $x = 0$).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn abs(self) -> Self {
        T::abs_dual(self)
    }

    /// Unit Heaviside step with Clarke subgradient $0$ everywhere.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn step(self) -> Self {
        T::step_dual(self)
    }

    /// Unit rectangular boxcar pulse with Clarke subgradient $0$ everywhere.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn rect(self) -> Self {
        T::rect_dual(self)
    }

    /// Minimum $\min(a, b)$ with midpoint Clarke subgradient at $a = b$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn min(self, other: Self) -> Self {
        T::min_dual(self, other)
    }

    /// Maximum $\max(a, b)$ with midpoint Clarke subgradient at $a = b$.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn max(self, other: Self) -> Self {
        T::max_dual(self, other)
    }

    /// Clamping $\operatorname{clamp}(x, \text{lo}, \text{hi})$ with Clarke subgradient at active bounds.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        T::clamp_dual(self, lo, hi)
    }
}

#[verified_engine::verified]
pub(crate) fn pow_f64_dual(a: Dual<f64>, b: Dual<f64>) -> Dual<f64> {
    let val = Float::powf(a.re, b.re);
    if a.re == 0.0 {
        if b.re == 1.0 {
            return Dual::new(0.0, a.eps);
        }
        if b.re > 1.0 {
            return Dual::new(0.0, 0.0);
        }
        if b.re == 0.0 {
            return Dual::new(1.0, 0.0);
        }
    }
    let term_a = if a.eps == 0.0 {
        0.0
    } else {
        b.re * Float::powf(a.re, b.re - 1.0) * a.eps
    };
    let term_b = if b.eps == 0.0 {
        0.0
    } else if a.re > 0.0 {
        val * Float::ln(a.re) * b.eps
    } else {
        f64::NAN
    };
    Dual::new(val, term_a + term_b)
}
