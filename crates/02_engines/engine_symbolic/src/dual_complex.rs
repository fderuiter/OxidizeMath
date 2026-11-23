//! `DualScalar` implementation for holomorphic `Complex<f64>` forward-mode differentiation.
// [cite:engine_symbolic]

use num_complex::Complex;

use crate::dual::{Dual, DualScalar};

impl DualScalar for Complex<f64> {
    const ZERO: Self = Complex::new(0.0, 0.0);
    const ONE: Self = Complex::new(1.0, 0.0);
    const NAN: Self = Complex::new(f64::NAN, f64::NAN);

    #[inline]
    #[verified_engine::verified]
    fn from_f64(val: f64) -> Self {
        Complex::new(val, 0.0)
    }

    #[inline]
    #[verified_engine::verified]
    fn real_part(self) -> f64 {
        self.re
    }

    #[inline]
    #[verified_engine::verified]
    fn is_nan_val(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    #[inline]
    #[verified_engine::verified]
    fn is_finite_val(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[inline]
    #[verified_engine::verified]
    fn is_zero_val(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    #[inline]
    #[verified_engine::verified]
    fn sin_with_deriv(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    #[inline]
    #[verified_engine::verified]
    fn cos_with_deriv(self) -> (Self, Self) {
        (self.cos(), -self.sin())
    }

    #[inline]
    #[verified_engine::verified]
    fn tan_with_deriv(self) -> (Self, Self) {
        let c = self.cos();
        (self.tan(), Self::ONE / (c * c))
    }

    #[inline]
    #[verified_engine::verified]
    fn exp_with_deriv(self) -> (Self, Self) {
        let e = self.exp();
        (e, e)
    }

    #[inline]
    #[verified_engine::verified]
    fn ln_with_deriv(self) -> (Self, Self) {
        (self.ln(), Self::ONE / self)
    }

    #[inline]
    #[verified_engine::verified]
    fn log10_with_deriv(self) -> (Self, Self) {
        (
            self.log10(),
            Self::ONE / (self * Self::from_f64(core::f64::consts::LN_10)),
        )
    }

    #[inline]
    #[verified_engine::verified]
    fn sqrt_with_deriv(self) -> (Self, Self) {
        let s = self.sqrt();
        let ds = if s.is_zero_val() {
            Self::ZERO
        } else {
            Self::from_f64(0.5) / s
        };
        (s, ds)
    }

    #[inline]
    #[verified_engine::verified]
    fn sinh_with_deriv(self) -> (Self, Self) {
        (self.sinh(), self.cosh())
    }

    #[inline]
    #[verified_engine::verified]
    fn cosh_with_deriv(self) -> (Self, Self) {
        (self.cosh(), self.sinh())
    }

    #[inline]
    #[verified_engine::verified]
    fn tanh_with_deriv(self) -> (Self, Self) {
        let c = self.cosh();
        (self.tanh(), Self::ONE / (c * c))
    }

    #[inline]
    #[verified_engine::verified]
    fn asin_with_deriv(self) -> (Self, Self) {
        (self.asin(), Self::ONE / (Self::ONE - self * self).sqrt())
    }

    #[inline]
    #[verified_engine::verified]
    fn acos_with_deriv(self) -> (Self, Self) {
        (self.acos(), -Self::ONE / (Self::ONE - self * self).sqrt())
    }

    #[inline]
    #[verified_engine::verified]
    fn atan_with_deriv(self) -> (Self, Self) {
        (self.atan(), Self::ONE / (Self::ONE + self * self))
    }

    #[inline]
    #[verified_engine::verified]
    fn pow_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self> {
        if a.re.is_zero_val() {
            if b.re == Self::ONE {
                return Dual::new(Self::ZERO, a.eps);
            }
            if b.re.im == 0.0 && b.re.re > 1.0 {
                return Dual::constant(Self::ZERO);
            }
            if b.re.is_zero_val() {
                return Dual::constant(Self::ONE);
            }
        }
        let val = a.re.powc(b.re);
        let da = if a.eps.is_zero_val() {
            Self::ZERO
        } else {
            b.re * a.re.powc(b.re - Self::ONE) * a.eps
        };
        let db = if b.eps.is_zero_val() {
            Self::ZERO
        } else {
            val * a.re.ln() * b.eps
        };
        Dual::new(val, da + db)
    }

    #[inline]
    #[verified_engine::verified]
    fn abs_dual(x: Dual<Self>) -> Dual<Self> {
        let n = x.re.norm();
        let dn = if n == 0.0 {
            0.0
        } else {
            (x.re.re * x.eps.re + x.re.im * x.eps.im) / n
        };
        Dual::new(Self::from_f64(n), Self::from_f64(dn))
    }

    #[inline]
    #[verified_engine::verified]
    fn step_dual(x: Dual<Self>) -> Dual<Self> {
        let r = f64::step_dual(Dual::new(x.re.re, x.eps.re));
        Dual::new(Self::from_f64(r.re), Self::from_f64(r.eps))
    }

    #[inline]
    #[verified_engine::verified]
    fn rect_dual(x: Dual<Self>) -> Dual<Self> {
        let r = f64::rect_dual(Dual::new(x.re.re, x.eps.re));
        Dual::new(Self::from_f64(r.re), Self::from_f64(r.eps))
    }

    #[inline]
    #[verified_engine::verified]
    fn min_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self> {
        let r = f64::min_dual(Dual::new(a.re.re, a.eps.re), Dual::new(b.re.re, b.eps.re));
        Dual::new(Self::from_f64(r.re), Self::from_f64(r.eps))
    }

    #[inline]
    #[verified_engine::verified]
    fn max_dual(a: Dual<Self>, b: Dual<Self>) -> Dual<Self> {
        let r = f64::max_dual(Dual::new(a.re.re, a.eps.re), Dual::new(b.re.re, b.eps.re));
        Dual::new(Self::from_f64(r.re), Self::from_f64(r.eps))
    }

    #[inline]
    #[verified_engine::verified]
    fn clamp_dual(x: Dual<Self>, lo: Dual<Self>, hi: Dual<Self>) -> Dual<Self> {
        let r = f64::clamp_dual(
            Dual::new(x.re.re, x.eps.re),
            Dual::new(lo.re.re, lo.eps.re),
            Dual::new(hi.re.re, hi.eps.re),
        );
        Dual::new(Self::from_f64(r.re), Self::from_f64(r.eps))
    }

    #[inline]
    #[verified_engine::verified]
    fn atan2_dual(y: Dual<Self>, x: Dual<Self>) -> Dual<Self> {
        let r = f64::atan2_dual(Dual::new(y.re.re, y.eps.re), Dual::new(x.re.re, x.eps.re));
        Dual::new(Self::from_f64(r.re), Self::from_f64(r.eps))
    }
}
