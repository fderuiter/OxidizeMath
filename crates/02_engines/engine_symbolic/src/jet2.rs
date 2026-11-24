//! Second-order $N$-variate Taylor jet `Jet2<N>` for exact single-pass gradients and Hessians.
// [cite:engine_symbolic]

use num_traits::Float;

/// Multivariate second-order forward jet storing value, gradient $\nabla f \in \mathbb{R}^N$,
/// and symmetric Hessian $\nabla^2 f \in \mathbb{R}^{N \times N}$ on stack arrays.
///
/// Modeled on `num-dual` hyper-dual / second-order jets to eliminate finite-difference
/// cancellation in surface curvature, Laplace-Beltrami, and curvilinear Laplacian operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jet2<const N: usize = 2> {
    /// Primal scalar value $f(x)$.
    pub val: f64,
    /// First-order partial derivatives $\partial f / \partial x_i$.
    pub grad: [f64; N],
    /// Second-order partial derivatives $\partial^2 f / (\partial x_i \partial x_j)$.
    pub hess: [[f64; N]; N],
}

/// Alias for 2-variable second-order jet (`Jet2<2>`).
pub type Jet2D = Jet2<2>;

/// Alias for 3-variable second-order jet (`Jet2<3>`).
pub type Jet2_3D = Jet2<3>;

impl<const N: usize> Default for Jet2<N> {
    #[inline]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const N: usize> Jet2<N> {
    /// Additive identity jet ($0$).
    pub const ZERO: Self = Self::constant(0.0);

    /// Multiplicative identity jet ($1$).
    pub const ONE: Self = Self::constant(1.0);

    /// Canonical quiet NaN jet.
    pub const NAN: Self = Self::constant(f64::NAN);

    /// Constructs a constant [`Jet2`] with zero gradient and Hessian.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn constant(val: f64) -> Self {
        Self {
            val,
            grad: [0.0; N],
            hess: [[0.0; N]; N],
        }
    }

    /// Returns `true` if `val`, `grad`, and `hess` entries are all finite.
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn is_finite(&self) -> bool {
        if !self.val.is_finite() {
            return false;
        }
        let mut i = 0;
        while i < N {
            if !self.grad[i].is_finite() {
                return false;
            }
            let mut j = 0;
            while j < N {
                if !self.hess[i][j].is_finite() {
                    return false;
                }
                j += 1;
            }
            i += 1;
        }
        true
    }

    /// Constructs an independent coordinate variable $x_{\text{axis}}$ with unit gradient along `axis`.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn var(val: f64, axis: usize) -> Self {
        let mut grad = [0.0; N];
        if axis < N {
            grad[axis] = 1.0;
        }
        Self {
            val,
            grad,
            hess: [[0.0; N]; N],
        }
    }

    /// Returns `true` if `grad` and `hess` are identically zero.
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn is_constant_jet(&self) -> bool {
        let mut i = 0;
        while i < N {
            if self.grad[i] != 0.0 {
                return false;
            }
            let mut j = 0;
            while j < N {
                if self.hess[i][j] != 0.0 {
                    return false;
                }
                j += 1;
            }
            i += 1;
        }
        true
    }

    /// Propagates univariate derivatives $(g(u), g'(u), g''(u))$ through `self`.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn chain_unary(self, f: f64, df: f64, d2f: f64) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            grad[i] = df * self.grad[i];
            let mut j = 0;
            while j < N {
                hess[i][j] = d2f * self.grad[i] * self.grad[j] + df * self.hess[i][j];
                j += 1;
            }
            i += 1;
        }
        Self { val: f, grad, hess }
    }

    /// Multiplicative reciprocal $1 / u$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn recip(self) -> Self {
        let inv = 1.0 / self.val;
        let inv2 = inv * inv;
        self.chain_unary(inv, -inv2, 2.0 * inv2 * inv)
    }

    /// Integer power $u^n$ via binary exponentiation.
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(1.0);
        }
        let mut base = if n < 0 { self.recip() } else { self };
        let mut exp = n.unsigned_abs();
        let mut acc = Self::constant(1.0);
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

    /// Real power $u^p$ handling integer/constant exponents at $u = 0$ without spurious `NaN`s.
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn powf(self, p: Self) -> Self {
        if p.is_constant_jet() {
            let c = p.val;
            if c == 0.0 {
                return Self::constant(1.0);
            }
            if c == 1.0 {
                return self;
            }
            if c == 2.0 {
                return self * self;
            }
            let f = Float::powf(self.val, c);
            let df = if self.val == 0.0 && c > 1.0 {
                0.0
            } else {
                c * Float::powf(self.val, c - 1.0)
            };
            let d2f = if self.val == 0.0 && c > 2.0 {
                0.0
            } else {
                c * (c - 1.0) * Float::powf(self.val, c - 2.0)
            };
            return self.chain_unary(f, df, d2f);
        }
        (p * self.ln()).exp()
    }

    /// Square root $\sqrt{u}$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn sqrt(self) -> Self {
        let s = Float::sqrt(self.val);
        if s == 0.0 {
            Self::constant(0.0)
        } else {
            let ds = 0.5 / s;
            let d2s = -0.25 / (self.val * s);
            self.chain_unary(s, ds, d2s)
        }
    }

    /// Sine $\sin(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn sin(self) -> Self {
        let (s, c) = Float::sin_cos(self.val);
        self.chain_unary(s, c, -s)
    }

    /// Cosine $\cos(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn cos(self) -> Self {
        let (s, c) = Float::sin_cos(self.val);
        self.chain_unary(c, -s, -c)
    }

    /// Simultaneous $(\sin(u), \cos(u))$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = Float::sin_cos(self.val);
        (self.chain_unary(s, c, -s), self.chain_unary(c, -s, -c))
    }

    /// Tangent $\tan(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn tan(self) -> Self {
        let t = Float::tan(self.val);
        let sec2 = 1.0 + t * t;
        self.chain_unary(t, sec2, 2.0 * t * sec2)
    }

    /// Natural exponential $e^u$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn exp(self) -> Self {
        let e = Float::exp(self.val);
        self.chain_unary(e, e, e)
    }

    /// Natural logarithm $\ln(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn ln(self) -> Self {
        let inv = 1.0 / self.val;
        self.chain_unary(Float::ln(self.val), inv, -inv * inv)
    }

    /// Base-10 logarithm $\log_{10}(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn log10(self) -> Self {
        let inv = 1.0 / self.val;
        let scale = 1.0 / core::f64::consts::LN_10;
        self.chain_unary(Float::log10(self.val), inv * scale, -inv * inv * scale)
    }

    /// Hyperbolic sine $\sinh(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn sinh(self) -> Self {
        let sh = Float::sinh(self.val);
        let ch = Float::cosh(self.val);
        self.chain_unary(sh, ch, sh)
    }

    /// Hyperbolic cosine $\cosh(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn cosh(self) -> Self {
        let sh = Float::sinh(self.val);
        let ch = Float::cosh(self.val);
        self.chain_unary(ch, sh, ch)
    }

    /// Hyperbolic tangent $\tanh(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn tanh(self) -> Self {
        let t = Float::tanh(self.val);
        let sech2 = 1.0 - t * t;
        self.chain_unary(t, sech2, -2.0 * t * sech2)
    }

    /// Inverse sine $\arcsin(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn asin(self) -> Self {
        let q = 1.0 - self.val * self.val;
        let sq = Float::sqrt(q);
        self.chain_unary(Float::asin(self.val), 1.0 / sq, self.val / (q * sq))
    }

    /// Inverse cosine $\arccos(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn acos(self) -> Self {
        let q = 1.0 - self.val * self.val;
        let sq = Float::sqrt(q);
        self.chain_unary(Float::acos(self.val), -1.0 / sq, -self.val / (q * sq))
    }

    /// Inverse tangent $\arctan(u)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn atan(self) -> Self {
        let q = 1.0 + self.val * self.val;
        self.chain_unary(Float::atan(self.val), 1.0 / q, -2.0 * self.val / (q * q))
    }

    /// Four-quadrant inverse tangent $\operatorname{atan2}(y, x)$ without branch-cut derivative jumps.
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn atan2(self, x: Self) -> Self {
        let val = Float::atan2(self.val, x.val);
        if self.val == 0.0 && x.val == 0.0 {
            return Self::constant(val);
        }
        let deriv_jet = if Float::abs(x.val) >= Float::abs(self.val) {
            (self / x).atan()
        } else {
            -(x / self).atan()
        };
        Self {
            val,
            grad: deriv_jet.grad,
            hess: deriv_jet.hess,
        }
    }

    /// Absolute value $|u|$ with centered Clarke subgradient ($0$ at $u = 0$).
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn abs(self) -> Self {
        if self.val.is_nan() {
            Self::constant(f64::NAN)
        } else if self.val > 0.0 {
            self
        } else if self.val < 0.0 {
            -self
        } else {
            Self::constant(0.0)
        }
    }

    /// Unit Heaviside step with Clarke subgradient $0$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn step(self) -> Self {
        if self.val.is_nan() {
            Self::constant(f64::NAN)
        } else {
            Self::constant(if self.val >= 0.0 { 1.0 } else { 0.0 })
        }
    }

    /// Unit rectangular boxcar pulse with Clarke subgradient $0$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn rect(self) -> Self {
        if self.val.is_nan() {
            Self::constant(f64::NAN)
        } else {
            Self::constant(if Float::abs(self.val) <= 0.5 {
                1.0
            } else {
                0.0
            })
        }
    }

    /// Minimum $\min(a, b)$ with midpoint Clarke subgradient at $a = b$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn min(self, other: Self) -> Self {
        if self.val.is_nan() || other.val.is_nan() {
            Self::constant(f64::NAN)
        } else if self.val < other.val {
            self
        } else if other.val < self.val {
            other
        } else {
            (self + other) * 0.5
        }
    }

    /// Maximum $\max(a, b)$ with midpoint Clarke subgradient at $a = b$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn max(self, other: Self) -> Self {
        if self.val.is_nan() || other.val.is_nan() {
            Self::constant(f64::NAN)
        } else if self.val > other.val {
            self
        } else if other.val > self.val {
            other
        } else {
            (self + other) * 0.5
        }
    }

    /// Clamping $\operatorname{clamp}(x, \text{lo}, \text{hi})$ with Clarke subgradients.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        if self.val.is_nan() || lo.val.is_nan() || hi.val.is_nan() || lo.val > hi.val {
            Self::constant(f64::NAN)
        } else {
            self.max(lo).min(hi)
        }
    }
}

impl Jet2<2> {
    /// First partial derivative $\partial f / \partial u$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn du(&self) -> f64 {
        self.grad[0]
    }

    /// First partial derivative $\partial f / \partial v$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn dv(&self) -> f64 {
        self.grad[1]
    }

    /// Second partial derivative $\partial^2 f / \partial u^2$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn duu(&self) -> f64 {
        self.hess[0][0]
    }

    /// Mixed second partial derivative $\partial^2 f / (\partial u \partial v)$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn duv(&self) -> f64 {
        self.hess[0][1]
    }

    /// Second partial derivative $\partial^2 f / \partial v^2$.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn dvv(&self) -> f64 {
        self.hess[1][1]
    }
}
