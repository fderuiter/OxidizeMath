// [cite:pure_math]
use super::{DiskPoint, HalfPlanePoint, HyperbolicError};
use num_complex::Complex64;

/// Orientation-preserving Möbius isometry of the Poincaré disk $\mathbb{D}$ in $PSU(1,1)$,
/// represented as $\begin{pmatrix} a & b \\ \bar{b} & \bar{a} \end{pmatrix}$ with $|a|^2 - |b|^2 = 1$.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Psu11 {
    a_re: f64,
    a_im: f64,
    b_re: f64,
    b_im: f64,
}

impl Default for Psu11 {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::identity()
    }
}

impl Psu11 {
    #[verified_engine::verified]
    fn from_raw(a: Complex64, b: Complex64) -> Self {
        debug_assert!(a.re.is_finite() && b.re.is_finite());
        Self {
            a_re: a.re,
            a_im: a.im,
            b_re: b.re,
            b_im: b.im,
        }
    }

    /// Returns the identity isometry $f(z) = z$.
    #[must_use]
    #[verified_engine::verified]
    pub fn identity() -> Self {
        let iso = Self::from_raw(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        debug_assert!((iso.determinant() - 1.0).abs() < 1.0e-14);
        iso
    }

    /// Constructs and normalizes a [`Psu11`] isometry from $(a, b)$ when $|a|^2 - |b|^2 > 0$.
    #[verified_engine::verified]
    pub fn new(a: Complex64, b: Complex64) -> Result<Self, HyperbolicError> {
        let det = a.norm_sqr() - b.norm_sqr();
        if !a.re.is_finite()
            || !a.im.is_finite()
            || !b.re.is_finite()
            || !b.im.is_finite()
            || !det.is_finite()
            || det <= 1.0e-15
        {
            return Err(HyperbolicError::DegenerateIsometry { determinant: det });
        }
        let inv_sqrt = 1.0 / det.sqrt();
        let iso = Self::from_raw(a * inv_sqrt, b * inv_sqrt);
        assert!((iso.determinant() - 1.0).abs() < 1.0e-10);
        Ok(iso)
    }

    /// Returns the complex entry $a$ of the $SU(1,1)$ representative.
    #[must_use]
    #[verified_engine::verified]
    pub fn a(self) -> Complex64 {
        debug_assert!(self.a_re.is_finite() && self.a_im.is_finite());
        Complex64::new(self.a_re, self.a_im)
    }

    /// Returns the complex entry $b$ of the $SU(1,1)$ representative.
    #[must_use]
    #[verified_engine::verified]
    pub fn b(self) -> Complex64 {
        debug_assert!(self.b_re.is_finite() && self.b_im.is_finite());
        Complex64::new(self.b_re, self.b_im)
    }

    /// Evaluates the $SU(1,1)$ determinant $|a|^2 - |b|^2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn determinant(self) -> f64 {
        let det = self.a().norm_sqr() - self.b().norm_sqr();
        debug_assert!(det.is_finite());
        det
    }

    /// Renormalizes $(a, b)$ onto the hyperboloid $|a|^2 - |b|^2 = 1$, resetting to identity on numerical underflow.
    #[verified_engine::verified]
    pub fn renormalize(&mut self) {
        let det = self.determinant();
        if !det.is_finite() || det <= 1.0e-12 {
            *self = Self::identity();
        } else {
            let inv_sqrt = 1.0 / det.sqrt();
            self.a_re *= inv_sqrt;
            self.a_im *= inv_sqrt;
            self.b_re *= inv_sqrt;
            self.b_im *= inv_sqrt;
        }
        assert!((self.determinant() - 1.0).abs() < 1.0e-9);
    }

    /// Constructs a pure elliptic rotation $f(z) = e^{i\theta} z$ around the origin.
    #[must_use]
    #[verified_engine::verified]
    pub fn rotation(theta: f64) -> Self {
        let angle = if theta.is_finite() { theta * 0.5 } else { 0.0 };
        let iso = Self::from_raw(
            Complex64::new(angle.cos(), angle.sin()),
            Complex64::new(0.0, 0.0),
        );
        assert!((iso.determinant() - 1.0).abs() < 1.0e-12);
        iso
    }

    /// Constructs the pure hyperbolic translation mapping `p` to the origin $0$: $T_{p \to 0}(z) = \frac{z - p}{1 - \bar{p} z}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn translation_to_origin(p: DiskPoint) -> Self {
        let z0 = p.as_complex();
        let denom = (1.0 - z0.norm_sqr()).max(1.0e-15).sqrt();
        let inv = 1.0 / denom;
        let iso = Self::from_raw(Complex64::new(inv, 0.0), -z0 * inv);
        assert!((iso.determinant() - 1.0).abs() < 1.0e-9);
        iso
    }

    /// Constructs the pure hyperbolic translation mapping the origin $0$ to `p`: $T_{0 \to p}(z) = \frac{z + p}{1 + \bar{p} z}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn translation_from_origin(p: DiskPoint) -> Self {
        let z0 = p.as_complex();
        let denom = (1.0 - z0.norm_sqr()).max(1.0e-15).sqrt();
        let inv = 1.0 / denom;
        let iso = Self::from_raw(Complex64::new(inv, 0.0), z0 * inv);
        assert!((iso.determinant() - 1.0).abs() < 1.0e-9);
        iso
    }

    /// Constructs a drag-isometry sending `from` to `to` via $T_{0 \to \text{to}} \circ T_{\text{from} \to 0}$ and renormalizing.
    #[must_use]
    #[verified_engine::verified]
    pub fn drag_between(from: DiskPoint, to: DiskPoint) -> Self {
        let to_orig = Self::translation_to_origin(from);
        let from_orig = Self::translation_from_origin(to);
        let mut composed = from_orig.compose(to_orig);
        composed.renormalize();
        assert!((composed.determinant() - 1.0).abs() < 1.0e-9);
        composed
    }

    /// Returns the group composition `self ∘ rhs` applying `rhs` first and `self` second.
    #[must_use]
    #[verified_engine::verified]
    pub fn compose(self, rhs: Self) -> Self {
        let sa = self.a();
        let sb = self.b();
        let ra = rhs.a();
        let rb = rhs.b();
        let a = sa * ra + sb * rb.conj();
        let b = sa * rb + sb * ra.conj();
        let mut out = Self::from_raw(a, b);
        out.renormalize();
        assert!((out.determinant() - 1.0).abs() < 1.0e-9);
        out
    }

    /// Returns the exact group inverse $\begin{pmatrix} \bar{a} & -b \\ -\bar{b} & a \end{pmatrix}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn inverse(self) -> Self {
        let inv = Self::from_raw(self.a().conj(), -self.b());
        debug_assert!((inv.determinant() - self.determinant()).abs() < 1.0e-12);
        inv
    }

    /// Applies the Möbius isometry $f(z) = \frac{a z + b}{\bar{b} z + \bar{a}}$ to an interior [`DiskPoint`], clamping $|f(z)| \le 1 - 10^{-6}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn apply(self, p: DiskPoint) -> DiskPoint {
        let z = p.as_complex();
        let a = self.a();
        let b = self.b();
        let num = a * z + b;
        let den = b.conj() * z + a.conj();
        assert!(den.norm_sqr() > 0.0);
        DiskPoint::from_complex_clamped(num / den)
    }

    /// Applies the Möbius transformation to a general point $z \in \overline{\mathbb{D}}$ (interior or ideal boundary $|z| = 1$).
    #[must_use]
    #[verified_engine::verified]
    pub fn apply_complex(self, z: Complex64) -> Complex64 {
        assert!(z.re.is_finite() && z.im.is_finite());
        let is_ideal = z.norm_sqr() >= 1.0 - 1.0e-12;
        let a = self.a();
        let b = self.b();
        let num = a * z + b;
        let den = b.conj() * z + a.conj();
        if den.norm_sqr() <= 1.0e-24 {
            return Complex64::new(1.0, 0.0);
        }
        let raw = num / den;
        if is_ideal {
            let n = raw.re.hypot(raw.im).max(1.0e-15);
            Complex64::new(raw.re / n, raw.im / n)
        } else {
            DiskPoint::from_complex_clamped(raw).as_complex()
        }
    }

    /// Conjugates this `PSU(1,1)` isometry by the Cayley map $C(z) = i \frac{1+z}{1-z}$ into `PSL(2,R)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn to_psl2r(self) -> Psl2R {
        let alpha = self.a_re + self.b_re;
        let beta = self.a_im - self.b_im;
        let gamma = -(self.a_im + self.b_im);
        let delta = self.a_re - self.b_re;
        let mut out = Psl2R {
            alpha,
            beta,
            gamma,
            delta,
        };
        out.renormalize();
        assert!((out.determinant() - 1.0).abs() < 1.0e-9);
        out
    }
}

/// Orientation-preserving Möbius isometry of the upper half-plane $\mathbb{H}$ in $PSL(2,\mathbb{R})$,
/// represented as $\begin{pmatrix} \alpha & \beta \\ \gamma & \delta \end{pmatrix}$ with $\alpha\delta - \beta\gamma = 1$.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Psl2R {
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
}

impl Default for Psl2R {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::identity()
    }
}

impl Psl2R {
    /// Returns the identity transformation $g(w) = w$.
    #[must_use]
    #[verified_engine::verified]
    pub fn identity() -> Self {
        let iso = Self {
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            delta: 1.0,
        };
        debug_assert!((iso.determinant() - 1.0).abs() < 1.0e-14);
        iso
    }

    /// Constructs and normalizes a [`Psl2R`] matrix when $\alpha\delta - \beta\gamma > 0$.
    #[verified_engine::verified]
    pub fn new(alpha: f64, beta: f64, gamma: f64, delta: f64) -> Result<Self, HyperbolicError> {
        let det = alpha * delta - beta * gamma;
        if !alpha.is_finite()
            || !beta.is_finite()
            || !gamma.is_finite()
            || !delta.is_finite()
            || !det.is_finite()
            || det <= 1.0e-15
        {
            return Err(HyperbolicError::DegenerateIsometry { determinant: det });
        }
        let inv_sqrt = 1.0 / det.sqrt();
        let iso = Self {
            alpha: alpha * inv_sqrt,
            beta: beta * inv_sqrt,
            gamma: gamma * inv_sqrt,
            delta: delta * inv_sqrt,
        };
        assert!((iso.determinant() - 1.0).abs() < 1.0e-10);
        Ok(iso)
    }

    /// Returns the entries `[alpha, beta, gamma, delta]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn entries(self) -> [f64; 4] {
        debug_assert!(self.alpha.is_finite() && self.delta.is_finite());
        [self.alpha, self.beta, self.gamma, self.delta]
    }

    /// Evaluates $\alpha\delta - \beta\gamma$.
    #[must_use]
    #[verified_engine::verified]
    pub fn determinant(self) -> f64 {
        let det = self.alpha * self.delta - self.beta * self.gamma;
        debug_assert!(det.is_finite());
        det
    }

    /// Renormalizes the matrix to unit determinant $\alpha\delta - \beta\gamma = 1$.
    #[verified_engine::verified]
    pub fn renormalize(&mut self) {
        let det = self.determinant();
        if !det.is_finite() || det <= 1.0e-12 {
            *self = Self::identity();
        } else {
            let inv_sqrt = 1.0 / det.sqrt();
            self.alpha *= inv_sqrt;
            self.beta *= inv_sqrt;
            self.gamma *= inv_sqrt;
            self.delta *= inv_sqrt;
        }
        assert!((self.determinant() - 1.0).abs() < 1.0e-9);
    }

    /// Constructs a parabolic horizontal translation $g(w) = w + \Delta u$.
    #[must_use]
    #[verified_engine::verified]
    pub fn translation(du: f64) -> Self {
        let shift = if du.is_finite() { du } else { 0.0 };
        let iso = Self {
            alpha: 1.0,
            beta: shift,
            gamma: 0.0,
            delta: 1.0,
        };
        assert!((iso.determinant() - 1.0).abs() < 1.0e-14);
        iso
    }

    /// Constructs a hyperbolic dilation $g(w) = \lambda w$ for $\lambda > 0$.
    #[verified_engine::verified]
    pub fn dilation(lambda: f64) -> Result<Self, HyperbolicError> {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(HyperbolicError::DegenerateIsometry {
                determinant: lambda,
            });
        }
        let s = lambda.sqrt();
        let iso = Self {
            alpha: s,
            beta: 0.0,
            gamma: 0.0,
            delta: 1.0 / s,
        };
        assert!((iso.determinant() - 1.0).abs() < 1.0e-10);
        Ok(iso)
    }

    /// Constructs an affine `PSL(2,R)` drag isometry mapping `from` to `to` via $w \mapsto \frac{v_2}{v_1}(w - u_1) + u_2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn drag_between(from: HalfPlanePoint, to: HalfPlanePoint) -> Self {
        let ratio = (to.im() / from.im()).clamp(1.0e-4, 1.0e4);
        let s = ratio.sqrt();
        let inv_s = 1.0 / s;
        let beta = to.re() * inv_s - s * from.re();
        let mut iso = Self {
            alpha: s,
            beta,
            gamma: 0.0,
            delta: inv_s,
        };
        iso.renormalize();
        assert!((iso.determinant() - 1.0).abs() < 1.0e-9);
        iso
    }

    /// Returns the group composition `self ∘ rhs`.
    #[must_use]
    #[verified_engine::verified]
    pub fn compose(self, rhs: Self) -> Self {
        let mut out = Self {
            alpha: self.alpha * rhs.alpha + self.beta * rhs.gamma,
            beta: self.alpha * rhs.beta + self.beta * rhs.delta,
            gamma: self.gamma * rhs.alpha + self.delta * rhs.gamma,
            delta: self.gamma * rhs.beta + self.delta * rhs.delta,
        };
        out.renormalize();
        assert!((out.determinant() - 1.0).abs() < 1.0e-9);
        out
    }

    /// Returns the group inverse $\begin{pmatrix} \delta & -\beta \\ -\gamma & \alpha \end{pmatrix}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn inverse(self) -> Self {
        let inv = Self {
            alpha: self.delta,
            beta: -self.beta,
            gamma: -self.gamma,
            delta: self.alpha,
        };
        debug_assert!((inv.determinant() - self.determinant()).abs() < 1.0e-12);
        inv
    }

    /// Applies $g(w) = \frac{\alpha w + \beta}{\gamma w + \delta}$ to a [`HalfPlanePoint`].
    #[must_use]
    #[verified_engine::verified]
    pub fn apply(self, q: HalfPlanePoint) -> HalfPlanePoint {
        let w = q.as_complex();
        let num = Complex64::new(self.alpha * w.re + self.beta, self.alpha * w.im);
        let den = Complex64::new(self.gamma * w.re + self.delta, self.gamma * w.im);
        assert!(den.norm_sqr() > 0.0);
        let out = num / den;
        HalfPlanePoint::new_clamped(out.re, out.im)
    }

    /// Conjugates this `PSL(2,R)` isometry back to `PSU(1,1)` via $C^{-1} M C$.
    #[must_use]
    #[verified_engine::verified]
    pub fn to_psu11(self) -> Psu11 {
        let a = Complex64::new(
            0.5 * (self.alpha + self.delta),
            0.5 * (self.beta - self.gamma),
        );
        let b = Complex64::new(
            0.5 * (self.alpha - self.delta),
            -0.5 * (self.beta + self.gamma),
        );
        let mut out = Psu11::from_raw(a, b);
        out.renormalize();
        assert!((out.determinant() - 1.0).abs() < 1.0e-9);
        out
    }
}

/// Orientation-reversing Coxeter reflection across a Poincaré disk geodesic (diameter or orthogonal circle).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CircleInversion {
    /// Reflection across a diameter with unit tangent $u = e^{i\alpha}$: $\iota(z) = u^2 \bar{z}$.
    Diameter {
        /// Unit direction vector $u$ ($|u| = 1$) of the diameter.
        unit_dir: Complex64,
    },
    /// Inversion across an orthogonal circle $|z - c|^2 = R^2$ ($|c|^2 - R^2 = 1$): $\iota(z) = \frac{c \bar{z} - 1}{\bar{z} - \bar{c}}$.
    OrthogonalCircle {
        /// Euclidean center $c$ ($|c| > 1$).
        center: Complex64,
        /// Euclidean radius $R = \sqrt{|c|^2 - 1}$.
        radius: f64,
    },
}

impl CircleInversion {
    /// Reflects a point $z \in \overline{\mathbb{D}}$ across this geodesic mirror.
    #[must_use]
    #[verified_engine::verified]
    pub fn reflect_complex(self, z: Complex64) -> Complex64 {
        assert!(z.re.is_finite() && z.im.is_finite());
        let is_ideal = z.norm_sqr() >= 1.0 - 1.0e-12;
        let raw = match self {
            Self::Diameter { unit_dir } => unit_dir * unit_dir * z.conj(),
            Self::OrthogonalCircle { center, .. } => {
                let zc = z.conj();
                let num = center * zc - Complex64::new(1.0, 0.0);
                let den = zc - center.conj();
                if den.norm_sqr() <= 1.0e-24 {
                    center / center.re.hypot(center.im).max(1.0)
                } else {
                    num / den
                }
            }
        };
        if is_ideal {
            let n = raw.re.hypot(raw.im).max(1.0e-15);
            Complex64::new(raw.re / n, raw.im / n)
        } else {
            DiskPoint::from_complex_clamped(raw).as_complex()
        }
    }

    /// Reflects a validated interior [`DiskPoint`] across this geodesic mirror.
    #[must_use]
    #[verified_engine::verified]
    pub fn reflect_disk_point(self, p: DiskPoint) -> DiskPoint {
        let out = DiskPoint::from_complex_clamped(self.reflect_complex(p.as_complex()));
        debug_assert!(out.norm_sq() < 1.0);
        out
    }
}
