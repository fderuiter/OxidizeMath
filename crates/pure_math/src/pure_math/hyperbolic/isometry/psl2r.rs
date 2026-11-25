use super::super::HalfPlanePoint;
use super::*;

/// Orientation-preserving Möbius isometry of the upper half-plane $\mathbb{H}$ in $PSL(2,\mathbb{R})$,
/// represented as $\begin{pmatrix} \alpha & \beta \\ \gamma & \delta \end{pmatrix}$ with $\alpha\delta - \beta\gamma = 1$.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Psl2R {
    pub(super) alpha: f64,
    pub(super) beta: f64,
    pub(super) gamma: f64,
    pub(super) delta: f64,
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

    /// Fallible companion to [`renormalize`](Self::renormalize).
    #[verified_engine::verified]
    pub fn try_renormalize(&mut self) -> core::result::Result<(), math_commons::error::MathError> {
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
        __ensure_math(
            (self.determinant() - 1.0).abs() < 1.0e-9,
            "renormalize: precondition or invariant violated",
        )?;
        debug_assert!((self.determinant() - 1.0).abs() < 1.0e-9);
        Ok(())
    }

    /// Renormalizes the matrix to unit determinant $\alpha\delta - \beta\gamma = 1$.
    #[verified_engine::verified]
    pub fn renormalize(&mut self) {
        self.try_renormalize()
            .expect("renormalize preconditions must hold");
    }

    /// Fallible companion to [`translation`](Self::translation).
    #[verified_engine::verified]
    pub fn try_translation(du: f64) -> core::result::Result<Self, math_commons::error::MathError> {
        let shift = if du.is_finite() { du } else { 0.0 };
        let iso = Self {
            alpha: 1.0,
            beta: shift,
            gamma: 0.0,
            delta: 1.0,
        };
        __ensure_math(
            (iso.determinant() - 1.0).abs() < 1.0e-14,
            "translation: precondition or invariant violated",
        )?;
        debug_assert!((iso.determinant() - 1.0).abs() < 1.0e-14);
        Ok(iso)
    }

    /// Constructs a parabolic horizontal translation $g(w) = w + \Delta u$.
    #[must_use]
    #[verified_engine::verified]
    pub fn translation(du: f64) -> Self {
        Self::try_translation(du).expect("translation preconditions must hold")
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

    /// Fallible companion to [`drag_between`](Self::drag_between).
    #[verified_engine::verified]
    pub fn try_drag_between(
        from: HalfPlanePoint,
        to: HalfPlanePoint,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
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
        __ensure_math(
            (iso.determinant() - 1.0).abs() < 1.0e-9,
            "drag_between: precondition or invariant violated",
        )?;
        debug_assert!((iso.determinant() - 1.0).abs() < 1.0e-9);
        Ok(iso)
    }

    /// Constructs an affine `PSL(2,R)` drag isometry mapping `from` to `to` via $w \mapsto \frac{v_2}{v_1}(w - u_1) + u_2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn drag_between(from: HalfPlanePoint, to: HalfPlanePoint) -> Self {
        Self::try_drag_between(from, to).expect("drag_between preconditions must hold")
    }

    /// Fallible companion to [`compose`](Self::compose).
    #[verified_engine::verified]
    pub fn try_compose(
        self,
        rhs: Self,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
        let mut out = Self {
            alpha: self.alpha * rhs.alpha + self.beta * rhs.gamma,
            beta: self.alpha * rhs.beta + self.beta * rhs.delta,
            gamma: self.gamma * rhs.alpha + self.delta * rhs.gamma,
            delta: self.gamma * rhs.beta + self.delta * rhs.delta,
        };
        out.renormalize();
        __ensure_math(
            (out.determinant() - 1.0).abs() < 1.0e-9,
            "compose: precondition or invariant violated",
        )?;
        debug_assert!((out.determinant() - 1.0).abs() < 1.0e-9);
        Ok(out)
    }

    /// Returns the group composition `self ∘ rhs`.
    #[must_use]
    #[verified_engine::verified]
    pub fn compose(self, rhs: Self) -> Self {
        self.try_compose(rhs)
            .expect("compose preconditions must hold")
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

    /// Fallible companion to [`apply`](Self::apply).
    #[verified_engine::verified]
    pub fn try_apply(
        self,
        q: HalfPlanePoint,
    ) -> core::result::Result<HalfPlanePoint, math_commons::error::MathError> {
        let w = q.as_complex();
        let num = Complex64::new(self.alpha * w.re + self.beta, self.alpha * w.im);
        let den = Complex64::new(self.gamma * w.re + self.delta, self.gamma * w.im);
        __ensure_math(
            den.norm_sqr() > 0.0,
            "apply: precondition or invariant violated",
        )?;
        let out = num / den;
        let clamped = HalfPlanePoint::new_clamped(out.re, out.im);
        debug_assert!(clamped.im() > 0.0);
        Ok(clamped)
    }

    /// Applies $g(w) = \frac{\alpha w + \beta}{\gamma w + \delta}$ to a [`HalfPlanePoint`].
    #[must_use]
    #[verified_engine::verified]
    pub fn apply(self, q: HalfPlanePoint) -> HalfPlanePoint {
        self.try_apply(q).expect("apply preconditions must hold")
    }

    /// Fallible companion to [`to_psu11`](Self::to_psu11).
    #[verified_engine::verified]
    pub fn try_to_psu11(self) -> core::result::Result<Psu11, math_commons::error::MathError> {
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
        __ensure_math(
            (out.determinant() - 1.0).abs() < 1.0e-9,
            "to_psu11: precondition or invariant violated",
        )?;
        debug_assert!((out.determinant() - 1.0).abs() < 1.0e-9);
        Ok(out)
    }

    /// Conjugates this `PSL(2,R)` isometry back to `PSU(1,1)` via $C^{-1} M C$.
    #[must_use]
    #[verified_engine::verified]
    pub fn to_psu11(self) -> Psu11 {
        self.try_to_psu11()
            .expect("to_psu11 preconditions must hold")
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
    /// Fallible companion to [`reflect_complex`](Self::reflect_complex).
    #[verified_engine::verified]
    pub fn try_reflect_complex(
        self,
        z: Complex64,
    ) -> core::result::Result<Complex64, math_commons::error::MathError> {
        __ensure_math(
            z.re.is_finite() && z.im.is_finite(),
            "reflect_complex: precondition or invariant violated",
        )?;
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
        let out = if is_ideal {
            let n = raw.re.hypot(raw.im).max(1.0e-15);
            Complex64::new(raw.re / n, raw.im / n)
        } else {
            DiskPoint::from_complex_clamped(raw).as_complex()
        };
        debug_assert!(out.re.is_finite() && out.im.is_finite());
        Ok(out)
    }

    /// Reflects a point $z \in \overline{\mathbb{D}}$ across this geodesic mirror.
    #[must_use]
    #[verified_engine::verified]
    pub fn reflect_complex(self, z: Complex64) -> Complex64 {
        self.try_reflect_complex(z)
            .expect("reflect_complex preconditions must hold")
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
