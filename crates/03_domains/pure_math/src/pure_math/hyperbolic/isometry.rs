// [cite:pure_math]
mod psl2r;

pub use psl2r::*;

use super::{DiskPoint, HyperbolicError};
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

    /// Fallible companion to [`renormalize`](Self::renormalize).
    #[verified_engine::verified]
    pub fn try_renormalize(&mut self) -> core::result::Result<(), math_commons::error::MathError> {
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
        crate::error::ensure_math(
            (self.determinant() - 1.0).abs() < 1.0e-9,
            "renormalize: precondition or invariant violated",
        )?;
        debug_assert!((self.determinant() - 1.0).abs() < 1.0e-9);
        Ok(())
    }

    /// Renormalizes $(a, b)$ onto the hyperboloid $|a|^2 - |b|^2 = 1$, resetting to identity on numerical underflow.
    #[verified_engine::verified]
    pub fn renormalize(&mut self) {
        self.try_renormalize()
            .expect("renormalize preconditions must hold");
    }

    /// Fallible companion to [`rotation`](Self::rotation).
    #[verified_engine::verified]
    pub fn try_rotation(theta: f64) -> core::result::Result<Self, math_commons::error::MathError> {
        let angle = if theta.is_finite() { theta * 0.5 } else { 0.0 };
        let iso = Self::from_raw(
            Complex64::new(angle.cos(), angle.sin()),
            Complex64::new(0.0, 0.0),
        );
        crate::error::ensure_math(
            (iso.determinant() - 1.0).abs() < 1.0e-12,
            "rotation: precondition or invariant violated",
        )?;
        debug_assert!((iso.determinant() - 1.0).abs() < 1.0e-12);
        Ok(iso)
    }

    /// Constructs a pure elliptic rotation $f(z) = e^{i\theta} z$ around the origin.
    #[must_use]
    #[verified_engine::verified]
    pub fn rotation(theta: f64) -> Self {
        Self::try_rotation(theta).expect("rotation preconditions must hold")
    }

    /// Fallible companion to [`translation_to_origin`](Self::translation_to_origin).
    #[verified_engine::verified]
    pub fn try_translation_to_origin(
        p: DiskPoint,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
        let z0 = p.as_complex();
        let denom = (1.0 - z0.norm_sqr()).max(1.0e-15).sqrt();
        let inv = 1.0 / denom;
        let iso = Self::from_raw(Complex64::new(inv, 0.0), -z0 * inv);
        crate::error::ensure_math(
            (iso.determinant() - 1.0).abs() < 1.0e-9,
            "translation_to_origin: precondition or invariant violated",
        )?;
        debug_assert!((iso.determinant() - 1.0).abs() < 1.0e-9);
        Ok(iso)
    }

    /// Constructs the pure hyperbolic translation mapping `p` to the origin $0$: $T_{p \to 0}(z) = \frac{z - p}{1 - \bar{p} z}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn translation_to_origin(p: DiskPoint) -> Self {
        Self::try_translation_to_origin(p).expect("translation_to_origin preconditions must hold")
    }

    /// Fallible companion to [`translation_from_origin`](Self::translation_from_origin).
    #[verified_engine::verified]
    pub fn try_translation_from_origin(
        p: DiskPoint,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
        let z0 = p.as_complex();
        let denom = (1.0 - z0.norm_sqr()).max(1.0e-15).sqrt();
        let inv = 1.0 / denom;
        let iso = Self::from_raw(Complex64::new(inv, 0.0), z0 * inv);
        crate::error::ensure_math(
            (iso.determinant() - 1.0).abs() < 1.0e-9,
            "translation_from_origin: precondition or invariant violated",
        )?;
        debug_assert!((iso.determinant() - 1.0).abs() < 1.0e-9);
        Ok(iso)
    }

    /// Constructs the pure hyperbolic translation mapping the origin $0$ to `p`: $T_{0 \to p}(z) = \frac{z + p}{1 + \bar{p} z}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn translation_from_origin(p: DiskPoint) -> Self {
        Self::try_translation_from_origin(p)
            .expect("translation_from_origin preconditions must hold")
    }

    /// Fallible companion to [`drag_between`](Self::drag_between).
    #[verified_engine::verified]
    pub fn try_drag_between(
        from: DiskPoint,
        to: DiskPoint,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
        let to_orig = Self::translation_to_origin(from);
        let from_orig = Self::translation_from_origin(to);
        let mut composed = from_orig.compose(to_orig);
        composed.renormalize();
        crate::error::ensure_math(
            (composed.determinant() - 1.0).abs() < 1.0e-9,
            "drag_between: precondition or invariant violated",
        )?;
        debug_assert!((composed.determinant() - 1.0).abs() < 1.0e-9);
        Ok(composed)
    }

    /// Constructs a drag-isometry sending `from` to `to` via $T_{0 \to \text{to}} \circ T_{\text{from} \to 0}$ and renormalizing.
    #[must_use]
    #[verified_engine::verified]
    pub fn drag_between(from: DiskPoint, to: DiskPoint) -> Self {
        Self::try_drag_between(from, to).expect("drag_between preconditions must hold")
    }

    /// Fallible companion to [`compose`](Self::compose).
    #[verified_engine::verified]
    pub fn try_compose(
        self,
        rhs: Self,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
        let sa = self.a();
        let sb = self.b();
        let ra = rhs.a();
        let rb = rhs.b();
        let a = sa * ra + sb * rb.conj();
        let b = sa * rb + sb * ra.conj();
        let mut out = Self::from_raw(a, b);
        out.renormalize();
        crate::error::ensure_math(
            (out.determinant() - 1.0).abs() < 1.0e-9,
            "compose: precondition or invariant violated",
        )?;
        debug_assert!((out.determinant() - 1.0).abs() < 1.0e-9);
        Ok(out)
    }

    /// Returns the group composition `self ∘ rhs` applying `rhs` first and `self` second.
    #[must_use]
    #[verified_engine::verified]
    pub fn compose(self, rhs: Self) -> Self {
        self.try_compose(rhs)
            .expect("compose preconditions must hold")
    }

    /// Returns the exact group inverse $\begin{pmatrix} \bar{a} & -b \\ -\bar{b} & a \end{pmatrix}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn inverse(self) -> Self {
        let inv = Self::from_raw(self.a().conj(), -self.b());
        debug_assert!((inv.determinant() - self.determinant()).abs() < 1.0e-12);
        inv
    }

    /// Fallible companion to [`apply`](Self::apply).
    #[verified_engine::verified]
    pub fn try_apply(
        self,
        p: DiskPoint,
    ) -> core::result::Result<DiskPoint, math_commons::error::MathError> {
        let z = p.as_complex();
        let a = self.a();
        let b = self.b();
        let num = a * z + b;
        let den = b.conj() * z + a.conj();
        crate::error::ensure_math(
            den.norm_sqr() > 0.0,
            "apply: precondition or invariant violated",
        )?;
        let out = DiskPoint::from_complex_clamped(num / den);
        debug_assert!(out.norm_sq() < 1.0);
        Ok(out)
    }

    /// Applies the Möbius isometry $f(z) = \frac{a z + b}{\bar{b} z + \bar{a}}$ to an interior [`DiskPoint`], clamping $|f(z)| \le 1 - 10^{-6}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn apply(self, p: DiskPoint) -> DiskPoint {
        self.try_apply(p).expect("apply preconditions must hold")
    }

    /// Fallible companion to [`apply_complex`](Self::apply_complex).
    #[verified_engine::verified]
    pub fn try_apply_complex(
        self,
        z: Complex64,
    ) -> core::result::Result<Complex64, math_commons::error::MathError> {
        crate::error::ensure_math(
            z.re.is_finite() && z.im.is_finite(),
            "apply_complex: precondition or invariant violated",
        )?;
        let is_ideal = z.norm_sqr() >= 1.0 - 1.0e-12;
        let a = self.a();
        let b = self.b();
        let num = a * z + b;
        let den = b.conj() * z + a.conj();
        if den.norm_sqr() <= 1.0e-24 {
            return Ok(Complex64::new(1.0, 0.0));
        }
        let raw = num / den;
        let out = if is_ideal {
            let n = raw.re.hypot(raw.im).max(1.0e-15);
            Complex64::new(raw.re / n, raw.im / n)
        } else {
            DiskPoint::from_complex_clamped(raw).as_complex()
        };
        debug_assert!(out.re.is_finite() && out.im.is_finite());
        Ok(out)
    }

    /// Applies the Möbius transformation to a general point $z \in \overline{\mathbb{D}}$ (interior or ideal boundary $|z| = 1$).
    #[must_use]
    #[verified_engine::verified]
    pub fn apply_complex(self, z: Complex64) -> Complex64 {
        self.try_apply_complex(z)
            .expect("apply_complex preconditions must hold")
    }

    /// Fallible companion to [`to_psl2r`](Self::to_psl2r).
    #[verified_engine::verified]
    pub fn try_to_psl2r(self) -> core::result::Result<Psl2R, math_commons::error::MathError> {
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
        crate::error::ensure_math(
            (out.determinant() - 1.0).abs() < 1.0e-9,
            "to_psl2r: precondition or invariant violated",
        )?;
        debug_assert!((out.determinant() - 1.0).abs() < 1.0e-9);
        Ok(out)
    }

    /// Conjugates this `PSU(1,1)` isometry by the Cayley map $C(z) = i \frac{1+z}{1-z}$ into `PSL(2,R)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn to_psl2r(self) -> Psl2R {
        self.try_to_psl2r()
            .expect("to_psl2r preconditions must hold")
    }
}
