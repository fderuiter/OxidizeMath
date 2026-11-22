//! Orientation-preserving $\mathrm{SU}(1,1)$ Möbius transformations and disk projections.

use super::{HyperbolicError, HyperbolicModel, POINCARE_DISK_RADIUS_MAX};
use num_complex::Complex64;

/// Element $\begin{pmatrix} a & b \\ \bar{b} & \bar{a} \end{pmatrix} \in \mathrm{SU}(1,1)$
/// satisfying $|a|^2 - |b|^2 = 1$, acting isometrically on the Poincaré disk via
/// $z \mapsto \frac{a z + b}{\bar{b} z + \bar{a}}$.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::num_complex::Complex64;
/// use math_explorer::pure_math::differential_geometry::hyperbolic::Su11Matrix;
///
/// let m1 = Su11Matrix::hyperbolic_pan(Complex64::new(0.35, -0.2));
/// let m2 = Su11Matrix::rotation(0.75);
/// let prod = m1.compose(&m2);
/// assert!((prod.determinant() - 1.0).abs() < 1e-12);
///
/// let id = prod.compose(&prod.inverse());
/// let p = Complex64::new(0.25, 0.4);
/// assert!((id.apply(p) - p).norm() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su11Matrix {
    /// Upper-left complex coefficient $a$.
    pub a: Complex64,
    /// Upper-right complex coefficient $b$.
    pub b: Complex64,
}

impl Default for Su11Matrix {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::identity()
    }
}

impl Su11Matrix {
    /// Returns the identity transformation $I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn identity() -> Self {
        Self {
            a: Complex64::new(1.0, 0.0),
            b: Complex64::new(0.0, 0.0),
        }
    }

    /// Exports `[a.re, a.im, b.re, b.im]` for compact state serialization.
    #[must_use]
    #[verified_engine::verified]
    pub fn to_coeffs(&self) -> [f64; 4] {
        [self.a.re, self.a.im, self.b.re, self.b.im]
    }

    /// Reconstructs and normalizes an $\mathrm{SU}(1,1)$ matrix from `[a.re, a.im, b.re, b.im]`.
    #[verified_engine::verified]
    pub fn from_coeffs(coeffs: [f64; 4]) -> Result<Self, HyperbolicError> {
        Self::new(
            Complex64::new(coeffs[0], coeffs[1]),
            Complex64::new(coeffs[2], coeffs[3]),
        )
    }

    /// Constructs a normalized $\mathrm{SU}(1,1)$ matrix when $|a|^2 - |b|^2 > 0$ and finite.
    #[verified_engine::verified]
    pub fn new(a: Complex64, b: Complex64) -> Result<Self, HyperbolicError> {
        if !a.re.is_finite() || !a.im.is_finite() || !b.re.is_finite() || !b.im.is_finite() {
            return Err(HyperbolicError::DegenerateSu11Matrix {
                determinant: f64::NAN,
            });
        }
        let det = a.norm_sqr() - b.norm_sqr();
        if !det.is_finite() || det <= 1e-15 {
            return Err(HyperbolicError::DegenerateSu11Matrix { determinant: det });
        }
        if (det - 1.0).abs() <= 1e-13 {
            return Ok(Self { a, b });
        }
        let inv_scale = 1.0 / det.sqrt();
        Ok(Self {
            a: a * inv_scale,
            b: b * inv_scale,
        })
    }

    /// Normalizes $(a, b)$ onto $\mathrm{SU}(1,1)$, falling back to [`Self::identity`] if degenerate.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_unnormalized(a: Complex64, b: Complex64) -> Self {
        Self::new(a, b).unwrap_or_else(|_| Self::identity())
    }

    /// Computes the $\mathrm{SU}(1,1)$ pseudo-determinant $|a|^2 - |b|^2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn determinant(&self) -> f64 {
        self.a.norm_sqr() - self.b.norm_sqr()
    }

    /// Projects the matrix back onto the $|a|^2 - |b|^2 = 1$ hypersurface to eliminate float drift.
    #[verified_engine::verified]
    pub fn renormalize(&mut self) {
        let det = self.determinant();
        if !self.a.re.is_finite()
            || !self.a.im.is_finite()
            || !self.b.re.is_finite()
            || !self.b.im.is_finite()
            || !det.is_finite()
            || det <= 1e-15
        {
            *self = Self::identity();
            return;
        }
        if (det - 1.0).abs() <= 1e-13 {
            return;
        }
        let inv_scale = 1.0 / det.sqrt();
        self.a *= inv_scale;
        self.b *= inv_scale;
    }

    /// Group multiplication $M_1 \circ M_2$ in $\mathrm{SU}(1,1)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn compose(&self, rhs: &Self) -> Self {
        let a = self.a * rhs.a + self.b * rhs.b.conj();
        let b = self.a * rhs.b + self.b * rhs.a.conj();
        Self { a, b }
    }

    /// Exact group inverse $M^{-1} = \begin{pmatrix} \bar{a} & -b \\ -\bar{b} & a \end{pmatrix}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn inverse(&self) -> Self {
        Self {
            a: self.a.conj(),
            b: -self.b,
        }
    }

    /// Hyperbolic rotation around the disk origin by Euclidean/conformal angle `theta` radians.
    #[must_use]
    #[verified_engine::verified]
    pub fn rotation(theta: f64) -> Self {
        if !theta.is_finite() {
            return Self::identity();
        }
        let half = 0.5 * theta;
        Self {
            a: Complex64::new(half.cos(), half.sin()),
            b: Complex64::new(0.0, 0.0),
        }
    }

    /// Pure hyperbolic boost mapping `z0` to the disk origin $0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn translation_to_origin(z0: Complex64) -> Self {
        let clamped = clamp_to_disk(z0);
        let denom = (1.0 - clamped.norm_sqr()).max(1e-18).sqrt();
        let inv = 1.0 / denom;
        Self {
            a: Complex64::new(inv, 0.0),
            b: -clamped * inv,
        }
    }

    /// Pure hyperbolic boost mapping the disk origin $0$ to `z0`.
    #[must_use]
    #[verified_engine::verified]
    pub fn translation_from_origin(z0: Complex64) -> Self {
        let clamped = clamp_to_disk(z0);
        let denom = (1.0 - clamped.norm_sqr()).max(1e-18).sqrt();
        let inv = 1.0 / denom;
        Self {
            a: Complex64::new(inv, 0.0),
            b: clamped * inv,
        }
    }

    /// Smooth hyperbolic pan displacement whose intrinsic arc-length at the origin equals `|delta|`.
    #[must_use]
    #[verified_engine::verified]
    pub fn hyperbolic_pan(delta: Complex64) -> Self {
        if !delta.re.is_finite() || !delta.im.is_finite() {
            return Self::identity();
        }
        let mag = delta.norm();
        if mag <= 1e-15 {
            return Self::identity();
        }
        let r = (0.5 * mag).tanh().min(POINCARE_DISK_RADIUS_MAX);
        let target = delta * (r / mag);
        Self::translation_from_origin(target)
    }

    /// Orientation-preserving $\pi$-rotation around `midpoint` $m \in \mathbb{D}$ ($g^2 = -I$).
    #[must_use]
    #[verified_engine::verified]
    pub fn edge_half_turn(midpoint: Complex64) -> Self {
        let m = clamp_to_disk(midpoint);
        let m2 = m.norm_sqr();
        let inv = 1.0 / (1.0 - m2).max(1e-18);
        let a = Complex64::new(0.0, (1.0 + m2) * inv);
        let b = Complex64::new(0.0, -2.0 * inv) * m;
        Self::from_unnormalized(a, b)
    }

    /// Applies the Möbius transformation $z \mapsto \frac{a z + b}{\bar{b} z + \bar{a}}$, clamped to $\mathbb{D}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn apply(&self, z: Complex64) -> Complex64 {
        let zc = clamp_to_disk(z);
        let num = self.a * zc + self.b;
        let den = self.b.conj() * zc + self.a.conj();
        let den_norm_sqr = den.norm_sqr();
        if !den_norm_sqr.is_finite() || den_norm_sqr <= 1e-30 {
            return Complex64::new(0.0, 0.0);
        }
        clamp_to_disk(num / den)
    }

    /// Pushes forward a unit tangent vector `unit_dir` at `z` to the unit tangent at `self.apply(z)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn apply_direction(&self, z: Complex64, unit_dir: Complex64) -> Complex64 {
        let zc = clamp_to_disk(z);
        let dir_norm = unit_dir.norm();
        if !dir_norm.is_finite() || dir_norm <= 1e-15 {
            return Complex64::new(1.0, 0.0);
        }
        let u = unit_dir / dir_norm;
        let den = self.b.conj() * zc + self.a.conj();
        if den.norm_sqr() <= 1e-30 {
            return u;
        }
        let phase = den.conj() / den;
        let rotated = u * phase;
        let rn = rotated.norm();
        if rn > 1e-15 {
            rotated / rn
        } else {
            Complex64::new(1.0, 0.0)
        }
    }
}

/// Clamps any complex coordinate safely inside $|z| \le 1 - 10^{-9}$, sanitizing non-finite values.
#[must_use]
#[verified_engine::verified]
pub fn clamp_to_disk(z: Complex64) -> Complex64 {
    if !z.re.is_finite() || !z.im.is_finite() {
        return Complex64::new(0.0, 0.0);
    }
    let norm = z.norm();
    if norm > POINCARE_DISK_RADIUS_MAX {
        z * (POINCARE_DISK_RADIUS_MAX / norm)
    } else {
        z
    }
}

/// Maps a conformal Poincaré disk point $z \in \mathbb{D}$ to the Beltrami-Klein disk:
/// $w = \frac{2 z}{1 + |z|^2}$.
#[must_use]
#[verified_engine::verified]
pub fn poincare_to_klein(z: Complex64) -> Complex64 {
    let zc = clamp_to_disk(z);
    let w = (2.0 * zc) / (1.0 + zc.norm_sqr());
    clamp_to_disk(w)
}

/// Maps a projective Beltrami-Klein disk point $w \in \mathbb{K}$ to the Poincaré disk:
/// $z = \frac{w}{1 + \sqrt{1 - |w|^2}}$.
#[must_use]
#[verified_engine::verified]
pub fn klein_to_poincare(w: Complex64) -> Complex64 {
    let wc = clamp_to_disk(w);
    let radical = (1.0 - wc.norm_sqr()).max(0.0).sqrt();
    clamp_to_disk(wc / (1.0 + radical))
}

/// Projects an intrinsic Poincaré point into the requested disk model (`Poincare` or `Klein`).
#[must_use]
#[verified_engine::verified]
pub fn project_disk_point(z_poincare: Complex64, model: HyperbolicModel) -> Complex64 {
    match model {
        HyperbolicModel::Poincare => clamp_to_disk(z_poincare),
        HyperbolicModel::Klein => poincare_to_klein(z_poincare),
    }
}

/// Unprojects a screen/model point (`Poincare` or `Klein`) back to conformal Poincaré coordinates.
#[must_use]
#[verified_engine::verified]
pub fn unproject_disk_point(point: Complex64, model: HyperbolicModel) -> Complex64 {
    match model {
        HyperbolicModel::Poincare => clamp_to_disk(point),
        HyperbolicModel::Klein => klein_to_poincare(point),
    }
}

/// Inverse Cayley transform mapping $z \in \mathbb{D}$ to the upper half-plane $\tau = i \frac{1 + z}{1 - z} \in \mathbb{H}$.
#[must_use]
#[verified_engine::verified]
pub fn poincare_to_upper_half_plane(z: Complex64) -> Complex64 {
    let zc = clamp_to_disk(z);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    i * (one + zc) / (one - zc)
}

/// Cayley transform mapping $\tau \in \mathbb{H}$ ($\operatorname{Im}(\tau) > 0$) to $z = \frac{\tau - i}{\tau + i} \in \mathbb{D}$.
#[verified_engine::verified]
pub fn upper_half_plane_to_poincare(tau: Complex64) -> Result<Complex64, HyperbolicError> {
    if !tau.re.is_finite() || !tau.im.is_finite() || tau.im <= 0.0 {
        return Err(HyperbolicError::InvalidDomainPoint {
            reason: "upper half-plane coordinate requires finite Re(tau) and Im(tau) > 0",
        });
    }
    let i = Complex64::new(0.0, 1.0);
    Ok(clamp_to_disk((tau - i) / (tau + i)))
}

// [cite:modular_polynomials_review]
