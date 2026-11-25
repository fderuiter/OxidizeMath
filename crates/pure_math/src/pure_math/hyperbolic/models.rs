// [cite:pure_math]
use super::HyperbolicError;
use num_complex::Complex64;

/// Maximum Euclidean modulus permitted for clamped Poincaré disk points ($1 - 10^{-6}$).
pub const MAX_DISK_MODULUS: f64 = 1.0 - 1.0e-6;

/// Minimum imaginary part enforced when clamping points into the upper half-plane $\mathbb{H}$.
const MIN_HALF_PLANE_IMAG: f64 = 1.0e-6;

/// Maximum coordinate magnitude enforced when clamping points into the upper half-plane $\mathbb{H}$.
const MAX_HALF_PLANE_COORD: f64 = 1.0e6;

/// Validated point $z = x + i y$ inside the open unit Poincaré disk $\mathbb{D} = \{z \in \mathbb{C} : |z| < 1\}$.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiskPoint {
    re: f64,
    im: f64,
}

impl Default for DiskPoint {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::origin()
    }
}

impl DiskPoint {
    /// Returns the origin $z = 0$ of the Poincaré disk.
    #[must_use]
    #[verified_engine::verified]
    pub fn origin() -> Self {
        let p = Self { re: 0.0, im: 0.0 };
        debug_assert!(p.norm_sq() < 1.0);
        debug_assert!(p.re.is_finite() && p.im.is_finite());
        p
    }

    /// Constructs a validated [`DiskPoint`] if `re` and `im` are finite and $re^2 + im^2 < 1$.
    #[verified_engine::verified]
    pub fn new(re: f64, im: f64) -> Result<Self, HyperbolicError> {
        let norm_sq = re * re + im * im;
        if !re.is_finite() || !im.is_finite() || !norm_sq.is_finite() || norm_sq >= 1.0 {
            return Err(HyperbolicError::InvalidDiskPoint { re, im });
        }
        let p = Self { re, im };
        assert!(p.norm_sq() < 1.0);
        assert!(p.re.is_finite() && p.im.is_finite());
        Ok(p)
    }

    /// Constructs a validated [`DiskPoint`] from a complex coordinate $z$.
    #[verified_engine::verified]
    pub fn from_complex(z: Complex64) -> Result<Self, HyperbolicError> {
        let res = Self::new(z.re, z.im);
        debug_assert!(res.is_err() || z.norm_sqr() < 1.0);
        res
    }

    /// Fallible companion to [`new_clamped`](Self::new_clamped).
    #[verified_engine::verified]
    pub fn try_new_clamped(
        re: f64,
        im: f64,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
        if !re.is_finite() || !im.is_finite() {
            return Ok(Self::origin());
        }
        let norm = re.hypot(im);
        let z = if !norm.is_finite() {
            Complex64::new(0.0, 0.0)
        } else if norm > MAX_DISK_MODULUS {
            let scale = MAX_DISK_MODULUS / norm;
            Complex64::new(re * scale, im * scale)
        } else {
            Complex64::new(re, im)
        };
        __ensure_math(
            z.re.is_finite() && z.im.is_finite() && z.norm_sqr() < 1.0,
            "new_clamped: precondition or invariant violated",
        )?;
        debug_assert!(z.norm_sqr() < 1.0);
        Ok(Self { re: z.re, im: z.im })
    }

    /// Constructs a [`DiskPoint`] clamping $|z| \le 1 - 10^{-6}$ and mapping non-finite inputs to $0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn new_clamped(re: f64, im: f64) -> Self {
        Self::try_new_clamped(re, im).expect("new_clamped preconditions must hold")
    }

    /// Constructs a [`DiskPoint`] from `z`, clamping $|z| \le 1 - 10^{-6}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_complex_clamped(z: Complex64) -> Self {
        let pt = Self::new_clamped(z.re, z.im);
        debug_assert!(pt.norm_sq() < 1.0);
        debug_assert!(pt.re.is_finite());
        pt
    }

    /// Returns the underlying complex coordinate $z \in \mathbb{D}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn as_complex(self) -> Complex64 {
        debug_assert!(self.re.is_finite() && self.im.is_finite());
        debug_assert!(self.norm_sq() < 1.0);
        Complex64::new(self.re, self.im)
    }

    /// Returns the real coordinate $\operatorname{Re}(z)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn re(self) -> f64 {
        debug_assert!(self.re.is_finite());
        debug_assert!(self.norm_sq() < 1.0);
        self.re
    }

    /// Returns the imaginary coordinate $\operatorname{Im}(z)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn im(self) -> f64 {
        debug_assert!(self.im.is_finite());
        debug_assert!(self.norm_sq() < 1.0);
        self.im
    }

    /// Returns the squared Euclidean modulus $|z|^2 < 1$.
    #[must_use]
    #[verified_engine::verified]
    pub fn norm_sq(self) -> f64 {
        let n2 = self.re * self.re + self.im * self.im;
        debug_assert!(n2.is_finite() && n2 < 1.0);
        n2
    }

    /// Returns the Euclidean modulus $|z| < 1$.
    #[must_use]
    #[verified_engine::verified]
    pub fn modulus(self) -> f64 {
        let m = self.re.hypot(self.im);
        debug_assert!(m.is_finite() && m < 1.0);
        m
    }
}

/// Validated point $w = u + i v$ in the Poincaré upper half-plane $\mathbb{H} = \{w \in \mathbb{C} : \operatorname{Im}(w) > 0\}$.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HalfPlanePoint {
    re: f64,
    im: f64,
}

impl Default for HalfPlanePoint {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::base_i()
    }
}

impl HalfPlanePoint {
    /// Returns the canonical basepoint $w = i \in \mathbb{H}$ (image of $z = 0$ under the Cayley map).
    #[must_use]
    #[verified_engine::verified]
    pub fn base_i() -> Self {
        let q = Self { re: 0.0, im: 1.0 };
        debug_assert!(q.im > 0.0);
        debug_assert!(q.re.is_finite());
        q
    }

    /// Constructs a validated [`HalfPlanePoint`] if `re` and `im` are finite and `im > 0`.
    #[verified_engine::verified]
    pub fn new(re: f64, im: f64) -> Result<Self, HyperbolicError> {
        if !re.is_finite() || !im.is_finite() || im <= 0.0 {
            return Err(HyperbolicError::InvalidHalfPlanePoint { re, im });
        }
        let q = Self { re, im };
        assert!(q.im > 0.0);
        assert!(q.re.is_finite());
        Ok(q)
    }

    /// Constructs a validated [`HalfPlanePoint`] from a complex coordinate $w$.
    #[verified_engine::verified]
    pub fn from_complex(w: Complex64) -> Result<Self, HyperbolicError> {
        let res = Self::new(w.re, w.im);
        debug_assert!(res.is_err() || w.im > 0.0);
        res
    }

    /// Fallible companion to [`new_clamped`](Self::new_clamped).
    #[verified_engine::verified]
    pub fn try_new_clamped(
        re: f64,
        im: f64,
    ) -> core::result::Result<Self, math_commons::error::MathError> {
        let u = if re.is_finite() {
            re.clamp(-MAX_HALF_PLANE_COORD, MAX_HALF_PLANE_COORD)
        } else {
            0.0
        };
        let v = if im.is_finite() {
            im.clamp(MIN_HALF_PLANE_IMAG, MAX_HALF_PLANE_COORD)
        } else {
            1.0
        };
        __ensure_math(
            u.is_finite() && v >= MIN_HALF_PLANE_IMAG,
            "new_clamped: precondition or invariant violated",
        )?;
        debug_assert!(u.is_finite() && v >= MIN_HALF_PLANE_IMAG);
        Ok(Self { re: u, im: v })
    }

    /// Constructs a [`HalfPlanePoint`] clamping coordinates to a finite upper half-plane region.
    #[must_use]
    #[verified_engine::verified]
    pub fn new_clamped(re: f64, im: f64) -> Self {
        Self::try_new_clamped(re, im).expect("new_clamped preconditions must hold")
    }

    /// Returns the underlying complex coordinate $w \in \mathbb{H}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn as_complex(self) -> Complex64 {
        debug_assert!(self.re.is_finite() && self.im > 0.0);
        Complex64::new(self.re, self.im)
    }

    /// Returns $\operatorname{Re}(w)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn re(self) -> f64 {
        debug_assert!(self.re.is_finite());
        debug_assert!(self.im > 0.0);
        self.re
    }

    /// Returns $\operatorname{Im}(w) > 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn im(self) -> f64 {
        debug_assert!(self.im > 0.0);
        debug_assert!(self.re.is_finite());
        self.im
    }
}

/// Fallible companion to [`cayley_disk_to_half_plane`].
#[verified_engine::verified]
pub fn try_cayley_disk_to_half_plane(
    p: DiskPoint,
) -> core::result::Result<HalfPlanePoint, math_commons::error::MathError> {
    #[allow(unused_braces)]
    Ok({
        let z = p.as_complex();
        let dx = 1.0 - z.re;
        let denom = dx * dx + z.im * z.im;
        __ensure_math(
            denom > 0.0 && denom.is_finite(),
            "cayley_disk_to_half_plane: precondition or invariant violated",
        )?;
        let re = (-2.0 * z.im) / denom;
        let im = ((1.0 - z.norm_sqr()) / denom).max(f64::MIN_POSITIVE);
        __ensure_math(
            re.is_finite() && im > 0.0,
            "cayley_disk_to_half_plane: precondition or invariant violated",
        )?;
        debug_assert!(re.is_finite() && im > 0.0);
        HalfPlanePoint { re, im }
    })
}

/// Maps a [`DiskPoint`] $z \in \mathbb{D}$ isometrically to $\mathbb{H}$ via the Cayley map $w = i \frac{1 + z}{1 - z}$.
#[must_use]
#[verified_engine::verified]
pub fn cayley_disk_to_half_plane(p: DiskPoint) -> HalfPlanePoint {
    try_cayley_disk_to_half_plane(p).expect("cayley_disk_to_half_plane preconditions must hold")
}

/// Fallible companion to [`cayley_half_plane_to_disk`].
#[verified_engine::verified]
pub fn try_cayley_half_plane_to_disk(
    q: HalfPlanePoint,
) -> core::result::Result<DiskPoint, math_commons::error::MathError> {
    #[allow(unused_braces)]
    Ok({
        let w = q.as_complex();
        let vp1 = w.im + 1.0;
        let denom = w.re * w.re + vp1 * vp1;
        __ensure_math(
            denom > 0.0 && denom.is_finite(),
            "cayley_half_plane_to_disk: precondition or invariant violated",
        )?;
        let re = (w.re * w.re + w.im * w.im - 1.0) / denom;
        let im = (-2.0 * w.re) / denom;
        let z = Complex64::new(re, im);
        if z.norm_sqr() < 1.0 {
            __ensure_math(
                z.re.is_finite() && z.im.is_finite(),
                "cayley_half_plane_to_disk: precondition or invariant violated",
            )?;
            debug_assert!(z.norm_sqr() < 1.0);
            DiskPoint { re: z.re, im: z.im }
        } else {
            DiskPoint::from_complex_clamped(z)
        }
    })
}

/// Maps a [`HalfPlanePoint`] $w \in \mathbb{H}$ isometrically to $\mathbb{D}$ via the inverse Cayley map $z = \frac{w - i}{w + i}$.
#[must_use]
#[verified_engine::verified]
pub fn cayley_half_plane_to_disk(q: HalfPlanePoint) -> DiskPoint {
    try_cayley_half_plane_to_disk(q).expect("cayley_half_plane_to_disk preconditions must hold")
}

/// Fallible companion to [`cayley_boundary_to_real`].
#[verified_engine::verified]
pub fn try_cayley_boundary_to_real(
    z_unit: Complex64,
) -> core::result::Result<Option<f64>, math_commons::error::MathError> {
    #[allow(unused_braces)]
    Ok({
        __ensure_math(
            z_unit.re.is_finite() && z_unit.im.is_finite(),
            "cayley_boundary_to_real: precondition or invariant violated",
        )?;
        let dx = 1.0 - z_unit.re;
        let denom = dx * dx + z_unit.im * z_unit.im;
        if denom <= 1.0e-12 {
            return Ok(None);
        }
        let u = (-2.0 * z_unit.im) / denom;
        debug_assert!(u.is_finite());
        Some(u)
    })
}

/// Maps an ideal unit-circle boundary point $\xi = e^{i\theta} \ne 1$ to $\mathbb{R} \cup \{\infty\}$ via $u = \frac{-2 \operatorname{Im}(\xi)}{|1 - \xi|^2}$.
///
/// Returns `None` when $\xi$ lies at the pole $\xi = 1$ ($w = \infty$).
#[must_use]
#[verified_engine::verified]
pub fn cayley_boundary_to_real(z_unit: Complex64) -> Option<f64> {
    try_cayley_boundary_to_real(z_unit).expect("cayley_boundary_to_real preconditions must hold")
}

/// Fallible companion to [`cayley_complex_to_half_plane_clamped`].
#[verified_engine::verified]
pub fn try_cayley_complex_to_half_plane_clamped(
    z: Complex64,
) -> core::result::Result<Complex64, math_commons::error::MathError> {
    #[allow(unused_braces)]
    Ok({
        __ensure_math(
            z.re.is_finite() && z.im.is_finite(),
            "cayley_complex_to_half_plane_clamped: precondition or invariant violated",
        )?;
        let dx = 1.0 - z.re;
        let denom = (dx * dx + z.im * z.im).max(1.0e-6);
        let u = ((-2.0 * z.im) / denom).clamp(-500.0, 500.0);
        let v = (((1.0 - z.norm_sqr()).max(0.0)) / denom).clamp(0.0, 500.0);
        debug_assert!(u.is_finite() && v.is_finite());
        Complex64::new(u, v)
    })
}

/// Maps any disk or ideal point $z \in \overline{\mathbb{D}}$ into a bounded upper half-plane coordinate for viewport rendering.
#[must_use]
#[verified_engine::verified]
pub fn cayley_complex_to_half_plane_clamped(z: Complex64) -> Complex64 {
    try_cayley_complex_to_half_plane_clamped(z)
        .expect("cayley_complex_to_half_plane_clamped preconditions must hold")
}

/// Fallible companion to [`hyperbolic_distance_disk`].
#[verified_engine::verified]
pub fn try_hyperbolic_distance_disk(
    p1: DiskPoint,
    p2: DiskPoint,
) -> core::result::Result<f64, math_commons::error::MathError> {
    #[allow(unused_braces)]
    Ok({
        let z1 = p1.as_complex();
        let z2 = p2.as_complex();
        let diff = z1 - z2;
        let num = diff.re.hypot(diff.im);
        let denom_c = Complex64::new(1.0, 0.0) - z1.conj() * z2;
        let den = denom_c.re.hypot(denom_c.im);
        __ensure_math(
            den > 0.0,
            "hyperbolic_distance_disk: precondition or invariant violated",
        )?;
        let rho = (num / den).clamp(0.0, 1.0 - 1.0e-15);
        let dist = if rho < 0.5 {
            2.0 * rho.atanh()
        } else {
            let delta = (2.0 * diff.norm_sqr()) / ((1.0 - z1.norm_sqr()) * (1.0 - z2.norm_sqr()));
            (1.0 + delta.max(0.0)).acosh()
        };
        __ensure_math(
            dist.is_finite() && dist >= 0.0,
            "hyperbolic_distance_disk: precondition or invariant violated",
        )?;
        debug_assert!(dist.is_finite() && dist >= 0.0);
        dist
    })
}

/// Computes the exact hyperbolic distance $d_{\mathbb{D}}(p_1, p_2)$ on the Poincaré disk ($K = -1$).
#[must_use]
#[verified_engine::verified]
pub fn hyperbolic_distance_disk(p1: DiskPoint, p2: DiskPoint) -> f64 {
    try_hyperbolic_distance_disk(p1, p2).expect("hyperbolic_distance_disk preconditions must hold")
}

/// Fallible companion to [`hyperbolic_distance_half_plane`].
#[verified_engine::verified]
pub fn try_hyperbolic_distance_half_plane(
    q1: HalfPlanePoint,
    q2: HalfPlanePoint,
) -> core::result::Result<f64, math_commons::error::MathError> {
    #[allow(unused_braces)]
    Ok({
        let w1 = q1.as_complex();
        let w2 = q2.as_complex();
        __ensure_math(
            w1.im > 0.0 && w2.im > 0.0,
            "hyperbolic_distance_half_plane: precondition or invariant violated",
        )?;
        let diff = w1 - w2;
        let conj_diff = w1 - w2.conj();
        let num = diff.re.hypot(diff.im);
        let den = conj_diff.re.hypot(conj_diff.im);
        __ensure_math(
            den > 0.0,
            "hyperbolic_distance_half_plane: precondition or invariant violated",
        )?;
        let rho = (num / den).clamp(0.0, 1.0 - 1.0e-15);
        let dist = if rho < 0.5 {
            2.0 * rho.atanh()
        } else {
            let delta = diff.norm_sqr() / (2.0 * w1.im * w2.im);
            (1.0 + delta.max(0.0)).acosh()
        };
        __ensure_math(
            dist.is_finite() && dist >= 0.0,
            "hyperbolic_distance_half_plane: precondition or invariant violated",
        )?;
        dist
    })
}

/// Computes the exact hyperbolic distance $d_{\mathbb{H}}(q_1, q_2)$ on the upper half-plane ($K = -1$).
#[must_use]
#[verified_engine::verified]
pub fn hyperbolic_distance_half_plane(q1: HalfPlanePoint, q2: HalfPlanePoint) -> f64 {
    try_hyperbolic_distance_half_plane(q1, q2)
        .expect("hyperbolic_distance_half_plane preconditions must hold")
}

#[inline]
fn __ensure_math(
    cond: bool,
    reason: &'static str,
) -> core::result::Result<(), math_commons::error::MathError> {
    if cond {
        Ok(())
    } else {
        Err(math_commons::error::MathError::NumericalError { reason })
    }
}
