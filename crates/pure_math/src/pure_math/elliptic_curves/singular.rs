// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]
//! # Zero-Allocation Singular Weierstrass Cubic Companion
//!
//! Classifies and evaluates singular short-Weierstrass cubics $y^2 = x^3 + ax + b$
//! with $\Delta = -16(4a^3 + 27b^2) = 0$ over $\mathbb{R}$ and prime fields $\mathbb{F}_p$ ($p \ge 5$)
//! without modifying [`super::WeierstrassCurve::new`] or [`super::PrimeWeierstrassCurve::new`].

use crate::pure_math::elliptic_curves::EllipticCurveError;
use crate::pure_math::elliptic_curves::modular_fp::{
    ChordStepFp, mod_inv_fp, mod_mul, mod_pow_fp, mod_reduce, shortest_torus_direction,
    tonelli_shanks_fp,
};
use crate::pure_math::elliptic_curves::weierstrass::{
    AffinePointFp, CurvePoint, MAX_LATTICE_PRIME, MAX_WEIERSTRASS_PRIME,
};
use crate::pure_math::number_theory::primes::is_prime;

const REAL_SINGULAR_TOL: f64 = 1e-12;

/// Algebraic singularity class of a short-Weierstrass cubic with $4a^3 + 27b^2 = 0$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SingularityKind {
    /// Triple-root origin cusp $y^2 = x^3$ ($a = b = 0$, additive smooth locus $\mathbb{G}_a$).
    Cusp,
    /// Self-intersecting double point $(x_s, 0)$ with rational/real tangents (split $\mathbb{G}_m$).
    Crunode,
    /// Isolated double point $(x_s, 0)$ with quadratic-conjugate tangents (non-split $\mathbb{G}_m$).
    Acnode,
}

impl SingularityKind {
    /// Canonical display label for UI badges and diagnostic readouts.
    #[must_use]
    #[verified_engine::verified]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cusp => "Cusp",
            Self::Crunode => "Crunode",
            Self::Acnode => "Acnode",
        }
    }
}

/// Classified singular short-Weierstrass cubic $y^2 = x^3 + ax + b$ over $\mathbb{R}$.
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::elliptic_curves::{RealSingularCubic, SingularityKind};
/// assert_eq!(RealSingularCubic::classify(-3.0, 2.0).unwrap().kind, SingularityKind::Crunode);
/// assert_eq!(RealSingularCubic::classify(-3.0, -2.0).unwrap().kind, SingularityKind::Acnode);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealSingularCubic {
    /// Linear coefficient $a$.
    pub a: f64,
    /// Constant coefficient $b$.
    pub b: f64,
    /// Singularity class (`Cusp`, `Crunode`, or `Acnode`).
    pub kind: SingularityKind,
    /// Double/triple root $x_s$ where the singular vertex $(x_s, 0)$ sits.
    pub singular_x: f64,
    /// Minimum $x$ coordinate of the continuous smooth branch ($0$ for cusp, $-2x_s$ for nodes).
    pub branch_min_x: f64,
    /// Absolute tangent slope $\sqrt{3 x_s}$ at a real crunode (`0.0` for cusp or acnode).
    pub tangent_slope_abs: f64,
}

impl RealSingularCubic {
    /// Classifies $(a, b)$ when $|4a^3 + 27b^2| \le 10^{-12}$, returning `None` if smooth or non-finite.
    #[must_use]
    #[verified_engine::verified]
    pub fn classify(a: f64, b: f64) -> Option<Self> {
        Self::classify_with_tol(a, b, REAL_SINGULAR_TOL)
    }

    /// Classifies $(a, b)$ when $|4a^3 + 27b^2| \le \text{tol}$, returning `None` if smooth or non-finite.
    #[must_use]
    #[verified_engine::verified]
    pub fn classify_with_tol(a: f64, b: f64, tol: f64) -> Option<Self> {
        if !a.is_finite() || !b.is_finite() || !tol.is_finite() {
            return None;
        }
        let eps = tol.max(REAL_SINGULAR_TOL);
        let core = 4.0 * a * a * a + 27.0 * b * b;
        if !core.is_finite() || core.abs() > eps {
            return None;
        }
        if a.abs() <= eps.cbrt() && b.abs() <= eps.sqrt() {
            return Some(Self {
                a: 0.0,
                b: 0.0,
                kind: SingularityKind::Cusp,
                singular_x: 0.0,
                branch_min_x: 0.0,
                tangent_slope_abs: 0.0,
            });
        }
        if a >= 0.0 {
            return None;
        }
        let xs = (-3.0 * b) / (2.0 * a);
        let (kind, slope) = if xs > 0.0 {
            (SingularityKind::Crunode, (3.0 * xs).sqrt())
        } else {
            (SingularityKind::Acnode, 0.0)
        };
        Some(Self {
            a,
            b,
            kind,
            singular_x: xs,
            branch_min_x: -2.0 * xs,
            tangent_slope_abs: slope,
        })
    }

    /// Evaluates the factored cubic $(x - x_s)^2 (x + 2x_s)$ (or $x^3$ for cusp) accurately near $x_s$.
    #[must_use]
    #[verified_engine::verified]
    pub fn rhs(&self, x: f64) -> f64 {
        match self.kind {
            SingularityKind::Cusp => x * x * x,
            SingularityKind::Crunode | SingularityKind::Acnode => {
                let d = x - self.singular_x;
                d * d * (x - self.branch_min_x)
            }
        }
    }

    /// Returns `true` if `point` coincides with the singular vertex $(x_s, 0)$ within `1e-9`.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_singular_vertex(&self, point: CurvePoint<f64>) -> bool {
        match point {
            CurvePoint::Infinity => false,
            CurvePoint::Affine { x, y } => (x - self.singular_x).abs() <= 1e-7 && y.abs() <= 1e-7,
        }
    }

    /// Negates a point $-P = (x, -y)$ on the cubic.
    #[must_use]
    #[verified_engine::verified]
    pub fn negate(&self, point: CurvePoint<f64>) -> CurvePoint<f64> {
        match point {
            CurvePoint::Infinity => CurvePoint::Infinity,
            CurvePoint::Affine { x, y } => CurvePoint::Affine {
                x,
                y: if y == 0.0 { 0.0 } else { -y },
            },
        }
    }

    /// Computes smooth secant/tangent slope, returning `None` at the singular vertex $(x_s, 0)$ or $\mathcal{O}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn secant_or_tangent_slope(&self, p: CurvePoint<f64>, q: CurvePoint<f64>) -> Option<f64> {
        if self.is_singular_vertex(p) || self.is_singular_vertex(q) {
            return None;
        }
        let (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: y2 }) = (p, q)
        else {
            return None;
        };
        let dx = x2 - x1;
        if dx.abs() <= REAL_SINGULAR_TOL {
            if (y1 - y2).abs() <= 1e-9 && y1.abs() > REAL_SINGULAR_TOL {
                Some((3.0 * x1 * x1 + self.a) / (2.0 * y1))
            } else {
                None
            }
        } else {
            Some((y2 - y1) / dx)
        }
    }

    /// Adds two smooth points on $E_{\mathrm{ns}}(\mathbb{R})$, returning `None` if either is $(x_s, 0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn add_smooth(&self, p: CurvePoint<f64>, q: CurvePoint<f64>) -> Option<CurvePoint<f64>> {
        if self.is_singular_vertex(p) || self.is_singular_vertex(q) {
            return None;
        }
        match (p, q) {
            (CurvePoint::Infinity, other) | (other, CurvePoint::Infinity) => Some(other),
            (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: _ }) => {
                let Some(slope) = self.secant_or_tangent_slope(p, q) else {
                    return Some(CurvePoint::Infinity);
                };
                let x3 = slope * slope - x1 - x2;
                Some(CurvePoint::Affine {
                    x: x3,
                    y: slope * (x1 - x3) - y1,
                })
            }
        }
    }
}

/// Classified singular short-Weierstrass cubic $y^2 \equiv x^3 + ax + b \pmod p$ for prime $p \ge 5$.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::elliptic_curves::{CurvePoint, FpSingularCubic, SingularityKind};
///
/// let sing = FpSingularCubic::classify(-3, 2, 17).unwrap().unwrap();
/// assert_eq!(sing.kind, SingularityKind::Acnode);
/// assert_eq!(sing.singular_x, 1);
/// let mut pts = [CurvePoint::Infinity; 24];
/// let count = sing.write_affine_points(&mut pts).unwrap();
/// assert_eq!(count, 18);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FpSingularCubic {
    /// Reduced linear coefficient $a \in [0, p)$.
    pub a: i64,
    /// Reduced constant coefficient $b \in [0, p)$.
    pub b: i64,
    /// Prime field modulus $p \ge 5$.
    pub p: i64,
    /// Singularity class over $\mathbb{F}_p$ (`Cusp`, `Crunode`, or `Acnode`).
    pub kind: SingularityKind,
    /// Residue $x_s \in [0, p)$ of the singular vertex $(x_s, 0)$.
    pub singular_x: i64,
}

impl FpSingularCubic {
    /// Classifies $(a, b) \bmod p$ when $4a^3 + 27b^2 \equiv 0 \pmod p$, returning `Ok(None)` if smooth.
    ///
    /// # Errors
    /// Returns [`EllipticCurveError::InvalidModulus`] if $p$ is outside $5..=10^9$ or composite.
    #[verified_engine::verified]
    pub fn classify(a: i64, b: i64, p: i64) -> Result<Option<Self>, EllipticCurveError> {
        if !(5..=MAX_WEIERSTRASS_PRIME).contains(&p) || !is_prime(p as u64) {
            return Err(EllipticCurveError::InvalidModulus { p });
        }
        let a_red = a.rem_euclid(p);
        let b_red = b.rem_euclid(p);
        let a3 = mod_mul(mod_mul(a_red, a_red, p), a_red, p);
        let b2 = mod_mul(b_red, b_red, p);
        if (mod_mul(4, a3, p) + mod_mul(27, b2, p)).rem_euclid(p) != 0 {
            return Ok(None);
        }
        if a_red == 0 && b_red == 0 {
            return Ok(Some(Self {
                a: 0,
                b: 0,
                p,
                kind: SingularityKind::Cusp,
                singular_x: 0,
            }));
        }
        let inv_2a = mod_inv_fp(mod_mul(2, a_red, p), p).unwrap_or(0);
        let xs = mod_mul(mod_reduce(-3_i128 * (b_red as i128), p), inv_2a, p);
        let euler = mod_pow_fp(mod_mul(3, xs, p), ((p - 1) / 2) as u64, p);
        let kind = if euler == 1 {
            SingularityKind::Crunode
        } else {
            SingularityKind::Acnode
        };
        Ok(Some(Self {
            a: a_red,
            b: b_red,
            p,
            kind,
            singular_x: xs,
        }))
    }

    /// Evaluates $x^3 + ax + b \bmod p \in [0, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn rhs_mod_p(&self, x: i64) -> i64 {
        let xr = x.rem_euclid(self.p);
        (mod_mul(mod_mul(xr, xr, self.p), xr, self.p) + mod_mul(self.a, xr, self.p) + self.b)
            .rem_euclid(self.p)
    }

    /// Exact number of affine solutions $(x, y) \in \mathbb{F}_p^2$ (including $(x_s, 0)$).
    #[must_use]
    #[verified_engine::verified]
    pub const fn affine_point_count(&self) -> usize {
        match self.kind {
            SingularityKind::Cusp => self.p as usize,
            SingularityKind::Crunode => (self.p - 1) as usize,
            SingularityKind::Acnode => (self.p + 1) as usize,
        }
    }

    /// Exact number of projective $\mathbb{F}_p$-points (affine points plus $\mathcal{O}$).
    #[must_use]
    #[verified_engine::verified]
    pub const fn projective_point_count(&self) -> usize {
        self.affine_point_count() + 1
    }

    /// Group order of the smooth locus $E_{\mathrm{ns}}(\mathbb{F}_p)$ ($p$, $p - 1$, or $p + 1$).
    #[must_use]
    #[verified_engine::verified]
    pub const fn smooth_group_order(&self) -> usize {
        self.affine_point_count()
    }

    /// Returns `true` if `point` is the singular vertex $(x_s, 0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_singular_vertex(&self, point: CurvePoint<i64>) -> bool {
        matches!(point, CurvePoint::Affine { x, y } if x.rem_euclid(self.p) == self.singular_x && y.rem_euclid(self.p) == 0)
    }

    /// Returns `true` if `point` is the singular vertex $(x_s, 0)$ in [`AffinePointFp`] form.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_singular_affine_fp(&self, point: AffinePointFp) -> bool {
        matches!(point, AffinePointFp::Finite { x, y } if x.rem_euclid(self.p) == self.singular_x && y.rem_euclid(self.p) == 0)
    }

    /// Returns `true` if `point` lies on $y^2 \equiv x^3 + ax + b \pmod p$ (or is $\mathcal{O}$).
    #[must_use]
    #[verified_engine::verified]
    pub fn contains_affine_fp(&self, point: AffinePointFp) -> bool {
        match point {
            AffinePointFp::Infinity => true,
            AffinePointFp::Finite { x, y } => {
                (0..self.p).contains(&x)
                    && (0..self.p).contains(&y)
                    && mod_mul(y, y, self.p) == self.rhs_mod_p(x)
            }
        }
    }

    #[verified_engine::verified]
    fn scan_affine<T>(
        &self,
        start: usize,
        out: &mut [T],
        make: fn(i64, i64) -> T,
    ) -> Result<usize, EllipticCurveError> {
        if self.p > MAX_LATTICE_PRIME {
            return Err(EllipticCurveError::ModulusExceedsLatticeCap {
                modulus: self.p,
                max_modulus: MAX_LATTICE_PRIME,
            });
        }
        let mut written = start;
        for x in 0..self.p {
            let (roots, count) = tonelli_shanks_fp(self.rhs_mod_p(x), self.p);
            for &y in roots.iter().take(count) {
                let len = out.len();
                let slot = out
                    .get_mut(written)
                    .ok_or(EllipticCurveError::BufferTooSmall {
                        required: written + 1,
                        provided: len,
                    })?;
                *slot = make(x, y);
                written += 1;
            }
        }
        Ok(written)
    }

    /// Writes all affine points $(x, y) \in \mathbb{F}_p^2$ (including $(x_s, 0)$) into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn write_affine_points(
        &self,
        out: &mut [CurvePoint<i64>],
    ) -> Result<usize, EllipticCurveError> {
        self.scan_affine(0, out, |x, y| CurvePoint::Affine { x, y })
    }

    /// Writes $\mathcal{O}$ followed by all affine points into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn enumerate_projective_points(
        &self,
        out: &mut [AffinePointFp],
    ) -> Result<usize, EllipticCurveError> {
        if self.p > MAX_LATTICE_PRIME {
            return Err(EllipticCurveError::ModulusExceedsLatticeCap {
                modulus: self.p,
                max_modulus: MAX_LATTICE_PRIME,
            });
        }
        let head = out.first_mut().ok_or(EllipticCurveError::BufferTooSmall {
            required: 1,
            provided: 0,
        })?;
        *head = AffinePointFp::Infinity;
        self.scan_affine(1, out, |x, y| AffinePointFp::Finite { x, y })
    }

    /// Computes modular secant/tangent slope between smooth points, suppressing slopes at $(x_s, 0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn slope_mod_p(&self, p1: CurvePoint<i64>, p2: CurvePoint<i64>) -> Option<i64> {
        if self.is_singular_vertex(p1) || self.is_singular_vertex(p2) {
            return None;
        }
        let (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: y2 }) = (p1, p2)
        else {
            return None;
        };
        let (x1, y1, x2, y2) = (
            x1.rem_euclid(self.p),
            y1.rem_euclid(self.p),
            x2.rem_euclid(self.p),
            y2.rem_euclid(self.p),
        );
        if x1 == x2 {
            (y1 == y2 && y1 != 0)
                .then(|| (mod_mul(3, mod_mul(x1, x1, self.p), self.p) + self.a).rem_euclid(self.p))
                .and_then(|n| {
                    mod_inv_fp(mod_mul(2, y1, self.p), self.p).map(|i| mod_mul(n, i, self.p))
                })
        } else {
            let n = (y2 - y1).rem_euclid(self.p);
            mod_inv_fp((x2 - x1).rem_euclid(self.p), self.p).map(|i| mod_mul(n, i, self.p))
        }
    }

    /// Adds two smooth points on $E_{\mathrm{ns}}(\mathbb{F}_p)$, returning `None` if either is $(x_s, 0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn add_smooth(&self, p1: CurvePoint<i64>, p2: CurvePoint<i64>) -> Option<CurvePoint<i64>> {
        if self.is_singular_vertex(p1) || self.is_singular_vertex(p2) {
            return None;
        }
        match (p1, p2) {
            (CurvePoint::Infinity, other) | (other, CurvePoint::Infinity) => Some(other),
            (CurvePoint::Affine { x: x1, y: y1 }, CurvePoint::Affine { x: x2, y: _ }) => {
                let Some(m) = self.slope_mod_p(p1, p2) else {
                    return Some(CurvePoint::Infinity);
                };
                let (x1, y1, x2) = (
                    x1.rem_euclid(self.p),
                    y1.rem_euclid(self.p),
                    x2.rem_euclid(self.p),
                );
                let x3 = (mod_mul(m, m, self.p) - x1 - x2).rem_euclid(self.p);
                Some(CurvePoint::Affine {
                    x: x3,
                    y: (mod_mul(m, x1 - x3, self.p) - y1).rem_euclid(self.p),
                })
            }
        }
    }

    /// Computes a torus geodesic [`ChordStepFp`] between two smooth points, returning `None` at $(x_s, 0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn chord_step_fp(&self, p1: AffinePointFp, p2: AffinePointFp) -> Option<ChordStepFp> {
        if self.is_singular_affine_fp(p1) || self.is_singular_affine_fp(p2) {
            return None;
        }
        let (AffinePointFp::Finite { x: x1, y: y1 }, AffinePointFp::Finite { x: x2, y: y2 }) =
            (p1, p2)
        else {
            return None;
        };
        let is_tangent = x1 == x2 && y1 == y2;
        let Some(m) = self.slope_mod_p(
            CurvePoint::Affine { x: x1, y: y1 },
            CurvePoint::Affine { x: x2, y: y2 },
        ) else {
            return Some(ChordStepFp {
                sum: AffinePointFp::Infinity,
                neg_sum: AffinePointFp::Infinity,
                slope: None,
                geodesic_dir: (0, 1),
                geodesic_span: self.p,
                is_tangent: false,
            });
        };
        let x3 = (mod_mul(m, m, self.p) - x1 - x2).rem_euclid(self.p);
        let y3 = (mod_mul(m, x1 - x3, self.p) - y1).rem_euclid(self.p);
        let dir = shortest_torus_direction(m, self.p);
        let span = mod_inv_fp(dir.0, self.p).map_or(0, |inv_u| {
            mod_mul((x3 - x1).rem_euclid(self.p), inv_u, self.p)
        });
        Some(ChordStepFp {
            sum: AffinePointFp::Finite { x: x3, y: y3 },
            neg_sum: AffinePointFp::Finite {
                x: x3,
                y: (-y3).rem_euclid(self.p),
            },
            slope: Some(m),
            geodesic_dir: dir,
            geodesic_span: span,
            is_tangent,
        })
    }
}

#[cfg(test)]
#[path = "singular_tests.rs"]
mod tests;
