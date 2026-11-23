//! Short Weierstrass elliptic curves over prime fields $\mathbb{F}_p$ ($p \ge 5$).
//!
//! Implements runtime-prime group arithmetic (`add_points`, `negate_point`, `scalar_mul`,
//! `point_order`), cyclic torsion subgroup orbits, and point enumeration via Tonelli-Shanks
//! writing into caller-supplied slices with zero dynamic heap allocations.

use crate::pure_math::elliptic_curves::EllipticCurveError;
pub use crate::pure_math::elliptic_curves::modular_fp::{
    ChordStepFp, HasseSweepEntry, mod_inv_fp, mod_pow_fp, sweep_hasse_bounds, tonelli_shanks_fp,
};
use crate::pure_math::elliptic_curves::modular_fp::{
    mod_mul, mod_reduce, shortest_torus_direction,
};
use crate::pure_math::number_theory::primes::is_prime;

/// Maximum prime modulus supported by `i64`/`i128` short Weierstrass arithmetic without overflow.
pub const MAX_WEIERSTRASS_PRIME: i64 = 1_000_000_000;

/// Point on a short Weierstrass curve $E(\mathbb{F}_p)$, including the identity $\mathcal{O}$.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AffinePointFp {
    /// Group identity element $\mathcal{O}$ (point at infinity).
    #[default]
    Infinity,
    /// Finite affine point $(x, y) \in \mathbb{F}_p \times \mathbb{F}_p$ with $0 \le x, y < p$.
    Finite {
        /// Horizontal residue $x \bmod p$.
        x: i64,
        /// Vertical residue $y \bmod p$.
        y: i64,
    },
}

impl AffinePointFp {
    /// Constructs a finite affine point reduced modulo `p` ($p > 0$).
    #[must_use]
    #[verified_engine::verified]
    pub fn from_coords(x: i64, y: i64, p: i64) -> Self {
        Self::Finite {
            x: x.rem_euclid(p),
            y: y.rem_euclid(p),
        }
    }

    /// Returns `true` when this point is the group identity $\mathcal{O}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_infinity(self) -> bool {
        matches!(self, Self::Infinity)
    }

    /// Returns `Some((x, y))` for a finite point or `None` for $\mathcal{O}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn coordinates(self) -> Option<(i64, i64)> {
        match self {
            Self::Infinity => None,
            Self::Finite { x, y } => Some((x, y)),
        }
    }
}

/// Short Weierstrass elliptic curve $E(\mathbb{F}_p): y^2 \equiv x^3 + ax + b \pmod p$ for prime $p \ge 5$.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::elliptic_curves::{AffinePointFp, WeierstrassCurveFp};
///
/// let curve = WeierstrassCurveFp::new(17, 2, 2).unwrap();
/// let p = AffinePointFp::from_coords(5, 1, 17);
/// let q = AffinePointFp::from_coords(6, 3, 17);
/// assert_eq!(curve.add_points(p, p).unwrap(), q);
/// assert_eq!(curve.point_order(p).unwrap(), 19);
/// assert_eq!(curve.frobenius_trace(), -1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct WeierstrassCurveFp {
    p: i64,
    a: i64,
    b: i64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for WeierstrassCurveFp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawCurve {
            p: i64,
            a: i64,
            b: i64,
        }
        let raw = RawCurve::deserialize(deserializer)?;
        Self::new(raw.p, raw.a, raw.b).map_err(serde::de::Error::custom)
    }
}

impl WeierstrassCurveFp {
    /// Computes the cubic discriminant factor $(4a^3 + 27b^2) \bmod p$ for $5 \le p \le 10^9$.
    #[must_use]
    #[verified_engine::verified]
    pub fn discriminant_factor(p: i64, a: i64, b: i64) -> Option<i64> {
        if !(5..=MAX_WEIERSTRASS_PRIME).contains(&p) {
            return None;
        }
        let a_m = (a as i128).rem_euclid(p as i128);
        let b_m = (b as i128).rem_euclid(p as i128);
        let a2 = (a_m * a_m).rem_euclid(p as i128);
        let a3 = (a2 * a_m).rem_euclid(p as i128);
        let b2 = (b_m * b_m).rem_euclid(p as i128);
        Some((4 * a3 + 27 * b2).rem_euclid(p as i128) as i64)
    }

    /// Constructs a validated non-singular curve over $\mathbb{F}_p$ ($5 \le p \le 10^9$ prime).
    #[verified_engine::verified]
    pub fn new(p: i64, a: i64, b: i64) -> Result<Self, EllipticCurveError> {
        if !(5..=MAX_WEIERSTRASS_PRIME).contains(&p) || !is_prime(p as u64) {
            return Err(EllipticCurveError::InvalidModulus { p });
        }
        let a_mod = a.rem_euclid(p);
        let b_mod = b.rem_euclid(p);
        let factor = Self::discriminant_factor(p, a_mod, b_mod).unwrap_or(0);
        if factor == 0 {
            return Err(EllipticCurveError::SingularCurve {
                p,
                a: a_mod,
                b: b_mod,
            });
        }
        Ok(Self {
            p,
            a: a_mod,
            b: b_mod,
        })
    }

    /// Prime field modulus $p$.
    #[must_use]
    #[verified_engine::verified]
    pub fn p(&self) -> i64 {
        self.p
    }

    /// Linear Weierstrass coefficient $a \in [0, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn a(&self) -> i64 {
        self.a
    }

    /// Constant Weierstrass coefficient $b \in [0, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn b(&self) -> i64 {
        self.b
    }

    /// Full Weierstrass discriminant $\Delta = -16(4a^3 + 27b^2) \bmod p \neq 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn discriminant(&self) -> i64 {
        let factor = Self::discriminant_factor(self.p, self.a, self.b).unwrap_or(0);
        mod_reduce(-16_i128 * (factor as i128), self.p)
    }

    #[verified_engine::verified]
    fn rhs_at(&self, x: i64) -> i64 {
        let x2 = mod_mul(x, x, self.p);
        let x3 = mod_mul(x2, x, self.p);
        let ax = mod_mul(self.a, x, self.p);
        mod_reduce((x3 as i128) + (ax as i128) + (self.b as i128), self.p)
    }

    /// Checks whether `point` belongs to $E(\mathbb{F}_p)$ with canonical coordinates in $[0, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn contains_point(&self, point: AffinePointFp) -> bool {
        match point {
            AffinePointFp::Infinity => true,
            AffinePointFp::Finite { x, y } => {
                (0..self.p).contains(&x)
                    && (0..self.p).contains(&y)
                    && mod_mul(y, y, self.p) == self.rhs_at(x)
            }
        }
    }

    #[verified_engine::verified]
    fn ensure_on_curve(&self, point: AffinePointFp) -> Result<(), EllipticCurveError> {
        if self.contains_point(point) {
            return Ok(());
        }
        let (x, y) = point.coordinates().unwrap_or((-1, -1));
        Err(EllipticCurveError::PointNotOnCurve { x, y, p: self.p })
    }

    /// Computes the group inverse $-P$ on $E(\mathbb{F}_p)$.
    #[verified_engine::verified]
    pub fn negate_point(&self, point: AffinePointFp) -> Result<AffinePointFp, EllipticCurveError> {
        self.ensure_on_curve(point)?;
        match point {
            AffinePointFp::Infinity => Ok(AffinePointFp::Infinity),
            AffinePointFp::Finite { x, y } => Ok(AffinePointFp::Finite {
                x,
                y: (self.p - y).rem_euclid(self.p),
            }),
        }
    }

    /// Computes the group sum $P + Q$ on $E(\mathbb{F}_p)$.
    #[verified_engine::verified]
    pub fn add_points(
        &self,
        p1: AffinePointFp,
        p2: AffinePointFp,
    ) -> Result<AffinePointFp, EllipticCurveError> {
        self.ensure_on_curve(p1)?;
        self.ensure_on_curve(p2)?;
        let (x1, y1) = match p1 {
            AffinePointFp::Infinity => return Ok(p2),
            AffinePointFp::Finite { x, y } => (x, y),
        };
        let (x2, y2) = match p2 {
            AffinePointFp::Infinity => return Ok(p1),
            AffinePointFp::Finite { x, y } => (x, y),
        };
        if x1 == x2 && (y1 + y2).rem_euclid(self.p) == 0 {
            return Ok(AffinePointFp::Infinity);
        }
        let lambda = if x1 == x2 {
            let num = mod_reduce(3 * (x1 as i128) * (x1 as i128) + (self.a as i128), self.p);
            let den = mod_inv_fp(mod_mul(2, y1, self.p), self.p).unwrap_or(0);
            mod_mul(num, den, self.p)
        } else {
            let num = mod_reduce((y2 as i128) - (y1 as i128), self.p);
            let den =
                mod_inv_fp(mod_reduce((x2 as i128) - (x1 as i128), self.p), self.p).unwrap_or(0);
            mod_mul(num, den, self.p)
        };
        let x3 = mod_reduce(
            (lambda as i128) * (lambda as i128) - (x1 as i128) - (x2 as i128),
            self.p,
        );
        let y3 = mod_reduce(
            (lambda as i128) * ((x1 - x3) as i128) - (y1 as i128),
            self.p,
        );
        Ok(AffinePointFp::Finite { x: x3, y: y3 })
    }

    /// Computes chord/tangent intersection metadata (`-R`, `P + Q`, slope, and torus geodesic).
    #[verified_engine::verified]
    pub fn chord_step(
        &self,
        p1: AffinePointFp,
        p2: AffinePointFp,
    ) -> Result<ChordStepFp, EllipticCurveError> {
        let sum = self.add_points(p1, p2)?;
        let neg_sum = self.negate_point(sum)?;
        let (Some((x1, y1)), Some((x2, y2))) = (p1.coordinates(), p2.coordinates()) else {
            return Ok(ChordStepFp {
                sum,
                neg_sum,
                slope: None,
                geodesic_dir: (0, 0),
                geodesic_span: 0,
                is_tangent: false,
            });
        };
        if x1 == x2 && (y1 + y2).rem_euclid(self.p) == 0 {
            return Ok(ChordStepFp {
                sum,
                neg_sum,
                slope: None,
                geodesic_dir: (0, 1),
                geodesic_span: self.p,
                is_tangent: y1 == y2,
            });
        }
        let is_tangent = x1 == x2 && y1 == y2;
        let slope = if is_tangent {
            let num = mod_reduce(3 * (x1 as i128) * (x1 as i128) + (self.a as i128), self.p);
            mod_mul(
                num,
                mod_inv_fp(mod_mul(2, y1, self.p), self.p).unwrap_or(0),
                self.p,
            )
        } else {
            let num = mod_reduce((y2 as i128) - (y1 as i128), self.p);
            let den =
                mod_inv_fp(mod_reduce((x2 as i128) - (x1 as i128), self.p), self.p).unwrap_or(0);
            mod_mul(num, den, self.p)
        };
        let (x3, _) = neg_sum.coordinates().unwrap_or((x1, y1));
        let (dir, span) = self.orient_torus_geodesic(x1, x2, x3, slope, is_tangent);
        Ok(ChordStepFp {
            sum,
            neg_sum,
            slope: Some(slope),
            geodesic_dir: dir,
            geodesic_span: span,
            is_tangent,
        })
    }

    #[verified_engine::verified]
    fn orient_torus_geodesic(
        &self,
        x1: i64,
        x2: i64,
        x3: i64,
        slope: i64,
        is_tangent: bool,
    ) -> ((i64, i64), i64) {
        let (mut u, mut v) = shortest_torus_direction(slope, self.p);
        let u_inv = mod_inv_fp(u, self.p).unwrap_or(1);
        let mut t_neg_r = mod_mul(mod_reduce((x3 - x1) as i128, self.p), u_inv, self.p);
        if is_tangent {
            if 2 * t_neg_r > self.p {
                u = -u;
                v = -v;
                t_neg_r = self.p - t_neg_r;
            }
            return ((u, v), t_neg_r);
        }
        let t_q = mod_mul(mod_reduce((x2 - x1) as i128, self.p), u_inv, self.p);
        if t_neg_r == 0 {
            return ((u, v), self.p);
        }
        if t_neg_r < t_q {
            u = -u;
            v = -v;
            t_neg_r = self.p - t_neg_r;
        }
        ((u, v), t_neg_r)
    }

    /// Computes scalar multiple `[k]P` via iterative double-and-add for any $k \in \mathbb{Z}$.
    #[verified_engine::verified]
    pub fn scalar_mul(
        &self,
        k: i64,
        point: AffinePointFp,
    ) -> Result<AffinePointFp, EllipticCurveError> {
        self.ensure_on_curve(point)?;
        if k == 0 || point.is_infinity() {
            return Ok(AffinePointFp::Infinity);
        }
        let mut base = if k < 0 {
            self.negate_point(point)?
        } else {
            point
        };
        let mut rem = k.unsigned_abs();
        let mut acc = AffinePointFp::Infinity;
        while rem > 0 {
            if rem & 1 == 1 {
                acc = self.add_points(acc, base)?;
            }
            rem >>= 1;
            if rem > 0 {
                base = self.add_points(base, base)?;
            }
        }
        Ok(acc)
    }

    /// Returns the torsion order $\operatorname{ord}(P) \ge 1$ of `point` in $E(\mathbb{F}_p)$.
    #[verified_engine::verified]
    pub fn point_order(&self, point: AffinePointFp) -> Result<usize, EllipticCurveError> {
        self.ensure_on_curve(point)?;
        if point.is_infinity() {
            return Ok(1);
        }
        let max_order = (self.p as usize) + 2 + (2.0 * (self.p as f64).sqrt()).ceil() as usize;
        let mut cur = point;
        for step in 1..=max_order {
            if cur.is_infinity() {
                return Ok(step);
            }
            cur = self.add_points(cur, point)?;
        }
        Ok(max_order)
    }

    /// Writes the cyclic subgroup orbit `[1]P, [2]P, ..., [ord(P)]P = O` into `out`.
    #[verified_engine::verified]
    pub fn torsion_orbit(
        &self,
        generator: AffinePointFp,
        out: &mut [AffinePointFp],
    ) -> Result<usize, EllipticCurveError> {
        self.ensure_on_curve(generator)?;
        if out.is_empty() {
            return Err(EllipticCurveError::BufferTooSmall {
                required: 1,
                provided: 0,
            });
        }
        if generator.is_infinity() {
            out[0] = AffinePointFp::Infinity;
            return Ok(1);
        }
        let max_order = (self.p as usize) + 2 + (2.0 * (self.p as f64).sqrt()).ceil() as usize;
        let mut cur = generator;
        for step in 0..max_order {
            if step >= out.len() {
                return Err(EllipticCurveError::BufferTooSmall {
                    required: step + 1,
                    provided: out.len(),
                });
            }
            out[step] = cur;
            if cur.is_infinity() {
                return Ok(step + 1);
            }
            cur = self.add_points(cur, generator)?;
        }
        Ok(max_order)
    }

    /// Enumerates all points of $E(\mathbb{F}_p)$ (starting with $\mathcal{O}$) into caller-supplied `out`.
    #[verified_engine::verified]
    pub fn enumerate_points(&self, out: &mut [AffinePointFp]) -> Result<usize, EllipticCurveError> {
        let total = self.cardinality();
        if out.len() < total {
            return Err(EllipticCurveError::BufferTooSmall {
                required: total,
                provided: out.len(),
            });
        }
        out[0] = AffinePointFp::Infinity;
        let mut idx = 1_usize;
        for x in 0..self.p {
            let rhs = self.rhs_at(x);
            let (roots, count) = tonelli_shanks_fp(rhs, self.p);
            for &y in roots.iter().take(count) {
                out[idx] = AffinePointFp::Finite { x, y };
                idx += 1;
            }
        }
        Ok(idx)
    }

    /// Computes exact group cardinality $\#E(\mathbb{F}_p) = p + 1 - a_p$ via Euler's Legendre criterion.
    #[must_use]
    #[verified_engine::verified]
    pub fn cardinality(&self) -> usize {
        let euler = ((self.p - 1) / 2) as u64;
        let mut count = 1_usize;
        for x in 0..self.p {
            let rhs = self.rhs_at(x);
            if rhs == 0 {
                count += 1;
            } else if mod_pow_fp(rhs, euler, self.p) == 1 {
                count += 2;
            }
        }
        count
    }

    /// Computes the Frobenius trace $a_p = p + 1 - \#E(\mathbb{F}_p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn frobenius_trace(&self) -> i64 {
        self.p + 1 - (self.cardinality() as i64)
    }

    /// Returns Hasse-Weil trace bounds `(-2 * sqrt(p), 2 * sqrt(p))` for this curve's modulus $p$.
    #[must_use]
    #[verified_engine::verified]
    pub fn hasse_bounds(&self) -> (f64, f64) {
        let arch = 2.0 * (self.p as f64).sqrt();
        (-arch, arch)
    }
}

// [cite:modular_polynomials_review]
