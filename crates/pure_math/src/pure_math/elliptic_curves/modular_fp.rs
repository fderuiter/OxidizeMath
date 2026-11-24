//! Stack-only prime-field arithmetic, torus geodesics, and Hasse-Weil sweeps.
//!
//! Provides iterative modular exponentiation, Extended Euclidean modular inversion,
//! Tonelli-Shanks square roots, 2D Gauss lattice basis reduction on $(\mathbb{Z}/p\mathbb{Z})^2$,
//! and Hasse-Weil Frobenius trace sweeps with zero dynamic heap allocations.

use crate::pure_math::elliptic_curves::EllipticCurveError;
use crate::pure_math::elliptic_curves::weierstrass::{AffinePointFp, WeierstrassCurveFp};
use math_scalars::modular::{
    mod_inv_i64, mod_mul_i64, mod_pow_i64, mod_reduce_i64, tonelli_shanks_i64,
};

/// Geometric chord-and-tangent step metadata on the modular torus $(\mathbb{R}/p\mathbb{Z})^2$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChordStepFp {
    /// Resulting group sum $R = P + Q$.
    pub sum: AffinePointFp,
    /// Unreflected third intersection $-R = -(P + Q)$ on the secant or tangent line.
    pub neg_sum: AffinePointFp,
    /// Modular slope $\lambda \in [0, p)$ when non-vertical (`None` for vertical or $\mathcal{O}$).
    pub slope: Option<i64>,
    /// Gauss-reduced torus direction vector $(u, v)$ tracing $P \to Q \to -R$.
    pub geodesic_dir: (i64, i64),
    /// Parameter span $T \ge 0$ along $(x_P + u t, y_P + v t) \bmod p$ reaching $-R$.
    pub geodesic_span: i64,
    /// Whether the step is a tangent doubling ($P = Q \neq \mathcal{O}$).
    pub is_tangent: bool,
}

/// Frobenius trace and Hasse-Weil bound sample for $y^2 \equiv x^3 + ax + b \pmod p$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HasseSweepEntry {
    /// Prime modulus $p \ge 5$.
    pub p: i64,
    /// Whether $4a^3 + 27b^2 \equiv 0 \pmod p$ (singular reduction at $p$).
    pub is_singular: bool,
    /// Group cardinality $\#E(\mathbb{F}_p)$ (`0` when singular).
    pub group_order: usize,
    /// Frobenius trace $a_p = p + 1 - \#E(\mathbb{F}_p)$ (`0` when singular).
    pub frobenius_trace: i64,
    /// Upper Hasse-Weil bound $2\sqrt{p}$ (lower bound is $-2\sqrt{p}$).
    pub hasse_bound: f64,
}

#[verified_engine::verified]
pub(crate) fn mod_reduce(val: i128, p: i64) -> i64 {
    mod_reduce_i64(val.rem_euclid(p as i128) as i64, p)
}

#[verified_engine::verified]
pub(crate) fn mod_mul(lhs: i64, rhs: i64, p: i64) -> i64 {
    mod_mul_i64(lhs, rhs, p)
}

/// Computes `base^exp mod p` in `O(log exp)` operations without heap allocation.
#[must_use]
#[verified_engine::verified]
pub fn mod_pow_fp(base: i64, exp: u64, p: i64) -> i64 {
    mod_pow_i64(base, exp, p)
}

/// Computes the multiplicative inverse $a^{-1} \bmod p$ via `math_scalars::modular`.
#[must_use]
#[verified_engine::verified]
pub fn mod_inv_fp(a: i64, p: i64) -> Option<i64> {
    mod_inv_i64(a, p)
}

/// Computes modular square roots of `n mod p` ($p \ge 3$ odd prime) via `math_scalars::modular::tonelli_shanks_i64`.
#[must_use]
#[verified_engine::verified]
pub fn tonelli_shanks_fp(n: i64, p: i64) -> ([i64; 2], usize) {
    match tonelli_shanks_i64(n, p) {
        None => ([0, 0], 0),
        Some([r0, r1]) if r0 == r1 => ([r0, 0], 1),
        Some(pair) => (pair, 2),
    }
}

#[verified_engine::verified]
pub(crate) fn shortest_torus_direction(slope: i64, p: i64) -> (i64, i64) {
    let lifted = if 2 * slope <= p { slope } else { slope - p };
    let mut u = (1_i128, lifted as i128);
    let mut v = (0_i128, p as i128);
    for _ in 0..64 {
        let norm_u = u.0 * u.0 + u.1 * u.1;
        let norm_v = v.0 * v.0 + v.1 * v.1;
        if norm_v < norm_u {
            std::mem::swap(&mut u, &mut v);
        }
        let denom = u.0 * u.0 + u.1 * u.1;
        if denom == 0 {
            break;
        }
        let dot = u.0 * v.0 + u.1 * v.1;
        let q = if dot >= 0 {
            (dot + denom / 2) / denom
        } else {
            (dot - denom / 2) / denom
        };
        if q == 0 {
            break;
        }
        v = (v.0 - q * u.0, v.1 - q * u.1);
    }
    let best = if (v.0 * v.0 + v.1 * v.1) < (u.0 * u.0 + u.1 * u.1) && v.0 != 0 {
        v
    } else {
        u
    };
    if best.0 < 0 {
        (-best.0 as i64, -best.1 as i64)
    } else {
        (best.0 as i64, best.1 as i64)
    }
}

/// Sweeps Frobenius traces $a_p$ and Hasse bounds $\pm 2\sqrt{p}$ for $y^2 \equiv x^3 + ax + b \pmod p$
/// across caller-supplied `primes` into `out` without dynamic heap allocation.
#[verified_engine::verified]
pub fn sweep_hasse_bounds(
    a: i64,
    b: i64,
    primes: &[i64],
    out: &mut [HasseSweepEntry],
) -> Result<usize, EllipticCurveError> {
    if out.len() < primes.len() {
        return Err(EllipticCurveError::BufferTooSmall {
            required: primes.len(),
            provided: out.len(),
        });
    }
    for (idx, &p) in primes.iter().enumerate() {
        let arch = 2.0 * (p.max(0) as f64).sqrt();
        match WeierstrassCurveFp::new(p, a, b) {
            Ok(curve) => {
                let order = curve.cardinality();
                out[idx] = HasseSweepEntry {
                    p,
                    is_singular: false,
                    group_order: order,
                    frobenius_trace: p + 1 - (order as i64),
                    hasse_bound: arch,
                };
            }
            Err(EllipticCurveError::SingularCurve { .. }) => {
                out[idx] = HasseSweepEntry {
                    p,
                    is_singular: true,
                    group_order: 0,
                    frobenius_trace: 0,
                    hasse_bound: arch,
                };
            }
            Err(err) => return Err(err),
        }
    }
    Ok(primes.len())
}

// [cite:modular_polynomials_review]
