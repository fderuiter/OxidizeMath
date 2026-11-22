// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]
//! # Modular Invariants & Number-Theory Bridges
//!
//! Connects short-Weierstrass elliptic curves to $j$-invariants, Breuer modular polynomial
//! valuation bounds (`psi`, `theorem_1_1_bounds`, `theorem_1_2_bounds`), imaginary quadratic
//! binary quadratic forms (`BQF`, `class_number`), Hurwitz-Kronecker trace sums
//! (`hurwitz_class_number`, `weighted_class_number`, `verify_summation_formula`), and
//! truncated modular $q$-series (`QSeries<i64>`).

use super::weierstrass::{
    FpWeierstrassCurve, PrimeWeierstrassCurve, WeierstrassCurve, mod_inv_i64, mod_mul_i64,
};
use super::{
    EllipticCurveError, Theorem11Bounds, Theorem12Bounds, psi, theorem_1_1_bounds,
    theorem_1_2_bounds,
};
use crate::pure_math::algebra::fields::Fp;
use crate::pure_math::algebra::traits::Field;
use crate::pure_math::number_theory::class_number::{BQF, class_number};
use crate::pure_math::number_theory::hurwitz_kronecker::{
    hurwitz_class_number, verify_summation_formula, weighted_class_number,
};
use crate::pure_math::number_theory::partitions::f_k;
use crate::pure_math::number_theory::q_series::QSeries;

/// Maximum $q$-series truncation length permitted for integer `j_invariant_q_scaled` before `i64` overflow.
pub const MAX_SCALED_J_PRECISION: usize = 12;

/// Summary pairing a curve's modular $j$-invariant with Breuer valuation lower bounds at level $N$.
#[derive(Debug, PartialEq)]
pub struct ModularValuationSummary {
    /// Curve $j$-invariant modulo $p$ in $[0, p)$.
    pub j_invariant_mod_p: i64,
    /// Modular polynomial level $N$.
    pub level_n: u64,
    /// Classical modular polynomial degree $\psi(N)$.
    pub psi_n: u64,
    /// Breuer Theorem 1.1 $p$-adic valuation lower bounds ($p \in \{2, 3, 5\}$) for $\Phi_N(X, Y)$.
    pub theorem_1_1: Theorem11Bounds,
    /// Breuer Theorem 1.2 $p$-adic valuation lower bounds ($p \in \{2, 3, 7\}$) for $\Phi_N(X+1728, Y+1728)$.
    pub theorem_1_2: Theorem12Bounds,
}

/// Summary linking Frobenius trace $a_p$ at prime $p$ to imaginary quadratic class numbers.
#[derive(Debug, PartialEq)]
pub struct FrobeniusClassSummary {
    /// Prime field modulus $p$.
    pub p: u64,
    /// Group cardinality $\#E(\mathbb{F}_p)$.
    pub point_count: u64,
    /// Frobenius trace $a_p = p + 1 - \#E(\mathbb{F}_p)$.
    pub frobenius_trace: i64,
    /// Frobenius endomorphism discriminant $D = a_p^2 - 4p \le 0$.
    pub frobenius_discriminant: i64,
    /// Reduced principal positive-definite binary quadratic form of discriminant $D$ when $D < 0$.
    pub principal_form: Option<BQF>,
    /// Ordinary class number $h(D)$ of primitive reduced forms of discriminant $D$.
    pub imaginary_class_number: u64,
    /// Weighted class number $h_w(D)$.
    pub weighted_class_num: f64,
    /// Hurwitz-Kronecker class number $H(D) = \sum_{f^2 \mid D} h_w(D / f^2)$.
    pub hurwitz_kronecker: f64,
}

/// Computes the real modular $j$-invariant $j(E) = 1728 \cdot \frac{4a^3}{4a^3 + 27b^2}$.
#[must_use]
#[verified_engine::verified]
pub fn j_invariant_real(curve: &WeierstrassCurve) -> f64 {
    let four_a3 = 4.0 * curve.a.powi(3);
    let denom = four_a3 + 27.0 * curve.b.powi(2);
    1728.0 * (four_a3 / denom)
}

/// Computes the modular $j$-invariant $j(E) \in \mathbb{F}_P$ for a const-generic prime curve.
#[must_use]
#[verified_engine::verified]
pub fn j_invariant_fp<const P: i64>(curve: &FpWeierstrassCurve<P>) -> Fp<P> {
    let four_a3 = Fp::<P>::new(4) * curve.a * curve.a * curve.a;
    let denom = four_a3 + Fp::<P>::new(27) * curve.b * curve.b;
    Fp::<P>::new(1728) * four_a3 * denom.multiplicative_inverse()
}

/// Computes the modular $j$-invariant $j(E) \equiv 1728 \cdot \frac{4a^3}{4a^3 + 27b^2} \pmod p$ in $[0, p)$.
#[must_use]
#[verified_engine::verified]
pub fn j_invariant_mod_p(curve: &PrimeWeierstrassCurve) -> i64 {
    let p = curve.p;
    let a3 = mod_mul_i64(mod_mul_i64(curve.a, curve.a, p), curve.a, p);
    let four_a3 = mod_mul_i64(4, a3, p);
    let b2 = mod_mul_i64(curve.b, curve.b, p);
    let denom = (four_a3 + mod_mul_i64(27, b2, p)).rem_euclid(p);
    let inv = mod_inv_i64(denom, p).unwrap_or(0);
    mod_mul_i64(mod_mul_i64(1728, four_a3, p), inv, p)
}

/// Pairs $j(E) \bmod p$ with classical modular degree $\psi(N)$ and Breuer Theorems 1.1 & 1.2 bounds.
///
/// # Errors
/// Returns [`EllipticCurveError::InvalidDegree`] if `deg_i + deg_j >= psi(level_n)`.
#[verified_engine::verified]
pub fn modular_valuation_summary(
    curve: &PrimeWeierstrassCurve,
    level_n: u64,
    deg_i: u64,
    deg_j: u64,
) -> Result<ModularValuationSummary, EllipticCurveError> {
    let psi_n = psi(level_n);
    let theorem_1_1 = theorem_1_1_bounds(level_n, deg_i, deg_j)?;
    let theorem_1_2 = theorem_1_2_bounds(level_n, deg_i, deg_j)?;
    Ok(ModularValuationSummary {
        j_invariant_mod_p: j_invariant_mod_p(curve),
        level_n,
        psi_n,
        theorem_1_1,
        theorem_1_2,
    })
}

/// Computes the Frobenius endomorphism discriminant $D = a_p^2 - 4p$.
#[must_use]
#[verified_engine::verified]
pub fn frobenius_discriminant(frobenius_trace: i64, p: u64) -> i64 {
    frobenius_trace * frobenius_trace - 4 * (p as i64)
}

/// Constructs the reduced principal positive-definite [`BQF`] for imaginary quadratic discriminant $D < 0$.
#[must_use]
#[verified_engine::verified]
pub fn principal_bqf_for_discriminant(discriminant: i64) -> Option<BQF> {
    if discriminant >= 0 {
        return None;
    }
    let rem = discriminant.rem_euclid(4);
    let form = if rem == 0 {
        BQF::new(1, 0, -discriminant / 4)
    } else if rem == 1 {
        BQF::new(1, 1, (1 - discriminant) / 4)
    } else {
        return None;
    };
    if form.discriminant() == discriminant && form.is_reduced() {
        Some(form)
    } else {
        None
    }
}

/// Computes Frobenius trace, discriminant $D = a_p^2 - 4p$, principal [`BQF`], `class_number`, and Hurwitz invariants.
///
/// # Errors
/// Propagates [`EllipticCurveError::ModulusExceedsLatticeCap`] when $p > 997$.
#[verified_engine::verified]
pub fn analyze_frobenius_prime(
    curve: &PrimeWeierstrassCurve,
) -> Result<FrobeniusClassSummary, EllipticCurveError> {
    let point_count = curve.point_count()?;
    let p_u64 = curve.p as u64;
    let frobenius_trace = curve.p + 1 - (point_count as i64);
    let frobenius_disc = frobenius_discriminant(frobenius_trace, p_u64);
    let principal_form = principal_bqf_for_discriminant(frobenius_disc);
    let imaginary_class_number = class_number(frobenius_disc);
    let weighted_class_num = weighted_class_number(frobenius_disc);
    let hurwitz_kronecker = hurwitz_class_number(frobenius_disc);
    Ok(FrobeniusClassSummary {
        p: p_u64,
        point_count,
        frobenius_trace,
        frobenius_discriminant: frobenius_disc,
        principal_form,
        imaginary_class_number,
        weighted_class_num,
        hurwitz_kronecker,
    })
}

/// Evaluates the Hurwitz-Kronecker trace sum $\sum_{t^2 < p} H(t^2 - p)$, theoretical $(p - 2)/3$, and identity check.
#[must_use]
#[verified_engine::verified]
pub fn hurwitz_trace_sum(p: u64) -> (f64, f64, bool) {
    if p < 2 {
        return (0.0, 0.0, false);
    }
    let p_i64 = p as i64;
    let limit = ((p - 1) as f64).sqrt() as i64;
    let mut sum = 0.0;
    for t in -limit..=limit {
        sum += hurwitz_class_number(t * t - p_i64);
    }
    let expected = (p_i64 - 2) as f64 / 3.0;
    (sum, expected, verify_summation_formula(p))
}

/// Computes $\Delta(q) / q = \prod_{n=1}^{\infty} (1 - q^n)^{24}$ truncated to `precision` terms.
#[must_use]
#[verified_engine::verified]
pub fn ramanujan_delta_shifted_q_series(precision: usize) -> QSeries<i64> {
    if precision == 0 {
        return QSeries::new();
    }
    f_k(1, precision).pow(24)
}

/// Computes Ramanujan's modular discriminant $\Delta(q) = \sum_{n \ge 1} \tau(n) q^n$ up to $O(q^{\text{precision}})$.
#[must_use]
#[verified_engine::verified]
pub fn ramanujan_delta_q_series(precision: usize) -> QSeries<i64> {
    if precision == 0 {
        return QSeries::new();
    }
    let mut coeffs = vec![0_i64; precision];
    if precision > 1 {
        let shifted = ramanujan_delta_shifted_q_series(precision - 1);
        for (idx, val) in shifted.coeffs.into_iter().enumerate() {
            coeffs[idx + 1] = val;
        }
    }
    QSeries::from_vec(coeffs)
}

#[verified_engine::verified]
fn divisor_sigma_3(n: usize) -> i64 {
    let mut sum = 0_i64;
    let mut d = 1_usize;
    while d * d <= n {
        if n.is_multiple_of(d) {
            let d_i64 = d as i64;
            sum += d_i64 * d_i64 * d_i64;
            let other = n / d;
            if other != d {
                let o_i64 = other as i64;
                sum += o_i64 * o_i64 * o_i64;
            }
        }
        d += 1;
    }
    sum
}

/// Computes the normalized weight-4 Eisenstein series $E_4(q) = 1 + 240 \sum_{n \ge 1} \sigma_3(n) q^n$.
#[must_use]
#[verified_engine::verified]
pub fn eisenstein_e4_q_series(precision: usize) -> QSeries<i64> {
    if precision == 0 {
        return QSeries::new();
    }
    let mut coeffs = Vec::with_capacity(precision);
    coeffs.push(1_i64);
    for n in 1..precision {
        coeffs.push(240_i64 * divisor_sigma_3(n));
    }
    QSeries::from_vec(coeffs)
}

/// Computes $q \cdot j(q) = \frac{E_4(q)^3}{\Delta(q)/q} = 1 + 744q + 196884q^2 + 21493760q^3 + \dots$
/// truncated to `precision.min(MAX_SCALED_J_PRECISION)` terms so `i64` arithmetic never overflows.
#[must_use]
#[verified_engine::verified]
pub fn j_invariant_q_scaled(precision: usize) -> QSeries<i64> {
    let safe_prec = precision.min(MAX_SCALED_J_PRECISION);
    if safe_prec == 0 {
        return QSeries::new();
    }
    let e4_cubed = eisenstein_e4_q_series(safe_prec).pow(3);
    let delta_shifted = ramanujan_delta_shifted_q_series(safe_prec);
    &e4_cubed / &delta_shifted
}
