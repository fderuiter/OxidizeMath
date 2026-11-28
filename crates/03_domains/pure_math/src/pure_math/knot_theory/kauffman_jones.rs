//! Kauffman bracket state sum, normalized Jones polynomial, and link chirality checks.

// [cite:knot_braid_topology]

use super::diagram::PlanarDiagram;
use super::{KnotError, MAX_KAUFFMAN_CROSSINGS};
use crate::pure_math::algebra::traits::Ring;
use crate::pure_math::algebra::{LaurentError, LaurentPolynomial};

/// Maximum loop exponent $k$ for $d^k = (-A^2 - A^{-2})^k$ before signed 64-bit binomial / evaluation bounds overflow.
pub const MAX_SAFE_KAUFFMAN_LOOPS: usize = 62;

/// Computes the Kauffman bracket $\langle D \rangle \in \mathbb{Z}[A, A^{-1}]$ by exact state-sum
/// expansion over all $2^c$ smoothings, guarded by [`MAX_KAUFFMAN_CROSSINGS`] (`16`).
///
/// - **A-smoothing** (weight $A$) pairs `(slots[0], slots[1])` and `(slots[2], slots[3])`.
/// - **B-smoothing** (weight $A^{-1}$) pairs `(slots[0], slots[3])` and `(slots[1], slots[2])`.
/// - Closed loop weight $d = -A^2 - A^{-2}$ with $\langle \bigcirc \rangle = 1$.
///
/// # Errors
/// Returns [`KnotError::CrossingCapExceeded`] if `diagram.crossing_count() > MAX_KAUFFMAN_CROSSINGS`,
/// [`KnotError::CoefficientOverflow`] above `62` loops or on `i64` state-sum overflow,
/// or [`KnotError::InvalidDiagram`] if `diagram.validate()` fails.
#[verified_engine::verified]
pub fn kauffman_bracket(diagram: &PlanarDiagram) -> Result<LaurentPolynomial<i64>, KnotError> {
    diagram.validate()?;
    let c = diagram.crossing_count();
    if c > MAX_KAUFFMAN_CROSSINGS {
        return Err(KnotError::CrossingCapExceeded {
            crossings: c,
            max_crossings: MAX_KAUFFMAN_CROSSINGS,
        });
    }

    let max_loops = c
        .checked_add(diagram.free_loops)
        .ok_or(KnotError::CoefficientOverflow)?;
    let d_powers = precompute_loop_powers(max_loops)?;
    if c == 0 {
        return Ok(d_powers[diagram.free_loops.saturating_sub(1)].clone());
    }

    let m = diagram.segments.len();
    let half_edges = 2 * m;
    let mut slot_half = vec![[0_usize; 4]; c];
    for (seg_id, seg) in diagram.segments.iter().enumerate() {
        slot_half[seg.tail.crossing][seg.tail.slot as usize] = 2 * seg_id;
        slot_half[seg.head.crossing][seg.head.slot as usize] = 2 * seg_id + 1;
    }

    let mut crossing_partner = vec![0_usize; half_edges];
    let mut visited = vec![false; half_edges];
    let mut total = LaurentPolynomial::<i64>::zero();
    let num_states = 1_usize << c;

    for state in 0..num_states {
        let mut a_minus_b = 0_i32;
        for (cid, sh) in slot_half.iter().enumerate() {
            let is_b = ((state >> cid) & 1) == 1;
            if is_b {
                a_minus_b -= 1;
                pair_halves(&mut crossing_partner, sh[0], sh[3]);
                pair_halves(&mut crossing_partner, sh[1], sh[2]);
            } else {
                a_minus_b += 1;
                pair_halves(&mut crossing_partner, sh[0], sh[1]);
                pair_halves(&mut crossing_partner, sh[2], sh[3]);
            }
        }
        let state_loops = count_smoothed_loops(half_edges, &crossing_partner, &mut visited);
        let total_loops = state_loops + diagram.free_loops;
        debug_assert!(total_loops >= 1);
        let term = d_powers[total_loops - 1].try_shift(a_minus_b)?;
        total = total.checked_add(&term).map_err(|e| match e {
            LaurentError::CoefficientOverflow => KnotError::CoefficientOverflow,
            other => KnotError::Laurent(other),
        })?;
    }
    assert!(!total.coeffs.is_empty());
    Ok(total)
}

/// Computes Kauffman's writhe-normalized ambient isotopy invariant
/// $$X_L(A) = (-A^3)^{-w(D)} \langle D \rangle \in \mathbb{Z}[A, A^{-1}].$$
///
/// # Errors
/// Propagates [`KnotError`] from [`kauffman_bracket`] or checked exponent/sign arithmetic.
#[verified_engine::verified]
pub fn normalized_kauffman_polynomial(
    diagram: &PlanarDiagram,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    let bracket = kauffman_bracket(diagram)?;
    let w = diagram.writhe();
    let sign = if w.rem_euclid(2) == 0 { 1_i64 } else { -1_i64 };
    let shift_i64 = w.checked_mul(-3).ok_or(LaurentError::ExponentOverflow)?;
    let shift_i32 = i32::try_from(shift_i64).map_err(|_| LaurentError::ExponentOverflow)?;
    let shifted = bracket.try_shift(shift_i32)?;
    if sign < 0 {
        shifted.checked_neg().map_err(|e| match e {
            LaurentError::CoefficientOverflow => KnotError::CoefficientOverflow,
            other => KnotError::Laurent(other),
        })
    } else {
        Ok(shifted)
    }
}

/// Computes the Jones polynomial $V_L(t) = X_L(t^{-1/4})$ in **half-powers** $q = t^{1/2} = A^{-2}$,
/// well-defined as a [`LaurentPolynomial<i64>`] for **every** classical knot and multi-component link.
///
/// Format human-readable $t$-expressions via [`LaurentPolynomial::format_half_power_in_var`]`("t")`.
///
/// # Errors
/// Returns [`KnotError::InvalidDiagram`] if any non-zero $A$-exponent is odd, or propagates [`KnotError`].
#[verified_engine::verified]
pub fn jones_polynomial_half_power(
    diagram: &PlanarDiagram,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    let x_poly = normalized_kauffman_polynomial(diagram)?;
    half_powers_from_kauffman_x(&x_poly)
}

#[verified_engine::verified]
fn half_powers_from_kauffman_x(
    x_poly: &LaurentPolynomial<i64>,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    let mut terms = Vec::with_capacity(x_poly.coeffs.len());
    for (idx, &coeff) in x_poly.coeffs.iter().enumerate() {
        if coeff != 0 {
            let idx_i32 = i32::try_from(idx).map_err(|_| LaurentError::ExponentOverflow)?;
            let exp_a = x_poly
                .min_exp
                .checked_add(idx_i32)
                .ok_or(LaurentError::ExponentOverflow)?;
            if exp_a % 2 != 0 {
                return Err(KnotError::InvalidDiagram {
                    reason: "normalized Kauffman polynomial produced an odd A-exponent on a non-classical diagram",
                });
            }
            let neg_exp_a = exp_a.checked_neg().ok_or(LaurentError::ExponentOverflow)?;
            terms.push((neg_exp_a / 2, coeff));
        }
    }
    Ok(LaurentPolynomial::checked_from_terms(&terms)?)
}

/// Computes the Jones polynomial $V_L(t)$:
/// - For knots and odd-component links (`component_count() % 2 == 1`), exponents are in whole
///   powers of $t = A^{-4}$ (e.g. right trefoil $t + t^3 - t^4$, left trefoil $-t^{-4} + t^{-3} + t^{-1}$).
/// - For even-component links (`component_count() % 2 == 0`, e.g. Hopf link), exponents are in
///   half-powers $q = t^{1/2} = A^{-2}$.
///
/// # Errors
/// Returns [`KnotError::InvalidDiagram`] if an odd-component diagram produces odd half-powers `exp_q % 2 != 0`,
/// or propagates [`KnotError`] from [`jones_polynomial_half_power`].
#[verified_engine::verified]
pub fn jones_polynomial(diagram: &PlanarDiagram) -> Result<LaurentPolynomial<i64>, KnotError> {
    let q_poly = jones_polynomial_half_power(diagram)?;
    if diagram.component_count().is_multiple_of(2) {
        return Ok(q_poly);
    }
    whole_powers_from_half_q(&q_poly)
}

#[verified_engine::verified]
fn whole_powers_from_half_q(
    q_poly: &LaurentPolynomial<i64>,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    let mut terms = Vec::with_capacity(q_poly.coeffs.len());
    for (idx, &coeff) in q_poly.coeffs.iter().enumerate() {
        if coeff != 0 {
            let idx_i32 = i32::try_from(idx).map_err(|_| LaurentError::ExponentOverflow)?;
            let exp_q = q_poly
                .min_exp
                .checked_add(idx_i32)
                .ok_or(LaurentError::ExponentOverflow)?;
            if exp_q % 2 != 0 {
                return Err(KnotError::InvalidDiagram {
                    reason: "odd-component link produced fractional Jones half-power; diagram is non-planar",
                });
            }
            terms.push((exp_q / 2, coeff));
        }
    }
    Ok(LaurentPolynomial::checked_from_terms(&terms)?)
}

/// Formats the normalized Jones polynomial in variable `"t"`, rendering fractional exponents
/// `t^(k/2)` automatically for even-component links.
///
/// # Errors
/// Propagates [`KnotError`] from [`jones_polynomial_half_power`].
pub fn format_jones_polynomial(diagram: &PlanarDiagram) -> Result<String, KnotError> {
    let q_poly = jones_polynomial_half_power(diagram)?;
    Ok(q_poly.format_half_power_in_var("t"))
}

/// Returns `true` when $V_L(t) \neq V_L(t^{-1})$, certifying that the oriented knot or link
/// is topologically chiral (not ambient-isotopic to its mirror image).
///
/// # Errors
/// Propagates [`KnotError`] from [`jones_polynomial_half_power`].
#[verified_engine::verified]
pub fn is_chiral_by_jones(diagram: &PlanarDiagram) -> Result<bool, KnotError> {
    let q_poly = jones_polynomial_half_power(diagram)?;
    Ok(!q_poly.is_reciprocal())
}

#[inline]
#[verified_engine::verified]
fn pair_halves(partner: &mut [usize], h1: usize, h2: usize) {
    debug_assert_ne!(h1, h2);
    partner[h1] = h2;
    partner[h2] = h1;
}

#[verified_engine::verified]
fn count_smoothed_loops(
    half_edges: usize,
    crossing_partner: &[usize],
    visited: &mut [bool],
) -> usize {
    visited.fill(false);
    let mut loops = 0;
    for start in 0..half_edges {
        if visited[start] {
            continue;
        }
        loops += 1;
        let mut cur = start;
        let mut guard = 0;
        while !visited[cur] && guard <= half_edges {
            visited[cur] = true;
            let other_end = cur ^ 1;
            visited[other_end] = true;
            cur = crossing_partner[other_end];
            guard += 2;
        }
    }
    debug_assert!(half_edges == 0 || loops >= 1);
    loops
}

#[verified_engine::verified]
fn precompute_loop_powers(max_power: usize) -> Result<Vec<LaurentPolynomial<i64>>, KnotError> {
    if max_power > MAX_SAFE_KAUFFMAN_LOOPS {
        return Err(KnotError::CoefficientOverflow);
    }
    let d = LaurentPolynomial::try_new(-2, vec![-1_i64, 0, 0, 0, -1])?;
    let mut powers = Vec::with_capacity(max_power + 1);
    powers.push(LaurentPolynomial::one());
    for k in 1..=max_power {
        let next = powers[k - 1].checked_mul(&d).map_err(|e| match e {
            LaurentError::CoefficientOverflow => KnotError::CoefficientOverflow,
            other => KnotError::Laurent(other),
        })?;
        powers.push(next);
    }
    assert_eq!(powers.len(), max_power + 1);
    Ok(powers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_exponents_in_kauffman_and_half_power_jones_return_invalid_diagram() {
        let odd_a = LaurentPolynomial::new(-1, vec![1_i64]);
        assert!(matches!(
            half_powers_from_kauffman_x(&odd_a),
            Err(KnotError::InvalidDiagram { .. })
        ));
        let odd_q = LaurentPolynomial::new(3, vec![1_i64]);
        assert!(matches!(
            whole_powers_from_half_q(&odd_q),
            Err(KnotError::InvalidDiagram { .. })
        ));
    }
}
