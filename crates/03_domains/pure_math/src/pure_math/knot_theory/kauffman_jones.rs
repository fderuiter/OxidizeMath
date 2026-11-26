//! Kauffman bracket state sum, normalized Jones polynomial, and link chirality checks.

// [cite:knot_braid_topology]

use super::diagram::PlanarDiagram;
use super::{KnotError, MAX_KAUFFMAN_CROSSINGS};
use crate::pure_math::algebra::LaurentPolynomial;
use crate::pure_math::algebra::traits::Ring;

/// Computes the Kauffman bracket $\langle D \rangle \in \mathbb{Z}[A, A^{-1}]$ by exact state-sum
/// expansion over all $2^c$ smoothings, guarded by [`MAX_KAUFFMAN_CROSSINGS`] (`16`).
///
/// - **A-smoothing** (weight $A$) pairs `(slots[0], slots[1])` and `(slots[2], slots[3])`.
/// - **B-smoothing** (weight $A^{-1}$) pairs `(slots[0], slots[3])` and `(slots[1], slots[2])`.
/// - Closed loop weight $d = -A^2 - A^{-2}$ with $\langle \bigcirc \rangle = 1$.
///
/// # Errors
/// Returns [`KnotError::CrossingCapExceeded`] if `diagram.crossing_count() > MAX_KAUFFMAN_CROSSINGS`,
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

    let d_powers = precompute_loop_powers(c + diagram.free_loops);
    if c == 0 {
        return Ok(d_powers[diagram.free_loops.saturating_sub(1)].clone());
    }

    let m = diagram.segments.len();
    let half_edges = 2 * m;
    // Map each crossing slot (cid, slot) to its unique segment half-edge index:
    // even index `2 * seg` = tail of `seg`, odd index `2 * seg + 1` = head of `seg`.
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
        let term = d_powers[total_loops - 1].shift(a_minus_b);
        total += term;
    }
    assert!(!total.coeffs.is_empty());
    Ok(total)
}

/// Computes Kauffman's writhe-normalized ambient isotopy invariant
/// $$X_L(A) = (-A^3)^{-w(D)} \langle D \rangle \in \mathbb{Z}[A, A^{-1}].$$
///
/// # Errors
/// Propagates [`KnotError`] from [`kauffman_bracket`].
#[verified_engine::verified]
pub fn normalized_kauffman_polynomial(
    diagram: &PlanarDiagram,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    let bracket = kauffman_bracket(diagram)?;
    let w = diagram.writhe();
    let sign = if w.rem_euclid(2) == 0 { 1_i64 } else { -1_i64 };
    let shifted = bracket.shift((-3 * w) as i32);
    Ok(if sign < 0 { -shifted } else { shifted })
}

/// Computes the Jones polynomial $V_L(t) = X_L(t^{-1/4})$ in **half-powers** $q = t^{1/2} = A^{-2}$,
/// well-defined as a [`LaurentPolynomial<i64>`] for **every** knot and multi-component link.
///
/// Format human-readable $t$-expressions via [`LaurentPolynomial::format_half_power_in_var`]`("t")`.
///
/// # Errors
/// Propagates [`KnotError`] from [`normalized_kauffman_polynomial`].
#[verified_engine::verified]
pub fn jones_polynomial_half_power(
    diagram: &PlanarDiagram,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    let x_poly = normalized_kauffman_polynomial(diagram)?;
    let mut terms = Vec::with_capacity(x_poly.coeffs.len());
    for (idx, &coeff) in x_poly.coeffs.iter().enumerate() {
        if coeff != 0 {
            let exp_a = x_poly.min_exp + idx as i32;
            terms.push((-exp_a / 2, coeff));
        }
    }
    Ok(LaurentPolynomial::from_terms(&terms))
}

/// Computes the Jones polynomial $V_L(t)$:
/// - For knots and odd-component links (`component_count() % 2 == 1`), exponents are in whole
///   powers of $t = A^{-4}$ (e.g. right trefoil $t + t^3 - t^4$, left trefoil $-t^{-4} + t^{-3} + t^{-1}$).
/// - For even-component links (`component_count() % 2 == 0`, e.g. Hopf link), exponents are in
///   half-powers $q = t^{1/2} = A^{-2}$.
///
/// # Errors
/// Propagates [`KnotError`] from [`normalized_kauffman_polynomial`].
#[verified_engine::verified]
pub fn jones_polynomial(diagram: &PlanarDiagram) -> Result<LaurentPolynomial<i64>, KnotError> {
    let q_poly = jones_polynomial_half_power(diagram)?;
    if diagram.component_count().is_multiple_of(2) {
        return Ok(q_poly);
    }
    let mut terms = Vec::with_capacity(q_poly.coeffs.len());
    for (idx, &coeff) in q_poly.coeffs.iter().enumerate() {
        if coeff != 0 {
            let exp_q = q_poly.min_exp + idx as i32;
            terms.push((exp_q / 2, coeff));
        }
    }
    Ok(LaurentPolynomial::from_terms(&terms))
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
            // Traverse along the segment (`2*s <-> 2*s + 1`), then across the smoothed crossing!
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
fn precompute_loop_powers(max_power: usize) -> Vec<LaurentPolynomial<i64>> {
    let d = LaurentPolynomial::new(-2, vec![-1_i64, 0, 0, 0, -1]);
    let mut powers = Vec::with_capacity(max_power + 1);
    powers.push(LaurentPolynomial::one());
    for k in 1..=max_power {
        let next = &powers[k - 1] * &d;
        powers.push(next);
    }
    assert_eq!(powers.len(), max_power + 1);
    powers
}
