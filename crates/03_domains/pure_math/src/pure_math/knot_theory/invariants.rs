//! Polynomial-time Alexander polynomials via Fox Wirtinger calculus and reduced Burau matrices.

// [cite:knot_braid_topology]

mod bareiss_ibig;

use super::KnotError;
use super::diagram::{CrossingSign, PlanarDiagram};
use crate::pure_math::algebra::traits::Ring;
use crate::pure_math::algebra::{BraidWord, LaurentPolynomial};
use bareiss_ibig::{
    bareiss_laurent_det, downcast_laurent_ibig_to_i64, exact_div_laurent,
    normalize_alexander_ibig_to_i64, reduced_burau_matrix_ibig,
};
use dashu::integer::IBig;

/// Computes the canonicalized single-variable Alexander polynomial $\Delta_L(t) \in \mathbb{Z}[t, t^{-1}]$
/// from an oriented [`PlanarDiagram`] via Fox free-differential calculus on its Wirtinger presentation.
///
/// Runs in polynomial time $O(c^3)$ using fraction-free Bareiss elimination over arbitrary-precision
/// $\mathbb{Z}[t, t^{-1}]$, remaining fast and free of intermediate overflow even above
/// [`super::MAX_KAUFFMAN_CROSSINGS`].
///
/// # Errors
/// Returns [`KnotError`] if `diagram.validate()` fails or if final normalized coefficients exceed `i64`.
#[verified_engine::verified]
pub fn alexander_polynomial_fox(
    diagram: &PlanarDiagram,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    diagram.validate()?;
    let c = diagram.crossings.len();
    if c == 0 {
        return Ok(if diagram.free_loops == 1 {
            LaurentPolynomial::one()
        } else {
            LaurentPolynomial::zero()
        });
    }
    if diagram.free_loops > 0 {
        return Ok(LaurentPolynomial::zero());
    }

    let (num_arcs, arc_of_seg) = build_wirtinger_arcs(diagram);
    if num_arcs > c {
        return Ok(LaurentPolynomial::zero());
    }
    if num_arcs <= 1 {
        return Ok(LaurentPolynomial::one());
    }

    let minor_dim = num_arcs - 1;
    let t_pos = LaurentPolynomial::<IBig>::monomial(IBig::ONE, 1);
    let one_minus_t = LaurentPolynomial::<IBig>::try_new(0, vec![IBig::ONE, IBig::NEG_ONE])?;
    let minus_one = LaurentPolynomial::<IBig>::constant(IBig::NEG_ONE);

    let mut mat = vec![vec![LaurentPolynomial::<IBig>::zero(); minor_dim]; minor_dim];
    for (row_idx, crossing) in diagram.crossings.iter().take(minor_dim).enumerate() {
        let a_over = arc_of_seg[crossing.over_in()];
        let a_ui = arc_of_seg[crossing.under_in()];
        let a_uo = arc_of_seg[crossing.under_out()];

        if a_over < minor_dim {
            mat[row_idx][a_over] = mat[row_idx][a_over].try_add(&one_minus_t)?;
        }
        match crossing.sign {
            CrossingSign::Positive => {
                if a_ui < minor_dim {
                    mat[row_idx][a_ui] = mat[row_idx][a_ui].try_add(&t_pos)?;
                }
                if a_uo < minor_dim {
                    mat[row_idx][a_uo] = mat[row_idx][a_uo].try_add(&minus_one)?;
                }
            }
            CrossingSign::Negative => {
                if a_ui < minor_dim {
                    mat[row_idx][a_ui] = mat[row_idx][a_ui].try_add(&minus_one)?;
                }
                if a_uo < minor_dim {
                    mat[row_idx][a_uo] = mat[row_idx][a_uo].try_add(&t_pos)?;
                }
            }
        }
    }

    let det_ibig = bareiss_laurent_det(&mut mat)?;
    normalize_alexander_ibig_to_i64(&det_ibig)
}

/// Fallibly evaluates the $(n - 1) \times (n - 1)$ symbolic reduced Burau matrix $\bar{\psi}(\beta)(t)$
/// of an Artin [`BraidWord`] over $\mathbb{Z}[t, t^{-1}]$, checking `i64` bounds on every entry.
///
/// # Errors
/// Returns [`KnotError::CoefficientOverflow`] if any matrix entry coefficient overflows `i64`.
#[verified_engine::verified]
pub fn try_reduced_burau_matrix(
    braid: &BraidWord,
) -> Result<Vec<Vec<LaurentPolynomial<i64>>>, KnotError> {
    let mat_ibig = reduced_burau_matrix_ibig(braid)?;
    let mut out = Vec::with_capacity(mat_ibig.len());
    for row in &mat_ibig {
        let mut out_row = Vec::with_capacity(row.len());
        for poly in row {
            out_row.push(downcast_laurent_ibig_to_i64(poly)?);
        }
        out.push(out_row);
    }
    Ok(out)
}

/// Evaluates the $(n - 1) \times (n - 1)$ symbolic reduced Burau matrix $\bar{\psi}(\beta)(t)$
/// of an Artin [`BraidWord`] over $\mathbb{Z}[t, t^{-1}]$.
///
/// # Panics
/// Panics if any matrix entry coefficient overflows `i64`; prefer [`try_reduced_burau_matrix`].
#[must_use]
#[verified_engine::verified]
pub fn reduced_burau_matrix(braid: &BraidWord) -> Vec<Vec<LaurentPolynomial<i64>>> {
    try_reduced_burau_matrix(braid).expect("reduced_burau_matrix entry overflowed i64")
}

/// Computes the canonicalized Alexander polynomial $\Delta_{\hat{\beta}}(t)$ of a braid closure
/// via Burau's determinant formula:
/// $$\Delta_{\hat{\beta}}(t) \doteq \frac{\det(I_{n-1} - \bar{\psi}(\beta)(t))}{1 + t + \dots + t^{n-1}}.$$
///
/// # Errors
/// Returns [`KnotError`] if final normalized coefficients exceed `i64`.
#[verified_engine::verified]
pub fn alexander_polynomial_burau(braid: &BraidWord) -> Result<LaurentPolynomial<i64>, KnotError> {
    let n = braid.strands();
    if n <= 1 {
        return Ok(LaurentPolynomial::one());
    }
    let dim = n - 1;
    let burau = reduced_burau_matrix_ibig(braid)?;
    let mut char_mat = vec![vec![LaurentPolynomial::<IBig>::zero(); dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            char_mat[i][j] = if i == j {
                LaurentPolynomial::<IBig>::one().try_sub(&burau[i][j])?
            } else {
                -&burau[i][j]
            };
        }
    }
    let det_num = bareiss_laurent_det(&mut char_mat)?;
    if det_num.is_zero() {
        return Ok(LaurentPolynomial::zero());
    }
    let cyclotomic_sum = LaurentPolynomial::<IBig>::try_new(0, vec![IBig::ONE; n])?;
    let quotient = exact_div_laurent(&det_num, &cyclotomic_sum)?;
    normalize_alexander_ibig_to_i64(&quotient)
}

#[verified_engine::verified]
fn build_wirtinger_arcs(diagram: &PlanarDiagram) -> (usize, Vec<usize>) {
    let m = diagram.segments.len();
    let mut parent: Vec<usize> = (0..m).collect();
    for c in &diagram.crossings {
        union_sets(&mut parent, c.over_in(), c.over_out());
    }
    let mut compact_id = vec![usize::MAX; m];
    let mut arc_of_seg = vec![0; m];
    let mut count = 0;
    for (s, slot) in arc_of_seg.iter_mut().enumerate().take(m) {
        let root = find_root(&mut parent, s);
        if compact_id[root] == usize::MAX {
            compact_id[root] = count;
            count += 1;
        }
        *slot = compact_id[root];
    }
    debug_assert!(count <= m);
    (count, arc_of_seg)
}

#[verified_engine::verified]
fn find_root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    debug_assert_eq!(parent[x], x);
    x
}

#[verified_engine::verified]
fn union_sets(parent: &mut [usize], a: usize, b: usize) {
    let ra = find_root(parent, a);
    let rb = find_root(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}
