//! Polynomial-time Alexander polynomials via Fox Wirtinger calculus and reduced Burau matrices.

// [cite:knot_braid_topology]

use super::KnotError;
use super::diagram::{CrossingSign, PlanarDiagram};
use crate::pure_math::algebra::traits::Ring;
use crate::pure_math::algebra::{BraidWord, LaurentPolynomial};

/// Computes the canonicalized single-variable Alexander polynomial $\Delta_L(t) \in \mathbb{Z}[t, t^{-1}]$
/// from an oriented [`PlanarDiagram`] via Fox free-differential calculus on its Wirtinger presentation.
///
/// Runs in polynomial time $O(c^3)$ using fraction-free Bareiss elimination over $\mathbb{Z}[t, t^{-1}]$,
/// remaining fast even above [`super::MAX_KAUFFMAN_CROSSINGS`].
///
/// # Errors
/// Returns [`KnotError`] if `diagram.validate()` fails.
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
    let t_pos = LaurentPolynomial::<i128>::monomial(1, 1);
    let one_minus_t = LaurentPolynomial::<i128>::new(0, vec![1, -1]);
    let minus_one = LaurentPolynomial::<i128>::constant(-1);

    let mut mat = vec![vec![LaurentPolynomial::<i128>::zero(); minor_dim]; minor_dim];
    for (row_idx, crossing) in diagram.crossings.iter().take(minor_dim).enumerate() {
        let a_over = arc_of_seg[crossing.over_in()];
        let a_ui = arc_of_seg[crossing.under_in()];
        let a_uo = arc_of_seg[crossing.under_out()];

        if a_over < minor_dim {
            mat[row_idx][a_over] += &one_minus_t;
        }
        match crossing.sign {
            CrossingSign::Positive => {
                if a_ui < minor_dim {
                    mat[row_idx][a_ui] += &t_pos;
                }
                if a_uo < minor_dim {
                    mat[row_idx][a_uo] += &minus_one;
                }
            }
            CrossingSign::Negative => {
                if a_ui < minor_dim {
                    mat[row_idx][a_ui] += &minus_one;
                }
                if a_uo < minor_dim {
                    mat[row_idx][a_uo] += &t_pos;
                }
            }
        }
    }

    let det_128 = bareiss_laurent_det(&mut mat)?;
    let norm = downcast_laurent_i64(&det_128)?.normalize_alexander();
    assert!(norm.eval_at_one() >= 0);
    Ok(norm)
}

/// Evaluates the $(n - 1) \times (n - 1)$ symbolic reduced Burau matrix $\bar{\psi}(\beta)(t)$
/// of an Artin [`BraidWord`] over $\mathbb{Z}[t, t^{-1}]$.
#[must_use]
#[verified_engine::verified]
pub fn reduced_burau_matrix(braid: &BraidWord) -> Vec<Vec<LaurentPolynomial<i64>>> {
    let mat128 = reduced_burau_matrix_i128(braid);
    mat128
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|p| {
                    LaurentPolynomial::new(
                        p.min_exp,
                        p.coeffs.into_iter().map(|v| v as i64).collect(),
                    )
                })
                .collect()
        })
        .collect()
}

/// Computes the canonicalized Alexander polynomial $\Delta_{\hat{\beta}}(t)$ of a braid closure
/// via Burau's determinant formula:
/// $$\Delta_{\hat{\beta}}(t) \doteq \frac{\det(I_{n-1} - \bar{\psi}(\beta)(t))}{1 + t + \dots + t^{n-1}}.$$
///
/// # Errors
/// Returns [`KnotError`] on internal arithmetic overflow.
#[verified_engine::verified]
pub fn alexander_polynomial_burau(braid: &BraidWord) -> Result<LaurentPolynomial<i64>, KnotError> {
    let n = braid.strands();
    if n <= 1 {
        return Ok(LaurentPolynomial::one());
    }
    let dim = n - 1;
    let burau = reduced_burau_matrix_i128(braid);
    let mut char_mat = vec![vec![LaurentPolynomial::<i128>::zero(); dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            char_mat[i][j] = if i == j {
                LaurentPolynomial::<i128>::one() - &burau[i][j]
            } else {
                -&burau[i][j]
            };
        }
    }
    let det_num = bareiss_laurent_det(&mut char_mat)?;
    if det_num.is_zero() {
        return Ok(LaurentPolynomial::zero());
    }
    let cyclotomic_sum = LaurentPolynomial::<i128>::new(0, vec![1_i128; n]);
    let quotient = exact_div_laurent(&det_num, &cyclotomic_sum)?;
    let norm = downcast_laurent_i64(&quotient)?.normalize_alexander();
    assert!(norm.eval_at_one() >= 0);
    Ok(norm)
}

#[verified_engine::verified]
fn reduced_burau_matrix_i128(braid: &BraidWord) -> Vec<Vec<LaurentPolynomial<i128>>> {
    let dim = braid.strands().saturating_sub(1);
    let mut mat = vec![vec![LaurentPolynomial::<i128>::zero(); dim]; dim];
    for (i, row) in mat.iter_mut().enumerate() {
        row[i] = LaurentPolynomial::one();
    }
    if dim == 0 {
        return mat;
    }
    let t_pos = LaurentPolynomial::<i128>::monomial(1, 1);
    let neg_t = LaurentPolynomial::<i128>::monomial(-1, 1);
    let t_inv = LaurentPolynomial::<i128>::monomial(1, -1);
    let neg_t_inv = LaurentPolynomial::<i128>::monomial(-1, -1);

    for &g in braid.generators() {
        let k = (g.unsigned_abs() as usize) - 1;
        for row in &mut mat {
            let m_rk = row[k].clone();
            if m_rk.is_zero() {
                continue;
            }
            if g > 0 {
                if k > 0 {
                    row[k - 1] += &(&m_rk * &t_pos);
                }
                row[k] = &m_rk * &neg_t;
                if k + 1 < dim {
                    row[k + 1] += &m_rk;
                }
            } else {
                if k > 0 {
                    row[k - 1] += &m_rk;
                }
                row[k] = &m_rk * &neg_t_inv;
                if k + 1 < dim {
                    row[k + 1] += &(&m_rk * &t_inv);
                }
            }
        }
    }
    mat
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

#[verified_engine::verified]
fn bareiss_laurent_det(
    mat: &mut [Vec<LaurentPolynomial<i128>>],
) -> Result<LaurentPolynomial<i128>, KnotError> {
    let n = mat.len();
    if n == 0 {
        return Ok(LaurentPolynomial::one());
    }
    let mut prev_pivot = LaurentPolynomial::<i128>::one();
    let mut sign = 1_i128;

    for k in 0..n {
        let Some(piv_row) = (k..n).find(|&r| !mat[r][k].is_zero()) else {
            return Ok(LaurentPolynomial::zero());
        };
        if piv_row != k {
            mat.swap(k, piv_row);
            sign = -sign;
        }
        let piv = mat[k][k].clone();
        let (top_rows, tail_rows) = mat.split_at_mut(k + 1);
        let row_k = &top_rows[k];
        for row_i in tail_rows.iter_mut() {
            let mik = row_i[k].clone();
            for (cell_ij, cell_kj) in row_i[(k + 1)..n].iter_mut().zip(&row_k[(k + 1)..n]) {
                let num = &piv * &*cell_ij - &mik * cell_kj;
                *cell_ij = exact_div_laurent(&num, &prev_pivot)?;
            }
            row_i[k] = LaurentPolynomial::zero();
        }
        prev_pivot = piv;
    }

    let last = mat[n - 1][n - 1].clone();
    Ok(if sign < 0 { -last } else { last })
}

#[verified_engine::verified]
fn exact_div_laurent(
    num: &LaurentPolynomial<i128>,
    den: &LaurentPolynomial<i128>,
) -> Result<LaurentPolynomial<i128>, KnotError> {
    if den.is_zero() {
        return Err(KnotError::InvalidParameters {
            reason: "division by zero Laurent polynomial in Bareiss elimination",
        });
    }
    if num.is_zero() {
        return Ok(LaurentPolynomial::zero());
    }
    let na = num.coeffs.len();
    let nb = den.coeffs.len();
    if na < nb {
        return Err(KnotError::InvalidParameters {
            reason: "non-exact Laurent polynomial division (degree underflow)",
        });
    }
    let lead_b = *den.coeffs.last().expect("non-zero denominator");
    let q_len = na - nb + 1;
    let mut rem = num.coeffs.clone();
    let mut q = vec![0_i128; q_len];

    for k in (0..q_len).rev() {
        let top = rem[k + nb - 1];
        if top % lead_b != 0 {
            return Err(KnotError::InvalidParameters {
                reason: "non-exact integer division in Bareiss Laurent determinant",
            });
        }
        let coeff = top / lead_b;
        q[k] = coeff;
        if coeff != 0 {
            for j in 0..nb {
                rem[k + j] -= coeff * den.coeffs[j];
            }
        }
    }
    debug_assert!(rem.iter().all(|&r| r == 0));
    Ok(LaurentPolynomial::new(num.min_exp - den.min_exp, q))
}

#[verified_engine::verified]
fn downcast_laurent_i64(
    poly: &LaurentPolynomial<i128>,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    let mut out = Vec::with_capacity(poly.coeffs.len());
    for &c in &poly.coeffs {
        let v = i64::try_from(c).map_err(|_| KnotError::InvalidParameters {
            reason: "Laurent polynomial coefficient overflowed i64",
        })?;
        out.push(v);
    }
    Ok(LaurentPolynomial::new(poly.min_exp, out))
}
