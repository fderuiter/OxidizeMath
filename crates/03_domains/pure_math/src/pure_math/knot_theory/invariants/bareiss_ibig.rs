//! Arbitrary-precision `IBig` Bareiss determinant, exact Laurent division, and Burau matrix helpers.

// [cite:knot_braid_topology]

use crate::pure_math::algebra::traits::Ring;
use crate::pure_math::algebra::{BraidWord, LaurentPolynomial};
use crate::pure_math::knot_theory::KnotError;
use dashu::integer::IBig;

#[verified_engine::verified]
pub(super) fn reduced_burau_matrix_ibig(
    braid: &BraidWord,
) -> Result<Vec<Vec<LaurentPolynomial<IBig>>>, KnotError> {
    let dim = braid.strands().saturating_sub(1);
    let mut mat = vec![vec![LaurentPolynomial::<IBig>::zero(); dim]; dim];
    for (i, row) in mat.iter_mut().enumerate() {
        row[i] = LaurentPolynomial::one();
    }
    if dim == 0 {
        return Ok(mat);
    }
    let t_pos = LaurentPolynomial::<IBig>::monomial(IBig::ONE, 1);
    let neg_t = LaurentPolynomial::<IBig>::monomial(IBig::NEG_ONE, 1);
    let t_inv = LaurentPolynomial::<IBig>::monomial(IBig::ONE, -1);
    let neg_t_inv = LaurentPolynomial::<IBig>::monomial(IBig::NEG_ONE, -1);

    for &g in braid.generators() {
        let idx = g.unsigned_abs() as usize;
        if idx == 0 || idx >= braid.strands() {
            return Err(KnotError::InvalidParameters {
                reason: "braid generator index out of range in reduced Burau matrix",
            });
        }
        let k = idx - 1;
        for row in &mut mat {
            let m_rk = row[k].clone();
            if m_rk.is_zero() {
                continue;
            }
            if g > 0 {
                if k > 0 {
                    let shifted = m_rk.try_mul(&t_pos)?;
                    row[k - 1] = row[k - 1].try_add(&shifted)?;
                }
                row[k] = m_rk.try_mul(&neg_t)?;
                if k + 1 < dim {
                    row[k + 1] = row[k + 1].try_add(&m_rk)?;
                }
            } else {
                if k > 0 {
                    row[k - 1] = row[k - 1].try_add(&m_rk)?;
                }
                row[k] = m_rk.try_mul(&neg_t_inv)?;
                if k + 1 < dim {
                    let shifted = m_rk.try_mul(&t_inv)?;
                    row[k + 1] = row[k + 1].try_add(&shifted)?;
                }
            }
        }
    }
    Ok(mat)
}

#[verified_engine::verified]
pub(super) fn bareiss_laurent_det(
    mat: &mut [Vec<LaurentPolynomial<IBig>>],
) -> Result<LaurentPolynomial<IBig>, KnotError> {
    let n = mat.len();
    if n == 0 {
        return Ok(LaurentPolynomial::one());
    }
    let mut prev_pivot = LaurentPolynomial::<IBig>::one();
    let mut neg_sign = false;

    for k in 0..n {
        let Some(piv_row) = (k..n).find(|&r| !mat[r][k].is_zero()) else {
            return Ok(LaurentPolynomial::zero());
        };
        if piv_row != k {
            mat.swap(k, piv_row);
            neg_sign = !neg_sign;
        }
        let piv = mat[k][k].clone();
        let (top_rows, tail_rows) = mat.split_at_mut(k + 1);
        let row_k = &top_rows[k];
        for row_i in tail_rows.iter_mut() {
            let mik = row_i[k].clone();
            for (cell_ij, cell_kj) in row_i[(k + 1)..n].iter_mut().zip(&row_k[(k + 1)..n]) {
                let lhs = piv.try_mul(cell_ij)?;
                let rhs = mik.try_mul(cell_kj)?;
                let num = lhs.try_sub(&rhs)?;
                *cell_ij = exact_div_laurent(&num, &prev_pivot)?;
            }
            row_i[k] = LaurentPolynomial::zero();
        }
        prev_pivot = piv;
    }

    let last = mat[n - 1][n - 1].clone();
    Ok(if neg_sign { -&last } else { last })
}

#[verified_engine::verified]
pub(super) fn exact_div_laurent(
    num: &LaurentPolynomial<IBig>,
    den: &LaurentPolynomial<IBig>,
) -> Result<LaurentPolynomial<IBig>, KnotError> {
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
    let lead_b = den.coeffs.last().expect("non-zero denominator");
    let q_len = na - nb + 1;
    let mut rem = num.coeffs.clone();
    let mut q = vec![IBig::ZERO; q_len];

    for k in (0..q_len).rev() {
        let top = &rem[k + nb - 1];
        if !(top % lead_b).is_zero() {
            return Err(KnotError::InvalidParameters {
                reason: "non-exact integer division in Bareiss Laurent determinant",
            });
        }
        let coeff = top / lead_b;
        if !coeff.is_zero() {
            for j in 0..nb {
                let prod = &coeff * &den.coeffs[j];
                rem[k + j] -= prod;
            }
        }
        q[k] = coeff;
    }
    if !rem.iter().all(IBig::is_zero) {
        return Err(KnotError::InvalidParameters {
            reason: "non-zero remainder in exact Laurent polynomial division",
        });
    }
    let min_exp = num
        .min_exp
        .checked_sub(den.min_exp)
        .ok_or(crate::pure_math::algebra::LaurentError::ExponentOverflow)?;
    Ok(LaurentPolynomial::try_new(min_exp, q)?)
}

#[verified_engine::verified]
pub(super) fn normalize_alexander_ibig_to_i64(
    poly: &LaurentPolynomial<IBig>,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    if poly.coeffs.is_empty() {
        return Ok(LaurentPolynomial::zero());
    }
    let span_i32 = i32::try_from(poly.span())
        .map_err(|_| crate::pure_math::algebra::LaurentError::ExponentOverflow)?;
    let target_min = -(span_i32 / 2);
    let mut sum = IBig::ZERO;
    for c in &poly.coeffs {
        sum += c;
    }
    let negate = if !sum.is_zero() {
        sum < IBig::ZERO
    } else {
        poly.coeffs.last().is_some_and(|c| *c < IBig::ZERO)
    };
    let mut out = Vec::with_capacity(poly.coeffs.len());
    for c in &poly.coeffs {
        let signed = if negate { -c } else { c.clone() };
        let val = i64::try_from(signed).map_err(|_| KnotError::CoefficientOverflow)?;
        out.push(val);
    }
    Ok(LaurentPolynomial::try_new(target_min, out)?)
}

#[verified_engine::verified]
pub(super) fn downcast_laurent_ibig_to_i64(
    poly: &LaurentPolynomial<IBig>,
) -> Result<LaurentPolynomial<i64>, KnotError> {
    let mut out = Vec::with_capacity(poly.coeffs.len());
    for c in &poly.coeffs {
        let v = i64::try_from(c.clone()).map_err(|_| KnotError::CoefficientOverflow)?;
        out.push(v);
    }
    Ok(LaurentPolynomial::try_new(poly.min_exp, out)?)
}
