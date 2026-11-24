#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Golub-Kahan Householder bidiagonalization and implicit Wilkinson-shifted QR SVD.

use crate::scalar::Scalar;
use crate::svd_jacobi::sort_thin_svd;
use crate::types::{LinalgError, SolverWorkspace, check_all_finite, mat_idx};

/// Workspace scalar capacity required by [`svd_bidiagonal`] for an `m x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn svd_bidiagonal_workspace_len(m: usize, n: usize) -> usize {
    if m >= n {
        m * n + n * n + 4 * n
    } else {
        2 * n * m + 2 * m * m + 4 * m
    }
}

#[inline]
fn givens_cs<T: Scalar>(f: T, g: T) -> (T, T, T) {
    if g == T::ZERO {
        (T::ONE, T::ZERO, f)
    } else if f == T::ZERO {
        let sign = if g >= T::ZERO { T::ONE } else { -T::ONE };
        (T::ZERO, sign, g.abs())
    } else {
        let r = f.hypot(g);
        (f / r, g / r, r)
    }
}

#[inline]
fn apply_col_givens<T: Scalar>(
    mat: &mut [T],
    nrows: usize,
    ncols: usize,
    col_a: usize,
    col_b: usize,
    c: T,
    s: T,
) {
    for r in 0..nrows {
        let ma = mat[mat_idx(r, col_a, ncols)];
        let mb = mat[mat_idx(r, col_b, ncols)];
        mat[mat_idx(r, col_a, ncols)] = c * ma + s * mb;
        mat[mat_idx(r, col_b, ncols)] = -s * ma + c * mb;
    }
}

fn col_tail_norm<T: Scalar>(mat: &[T], m: usize, n: usize, col: usize, start_row: usize) -> T {
    let mut scale = T::ZERO;
    let mut ssq = T::ONE;
    for r in start_row..m {
        let v = mat[mat_idx(r, col, n)].abs();
        if v > T::ZERO {
            if scale < v {
                let ratio = scale / v;
                ssq = T::ONE + ssq * ratio * ratio;
                scale = v;
            } else {
                let ratio = v / scale;
                ssq += ratio * ratio;
            }
        }
    }
    scale * ssq.sqrt()
}

fn row_tail_norm<T: Scalar>(mat: &[T], n: usize, row: usize, start_col: usize) -> T {
    let mut scale = T::ZERO;
    let mut ssq = T::ONE;
    for c in start_col..n {
        let v = mat[mat_idx(row, c, n)].abs();
        if v > T::ZERO {
            if scale < v {
                let ratio = scale / v;
                ssq = T::ONE + ssq * ratio * ratio;
                scale = v;
            } else {
                let ratio = v / scale;
                ssq += ratio * ratio;
            }
        }
    }
    scale * ssq.sqrt()
}

fn bidiag_left_householder<T: Scalar>(work_a: &mut [T], m: usize, n: usize, k: usize) -> (T, T) {
    let tail = if k + 1 < m {
        col_tail_norm(work_a, m, n, k, k + 1)
    } else {
        T::ZERO
    };
    if tail == T::ZERO {
        return (T::ZERO, work_a[mat_idx(k, k, n)]);
    }
    let x0 = work_a[mat_idx(k, k, n)];
    let norm_x = col_tail_norm(work_a, m, n, k, k);
    let alpha = -norm_x.copysign(x0);
    let v0 = x0 - alpha;
    let tau = -v0 / alpha;
    work_a[mat_idx(k, k, n)] = alpha;
    for i in (k + 1)..m {
        work_a[mat_idx(i, k, n)] /= v0;
    }
    for j in (k + 1)..n {
        let mut dot = work_a[mat_idx(k, j, n)];
        for i in (k + 1)..m {
            dot += work_a[mat_idx(i, k, n)] * work_a[mat_idx(i, j, n)];
        }
        let factor = tau * dot;
        work_a[mat_idx(k, j, n)] -= factor;
        for i in (k + 1)..m {
            let ui = work_a[mat_idx(i, k, n)];
            work_a[mat_idx(i, j, n)] -= factor * ui;
        }
    }
    (tau, alpha)
}

fn bidiag_right_householder<T: Scalar>(work_a: &mut [T], m: usize, n: usize, k: usize) -> (T, T) {
    let r_tail = if k + 2 < n {
        row_tail_norm(work_a, n, k, k + 2)
    } else {
        T::ZERO
    };
    if r_tail == T::ZERO {
        return (T::ZERO, work_a[mat_idx(k, k + 1, n)]);
    }
    let y0 = work_a[mat_idx(k, k + 1, n)];
    let norm_y = row_tail_norm(work_a, n, k, k + 1);
    let beta = -norm_y.copysign(y0);
    let w0 = y0 - beta;
    let tau = -w0 / beta;
    work_a[mat_idx(k, k + 1, n)] = beta;
    for j in (k + 2)..n {
        work_a[mat_idx(k, j, n)] /= w0;
    }
    for i in (k + 1)..m {
        let mut dot = work_a[mat_idx(i, k + 1, n)];
        for j in (k + 2)..n {
            dot += work_a[mat_idx(i, j, n)] * work_a[mat_idx(k, j, n)];
        }
        let factor = tau * dot;
        work_a[mat_idx(i, k + 1, n)] -= factor;
        for j in (k + 2)..n {
            let vj = work_a[mat_idx(k, j, n)];
            work_a[mat_idx(i, j, n)] -= factor * vj;
        }
    }
    (tau, beta)
}

fn bidiagonalize_tall<T: Scalar>(
    work_a: &mut [T],
    m: usize,
    n: usize,
    tau_u: &mut [T],
    tau_v: &mut [T],
    d: &mut [T],
    e: &mut [T],
) {
    for k in 0..n {
        let (tu, dk) = bidiag_left_householder(work_a, m, n, k);
        tau_u[k] = tu;
        d[k] = dk;
        if k + 1 < n {
            let (tv, ek) = bidiag_right_householder(work_a, m, n, k);
            tau_v[k] = tv;
            e[k] = ek;
        }
    }
}

fn accumulate_bidiag_uv<T: Scalar>(
    work_a: &[T],
    m: usize,
    n: usize,
    tau_u: &[T],
    tau_v: &[T],
    u: &mut [T],
    v: &mut [T],
) {
    u.fill(T::ZERO);
    v.fill(T::ZERO);
    for i in 0..n {
        u[mat_idx(i, i, n)] = T::ONE;
        v[mat_idx(i, i, n)] = T::ONE;
    }

    for k in (0..n).rev() {
        let tk = tau_u[k];
        if tk == T::ZERO {
            continue;
        }
        for c in k..n {
            let mut dot = u[mat_idx(k, c, n)];
            for i in (k + 1)..m {
                dot += work_a[mat_idx(i, k, n)] * u[mat_idx(i, c, n)];
            }
            let factor = tk * dot;
            u[mat_idx(k, c, n)] -= factor;
            for i in (k + 1)..m {
                u[mat_idx(i, c, n)] -= factor * work_a[mat_idx(i, k, n)];
            }
        }
    }

    for k in (0..n.saturating_sub(2)).rev() {
        let tk = tau_v[k];
        if tk == T::ZERO {
            continue;
        }
        for c in (k + 1)..n {
            let mut dot = v[mat_idx(k + 1, c, n)];
            for i in (k + 2)..n {
                dot += work_a[mat_idx(k, i, n)] * v[mat_idx(i, c, n)];
            }
            let factor = tk * dot;
            v[mat_idx(k + 1, c, n)] -= factor;
            for i in (k + 2)..n {
                v[mat_idx(i, c, n)] -= factor * work_a[mat_idx(k, i, n)];
            }
        }
    }
}

struct BidiagQrState<'a, T> {
    d: &'a mut [T],
    e: &'a mut [T],
    u: &'a mut [T],
    v: &'a mut [T],
    m: usize,
    n: usize,
}

fn decouple_zero_diagonal<T: Scalar>(
    st: &mut BidiagQrState<'_, T>,
    p: usize,
    q: usize,
    tol: T,
) -> bool {
    for i in p..q {
        if st.d[i].abs() <= tol {
            st.d[i] = T::ZERO;
            let mut rv = st.e[i];
            st.e[i] = T::ZERO;
            for k in (i + 1)..=q {
                let (c, s, r) = givens_cs(st.d[k], rv);
                st.d[k] = r;
                if k < q {
                    rv = -s * st.e[k];
                    st.e[k] = c * st.e[k];
                }
                apply_col_givens(st.u, st.m, st.n, k, i, c, s);
            }
            return true;
        }
    }
    if st.d[q].abs() <= tol {
        st.d[q] = T::ZERO;
        let mut rv = st.e[q - 1];
        st.e[q - 1] = T::ZERO;
        for k in (p..q).rev() {
            let (c, s, r) = givens_cs(st.d[k], rv);
            st.d[k] = r;
            if k > p {
                rv = -s * st.e[k - 1];
                st.e[k - 1] = c * st.e[k - 1];
            }
            apply_col_givens(st.v, st.n, st.n, k, q, c, s);
        }
        return true;
    }
    false
}

fn golub_kahan_qr_step<T: Scalar>(st: &mut BidiagQrState<'_, T>, p: usize, q: usize) {
    let prev_e = if q >= p + 2 { st.e[q - 2] } else { T::ZERO };
    let t11 = st.d[q - 1] * st.d[q - 1] + prev_e * prev_e;
    let t22 = st.d[q] * st.d[q] + st.e[q - 1] * st.e[q - 1];
    let t12 = st.d[q - 1] * st.e[q - 1];
    let delta = T::HALF * (t11 - t22);
    let mu = if delta == T::ZERO {
        t22 - t12.abs()
    } else {
        let denom = delta + delta.hypot(t12).copysign(delta);
        t22 - (t12 * t12) / denom
    };

    let mut f = st.d[p] * st.d[p] - mu;
    let mut g = st.d[p] * st.e[p];

    for k in p..q {
        let (c, s, r) = givens_cs(f, g);
        if k > p {
            st.e[k - 1] = r;
        }
        f = c * st.d[k] + s * st.e[k];
        st.e[k] = -s * st.d[k] + c * st.e[k];
        g = s * st.d[k + 1];
        st.d[k + 1] = c * st.d[k + 1];
        apply_col_givens(st.v, st.n, st.n, k, k + 1, c, s);

        let (c2, s2, r2) = givens_cs(f, g);
        st.d[k] = r2;
        f = c2 * st.e[k] + s2 * st.d[k + 1];
        st.d[k + 1] = -s2 * st.e[k] + c2 * st.d[k + 1];
        if k + 1 < q {
            g = s2 * st.e[k + 1];
            st.e[k + 1] = c2 * st.e[k + 1];
        }
        apply_col_givens(st.u, st.m, st.n, k, k + 1, c2, s2);
    }
    st.e[q - 1] = f;
}

fn run_bidiagonal_qr<T: Scalar>(st: &mut BidiagQrState<'_, T>) -> Result<(), LinalgError> {
    let (m, n) = (st.m, st.n);
    if n <= 1 {
        if n == 1 && st.d[0] < T::ZERO {
            st.d[0] = -st.d[0];
            for r in 0..m {
                st.u[mat_idx(r, 0, 1)] = -st.u[mat_idx(r, 0, 1)];
            }
        }
        return Ok(());
    }

    let mut b_norm = T::ZERO;
    for i in 0..n {
        b_norm = b_norm.max(st.d[i].abs());
        if i + 1 < n {
            b_norm = b_norm.max(st.e[i].abs());
        }
    }
    let zero_tol = T::EPSILON * b_norm.max(T::MIN_POSITIVE);
    let max_iter = 75 * n;
    let mut iter = 0;

    while iter < max_iter {
        for i in 0..(n - 1) {
            let thresh = T::EPSILON * (st.d[i].abs() + st.d[i + 1].abs()) + zero_tol;
            if st.e[i].abs() <= thresh {
                st.e[i] = T::ZERO;
            }
        }
        let mut q = n - 1;
        while q > 0 && st.e[q - 1] == T::ZERO {
            q -= 1;
        }
        if q == 0 {
            break;
        }
        let mut p = q;
        while p > 0 && st.e[p - 1] != T::ZERO {
            p -= 1;
        }

        iter += 1;
        if !decouple_zero_diagonal(st, p, q, zero_tol) {
            golub_kahan_qr_step(st, p, q);
        }
    }

    if iter >= max_iter && (0..(n - 1)).any(|i| st.e[i] != T::ZERO) {
        return Err(LinalgError::NonConvergence);
    }

    for j in 0..n {
        if !st.d[j].is_finite() {
            return Err(LinalgError::NonConvergence);
        }
        if st.d[j] < T::ZERO {
            st.d[j] = -st.d[j];
            for r in 0..m {
                st.u[mat_idx(r, j, n)] = -st.u[mat_idx(r, j, n)];
            }
        }
    }
    Ok(())
}

fn svd_bidiagonal_tall<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    u: &mut [T],
    s: &mut [T],
    vt: &mut [T],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    let (work_a, mut r1) = ws.try_split(m * n)?;
    let (v_mat, mut r2) = r1.try_split(n * n)?;
    let (tau_u, mut r3) = r2.try_split(n)?;
    let (tau_v, mut r4) = r3.try_split(n)?;
    let (e, _) = r4.try_split(n)?;

    work_a.copy_from_slice(a);
    bidiagonalize_tall(work_a, m, n, tau_u, tau_v, s, e);
    accumulate_bidiag_uv(work_a, m, n, tau_u, tau_v, u, v_mat);
    let mut st = BidiagQrState {
        d: s,
        e,
        u,
        v: v_mat,
        m,
        n,
    };
    run_bidiagonal_qr(&mut st)?;
    sort_thin_svd(u, m, n, s, v_mat, n, vt);
    Ok(())
}

/// Computes the thin Singular Value Decomposition $A = U \operatorname{diag}(S) V^T$ via Golub-Kahan bidiagonalization and implicit QR shifts.
///
/// Output dimensions for $k = \min(m, n)$: `u` is `m x k`, `s` has length `k` (non-increasing), and `vt` is `k x n`.
/// `[cite:advanced_linear_algebra]`
pub fn svd_bidiagonal<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    u: &mut [T],
    s: &mut [T],
    vt: &mut [T],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    let k = m.min(n);
    if m.checked_mul(n) != Some(a.len())
        || m.checked_mul(k) != Some(u.len())
        || s.len() != k
        || k.checked_mul(n) != Some(vt.len())
    {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    if k == 0 {
        return Ok(());
    }
    if m >= n {
        return svd_bidiagonal_tall(a, m, n, u, s, vt, ws);
    }

    let (at, mut r1) = ws.try_split(n * m)?;
    let (u_at, mut r2) = r1.try_split(n * m)?;
    let (vt_at, mut inner_ws) = r2.try_split(m * m)?;
    for r in 0..m {
        for c in 0..n {
            at[mat_idx(c, r, m)] = a[mat_idx(r, c, n)];
        }
    }
    svd_bidiagonal_tall(at, n, m, u_at, s, vt_at, &mut inner_ws)?;
    for r in 0..m {
        for c in 0..m {
            u[mat_idx(r, c, m)] = vt_at[mat_idx(c, r, m)];
        }
    }
    for r in 0..m {
        for c in 0..n {
            vt[mat_idx(r, c, n)] = u_at[mat_idx(c, r, m)];
        }
    }
    Ok(())
}
