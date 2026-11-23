use nalgebra::{DMatrix, DVector, RealField};

fn check_r_invertible<T: RealField>(
    r: &DMatrix<T>,
    m: usize,
    p: usize,
) -> Result<(), &'static str> {
    let scale: T = nalgebra::convert((m.max(p) * 64) as f64);
    let max_diag = (0..p)
        .map(|i| r[(i, i)].clone().abs())
        .fold(T::zero(), |a, b| if a > b { a } else { b });
    let tol = T::default_epsilon() * scale * max_diag.clone();
    if max_diag == T::zero() || (0..p).any(|i| r[(i, i)].clone().abs() <= tol) {
        return Err("Design matrix is rank-deficient or singular");
    }
    Ok(())
}

/// Estimates the beta parameters of the GLM using ordinary least squares via thin Householder QR.
///
/// Solves $R \hat{\beta} = Q^T Y$ directly from $X = QR$ without forming $X^T X$, avoiding
/// $\kappa(X)^2$ condition-number squaring on nearly collinear design matrices.
///
/// # Arguments
/// * `x` - The design matrix (subjects x predictors).
/// * `y` - The data vector (e.g., thickness at a vertex for all subjects).
///
/// # Returns
/// The estimated beta parameters, or an error if inputs are non-finite, non-conformable, or rank-deficient.
#[verified_engine::verified]
pub fn estimate_beta<T: RealField>(
    x: &DMatrix<T>,
    y: &DVector<T>,
) -> Result<DVector<T>, &'static str> {
    if !x.iter().all(|v| v.is_finite()) || !y.iter().all(|v| v.is_finite()) {
        return Err("Input contains non-finite values");
    }
    let (m, p) = (x.nrows(), x.ncols());
    if m == 0 || p == 0 || m != y.len() || m < p {
        return Err("Incompatible design matrix and observation dimensions");
    }
    let (q, r) = x.clone().qr().unpack();
    check_r_invertible(&r, m, p)?;
    let qty = q.transpose() * y;
    r.solve_upper_triangular(&qty)
        .ok_or("R factor is not invertible")
}

/// Calculates the t-statistic for a given contrast via the QR $R$-factor:
/// $$t = \frac{c^T \hat{\beta}}{\sqrt{\hat{\sigma}^2 \, \|R^{-T} c\|_2^2}}$$
///
/// Uses $X^T X = R^T R$ so $c^T (X^T X)^{-1} c = \|R^{-T} c\|_2^2$ is evaluated by forward
/// substitution on $R^T w = c$ without forming or inverting $X^T X$.
///
/// # Arguments
/// * `c` - The contrast vector.
/// * `beta` - The estimated beta parameters.
/// * `x` - The design matrix.
/// * `residual_variance` - $\hat{\sigma}^2$, the variance of the residuals ($Y - X\hat{\beta}$).
///
/// # Returns
/// The t-statistic, or an error if matrices are non-finite, non-conformable, or rank-deficient.
#[verified_engine::verified]
pub fn t_statistic<T: RealField + Copy>(
    c: &DVector<T>,
    beta: &DVector<T>,
    x: &DMatrix<T>,
    residual_variance: T,
) -> Result<T, &'static str> {
    if !residual_variance.is_finite()
        || residual_variance < T::zero()
        || !c.iter().all(|v| v.is_finite())
        || !beta.iter().all(|v| v.is_finite())
        || !x.iter().all(|v| v.is_finite())
    {
        return Err("Input contains non-finite or negative variance values");
    }
    let (m, p) = (x.nrows(), x.ncols());
    if m == 0 || p == 0 || m < p || c.len() != p || beta.len() != p {
        return Err("Incompatible contrast, parameter, or design matrix dimensions");
    }

    let r = x.clone().qr().unpack_r();
    check_r_invertible(&r, m, p)?;
    let w = r
        .transpose()
        .solve_lower_triangular(c)
        .ok_or("R^T factor is not invertible")?;

    let numerator = c.dot(beta);
    let variance_term = w.dot(&w);
    let denominator = (residual_variance * variance_term).sqrt();

    if denominator == T::zero() || !denominator.is_finite() {
        return Err("Denominator is zero, t-statistic is undefined.");
    }

    Ok(numerator / denominator)
}
