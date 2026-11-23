use crate::error::EngineeringError;
use math_commons::error::MathError;
use nalgebra::{DMatrix, DVector, RealField};

fn check_r_invertible<T: RealField>(
    r: &DMatrix<T>,
    m: usize,
    p: usize,
) -> Result<(), EngineeringError> {
    let scale: T = nalgebra::convert((m.max(p) * 64) as f64);
    let max_diag = (0..p)
        .map(|i| r[(i, i)].clone().abs())
        .fold(T::zero(), |a, b| if a > b { a } else { b });
    let tol = T::default_epsilon() * scale * max_diag.clone();
    if max_diag == T::zero() || (0..p).any(|i| r[(i, i)].clone().abs() <= tol) {
        return Err(EngineeringError::SingularDesignMatrix);
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
/// The estimated beta parameters, or an [`EngineeringError`] if inputs are non-finite, dimensions mismatch, or `X` is rank-deficient.
#[verified_engine::verified]
pub fn estimate_beta<T: RealField>(
    x: &DMatrix<T>,
    y: &DVector<T>,
) -> Result<DVector<T>, EngineeringError> {
    if !x.iter().all(|v| v.is_finite()) || !y.iter().all(|v| v.is_finite()) {
        return Err(EngineeringError::Math(MathError::NumericalError {
            reason: "GLM input contains non-finite values".to_string(),
        }));
    }
    let (m, p) = (x.nrows(), x.ncols());
    if m != y.len() {
        return Err(EngineeringError::DimensionMismatch {
            context: "Observation vector y length must match design matrix row count",
            expected: m,
            actual: y.len(),
        });
    }
    if m == 0 || p == 0 {
        return Err(EngineeringError::DimensionMismatch {
            context: "Design matrix dimensions must be non-empty",
            expected: 1,
            actual: 0,
        });
    }
    if m < p {
        return Err(EngineeringError::SingularDesignMatrix);
    }
    let (q, r) = x.clone().qr().unpack();
    check_r_invertible(&r, m, p)?;
    let qty = q.transpose() * y;
    r.solve_upper_triangular(&qty)
        .ok_or(EngineeringError::SingularDesignMatrix)
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
/// The t-statistic, or an [`EngineeringError`] if matrices are non-finite, non-conformable, or rank-deficient.
#[verified_engine::verified]
pub fn t_statistic<T: RealField + Copy>(
    c: &DVector<T>,
    beta: &DVector<T>,
    x: &DMatrix<T>,
    residual_variance: T,
) -> Result<T, EngineeringError> {
    if !residual_variance.is_finite()
        || !c.iter().all(|v| v.is_finite())
        || !beta.iter().all(|v| v.is_finite())
        || !x.iter().all(|v| v.is_finite())
    {
        return Err(EngineeringError::Math(MathError::NumericalError {
            reason: "GLM contrast input contains non-finite values".to_string(),
        }));
    }
    let (m, p) = (x.nrows(), x.ncols());
    if c.len() != p {
        return Err(EngineeringError::DimensionMismatch {
            context: "Contrast vector c length must match design matrix predictor count",
            expected: p,
            actual: c.len(),
        });
    }
    if beta.len() != p {
        return Err(EngineeringError::DimensionMismatch {
            context: "Parameter vector beta length must match design matrix predictor count",
            expected: p,
            actual: beta.len(),
        });
    }
    if m == 0 || p == 0 || m < p {
        return Err(EngineeringError::SingularDesignMatrix);
    }

    let r = x.clone().qr().unpack_r();
    check_r_invertible(&r, m, p)?;
    let w = r
        .transpose()
        .solve_lower_triangular(c)
        .ok_or(EngineeringError::SingularDesignMatrix)?;

    let numerator = c.dot(beta);
    let variance_term = w.dot(&w);
    let scaled_variance = residual_variance * variance_term;
    if scaled_variance <= T::zero() {
        return Err(EngineeringError::ZeroContrastVariance);
    }
    let denominator = scaled_variance.sqrt();
    if denominator == T::zero() || !denominator.is_finite() {
        return Err(EngineeringError::ZeroContrastVariance);
    }

    Ok(numerator / denominator)
}
