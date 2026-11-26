/// Result of certifying a single numerical solver under Tier 1 (MMS / grid refinement).
#[derive(Clone, Debug)]
pub struct SolverCertificate {
    /// Canonical solver identifier.
    pub name: &'static str,
    /// Owning workspace crate name.
    pub crate_name: &'static str,
    /// Minimum required asymptotic convergence order (or `0.0` for exact algebraic/contraction solvers).
    pub expected_min_order: f64,
    /// Observed log-log OLS convergence order (or contraction rate).
    pub observed_order: f64,
    /// Finest-grid or terminal $L^\infty$ / $L^2$ error residual.
    pub fine_error: f64,
    /// Maximum finest-grid error threshold allowed.
    pub max_allowed_error: f64,
    /// Whether both convergence order and error bound passed.
    pub passed: bool,
}

impl SolverCertificate {
    /// Constructs a [`SolverCertificate`] from observed convergence order and finest-grid error.
    #[must_use]
    pub fn evaluate(
        name: &'static str,
        crate_name: &'static str,
        expected_min_order: f64,
        observed_order: f64,
        fine_error: f64,
        max_allowed_error: f64,
    ) -> Self {
        let passed = observed_order.is_finite()
            && fine_error.is_finite()
            && observed_order >= expected_min_order
            && fine_error <= max_allowed_error;
        Self {
            name,
            crate_name,
            expected_min_order,
            observed_order,
            fine_error,
            max_allowed_error,
            passed,
        }
    }
}

/// Computes the ordinary least-squares slope $p$ of $\ln E_k = a + p \ln h_k$ across refinement levels.
#[must_use]
pub fn ols_convergence_order(spacings: &[f64], errors: &[f64]) -> f64 {
    assert_eq!(spacings.len(), errors.len());
    assert!(spacings.len() >= 2);
    let n = spacings.len() as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (&h, &e) in spacings.iter().zip(errors.iter()) {
        let lx = h.max(1e-300).ln();
        let ly = e.max(1e-300).ln();
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() <= 1e-30 {
        0.0
    } else {
        (n * sxy - sx * sy) / denom
    }
}

/// Computes discrete $L^2$ root-mean-square error $\sqrt{\frac{1}{N}\sum_i (a_i - b_i)^2}$.
#[must_use]
pub fn rms_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum();
    (sum_sq / a.len() as f64).sqrt()
}

/// Computes discrete $L^\infty$ maximum absolute error $\max_i |a_i - b_i|$.
#[must_use]
pub fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}
