//! UI presentation adapter over `math_commons::fit` zero-allocation curve solvers.

#[cfg(test)]
mod tests;

pub use math_commons::error::FitError;
pub use math_commons::fit::{
    CurveFit, FitModel, FitModel as FitType, MAX_LM_ITERATIONS, MAX_POLYNOMIAL_DEGREE,
};
pub use math_commons::series::{PointSeries, Sample2D};

thread_local! {
    static FIT_MODEL_WITH_AXES_CALL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Returns the total number of [`fit_model_with_axes`] solver executions on the current thread for regression diagnostics.
#[doc(hidden)]
#[must_use]
pub fn fit_model_with_axes_call_count() -> usize {
    FIT_MODEL_WITH_AXES_CALL_COUNT.with(std::cell::Cell::get)
}

/// Resets the [`fit_model_with_axes`] solver invocation counter on the current thread to zero.
#[doc(hidden)]
pub fn reset_fit_model_with_axes_call_count() {
    FIT_MODEL_WITH_AXES_CALL_COUNT.with(|c| c.set(0));
}

/// UI regression summary bundling fitted parameters, formula text, and residual stems.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// Selected canonical regression model preset.
    pub fit_type: FitType,
    /// Fitted model parameters (polynomial `[c0..cd]` or nonlinear model parameters).
    pub coefficients: Vec<f64>,
    /// Coefficient of determination ($R^2$).
    pub r_squared: f64,
    /// Root-mean-square error in fit space.
    pub rmse: f64,
    /// Vertical residuals `[x_i, y_i - y_hat(x_i)]` for valid fitted points.
    pub residuals: Vec<[f64; 2]>,
    /// Polynomial degree (`0` for nonlinear models).
    pub degree: usize,
    /// Mathematical equation string representation.
    pub formula: String,
    /// Name of the fitted dataset.
    pub dataset_name: String,
    /// Whether the independent variable was transformed via $\log_{10}(x)$.
    pub log_x: bool,
    /// Whether the dependent variable was transformed via $\log_{10}(y)$.
    pub log_y: bool,
    /// Whether transformed linearized fitting (Log Mode) was used.
    pub log_mode: bool,
    curve_fit: Option<CurveFit>,
}

impl FitResult {
    /// Builds a UI [`FitResult`] from a stack-solved [`CurveFit`] and source `points`.
    #[must_use]
    pub fn from_curve_fit<P: PointSeries + ?Sized>(
        curve: CurveFit,
        points: &P,
        dataset_name: &str,
    ) -> Self {
        Self {
            fit_type: curve.fit_model,
            coefficients: curve.coefficients().to_vec(),
            r_squared: curve.r_squared,
            rmse: curve.rmse,
            residuals: curve.residuals(points),
            degree: curve.degree,
            formula: curve.format_formula(),
            dataset_name: dataset_name.to_owned(),
            log_x: curve.log_x,
            log_y: curve.log_y,
            log_mode: curve.log_mode,
            curve_fit: Some(curve),
        }
    }

    /// Evaluates the fitted function at data-space coordinate `x`.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        if let Some(ref curve) = self.curve_fit {
            return curve.eval(x);
        }
        match self.fit_type {
            FitType::Linear | FitType::Quadratic | FitType::Cubic | FitType::Quartic => {
                if self.log_x && x <= 0.0 {
                    return f64::NAN;
                }
                let tx = if self.log_x { x.log10() } else { x };
                let ty = math_commons::fit::eval_polynomial_horner(&self.coefficients, tx);
                let out = if self.log_y {
                    10.0_f64.powf(ty.clamp(-300.0, 300.0)).clamp(1e-300, 1e300)
                } else {
                    ty
                };
                if out.is_finite() { out } else { f64::NAN }
            }
            FitType::Exponential
            | FitType::PowerLaw
            | FitType::Logarithmic
            | FitType::Logistic
            | FitType::Sinusoidal => {
                math_commons::fit::eval_nonlinear(self.fit_type, &self.coefficients, x)
            }
        }
    }
}

/// Computes least-squares polynomial regression (`1..=4`) over any [`PointSeries`].
pub fn fit_dataset<P: PointSeries + ?Sized>(
    points: &P,
    target_degree: usize,
    dataset_name: &str,
) -> Result<FitResult, FitError> {
    fit_dataset_scaled(points, target_degree, dataset_name, false, false)
}

/// Computes least-squares polynomial regression (`1..=4`) in linear, semilog, or log-log space.
pub fn fit_dataset_scaled<P: PointSeries + ?Sized>(
    points: &P,
    target_degree: usize,
    dataset_name: &str,
    log_x: bool,
    log_y: bool,
) -> Result<FitResult, FitError> {
    let curve = math_commons::fit::fit_polynomial_scaled(points, target_degree, log_x, log_y)?;
    Ok(FitResult::from_curve_fit(curve, points, dataset_name))
}

/// Fits any canonical [`FitType`] preset to `points`, returning typed [`FitError`] on failure.
pub fn fit_model<P: PointSeries + ?Sized>(
    points: &P,
    fit_type: FitType,
    dataset_name: &str,
    log_mode: bool,
) -> Result<FitResult, FitError> {
    let curve = math_commons::fit::fit_model(points, fit_type, log_mode)?;
    Ok(FitResult::from_curve_fit(curve, points, dataset_name))
}

/// Fits a canonical [`FitType`] respecting plot logarithmic axes for polynomial presets.
pub fn fit_model_with_axes<P: PointSeries + ?Sized>(
    points: &P,
    fit_type: FitType,
    dataset_name: &str,
    log_x: bool,
    log_y: bool,
    log_mode: bool,
) -> Result<FitResult, FitError> {
    FIT_MODEL_WITH_AXES_CALL_COUNT.with(|c| c.set(c.get() + 1));
    let curve = math_commons::fit::fit_model_with_axes(points, fit_type, log_mode, log_x, log_y)?;
    Ok(FitResult::from_curve_fit(curve, points, dataset_name))
}
