//! UI presentation adapter over `math_commons::fit` zero-allocation curve solvers and custom
//! callback-compiled regression models.

mod custom;
#[cfg(test)]
mod tests;

pub use custom::{
    CustomEvaluatorHandle, CustomFitCallback, CustomFitEvaluator, fit_custom_evaluator,
};
pub use math_commons::error::FitError;
pub use math_commons::fit::MAX_FIT_PARAMS;
use math_commons::fit::{CurveFit, FitModel};
use math_commons::series::PointSeries;

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

/// Supported curve fitting models for interactive plot overlays and programmatic regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum FitType {
    /// Degree-1 polynomial regression ($y = c_0 + c_1 x$).
    Linear,
    /// Degree-2 polynomial regression ($y = c_0 + c_1 x + c_2 x^2$).
    Quadratic,
    /// Degree-3 polynomial regression ($y = c_0 + c_1 x + c_2 x^2 + c_3 x^3$).
    Cubic,
    /// Degree-4 polynomial regression ($y = c_0 + c_1 x + c_2 x^2 + c_3 x^3 + c_4 x^4$).
    Quartic,
    /// Exponential model ($y = a e^{b x}$).
    Exponential,
    /// Power-law model ($y = a x^b$, $x > 0$).
    PowerLaw,
    /// Logarithmic model ($y = a + b \ln(x)$, $x > 0$).
    Logarithmic,
    /// 3-parameter logistic curve ($y = L / (1 + e^{-k(x - x_0)})$).
    Logistic,
    /// 4-parameter sinusoidal wave ($y = A \sin(B x + C) + D$).
    Sinusoidal,
    /// Callback-compiled custom formula ($0 \le k \le 6$ parameters).
    Custom,
}

impl FitType {
    /// Ordered list of canonical regression models for UI menus and batch sweeps.
    pub const MENU_PRESETS: [Self; 9] = [
        Self::Linear,
        Self::Quadratic,
        Self::Cubic,
        Self::Quartic,
        Self::Exponential,
        Self::PowerLaw,
        Self::Logarithmic,
        Self::Logistic,
        Self::Sinusoidal,
    ];

    /// Converts a canonical [`FitModel`] into its [`FitType`] counterpart.
    #[must_use]
    pub const fn from_fit_model(model: FitModel) -> Self {
        match model {
            FitModel::Linear => Self::Linear,
            FitModel::Quadratic => Self::Quadratic,
            FitModel::Cubic => Self::Cubic,
            FitModel::Quartic => Self::Quartic,
            FitModel::Exponential => Self::Exponential,
            FitModel::PowerLaw => Self::PowerLaw,
            FitModel::Logarithmic => Self::Logarithmic,
            FitModel::Logistic => Self::Logistic,
            FitModel::Sinusoidal => Self::Sinusoidal,
        }
    }

    /// Maps a preset [`FitType`] to its [`FitModel`], returning `None` for [`Self::Custom`].
    #[must_use]
    pub const fn to_fit_model(self) -> Option<FitModel> {
        match self {
            Self::Linear => Some(FitModel::Linear),
            Self::Quadratic => Some(FitModel::Quadratic),
            Self::Cubic => Some(FitModel::Cubic),
            Self::Quartic => Some(FitModel::Quartic),
            Self::Exponential => Some(FitModel::Exponential),
            Self::PowerLaw => Some(FitModel::PowerLaw),
            Self::Logarithmic => Some(FitModel::Logarithmic),
            Self::Logistic => Some(FitModel::Logistic),
            Self::Sinusoidal => Some(FitModel::Sinusoidal),
            Self::Custom => None,
        }
    }

    /// Human-readable menu label describing the model equation.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self.to_fit_model() {
            Some(model) => model.label(),
            None => "Custom Formula f(x)",
        }
    }

    /// Compact badge label for legends and plot callouts.
    #[must_use]
    pub const fn short_label(self) -> &'static str {
        match self.to_fit_model() {
            Some(model) => model.short_label(),
            None => "Custom",
        }
    }

    /// Whether this preset supports linearized log-space fitting (`Exponential`, `PowerLaw`).
    #[must_use]
    pub const fn supports_log_mode(self) -> bool {
        match self.to_fit_model() {
            Some(model) => model.supports_log_mode(),
            None => false,
        }
    }
}

impl From<FitModel> for FitType {
    fn from(model: FitModel) -> Self {
        Self::from_fit_model(model)
    }
}

impl std::fmt::Display for FitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// UI regression summary bundling fitted parameters, formula text, and residual stems.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// Selected regression model (`FitModel` preset or [`FitType::Custom`]).
    pub fit_type: FitType,
    /// Fitted model parameters (polynomial `[c0..cd]`, nonlinear preset, or custom parameters).
    pub coefficients: Vec<f64>,
    /// Coefficient of determination ($R^2$).
    pub r_squared: f64,
    /// Root-mean-square error in fit space.
    pub rmse: f64,
    /// Vertical residuals `[x_i, y_i - y_hat(x_i)]` for valid fitted points.
    pub residuals: crate::SharedPoints,
    /// Polynomial degree (`0` for nonlinear and custom models).
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
    /// Optional compiled callback evaluator when `fit_type == FitType::Custom`.
    pub custom_evaluator: Option<CustomEvaluatorHandle>,
    pub(crate) curve_fit: Option<CurveFit>,
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
            fit_type: FitType::from_fit_model(curve.fit_model),
            coefficients: curve.coefficients().to_vec(),
            r_squared: curve.r_squared,
            rmse: curve.rmse,
            residuals: crate::SharedPoints::from(curve.residuals(points)),
            degree: curve.degree,
            formula: curve.format_formula(),
            dataset_name: dataset_name.to_owned(),
            log_x: curve.log_x,
            log_y: curve.log_y,
            log_mode: curve.log_mode,
            custom_evaluator: None,
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
        let Some(model) = self.fit_type.to_fit_model() else {
            return self
                .custom_evaluator
                .as_ref()
                .and_then(|h| h.0.eval(x, &self.coefficients))
                .unwrap_or(f64::NAN);
        };
        match model {
            FitModel::Linear | FitModel::Quadratic | FitModel::Cubic | FitModel::Quartic => {
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
            FitModel::Exponential
            | FitModel::PowerLaw
            | FitModel::Logarithmic
            | FitModel::Logistic
            | FitModel::Sinusoidal => {
                math_commons::fit::eval_nonlinear(model, &self.coefficients, x)
            }
        }
    }
}

/// Fits a canonical [`FitType`] respecting plot logarithmic axes for polynomial presets and `log_mode` for nonlinear presets.
pub fn fit_model_with_axes<P: PointSeries + ?Sized>(
    points: &P,
    fit_type: FitType,
    dataset_name: &str,
    log_x: bool,
    log_y: bool,
    log_mode: bool,
) -> Result<FitResult, FitError> {
    FIT_MODEL_WITH_AXES_CALL_COUNT.with(|c| c.set(c.get() + 1));
    let Some(model) = fit_type.to_fit_model() else {
        return Err(FitError::InvalidDomain {
            reason: "custom formula requires fit_custom_evaluator",
        });
    };
    let curve = math_commons::fit::fit_model_with_axes(points, model, log_mode, log_x, log_y)?;
    Ok(FitResult::from_curve_fit(curve, points, dataset_name))
}
